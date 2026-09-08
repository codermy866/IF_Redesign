"""Build private multimodal SFT/DPO data from audited existing OOF artifacts."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from cervix_cogalign.evidence_alignment import clinical_prompt, chain_target, counterfactual_pair, preference_text, validate_oof

ROOT = Path(__file__).resolve().parents[1]
IMAGE_POOL = ThreadPoolExecutor(max_workers=4)


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def strip_image(unit, cfg, output):
    paths = [unit["frame_paths"][i] for i in cfg["frames_per_site"]]
    if not all(Path(p).is_file() for p in paths):
        raise FileNotFoundError("Missing OCT input frame")
    key = hashlib.sha256(json.dumps([paths, cfg["tile_size"]]).encode()).hexdigest()
    out = output / "images" / f"{key}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        size = cfg["tile_size"]
        canvas = Image.new("RGB", (size * len(paths), size))
        for i, p in enumerate(paths):
            with Image.open(p) as im:
                canvas.paste(im.convert("RGB").resize((size, size)), (i * size, 0))
        canvas.save(out)
    return str(out)


def build(cfg, folds=None, seeds=None, max_cases=None):
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    cache = torch.load(ROOT / cfg["features"], map_location="cpu", weights_only=False)
    cases = [json.loads(line) for line in (ROOT / cfg["manifest"]).read_text().splitlines()]
    side = np.load(ROOT / cfg["clinical_sidecar"], allow_pickle=False)
    labels = np.load(ROOT / cfg["site_labels"], allow_pickle=False)
    assert cache["ids"] == [c["id"] for c in cases] == list(side["ids"]) == list(labels["ids"])
    assert len(set(cache["patient_ids"])) == len(cases), "Repeated patients need grouped splitting"
    assignments = list(csv.DictReader((ROOT / cfg["fold_assignments"]).open()))
    statuses = []
    for fold in folds or cfg["folds"]:
        mapping = {r["case_id"]: r["split"] for r in assignments if r["fold"] == fold}
        groups = {s: {cases[i]["patient_id"] for i in range(len(cases)) if mapping[cases[i]["id"]] == s}
                  for s in ("train", "val", "test")}
        assert not groups["train"] & (groups["val"] | groups["test"]) and not groups["val"] & groups["test"]
        for seed in seeds or cfg["seeds"]:
            gain_dir = ROOT / cfg["gain_root"] / fold / f"seed_{seed}"
            if not (gain_dir / "gain_targets.npz").exists():
                statuses.append(dict(fold=fold, seed=seed, status="blocked_missing_oof_targets"))
                continue
            targets = np.load(gain_dir / "gain_targets.npz", allow_pickle=False)
            # Only step 0 is a common clinical+non-OCT context. Never average
            # gains over incompatible acquired-evidence states.
            gain_map = {(int(i), int(slot)-25+1): float(g) for (i, step, slot), g in zip(targets["rows"], targets["oof_gain"]) if step == 0}
            provenance = json.loads((gain_dir / "target_provenance.json").read_text())
            owner = {p["patient_index"]: p["teacher"] for p in provenance if p["kind"] == "oof"}
            memberships = {k: json.loads((gain_dir / f"teacher_{k}/complete.json").read_text()) for k in set(owner.values())}
            parts = {s: [] for s in ("train", "val", "test")}
            preferences = []
            skipped = 0
            for i, case in enumerate(cases):
                if max_cases and i >= max_cases:
                    break
                if i % 100 == 0:
                    print(json.dumps(dict(fold=fold, seed=seed, processed=i)), flush=True)
                part = mapping[case["id"]]
                if part == "train":
                    if i not in owner:
                        skipped += 1
                        continue
                    validate_oof(i, memberships[owner[i]])
                    if any((i, s) not in gain_map for s in range(1, 13)):
                        raise ValueError("Incomplete 12-site OOF utility")
                units = sorted((u for u in case["evidence_units"] if u["kind"] == "oct_position_cluster"), key=lambda u: u["scanner_position"][0])
                assert [u["scanner_position"][0] for u in units] == list(range(1, 13))
                # Label-independent candidate budget shared by all four VLM arms.
                local_seed = int(hashlib.sha256(f"{case['id']}:{seed}".encode()).hexdigest()[:8], 16)
                sites = np.random.default_rng(local_seed).choice(np.arange(1, 13), cfg["evidence_budget"], replace=False).tolist()
                images = list(IMAGE_POOL.map(lambda s: strip_image(units[s-1], cfg, out), sites))
                clinical = {"HPV": str(side["hpvs"][i]), "TCT": str(side["tcts"][i]), "age": str(cache["ages"][i])}
                row = dict(id=hashlib.sha256(case["id"].encode()).hexdigest(), index=i, split=part,
                           center=case["center"], sites=sites, images=images,
                           prompt=clinical_prompt(clinical, sites), label=int(cache["labels"][i]),
                           feature_ref=dict(path=str(ROOT / cfg["features"]), index=i, slots=[s+24 for s in sites]),
                           supervision_scope="site-level weak labels, OOF model utility; morphology unreviewed")
                negative_donors = [s for s in range(1,13) if labels["labels"][i,s-1] == 0 and s not in sites]
                if negative_donors:
                    donor = negative_donors[0]
                    row["audit_normal_donor"] = dict(site=donor, image=strip_image(units[donor-1], cfg, out),
                                                     scope="label-informed stress audit only; never primary input")
                if part == "train":
                    gains = [gain_map[i, s] for s in sites]
                    row.update(gains=gains, target=chain_target(sites, labels["labels"][i, np.array(sites)-1], gains, row["label"]), teacher=owner[i])
                    ordinary = json.loads(row["target"])
                    for evidence in ordinary["evidence"]:
                        evidence["importance"] = "unassessed"
                    row["sft_target"] = json.dumps(ordinary)
                    cf = counterfactual_pair(sites, gains, labels["labels"][i], list(range(1, 13)))
                    if cf:
                        for intervention in ("deletion", "replacement"):
                            if intervention == "replacement" and cf["donor"] is None:
                                continue
                            cf_sites = [s for s in sites if s != cf["removed"]]
                            cf_images = [img for s, img in zip(sites, images) if s != cf["removed"]]
                            if intervention == "replacement":
                                cf_sites.append(cf["donor"])
                                cf_images.append(strip_image(units[cf["donor"]-1], cfg, out))
                            chosen, rejected = preference_text(cf["removed"], intervention)
                            preferences.append(dict(id=row["id"], index=i, images=cf_images, sites=cf_sites,
                                prompt=clinical_prompt(clinical, cf_sites) + f" Original site {cf['removed']} evidence underwent {intervention}. Audit whether that original evidence may still be cited.",
                                chosen=chosen, rejected=rejected, intervention=intervention,
                                removed=cf["removed"], donor=cf["donor"] if intervention == "replacement" else None,
                                counterfactual_pathology_known=False, label_used_for_cf_diagnosis=False))
                        if cf["donor"] is not None:
                            # Balanced visual grounding pairs: exactly identical text
                            # and slot IDs, opposite preferences, changed pixels only.
                            # Donor acquisition position is metadata, not model input.
                            audit_prompt = row["prompt"] + f" Assess whether the image currently occupying site slot {cf['removed']} supports a positive site-level OCT finding. Do not infer patient pathology from a single site."
                            positive = f"The image at site slot {cf['removed']} supports a positive site-level OCT finding. Patient diagnosis requires the remaining evidence and clinical information."
                            negative = f"The image at site slot {cf['removed']} does not support a positive site-level OCT finding. Patient diagnosis requires the remaining evidence and clinical information."
                            original = dict(id=row["id"], index=i, images=images, sites=sites, prompt=audit_prompt,
                                chosen=positive,rejected=negative,intervention="matched_original",removed=cf["removed"],
                                counterfactual_pathology_known=False,label_used_for_cf_diagnosis=False,anchor_id=row["id"])
                            replaced_images = list(images)
                            replaced_images[sites.index(cf["removed"])] = strip_image(units[cf["donor"]-1],cfg,out)
                            replaced = dict(original, images=replaced_images, chosen=negative, rejected=positive,
                                            intervention="matched_replacement", donor=cf["donor"])
                            preferences.extend([original,replaced])
                parts[part].append(row)
            destination = out / "datasets" / fold / f"seed_{seed}"
            for part, rows in parts.items():
                dump_jsonl(destination / f"{part}.jsonl", rows)
            dump_jsonl(destination / "dpo.jsonl", preferences)
            audit = dict(fold=fold, seed=seed, status="ready", counts={k: len(v) for k, v in parts.items()},
                         schema_version=2,
                         preference_pairs=len(preferences), train_missing_oof=skipped,
                         candidate_policy="seeded label-independent fixed-budget sites shared across VLM arms",
                         gain_context="step0 clinical plus colposcopy; NOT OCT-only teacher utility",
                         oof_target_sha256=digest(gain_dir / "gain_targets.npz"),
                         teacher_provenance_sha256=digest(gain_dir / "target_provenance.json"),
                         clinical_sha256=digest(ROOT / cfg["clinical_sidecar"]),
                         limitations=["Site utility transferred to three representative frames, not per-frame ground truth",
                                      "Within-patient replacement preserves patient/session but not identical pixel background",
                                      "Preference pairs supervise citation validity, not counterfactual pathology",
                                      "Morphology is unverified; no expert-level reasoning claim"])
            (destination / "audit.json").write_text(json.dumps(audit, indent=2))
            statuses.append(audit)
            print(json.dumps({k: audit[k] for k in ("fold", "seed", "counts", "preference_pairs")}), flush=True)
    (out / "dataset_status.json").write_text(json.dumps(statuses, indent=2))
    return statuses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dea_v1.json")
    parser.add_argument("--fold", nargs="+")
    parser.add_argument("--seed", nargs="+", type=int)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.max_cases:
        cfg["output"] += "_smoke_data_v2"
        cfg["gradient_accumulation"] = 1
        config_out = ROOT / cfg["output"] / "smoke_config.json"
        config_out.parent.mkdir(parents=True, exist_ok=True)
        config_out.write_text(json.dumps(cfg, indent=2))
    build(cfg, args.fold, args.seed, args.max_cases)
