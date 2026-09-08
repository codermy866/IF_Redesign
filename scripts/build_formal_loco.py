#!/usr/bin/env python3
"""Build the complete multimodal cohort and deterministic five-centre LOCO folds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cervix_cogalign.io import (  # noqa: E402
    build_contact_sheet,
    evenly_spaced,
    existing_paths,
    read_json,
    select_colposcopy,
    stable_score,
    write_jsonl,
)
from cervix_cogalign.prompts import SYSTEM_PROMPT, diagnostic_prompt, evidence_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/formal_v1.json"))
    parser.add_argument("--rebuild-panels", action="store_true")
    parser.add_argument("--asset-workers", type=int, default=8)
    return parser.parse_args()


def prepare_assets(task: tuple[dict, str, dict, bool]) -> dict:
    raw, output_text, config, rebuild = task
    output = Path(output_text)
    case_id = str(raw["case_id"])
    centre_slug = config["centres"][raw["hospital_name"]]
    asset_dir = output / "assets" / centre_slug
    colposcopy_selected = select_colposcopy(
        raw["colposcopy_existing"], int(config["max_colposcopy_images"])
    )
    oct_selected = evenly_spaced(sorted(raw["oct_existing"]), int(config["max_oct_images"]))
    colposcopy_panel = asset_dir / f"{case_id}_colposcopy.jpg"
    oct_panel = asset_dir / f"{case_id}_oct.jpg"
    if rebuild or not colposcopy_panel.is_file():
        build_contact_sheet(
            colposcopy_selected,
            colposcopy_panel,
            tile_size=int(config["panel_tile_size"]),
            columns=2,
        )
    if rebuild or not oct_panel.is_file():
        build_contact_sheet(
            oct_selected,
            oct_panel,
            tile_size=int(config["panel_tile_size"]),
            columns=3,
        )
    return {
        "case_id": case_id,
        "center_slug": centre_slug,
        "colposcopy_available": len(raw["colposcopy_existing"]),
        "oct_available": len(raw["oct_existing"]),
        "colposcopy_panel": str(colposcopy_panel.resolve()),
        "oct_panel": str(oct_panel.resolve()),
        "colposcopy_selection_digest": f"{stable_score(*colposcopy_selected):016x}",
        "oct_selection_digest": f"{stable_score(*oct_selected):016x}",
    }


def model_row(raw: dict, prompt: str, images: list[str], centre_slug: str) -> dict:
    return {
        "id": str(raw["case_id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>\n<image>\n" + prompt},
        ],
        "images": images,
        "label": int(raw["binary_label"]),
        "patient_id": f"p_{stable_score(raw['patient_id']):016x}",
        "center": str(raw["hospital_name"]),
        "center_slug": centre_slug,
        "split": "all",
    }


def select_inner_validation(source: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    selected: set[str] = set()
    for (_, _), part in source.groupby(["hospital_name", "binary_label"], sort=True):
        ordered = part.assign(
            _order=[stable_score(case, "inner_val", seed=seed) for case in part.case_id]
        ).sort_values("_order")
        requested = max(1, round(len(ordered) * fraction))
        requested = min(requested, len(ordered) - 1)
        if requested < 1:
            raise ValueError("Every source centre/label stratum needs at least two cases")
        selected.update(ordered.head(requested).case_id.astype(str))
    return selected


def with_split(rows: list[dict], split: str, fold: str) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        item["outer_fold"] = fold
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(config["manifest"])
    frame["colposcopy_existing"] = frame.colposcopy_paths.map(existing_paths)
    frame["oct_existing"] = frame.oct_paths.map(existing_paths)
    eligible = frame[
        frame.colposcopy_existing.map(bool) & frame.oct_existing.map(bool)
    ].copy()
    eligible["case_id"] = eligible.case_id.astype(str)
    if eligible.patient_id.duplicated().any():
        raise AssertionError("Formal complete-modality cohort must contain one row per patient")
    unknown = set(eligible.hospital_name) - set(config["centres"])
    if unknown:
        raise ValueError(f"Centres missing from configuration: {sorted(unknown)}")

    records = eligible.to_dict(orient="records")
    total = len(eligible)
    tasks = [(raw, str(output), config, args.rebuild_panels) for raw in records]
    asset_by_id = {}
    with ProcessPoolExecutor(max_workers=args.asset_workers) as executor:
        for index, asset in enumerate(executor.map(prepare_assets, tasks), start=1):
            asset_by_id[asset["case_id"]] = asset
            if index % 50 == 0 or index == total:
                print(f"panels {index}/{total}", flush=True)

    baseline_by_id: dict[str, dict] = {}
    evidence_by_id: dict[str, dict] = {}
    audit_rows = []
    for raw in records:
        case_id = str(raw["case_id"])
        centre_slug = config["centres"][raw["hospital_name"]]
        asset = asset_by_id[case_id]
        prompt_values = {"age": raw.get("age"), "hpv": raw.get("hpv"), "tct": raw.get("tct")}
        images = [asset["colposcopy_panel"], asset["oct_panel"]]
        baseline_by_id[case_id] = model_row(
            raw, diagnostic_prompt(prompt_values), images, centre_slug
        )
        evidence_by_id[case_id] = model_row(
            raw, evidence_prompt(prompt_values), images, centre_slug
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "patient_id": f"p_{stable_score(raw['patient_id']):016x}",
                "center": raw["hospital_name"],
                "center_slug": centre_slug,
                "label": int(raw["binary_label"]),
                "colposcopy_available": asset["colposcopy_available"],
                "oct_available": asset["oct_available"],
                "colposcopy_panel": asset["colposcopy_panel"],
                "oct_panel": asset["oct_panel"],
                "colposcopy_selection_digest": asset["colposcopy_selection_digest"],
                "oct_selection_digest": asset["oct_selection_digest"],
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(["center_slug", "case_id"])
    audit.to_csv(output / "cohort_audit.csv", index=False)
    ordered_ids = audit.case_id.tolist()
    write_jsonl(output / "all" / "baseline.jsonl", [baseline_by_id[x] for x in ordered_ids])
    write_jsonl(output / "all" / "evidence_prompts.jsonl", [evidence_by_id[x] for x in ordered_ids])

    fold_summaries = []
    assignment_rows = []
    fraction = float(config["inner_validation_fraction"])
    seed = int(config["seed"])
    for held_center, fold in config["centres"].items():
        test = eligible[eligible.hospital_name == held_center]
        source = eligible[eligible.hospital_name != held_center]
        validation_ids = select_inner_validation(source, fraction, seed)
        split_ids = {
            "train": set(source.case_id.astype(str)) - validation_ids,
            "val": validation_ids,
            "test": set(test.case_id.astype(str)),
        }
        if any(split_ids[a] & split_ids[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
            raise AssertionError(f"Overlap in fold {fold}")
        if set().union(*split_ids.values()) != set(eligible.case_id.astype(str)):
            raise AssertionError(f"Incomplete fold {fold}")
        fold_dir = output / "folds" / fold
        for split, identifiers in split_ids.items():
            ids = sorted(identifiers, key=lambda value: stable_score(value, fold, seed=seed))
            write_jsonl(
                fold_dir / "baseline" / f"{split}.jsonl",
                with_split([baseline_by_id[x] for x in ids], split, fold),
            )
            write_jsonl(
                fold_dir / "evidence_prompts" / f"{split}.jsonl",
                with_split([evidence_by_id[x] for x in ids], split, fold),
            )
            selected = eligible[eligible.case_id.astype(str).isin(ids)]
            for raw in selected.to_dict(orient="records"):
                assignment_rows.append(
                    {
                        "fold": fold,
                        "held_out_center": held_center,
                        "case_id": str(raw["case_id"]),
                        "patient_id": f"p_{stable_score(raw['patient_id']):016x}",
                        "center": raw["hospital_name"],
                        "split": split,
                        "label": int(raw["binary_label"]),
                    }
                )
            counts = selected.binary_label.value_counts().to_dict()
            fold_summaries.append(
                {
                    "fold": fold,
                    "held_out_center": held_center,
                    "split": split,
                    "n": len(selected),
                    "negative": int(counts.get(0, 0)),
                    "positive": int(counts.get(1, 0)),
                    "positive_rate": float(selected.binary_label.mean()),
                }
            )

    pd.DataFrame(assignment_rows).to_csv(output / "fold_assignments.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(output / "fold_summary.csv", index=False)
    label_disagreements = None
    old_index = ROOT.parent / "02_original_result_evidence/links/locked_splits/full_multimodal_resplit/final_1897_case_index.csv"
    if old_index.is_file():
        old = pd.read_csv(old_index)
        joined = old.merge(frame[["case_id", "binary_label"]], left_on="oct_id", right_on="case_id")
        label_disagreements = int((joined.label.astype(int) != joined.binary_label.astype(int)).sum())
    formal_manifest = {
        "experiment_id": config["experiment_id"],
        "endpoint": config["endpoint"],
        "eligible_cases": len(eligible),
        "eligible_patients": int(eligible.patient_id.nunique()),
        "centres": config["centres"],
        "outer_folds": len(config["centres"]),
        "old_index_label_disagreements": label_disagreements,
        "prohibited_supervision_columns": [
            "pseudo_report_text",
            "training_report_text",
            "reference_diagnostic_summary",
        ],
        "test_policy": "each held-out centre is opened only by its fold-specific frozen model",
    }
    with (output / "formal_manifest.json").open("w", encoding="utf-8") as sink:
        json.dump(formal_manifest, sink, ensure_ascii=False, indent=2)
    print(json.dumps(formal_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
