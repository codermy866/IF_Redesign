#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Build a leakage-aware multimodal pilot dataset")
    parser.add_argument("--config", default=str(ROOT / "configs/pilot_v0.json"))
    parser.add_argument("--rebuild-panels", action="store_true")
    return parser.parse_args()


def select_cohort(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["colposcopy_existing"] = df["colposcopy_paths"].map(existing_paths)
    df["oct_existing"] = df["oct_paths"].map(existing_paths)
    if config["require_colposcopy"]:
        df = df[df["colposcopy_existing"].map(bool)]
    if config["require_oct"]:
        df = df[df["oct_existing"].map(bool)]

    seed = int(config["seed"])
    df["stable_order"] = [stable_score(p, c, seed=seed) for p, c in zip(df.patient_id, df.case_id)]
    if config["one_case_per_patient"]:
        df = df.sort_values("stable_order").drop_duplicates("patient_id", keep="first")

    selected: list[pd.DataFrame] = []
    frac = float(config["positive_fraction"])

    def center_round_robin(part: pd.DataFrame, n: int) -> pd.DataFrame:
        center_order = sorted(
            part["hospital_name"].unique(),
            key=lambda center: stable_score(center, seed=seed),
        )
        queues = {
            center: list(part[part["hospital_name"] == center].sort_values("stable_order").index)
            for center in center_order
        }
        chosen: list[int] = []
        while len(chosen) < n:
            before = len(chosen)
            for center in center_order:
                if queues[center] and len(chosen) < n:
                    chosen.append(queues[center].pop(0))
            if len(chosen) == before:
                break
        return part.loc[chosen]

    for split, requested in config["samples_per_split"].items():
        part = df[df["split"] == split]
        n_pos = round(int(requested) * frac)
        n_neg = int(requested) - n_pos
        pos = center_round_robin(part[part["binary_label"] == 1], n_pos)
        neg = center_round_robin(part[part["binary_label"] == 0], n_neg)
        if len(pos) != n_pos or len(neg) != n_neg:
            raise RuntimeError(
                f"Insufficient eligible cases in {split}: requested positive={n_pos}, negative={n_neg}; "
                f"found positive={len(pos)}, negative={len(neg)}"
            )
        selected.extend([pos, neg])
    cohort = pd.concat(selected, ignore_index=True).sort_values(["split", "stable_order"])
    if cohort["patient_id"].duplicated().any():
        raise AssertionError("A patient appeared more than once in the pilot")
    return cohort


def model_row(row: dict, prompt: str, images: list[str]) -> dict:
    patient_key = f"p_{stable_score(row['patient_id']):016x}"
    return {
        "id": str(row["case_id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>\n<image>\n" + prompt},
        ],
        "images": images,
        "label": int(row["binary_label"]),
        "patient_id": patient_key,
        "center": str(row["hospital_name"]),
        "split": str(row["split"]),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config["manifest"])
    cohort = select_cohort(df, config)

    audit_rows: list[dict] = []
    baseline_rows: dict[str, list[dict]] = {s: [] for s in config["samples_per_split"]}
    evidence_rows: dict[str, list[dict]] = {s: [] for s in config["samples_per_split"]}
    for raw in cohort.to_dict(orient="records"):
        case_id = str(raw["case_id"])
        colpo_selected = select_colposcopy(
            raw["colposcopy_existing"], int(config["max_colposcopy_images"])
        )
        oct_selected = evenly_spaced(
            sorted(raw["oct_existing"]), int(config["max_oct_images"])
        )
        asset_dir = out / "assets" / str(raw["split"])
        colpo_panel = asset_dir / f"{case_id}_colposcopy.jpg"
        oct_panel = asset_dir / f"{case_id}_oct.jpg"
        if args.rebuild_panels or not colpo_panel.is_file():
            build_contact_sheet(
                colpo_selected,
                colpo_panel,
                tile_size=int(config["panel_tile_size"]),
                columns=2,
            )
        if args.rebuild_panels or not oct_panel.is_file():
            build_contact_sheet(
                oct_selected,
                oct_panel,
                tile_size=int(config["panel_tile_size"]),
                columns=3,
            )
        row_for_prompt = {
            "age": raw.get("age"),
            "hpv": raw.get("hpv"),
            "tct": raw.get("tct"),
        }
        images = [str(colpo_panel.resolve()), str(oct_panel.resolve())]
        baseline_rows[str(raw["split"])].append(
            model_row(raw, diagnostic_prompt(row_for_prompt), images)
        )
        evidence_rows[str(raw["split"])].append(
            model_row(raw, evidence_prompt(row_for_prompt), images)
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "patient_id": f"p_{stable_score(raw['patient_id']):016x}",
                "split": raw["split"],
                "center": raw["hospital_name"],
                "label": int(raw["binary_label"]),
                "colposcopy_available": len(raw["colposcopy_existing"]),
                "oct_available": len(raw["oct_existing"]),
                "colposcopy_selection_digest": f"{stable_score(*colpo_selected):016x}",
                "oct_selection_digest": f"{stable_score(*oct_selected):016x}",
                "colposcopy_panel": str(colpo_panel.resolve()),
                "oct_panel": str(oct_panel.resolve()),
            }
        )

    pd.DataFrame(audit_rows).to_csv(out / "cohort_audit.csv", index=False)
    for split in config["samples_per_split"]:
        write_jsonl(out / "baseline" / f"{split}.jsonl", baseline_rows[split])
        write_jsonl(out / "evidence_prompts" / f"{split}.jsonl", evidence_rows[split])
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_manifest": config["manifest"],
        "rows": len(audit_rows),
        "counts": [
            {"split": split, "label": int(label), "n": int(count)}
            for (split, label), count in pd.DataFrame(audit_rows).groupby(["split", "label"]).size().items()
        ],
        "prohibited_supervision_columns": [
            "pseudo_report_text",
            "training_report_text",
            "reference_diagnostic_summary",
        ],
        "selection_unit": "one case per patient for pilot only",
        "formal_unit_note": "case/site predictions; uncertainty must be clustered by patient",
    }
    with (out / "build_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
