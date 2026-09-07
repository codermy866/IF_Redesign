#!/usr/bin/env python3
"""Build a deterministic raw-frame atom manifest for CESL-v2.

The manifest preserves the Formal v1 patient/case cohort but does not change
its splits or train a model. It is an auditable input to a future CESL policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import evenly_spaced, existing_paths, read_json, select_colposcopy, stable_score, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2.json"))
    parser.add_argument(
        "--formal-config", default=str(ROOT / "configs/formal_v1.json"),
        help="Used only to reuse Formal v1 maximum atom counts and centre aliases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    formal = read_json(args.formal_config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(config["manifest"])
    frame["case_id"] = frame.case_id.astype(str)
    frame["colposcopy_existing"] = frame.colposcopy_paths.map(existing_paths)
    frame["oct_existing"] = frame.oct_paths.map(existing_paths)
    eligible = frame[frame.colposcopy_existing.map(bool) & frame.oct_existing.map(bool)].copy()
    if eligible.patient_id.duplicated().any():
        raise ValueError("CESL atom cohort must preserve one selected site per patient")
    rows = []
    for raw in eligible.to_dict(orient="records"):
        colpo = select_colposcopy(raw["colposcopy_existing"], int(formal["max_colposcopy_images"]))
        oct_paths = evenly_spaced(sorted(raw["oct_existing"]), int(formal["max_oct_images"]))
        atoms = [
            {
                "atom_id": "clinical",
                "kind": "clinical",
                "relative_cost": float(config["relative_evidence_cost_units"]["clinical"]),
                "fields": ["age", "hpv", "tct"],
            }
        ]
        atoms.extend(
            {
                "atom_id": f"colposcopy_{index:02d}",
                "kind": "colposcopy_frame",
                "position": index,
                "relative_cost": float(config["relative_evidence_cost_units"]["colposcopy_frame"]),
                "path": str(Path(path).resolve()),
            }
            for index, path in enumerate(colpo)
        )
        atoms.extend(
            {
                "atom_id": f"oct_{index:02d}",
                "kind": "oct_bscan",
                "position": index,
                "relative_cost": float(config["relative_evidence_cost_units"]["oct_bscan"]),
                "path": str(Path(path).resolve()),
            }
            for index, path in enumerate(oct_paths)
        )
        rows.append(
            {
                "id": str(raw["case_id"]),
                "patient_id": f"p_{stable_score(raw['patient_id']):016x}",
                "center": str(raw["hospital_name"]),
                "center_slug": formal["centres"][str(raw["hospital_name"])],
                "label": int(raw["binary_label"]),
                "atoms": atoms,
            }
        )
    if len(rows) != len(eligible) or len({row["patient_id"] for row in rows}) != len(rows):
        raise AssertionError("CESL atom manifest lost case/patient uniqueness")
    write_jsonl(output / "raw_atoms.jsonl", rows)
    report = {
        "experiment_id": config["experiment_id"],
        "n_cases": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "atom_schema": "clinical + deterministically selected raw colposcopy frames + raw OCT B-scans",
        "mean_atoms_per_case": sum(len(row["atoms"]) for row in rows) / len(rows),
        "selection": {
            "max_colposcopy_images": int(formal["max_colposcopy_images"]),
            "max_oct_images": int(formal["max_oct_images"]),
            "colposcopy": "deterministic select_colposcopy",
            "oct": "deterministic evenly spaced selection",
        },
        "claim_boundary": config["claim_boundary"],
    }
    (output / "raw_atoms_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
