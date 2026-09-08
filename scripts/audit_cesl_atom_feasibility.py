#!/usr/bin/env python3
"""Audit whether raw colposcopy/OCT evidence atoms can support CESL-v2.

This script is descriptive only: it reads the locked manifest, records
accessible raw-frame counts, and writes no labels, models, or counterfactual
examples. It does not open held-out outcome results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import existing_paths, read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(config["manifest"])
    frame["case_id"] = frame.case_id.astype(str)
    frame["colposcopy_accessible"] = frame.colposcopy_paths.map(existing_paths)
    frame["oct_accessible"] = frame.oct_paths.map(existing_paths)
    frame["n_colposcopy_atoms"] = frame.colposcopy_accessible.map(len)
    frame["n_oct_atoms"] = frame.oct_accessible.map(len)
    eligible = frame[(frame.n_colposcopy_atoms > 0) & (frame.n_oct_atoms > 0)].copy()
    max_colpo = 4
    max_oct = 6
    eligible["usable_colposcopy_atoms"] = eligible.n_colposcopy_atoms.clip(upper=max_colpo)
    eligible["usable_oct_atoms"] = eligible.n_oct_atoms.clip(upper=max_oct)
    eligible["usable_visual_atoms"] = eligible.usable_colposcopy_atoms + eligible.usable_oct_atoms
    columns = [
        "case_id", "patient_id", "hospital_name", "binary_label",
        "n_colposcopy_atoms", "n_oct_atoms", "usable_colposcopy_atoms",
        "usable_oct_atoms", "usable_visual_atoms",
    ]
    eligible[columns].to_csv(output / "raw_atom_feasibility.csv", index=False)
    by_centre = (
        eligible.groupby("hospital_name", as_index=False)
        .agg(
            cases=("case_id", "size"),
            patients=("patient_id", "nunique"),
            positive_rate=("binary_label", "mean"),
            median_colposcopy_atoms=("n_colposcopy_atoms", "median"),
            median_oct_atoms=("n_oct_atoms", "median"),
            median_usable_visual_atoms=("usable_visual_atoms", "median"),
        )
        .sort_values("hospital_name")
    )
    report = {
        "experiment_id": config["experiment_id"],
        "purpose": "raw-frame CESL atom feasibility; not an outcome analysis",
        "manifest": str(Path(config["manifest"]).resolve()),
        "total_manifest_cases": int(len(frame)),
        "eligible_cases_with_both_modalities": int(len(eligible)),
        "eligible_unique_patients": int(eligible.patient_id.nunique()),
        "atom_caps": {"colposcopy": max_colpo, "oct": max_oct},
        "centres": by_centre.to_dict(orient="records"),
        "claim_boundary": config["claim_boundary"],
    }
    (output / "raw_atom_feasibility.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
