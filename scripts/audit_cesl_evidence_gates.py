#!/usr/bin/env python3
"""Write an explicit evidence-gate audit for CESL top-journal requirements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    args = parser.parse_args()
    config = read_json(args.config)
    manifest = pd.read_csv(config["manifest"], nrows=5)
    date_like = [
        column
        for column in manifest.columns
        if any(token in column.lower() for token in ("date", "time", "year", "month", "day", "visit", "acq"))
    ]
    atoms = read_jsonl(config["atom_manifest"])
    output = Path(config["output_dir"]) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "experiment_id": config["experiment_id"],
        "completed_or_running_local_evidence": {
            "raw_atom_cohort": {"n_cases": len(atoms), "one_case_per_patient_expected": True},
            "loco_transport": "five existing hospitals with source-only fitting",
            "three_seed_ensemble": list(config["seeds"]),
            "counterfactual_donor_audit": "per-edge artifacts retained for predicted-class policy scores",
            "calibration_decision_curve_robustness": "computed only after frozen CESL predictions complete",
        },
        "not_started_or_not_claimable_without_new_authority": config["unavailable_evidence_gates"],
        "structured_date_columns_detected": date_like,
        "temporal_validation_gate": "blocked" if not date_like else "requires provenance verification before use",
        "claim_boundary": config["claim_boundary"],
        "reporting_framework_status": {
            "STARD_AI": "planned coverage audit after final results; not a certification of reporting quality",
            "TRIPOD_AI": "planned coverage audit after final results; not a certification of reporting quality",
            "DECIDE_AI": "reserved for any future prospective/live clinical stage",
        },
    }
    (output / "top_journal_evidence_gates.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
