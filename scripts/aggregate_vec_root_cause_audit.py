#!/usr/bin/env python3
"""Aggregate source-only VEC root-cause diagnostics across fixed folds/seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("enshi", "jingzhou", "shiyan", "wuhan", "xiangyang")
SEEDS = (20260905, 20260906, 20260907)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "auto_research/verifiable_evidence_chain_20260905/root_cause_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    jobs = []
    for fold in FOLDS:
        for seed in SEEDS:
            path = root / "jobs" / fold / f"seed_{seed}" / "diagnostics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete root-cause audit: {path}")
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
    if any(job["test_labels_opened"] for job in jobs):
        raise AssertionError("Root-cause audit must not open test labels")

    score_rows = []
    job_rows = []
    for job in jobs:
        job_rows.append({
            "fold": job["fold"], "seed": job["seed"],
            "main_center_q_rho": job["q_rank_agreement"]["main_vs_center_only_mean_within_case_spearman"],
            "main_pooled_q_rho": job["q_rank_agreement"]["main_vs_pooled_unmatched_mean_within_case_spearman"],
            "d4_oof_r2": job["crossfit_estimator"]["atom_level_out_of_fold_r2"],
            "d4_attention_top_agreement": job["crossfit_estimator"]["top_atom_agreement_with_attention"],
            "d4_top_loss_delta": job["crossfit_estimator"]["mean_observed_loss_difference_d4_top_minus_attention_top"],
            "d1_attention_top_agreement": job["selection"]["posterior_necessity_top_agreement_with_attention"],
            "source_validation_cases": job["sample"]["source_validation_cases"],
            "source_validation_positive_rate": job["sample"]["source_validation_positive_rate"],
            "donor_same_center_stratum_fraction": job["donor_distribution"]["matching_level_fraction"].get("same_center_stratum", 0.0),
            "donor_same_center_fraction": job["donor_distribution"]["matching_level_fraction"].get("same_center", 0.0),
            "donor_transport_fraction": sum(value for key, value in job["donor_distribution"]["matching_level_fraction"].items() if key.startswith("transport")),
            "donor_pool_median": job["donor_distribution"]["candidate_pool_size"]["median"],
            "donor_three_unique_draw_rate": job["donor_distribution"]["unique_donors_in_three_draws"]["three_donor_rate"],
        })
        for name, score in job["score_recovery_of_observed_replacement_loss"].items():
            score_rows.append({"fold": job["fold"], "seed": job["seed"], "score": name, **{key: value for key, value in score.items() if key != "name"}})
    job_frame = pd.DataFrame(job_rows)
    score_frame = pd.DataFrame(score_rows)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    job_frame.to_csv(analysis / "per_fold_seed_measurement.csv", index=False)
    score_frame.to_csv(analysis / "per_fold_seed_score_recovery.csv", index=False)

    numeric = [column for column in job_frame if column not in {"fold", "seed"}]
    fold_job = job_frame.groupby("fold", as_index=False)[numeric].mean()
    score_numeric = [column for column in score_frame if column not in {"fold", "seed", "score"}]
    fold_score = score_frame.groupby(["fold", "score"], as_index=False)[score_numeric].mean()
    fold_job.to_csv(analysis / "per_fold_measurement.csv", index=False)
    fold_score.to_csv(analysis / "per_fold_score_recovery.csv", index=False)

    score_summary = fold_score.groupby("score")[score_numeric].median(numeric_only=True).to_dict(orient="index")
    summary = {
        "status": "complete",
        "test_labels_opened": False,
        "unit": "fold-specific source-validation case; three fixed seeds averaged within fold",
        "n_folds": len(FOLDS),
        "n_jobs": len(jobs),
        "median_across_folds": {column: float(fold_job[column].median()) for column in numeric},
        "score_recovery_median_across_folds": {
            score: {key: (float(value) if pd.notna(value) else None) for key, value in values.items()}
            for score, values in score_summary.items()
        },
    }
    (analysis / "root_cause_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
