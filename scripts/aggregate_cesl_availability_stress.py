#!/usr/bin/env python3
"""Aggregate prespecified fixed-policy CESL availability-stress outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json  # noqa: E402
from cervix_cogalign.statistics import macro_centre_metrics, paired_bootstrap_delta, point_metrics  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    args = parser.parse_args()
    config = read_json(args.config)
    folds = tuple(str(fold) for fold in config["folds"])
    output = Path(config["output_dir"])
    analysis = output / "analysis"
    threshold_path = analysis / "ensemble_thresholds.csv"
    if not threshold_path.is_file():
        raise FileNotFoundError("Aggregate primary CESL predictions before availability stress")
    paths = []
    for fold in folds:
        for seed in config["seeds"]:
            root = output / "evaluations" / fold / f"seed_{seed}"
            if not (root / "availability_stress_complete.json").is_file():
                raise FileNotFoundError(f"Missing completed availability stress artifact: {root}")
            paths.append(root / "availability_stress_predictions.csv")
    frame = pd.concat([pd.read_csv(path, dtype={"id": str, "patient_id": str}) for path in paths], ignore_index=True)
    keys = ["fold", "policy", "scenario", "id", "patient_id", "center", "label"]
    ensemble = frame.groupby(keys, as_index=False).agg(
        probability=("probability", "mean"),
        remaining_visual_atoms=("remaining_visual_atoms", "mean"),
        n_seeds=("seed", "nunique"),
    )
    if (ensemble.n_seeds != 3).any():
        raise AssertionError("Availability-stress ensembles require all three seeds")
    thresholds = pd.read_csv(threshold_path)
    ensemble = ensemble.merge(thresholds[["fold", "policy", "threshold_source_validation_ensemble"]], on=["fold", "policy"], how="left", validate="many_to_one")
    ensemble["predicted_label"] = (ensemble.probability >= ensemble.threshold_source_validation_ensemble).astype(int)
    ensemble.to_csv(analysis / "availability_stress_ensemble.csv", index=False)
    metric_rows = []
    for (policy, scenario), part in ensemble.groupby(["policy", "scenario"], sort=True):
        for scale, func in (("pooled", point_metrics), ("macro_centre", macro_centre_metrics)):
            metrics = func(part)
            metric_rows.extend({"policy": policy, "scenario": scenario, "scale": scale, "metric": metric, "value": value} for metric, value in metrics.items())
    pd.DataFrame(metric_rows).to_csv(analysis / "availability_stress_metrics.csv", index=False)
    comparisons = []
    for scenario in config["robustness_scenarios"]:
        c3 = ensemble[(ensemble.policy == "C3_cev_adaptive") & (ensemble.scenario == scenario)]
        c4 = ensemble[(ensemble.policy == "C4_cesl_css") & (ensemble.scenario == scenario)]
        report = paired_bootstrap_delta(c3, c4, int(config["bootstrap"]["replicates"]), 20264000 + len(comparisons), metrics=("auprc", "auroc", "sensitivity", "brier", "ece"))
        for scale, values in report["candidate_minus_reference"].items():
            for metric, result in values.items():
                comparisons.append(
                    {
                        "scenario": scenario,
                        "reference": "C3_cev_adaptive",
                        "candidate": "C4_cesl_css",
                        "scale": scale,
                        "metric": metric,
                        "delta": result["point"],
                        "ci95_low": result["ci95"][0],
                        "ci95_high": result["ci95"][1],
                        "bootstrap_p_two_sided": result["bootstrap_p_two_sided"],
                        "n_paired": report["n_paired"],
                    }
                )
    pd.DataFrame(comparisons).to_csv(analysis / "availability_stress_c4_vs_c3.csv", index=False)
    print(f"Wrote availability-stress summaries for {len(config['robustness_scenarios'])} scenarios")


if __name__ == "__main__":
    main()
