#!/usr/bin/env python3
"""Run prespecified CESL retrospective summaries, comparisons, and audits."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json  # noqa: E402
from cervix_cogalign.statistics import (  # noqa: E402
    REPORTED_METRICS,
    _cluster_strata,
    _draw_indices,
    bootstrap_intervals,
    holm_adjust,
    macro_centre_metrics,
    paired_bootstrap_delta,
    point_metrics,
)


PROPOSED = "C4_cesl_css"
PRIMARY_COMPARATORS = (
    "C0_full_evidence",
    "C1_attention_fixed_budget",
    "C2_cev_fixed_budget",
    "C3_cev_adaptive",
    "random_fixed_budget",
    "uncertainty_only",
)
STATIC_RAW_BASELINES = (
    "clinical_only_lr",
    "raw_colposcopy_meanpool_linear",
    "raw_oct_meanpool_linear",
    "raw_full_meanpool_linear",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    return parser.parse_args()


def method_summary(frame: pd.DataFrame, method: str, replicates: int, seed: int) -> tuple[dict, list[dict]]:
    intervals, standard_errors = bootstrap_intervals(frame, replicates, seed)
    points = {"pooled": point_metrics(frame), "macro_centre": macro_centre_metrics(frame)}
    report = {
        "method": method,
        "n_total": int(len(frame)),
        "n_scored": int(frame.probability.notna().sum()),
        "score_coverage": float(frame.probability.notna().mean()),
        "point": points,
        "ci95": intervals,
        "bootstrap_standard_error": standard_errors,
    }
    rows = []
    for scale in ("pooled", "macro_centre"):
        for metric in REPORTED_METRICS:
            rows.append(
                {
                    "method": method,
                    "scale": scale,
                    "metric": metric,
                    "value": points[scale][metric],
                    "ci95_low": intervals[scale][metric][0],
                    "ci95_high": intervals[scale][metric][1],
                    "bootstrap_se": standard_errors[scale][metric],
                    "n_total": len(frame),
                    "n_scored": frame.probability.notna().sum(),
                }
            )
    return report, rows


def flatten_comparison(group: str, reference: str, candidate: str, report: dict) -> list[dict]:
    rows = []
    for scale, metrics in report["candidate_minus_reference"].items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "comparison_group": group,
                    "reference": reference,
                    "candidate": candidate,
                    "scale": scale,
                    "metric": metric,
                    "delta": values["point"],
                    "ci95_low": values["ci95"][0],
                    "ci95_high": values["ci95"][1],
                    "bootstrap_p_two_sided": values["bootstrap_p_two_sided"],
                    "n_paired": report["n_paired"],
                }
            )
    return rows


def add_holm(rows: list[dict], group: str) -> None:
    selected = [
        row
        for row in rows
        if row["comparison_group"] == group and row["scale"] == "macro_centre" and row["metric"] == "auprc"
    ]
    adjusted = holm_adjust([row["bootstrap_p_two_sided"] for row in selected])
    for row, value in zip(selected, adjusted):
        row["holm_adjusted_p_within_family"] = value


def calibration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in frame.groupby("method", sort=True):
        probability = np.clip(part.probability.to_numpy(dtype=float), 1e-5, 1 - 1e-5)
        label = part.label.to_numpy(dtype=int)
        logit = np.log(probability / (1 - probability))
        try:
            fitted = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit[:, None], label)
            slope = float(fitted.coef_[0, 0])
            intercept = float(fitted.intercept_[0])
        except Exception:
            slope, intercept = float("nan"), float("nan")
        rows.append({"method": method, "calibration_slope": slope, "calibration_intercept": intercept, "n": len(part)})
    return pd.DataFrame(rows)


def decision_curve(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in frame.groupby("method", sort=True):
        y = part.label.to_numpy(dtype=int)
        probability = part.probability.to_numpy(dtype=float)
        n = len(part)
        prevalence = float(y.mean())
        for threshold in (0.05, 0.10, 0.20, 0.30):
            prediction = probability >= threshold
            tp = int(np.count_nonzero(prediction & (y == 1)))
            fp = int(np.count_nonzero(prediction & (y == 0)))
            weight = threshold / (1.0 - threshold)
            rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "net_benefit": tp / n - fp / n * weight,
                    "treat_all_net_benefit": prevalence - (1.0 - prevalence) * weight,
                    "treat_none_net_benefit": 0.0,
                    "n": n,
                    "interpretation": "retrospective decision-curve calculation; not a clinical recommendation",
                }
            )
    return pd.DataFrame(rows)


def paired_cost_difference(reference: pd.DataFrame, candidate: pd.DataFrame, replicates: int, seed: int) -> dict:
    columns = ["id", "patient_id", "center", "label", "relative_evidence_cost"]
    paired = reference[columns].merge(candidate[columns], on=["id", "patient_id", "center", "label"], suffixes=("_reference", "_candidate"), validate="one_to_one")
    if not len(paired):
        raise ValueError("Cost comparison has no paired cases")
    diff = paired.relative_evidence_cost_candidate.to_numpy(dtype=float) - paired.relative_evidence_cost_reference.to_numpy(dtype=float)
    strata = _cluster_strata(paired.rename(columns={"relative_evidence_cost_reference": "probability", "relative_evidence_cost_candidate": "predicted_label"}))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        indices = _draw_indices(strata, rng)
        draws.append(float(diff[indices].mean()))
    array = np.asarray(draws)
    lower = (np.count_nonzero(array <= 0) + 1) / (len(array) + 1)
    upper = (np.count_nonzero(array >= 0) + 1) / (len(array) + 1)
    return {
        "n_paired": len(paired),
        "candidate_minus_reference_mean_cost": float(diff.mean()),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "bootstrap_p_two_sided": float(min(1.0, 2.0 * min(lower, upper))),
    }


def bootstrap_mean_effect(frame: pd.DataFrame, value_column: str, replicates: int, seed: int) -> dict:
    rows = frame.dropna(subset=[value_column]).copy().reset_index(drop=True)
    if not len(rows):
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    if "label" not in rows:
        rows["label"] = 0
    strata = _cluster_strata(rows[["patient_id", "center", "label"]].assign(probability=0.0, predicted_label=0))
    value = rows[value_column].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = [float(value[_draw_indices(strata, rng)].mean()) for _ in range(replicates)]
    return {"n": len(rows), "mean": float(value.mean()), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def rank_stability(rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), part in rank.groupby(["model", "seed"], sort=True):
        pivots = {
            center: value.set_index("slot").mean_predicted_cev
            for center, value in part.groupby("center", sort=True)
        }
        centers = sorted(pivots)
        for left_index, left in enumerate(centers):
            for right in centers[left_index + 1 :]:
                joined = pd.concat([pivots[left], pivots[right]], axis=1, join="inner").dropna()
                correlation = float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic) if len(joined) >= 3 else float("nan")
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "center_left": left,
                        "center_right": right,
                        "n_common_slots": len(joined),
                        "spearman_cev_rank_stability": correlation,
                        "interpretation": "descriptive rank stability over atom slots; centers differ in prevalence and case mix",
                    }
                )
    return pd.DataFrame(rows)


def q_rank_sensitivity(q_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, center), part in q_frame.groupby(["seed", "center"], sort=True):
        pivot = part.pivot_table(index="slot", columns="q_variant", values="predicted_cev", aggfunc="mean")
        if "main_matched" not in pivot:
            continue
        for variant in pivot.columns:
            if variant == "main_matched":
                continue
            joined = pivot[["main_matched", variant]].dropna()
            correlation = float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic) if len(joined) >= 3 else float("nan")
            rows.append(
                {
                    "seed": seed,
                    "center": center,
                    "q_variant": variant,
                    "n_common_slots": len(joined),
                    "spearman_vs_main_matched": correlation,
                    "interpretation": "q sensitivity, not an estimate of causal intervention validity",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    analysis = Path(config["output_dir"]) / "analysis"
    prediction_path = analysis / "all_test_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError("Run aggregate_cesl_results.py before CESL statistics")
    predictions = pd.read_csv(prediction_path, dtype={"id": str, "patient_id": str})
    methods = sorted(predictions.method.unique().tolist())
    if PROPOSED not in methods:
        raise AssertionError("Full CESL policy is absent from aggregated predictions")
    replicates = int(config["bootstrap"]["replicates"])
    summaries = {}
    metric_rows: list[dict] = []
    for offset, method in enumerate(methods):
        report, rows = method_summary(predictions[predictions.method == method].copy(), method, replicates, 20260905 + offset)
        summaries[method] = report
        metric_rows.extend(rows)
    comparison_reports = {}
    comparison_rows: list[dict] = []
    candidate = predictions[predictions.method == PROPOSED].copy()
    for offset, reference_name in enumerate(PRIMARY_COMPARATORS):
        reference = predictions[predictions.method == reference_name].copy()
        report = paired_bootstrap_delta(reference, candidate, replicates, 20261000 + offset)
        comparison_reports[f"{PROPOSED}_vs_{reference_name}"] = report
        comparison_rows.extend(flatten_comparison("primary_c4_policy_family", reference_name, PROPOSED, report))
    for offset, reference_name in enumerate(STATIC_RAW_BASELINES):
        reference = predictions[predictions.method == reference_name].copy()
        report = paired_bootstrap_delta(reference, candidate, replicates, 20261100 + offset)
        comparison_reports[f"{PROPOSED}_vs_{reference_name}"] = report
        comparison_rows.extend(flatten_comparison("static_raw_baseline_family", reference_name, PROPOSED, report))
    add_holm(comparison_rows, "primary_c4_policy_family")
    add_holm(comparison_rows, "static_raw_baseline_family")
    cost_rows = []
    for offset, reference_name in enumerate(PRIMARY_COMPARATORS):
        reference = predictions[predictions.method == reference_name].copy()
        report = paired_cost_difference(reference, candidate, replicates, 20262000 + offset)
        cost_rows.append({"reference": reference_name, "candidate": PROPOSED, **report})
    pd.DataFrame(metric_rows).to_csv(analysis / "main_metrics_with_ci.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(analysis / "paired_comparisons.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(analysis / "paired_cost_comparisons.csv", index=False)
    calibration_summary(predictions).to_csv(analysis / "calibration_summary.csv", index=False)
    decision_curve(predictions).to_csv(analysis / "decision_curve.csv", index=False)

    mechanism = pd.read_csv(analysis / "mechanism_audit.csv", dtype={"id": str, "patient_id": str})
    label_map = predictions[predictions.method == PROPOSED][["id", "patient_id", "center", "label"]].drop_duplicates()
    mechanism = mechanism.merge(label_map, on=["id", "patient_id", "center"], how="left", validate="many_to_one")
    mechanism_ensemble = (
        mechanism.groupby(["model", "id", "patient_id", "center", "label"], as_index=False)[
            ["observed_replacement_loss_cev", "observed_replacement_loss_attention", "observed_replacement_loss_random"]
        ]
        .mean()
    )
    mechanism_ensemble["cev_minus_attention"] = (
        mechanism_ensemble.observed_replacement_loss_cev - mechanism_ensemble.observed_replacement_loss_attention
    )
    mechanism_ensemble["cev_minus_random"] = mechanism_ensemble.observed_replacement_loss_cev - mechanism_ensemble.observed_replacement_loss_random
    mechanism_rows = []
    for model, part in mechanism_ensemble.groupby("model", sort=True):
        for offset, column in enumerate(("cev_minus_attention", "cev_minus_random")):
            mechanism_rows.append({"model": model, "contrast": column, **bootstrap_mean_effect(part, column, replicates, 20263000 + offset)})
    pd.DataFrame(mechanism_rows).to_csv(analysis / "mechanism_effects.csv", index=False)
    mechanism_ensemble.to_csv(analysis / "mechanism_audit_ensemble.csv", index=False)

    ranks = pd.read_csv(analysis / "rank_profiles.csv")
    rank_stability(ranks).to_csv(analysis / "centre_rank_stability.csv", index=False)
    q_sensitivity = pd.read_csv(analysis / "q_sensitivity.csv", dtype={"id": str, "patient_id": str})
    q_rank_sensitivity(q_sensitivity).to_csv(analysis / "q_rank_sensitivity.csv", index=False)
    policy_stability = (
        predictions[predictions.method.isin(["C3_cev_adaptive", "C4_cesl_css"])]
        .groupby(["method", "center"], as_index=False)
        .agg(mean_visual_atom_count=("selected_visual_atoms", "mean"), stopping_rate=("stopping_criterion_met", "mean"), n=("id", "nunique"))
    )
    policy_stability.to_csv(analysis / "stopping_stability_by_centre.csv", index=False)
    worst_center = []
    for method, part in predictions.groupby("method", sort=True):
        values = []
        for center, center_part in part.groupby("center", sort=True):
            metrics = point_metrics(center_part)
            values.append({"method": method, "center": center, **metrics})
        worst_center.extend(values)
    pd.DataFrame(worst_center).to_csv(analysis / "per_centre_metrics.csv", index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "primary_estimand": config["primary_estimand"],
        "bootstrap_replicates": replicates,
        "bootstrap_unit": config["bootstrap"]["unit"],
        "multiplicity": config["bootstrap"]["multiplicity"],
        "method_summaries": summaries,
        "paired_comparisons": comparison_reports,
        "claim_boundary": config["claim_boundary"],
    }
    (analysis / "statistical_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote CESL statistics for {len(methods)} methods, {len(PRIMARY_COMPARATORS)} policy comparisons, and {len(STATIC_RAW_BASELINES)} static baselines")


if __name__ == "__main__":
    main()
