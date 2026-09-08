#!/usr/bin/env python3
"""Patient-clustered inference for the locked Formal v1 LOCO experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json  # noqa: E402
from cervix_cogalign.statistics import (  # noqa: E402
    REPORTED_METRICS,
    bootstrap_intervals,
    holm_adjust,
    macro_centre_metrics,
    paired_bootstrap_delta,
    point_metrics,
)


MAIN_METHODS = (
    "clinical_only_lr",
    "frozen_multimodal_linear",
    "qwen3vl_zero_shot",
    "diagnosis_only_3seed_ensemble",
    "full_hierarchy_3seed_ensemble",
)
PROPOSED = "full_hierarchy_3seed_ensemble"
ABLATION_REFERENCES = (
    ("cognition_structure", "diagnosis_only"),
    ("cognition_structure", "without_integration"),
    ("modality_contribution", "colposcopy_only"),
    ("modality_contribution", "oct_only"),
    ("supervision_integrity", "without_clinical"),
    ("supervision_integrity", "shuffled_cognition"),
)


def method_frame(predictions: pd.DataFrame, method: str, seed: str | None = None) -> pd.DataFrame:
    frame = predictions[predictions.method == method].copy()
    if seed is not None:
        frame = frame[frame.seed.astype(str) == str(seed)].copy()
    if frame.id.duplicated().any():
        raise ValueError(f"Duplicate cases for method={method}, seed={seed}")
    return frame


def method_summary(
    frame: pd.DataFrame, method: str, replicates: int, seed: int
) -> tuple[dict, list[dict]]:
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


def add_holm(rows: list[dict], group: str, scale: str, metric: str) -> None:
    selected = [
        row
        for row in rows
        if row["comparison_group"] == group
        and row["scale"] == scale
        and row["metric"] == metric
    ]
    adjusted = holm_adjust([row["bootstrap_p_two_sided"] for row in selected])
    for row, value in zip(selected, adjusted):
        row["holm_adjusted_p_within_family"] = value


def variance_decomposition(predictions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive two-way decomposition of centre and method variation."""
    rows = []
    for method in MAIN_METHODS:
        frame = method_frame(predictions, method)
        for centre, part in frame.groupby("center"):
            summary = point_metrics(part)
            for metric in ("auprc", "auroc", "balanced_accuracy"):
                rows.append({"method": method, "center": centre, "metric": metric, "value": summary[metric]})
    cell = pd.DataFrame(rows)
    output = []
    for metric, part in cell.groupby("metric"):
        table = part.pivot(index="method", columns="center", values="value")
        grand = float(table.to_numpy().mean())
        method_ss = table.shape[1] * float(((table.mean(axis=1) - grand) ** 2).sum())
        centre_ss = table.shape[0] * float(((table.mean(axis=0) - grand) ** 2).sum())
        fitted = (
            table.mean(axis=1).to_numpy()[:, None]
            + table.mean(axis=0).to_numpy()[None, :]
            - grand
        )
        residual_ss = float(((table.to_numpy() - fitted) ** 2).sum())
        total = method_ss + centre_ss + residual_ss
        output.append(
            {
                "metric": metric,
                "method_sum_squares": method_ss,
                "centre_sum_squares": centre_ss,
                "method_by_centre_residual_sum_squares": residual_ss,
                "method_fraction": method_ss / total,
                "centre_fraction": centre_ss / total,
                "method_by_centre_residual_fraction": residual_ss / total,
                "interpretation": "descriptive; AUPRC centre component includes prevalence effects",
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/formal_v1.json"))
    parser.add_argument("--predictions")
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    config = read_json(args.config)
    formal_root = Path(config["output_dir"])
    predictions_path = Path(args.predictions) if args.predictions else formal_root / "analysis/all_test_predictions.csv"
    predictions = pd.read_csv(predictions_path, dtype={"id": str, "patient_id": str, "seed": str})
    replicates = args.bootstrap or int(config["bootstrap_replicates"])
    output = formal_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    summaries = {}
    metric_rows: list[dict] = []
    for index, method in enumerate(MAIN_METHODS):
        report, rows = method_summary(
            method_frame(predictions, method), method, replicates, args.seed + index
        )
        summaries[method] = report
        metric_rows.extend(rows)

    comparison_reports = {}
    comparison_rows: list[dict] = []
    candidate = method_frame(predictions, PROPOSED)
    for index, reference_name in enumerate(MAIN_METHODS[:-1]):
        report = paired_bootstrap_delta(
            method_frame(predictions, reference_name),
            candidate,
            replicates,
            args.seed + 100 + index,
        )
        key = f"{PROPOSED}_vs_{reference_name}"
        comparison_reports[key] = report
        comparison_rows.extend(flatten_comparison("main", reference_name, PROPOSED, report))

    full_seed = method_frame(predictions, "full_hierarchy", str(config["seeds"][0]))
    for index, (block, reference_name) in enumerate(ABLATION_REFERENCES):
        report = paired_bootstrap_delta(
            method_frame(predictions, reference_name, str(config["seeds"][0])),
            full_seed,
            replicates,
            args.seed + 200 + index,
        )
        key = f"full_hierarchy_vs_{reference_name}"
        comparison_reports[key] = report
        comparison_rows.extend(flatten_comparison(block, reference_name, "full_hierarchy", report))

    for group in ("main", "cognition_structure", "modality_contribution", "supervision_integrity"):
        add_holm(comparison_rows, group, "macro_centre", "auprc")

    pd.DataFrame(metric_rows).to_csv(output / "main_metrics_with_ci.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "paired_comparisons.csv", index=False)
    variance = variance_decomposition(predictions)
    variance.to_csv(output / "data_vs_method_variance.csv", index=False)
    report = {
        "experiment_id": config["experiment_id"],
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "patients sampled within centre and outcome strata",
        "primary_estimand": "macro-centre AUPRC",
        "main_methods": list(MAIN_METHODS),
        "method_summaries": summaries,
        "paired_comparisons": comparison_reports,
        "data_vs_method_variance_decomposition": variance.to_dict(orient="records"),
        "multiplicity": "Holm adjustment within each comparison family for macro-centre AUPRC",
    }
    with (output / "statistical_analysis.json").open("w", encoding="utf-8") as sink:
        json.dump(report, sink, ensure_ascii=False, indent=2)
    print(f"Wrote formal statistics for {len(MAIN_METHODS)} main methods and {len(ABLATION_REFERENCES)} ablations")


if __name__ == "__main__":
    main()
