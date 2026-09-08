#!/usr/bin/env python3
"""Aggregate the post-screen clean-M3 source-only supplement.

The report labels every finding exploratory and never opens an outer-test label.
The unit of resampling is one validation case, stratified by source centre.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("cluster_selected_control", "cluster_sufficiency_control", "cluster_pure_m3")
FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_clean_m3_supplement.json"))
    return parser.parse_args()


def macro_ap(labels: np.ndarray, probabilities: np.ndarray, centres: np.ndarray) -> float:
    values = []
    for centre in sorted(set(centres.tolist())):
        include = centres == centre
        if len(np.unique(labels[include])) == 2:
            values.append(float(average_precision_score(labels[include], probabilities[include])))
    return float(np.mean(values)) if values else float("nan")


def paired_bootstrap(labels: np.ndarray, left: np.ndarray, right: np.ndarray, centres: np.ndarray, seed: int, draws: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(draws):
        index = np.concatenate([
            part[rng.integers(0, len(part), size=len(part))]
            for centre in sorted(set(centres.tolist()))
            if len(part := np.flatnonzero(centres == centre))
        ])
        value = macro_ap(labels[index], left[index], centres[index]) - macro_ap(labels[index], right[index], centres[index])
        if np.isfinite(value):
            results.append(value)
    values = np.asarray(results, dtype=float)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(array)), "median": float(np.nanmedian(array)), "min": float(np.nanmin(array)), "max": float(np.nanmax(array)), "n": int(np.isfinite(array).sum())}


def load(output: Path, arm: str, fold: str, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    root = output / "runs" / arm / fold / f"seed_{seed}"
    metric_path, audit_path = root / "source_validation_metrics.json", root / "source_validation_audits.npz"
    if not metric_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"Missing clean-M3 supplementary run: {root}")
    with np.load(audit_path) as archive:
        audit = {key: archive[key] for key in archive.files}
    return json.loads(metric_path.read_text(encoding="utf-8")), audit


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    cache = torch.load(config["feature_cache"], map_location="cpu", weights_only=False)
    centres = np.asarray(cache["centers"], dtype=object)
    runs = {(arm, fold, int(seed)): load(output, arm, fold, int(seed)) for arm in ARMS for fold in FOLDS for seed in config["seeds"]}
    family = []
    for arm in ARMS:
        auprc, cost, contrast, matched = [], [], [], []
        for fold in FOLDS:
            for seed in config["seeds"]:
                metrics, _ = runs[(arm, fold, int(seed))]
                report = metrics["adaptive_retention"]
                auprc.append(float(report["macro_auprc"]))
                cost.append(float(report["mean_visual_cost"]))
                contrast.append(float(report["mean_full_context_selected_vs_matched_unselected_deletion_delta"]))
                matched.append(float(report["n_modality_matched_deletions"]))
        family.append({"arm": arm, "adaptive_macro_auprc": summarize(auprc), "adaptive_visual_cost": summarize(cost), "modality_matched_necessity_contrast": summarize(contrast), "n_modality_matched_deletions": summarize(matched)})
    comparisons = []
    for label, left, right in (
        ("selected_set_sufficiency_loss", "cluster_sufficiency_control", "cluster_selected_control"),
        ("pure_M3_minimality_loss", "cluster_pure_m3", "cluster_sufficiency_control"),
    ):
        deltas, bootstrap = [], []
        for fold_number, fold in enumerate(FOLDS):
            for seed in config["seeds"]:
                _, a = runs[(left, fold, int(seed))]
                _, b = runs[(right, fold, int(seed))]
                policy = "adaptive_retention"
                indices = a[f"{policy}__global_indices"]
                labels = a[f"{policy}__labels"]
                if not np.array_equal(indices, b[f"{policy}__global_indices"]) or not np.array_equal(labels, b[f"{policy}__labels"]):
                    raise AssertionError("Clean-M3 source-validation cases are not aligned")
                point = macro_ap(labels, a[f"{policy}__probabilities"], centres[indices]) - macro_ap(labels, b[f"{policy}__probabilities"], centres[indices])
                mean, low, high = paired_bootstrap(labels, a[f"{policy}__probabilities"], b[f"{policy}__probabilities"], centres[indices], int(seed) + 31 * fold_number)
                deltas.append(point)
                bootstrap.append({"fold": fold, "seed": int(seed), "observed_delta": point, "bootstrap_mean": mean, "ci95": [low, high]})
        comparisons.append({"comparison": label, "macro_auprc_delta_left_minus_right": summarize(deltas), "within_centre_case_bootstrap": bootstrap})
    report = {
        "experiment_id": config["experiment_id"], "analysis": "post-screen exploratory clean-M3 source-only supplement", "n_runs": len(runs),
        "test_labels_opened": False, "outer_test_predictions_written": False, "claim_boundary": config["claim_boundary"],
        "families": family, "clean_component_contrasts": comparisons,
    }
    (output / "clean_m3_supplement_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ICES clean-M3 post-screen supplement", "", "Exploratory source-only supplement; no outer-test outcome was evaluated.", "", "| Arm | median adaptive macro-AUPRC | median visual units | median modality-matched deletion contrast |", "|---|---:|---:|---:|"]
    for item in family:
        lines.append(f"| {item['arm']} | {item['adaptive_macro_auprc']['median']:.4f} | {item['adaptive_visual_cost']['median']:.2f} | {item['modality_matched_necessity_contrast']['median']:.4f} |")
    lines.extend(["", "The pure M3 contrast is `cluster_pure_m3 − cluster_sufficiency_control`: all classification and sufficiency terms are identical, so its difference isolates the declared deletion-minimality regularizer.", ""])
    (output / "CLEAN_M3_SUPPLEMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(output / "clean_m3_supplement_summary.json"), "n_runs": len(runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
