#!/usr/bin/env python3
"""Aggregate predeclared ICES M1--M3 source-validation ablations.

This report intentionally stops at source-train/source-validation evidence.
It cannot unlock or substitute for the single frozen exploratory LOCO test, and
it makes no confirmatory transport or clinical-efficiency claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("raw_m1_ablation", "cluster_m1_m2", "cluster_m1_m2_m3")
FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    return parser.parse_args()


def macro_ap(labels: np.ndarray, probabilities: np.ndarray, centres: np.ndarray) -> float:
    values = []
    for centre in sorted(set(centres.tolist())):
        include = centres == centre
        if len(np.unique(labels[include])) == 2:
            values.append(float(average_precision_score(labels[include], probabilities[include])))
    return float(np.mean(values)) if values else float("nan")


def paired_bootstrap_macro_ap_delta(labels: np.ndarray, a: np.ndarray, b: np.ndarray, centres: np.ndarray, seed: int, draws: int = 1000) -> tuple[float, float, float]:
    """Within-centre patient bootstrap of a source-validation macro-AUPRC delta."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(centres.tolist()))
    estimates = []
    for _ in range(draws):
        parts = []
        for centre in unique:
            local = np.flatnonzero(centres == centre)
            if len(local):
                parts.append(local[rng.integers(0, len(local), size=len(local))])
        index = np.concatenate(parts)
        delta = macro_ap(labels[index], a[index], centres[index]) - macro_ap(labels[index], b[index], centres[index])
        if np.isfinite(delta):
            estimates.append(delta)
    if not estimates:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(estimates)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def run_root(output: Path, backbone: str, fold: str, seed: int) -> Path:
    return output / "runs" / backbone / fold / f"seed_{seed}"


def load_run(output: Path, backbone: str, fold: str, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    root = run_root(output, backbone, fold, seed)
    metrics_path, audit_path = root / "source_validation_metrics.json", root / "source_validation_audits.npz"
    if not metrics_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"Incomplete ICES source run: {root}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(audit_path) as archive:
        audit = {key: archive[key] for key in archive.files}
    return metrics, audit


def metric(metrics: dict, policy: str, key: str) -> float:
    return float(metrics[policy][key])


def summarize(values: list[float]) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)), "min": float(np.nanmin(values)), "max": float(np.nanmax(values)), "n": int(np.isfinite(values).sum())}


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    cache = torch.load(output / config["image_encoder"]["feature_file"], map_location="cpu", weights_only=False)
    centre_by_index = np.asarray(cache["centers"], dtype=object)
    runs: dict[tuple[str, str, int], tuple[dict, dict[str, np.ndarray]]] = {}
    for backbone in BACKBONES:
        for fold in FOLDS:
            for seed in config["seeds"]:
                runs[(backbone, fold, int(seed))] = load_run(output, backbone, fold, int(seed))
    family = {
        "A1_raw_fixed_M1_removed": ("raw_m1_ablation", "attention_fixed_cost"),
        "A2_cluster_fixed_M1_only": ("cluster_m1_m2", "attention_fixed_cost"),
        "A3_cluster_adaptive_M1_M2": ("cluster_m1_m2", "adaptive_retention"),
        "A4_ICES_full_M1_M2_M3": ("cluster_m1_m2_m3", "adaptive_retention"),
        "C0_full_evidence": ("cluster_m1_m2_m3", "full_evidence"),
        "C1_attention_fixed_cost": ("cluster_m1_m2_m3", "attention_fixed_cost"),
        "C2_random_fixed_cost": ("cluster_m1_m2_m3", "random_fixed_cost"),
    }
    rows = []
    for name, (backbone, policy) in family.items():
        auprcs, costs, deletion, contrast = [], [], [], []
        for fold in FOLDS:
            for seed in config["seeds"]:
                metrics, _ = runs[(backbone, fold, int(seed))]
                report = metrics[policy]
                auprcs.append(float(report["macro_auprc"]))
                costs.append(float(report["mean_visual_cost"]))
                deletion.append(float(report["mean_selected_deletion_risk_increase"]))
                contrast.append(float(report["mean_full_context_selected_vs_matched_unselected_deletion_delta"]))
        rows.append({"family": name, "macro_auprc": summarize(auprcs), "mean_visual_cost": summarize(costs), "selected_set_deletion_delta": summarize(deletion), "full_context_selected_vs_matched_unselected": summarize(contrast)})
    comparisons = {
        "M1_cluster_pooling_vs_matched_single_frame": (("cluster_m1_m2", "attention_fixed_cost"), ("raw_m1_ablation", "attention_fixed_cost")),
        "M2_adaptive_retention_vs_fixed_budget": (("cluster_m1_m2", "adaptive_retention"), ("cluster_m1_m2", "attention_fixed_cost")),
        "M3_deletion_objective_vs_no_deletion_objective": (("cluster_m1_m2_m3", "adaptive_retention"), ("cluster_m1_m2", "adaptive_retention")),
        "Full_ICES_vs_random_matched_cost": (("cluster_m1_m2_m3", "adaptive_retention"), ("cluster_m1_m2_m3", "random_fixed_cost")),
    }
    comparison_rows = []
    for name, (left, right) in comparisons.items():
        deltas, cost_deltas, bootstrap = [], [], []
        for fold_index, fold in enumerate(FOLDS):
            for seed in config["seeds"]:
                _, left_audit = runs[(left[0], fold, int(seed))]
                _, right_audit = runs[(right[0], fold, int(seed))]
                lp = left_audit[f"{left[1]}__probabilities"]
                rp = right_audit[f"{right[1]}__probabilities"]
                labels = left_audit[f"{left[1]}__labels"]
                li, ri = left_audit[f"{left[1]}__global_indices"], right_audit[f"{right[1]}__global_indices"]
                if not np.array_equal(li, ri) or not np.array_equal(labels, right_audit[f"{right[1]}__labels"]):
                    raise AssertionError(f"Source validation case alignment failed for {name}, {fold}, {seed}")
                centres = centre_by_index[li]
                observed = macro_ap(labels, lp, centres) - macro_ap(labels, rp, centres)
                mean, low, high = paired_bootstrap_macro_ap_delta(labels, lp, rp, centres, seed=int(seed) + fold_index * 31)
                deltas.append(observed)
                cost_deltas.append(float(left_audit[f"{left[1]}__visual_cost"].mean() - right_audit[f"{right[1]}__visual_cost"].mean()))
                bootstrap.append({"fold": fold, "seed": int(seed), "observed_delta": observed, "bootstrap_mean": mean, "ci95": [low, high]})
        comparison_rows.append({"comparison": name, "macro_auprc_delta_left_minus_right": summarize(deltas), "mean_visual_cost_delta_left_minus_right": summarize(cost_deltas), "within_centre_bootstrap": bootstrap})
    # Locked screening gates.  Fold passes only when at least two of its three
    # fixed-seed source validations have a positive case-bootstrap lower bound.
    full_backbone, full_policy = family["A4_ICES_full_M1_M2_M3"]
    fold_gate = {}
    for fold in FOLDS:
        seed_reports = []
        for seed in config["seeds"]:
            metrics, _ = runs[(full_backbone, fold, int(seed))]
            ci = metrics[full_policy]["full_context_selected_vs_matched_unselected_deletion_delta_ci95"]
            seed_reports.append({"seed": int(seed), "ci95": ci, "passes": bool(np.isfinite(ci[0]) and ci[0] > 0)})
        fold_gate[fold] = {"seed_reports": seed_reports, "passes_majority_seed_rule": sum(item["passes"] for item in seed_reports) >= 2}
    a4_auprc = [metric(runs[(full_backbone, fold, int(seed))][0], full_policy, "macro_auprc") for fold in FOLDS for seed in config["seeds"]]
    c0_auprc = [metric(runs[(family["C0_full_evidence"][0], fold, int(seed))][0], family["C0_full_evidence"][1], "macro_auprc") for fold in FOLDS for seed in config["seeds"]]
    c1_auprc = [metric(runs[(family["C1_attention_fixed_cost"][0], fold, int(seed))][0], family["C1_attention_fixed_cost"][1], "macro_auprc") for fold in FOLDS for seed in config["seeds"]]
    a4_cost = [metric(runs[(full_backbone, fold, int(seed))][0], full_policy, "mean_visual_cost") for fold in FOLDS for seed in config["seeds"]]
    c0_cost = [metric(runs[(family["C0_full_evidence"][0], fold, int(seed))][0], family["C0_full_evidence"][1], "mean_visual_cost") for fold in FOLDS for seed in config["seeds"]]
    gates = {
        "necessity_contrast": {"criterion": "at least 4/5 folds pass the majority-seed positive bootstrap-lower-bound rule", "passing_folds": int(sum(item["passes_majority_seed_rule"] for item in fold_gate.values())), "passes": int(sum(item["passes_majority_seed_rule"] for item in fold_gate.values())) >= 4, "by_fold": fold_gate},
        "macro_auprc_vs_full": {"criterion": "median(A4 - C0) >= -0.02", "median_delta": float(np.median(np.asarray(a4_auprc) - np.asarray(c0_auprc))), "passes": bool(np.median(np.asarray(a4_auprc) - np.asarray(c0_auprc)) >= -0.02)},
        "macro_auprc_vs_attention": {"criterion": "median(A4 - C1) >= -0.01", "median_delta": float(np.median(np.asarray(a4_auprc) - np.asarray(c1_auprc))), "passes": bool(np.median(np.asarray(a4_auprc) - np.asarray(c1_auprc)) >= -0.01)},
        "visual_cost": {"criterion": "median A4 cost at least 25% lower than C0", "median_fraction_reduction": float(1.0 - np.median(a4_cost) / np.median(c0_cost)), "passes": bool(1.0 - np.median(a4_cost) / np.median(c0_cost) >= 0.25)},
    }
    gates["all_locked_source_gates_pass"] = bool(all(item["passes"] for key, item in gates.items() if key != "all_locked_source_gates_pass"))
    report = {
        "experiment_id": config["experiment_id"], "analysis": "locked source-train/source-validation M1-M3 ablations", "n_runs": len(runs),
        "test_labels_opened": False, "outer_test_predictions_written": False,
        "claim_boundary": config["claim_boundary"], "families": rows, "module_comparisons": comparison_rows, "source_gates": gates,
    }
    destination = output / "source_ablation_summary.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# ICES-v1 source-only M1-M3 ablation summary", "",
        "This is a developmental source-train/source-validation screen. It contains no held-out LOCO prediction or label evaluation and is not confirmatory clinical validation.", "",
        "| Family | median macro-AUPRC | median visual units | median selected-set deletion risk increase |", "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['family']} | {row['macro_auprc']['median']:.4f} | {row['mean_visual_cost']['median']:.2f} | {row['selected_set_deletion_delta']['median']:.4f} |")
    lines.extend(["", "## Locked source gates", ""])
    for key, item in gates.items():
        if key != "all_locked_source_gates_pass":
            lines.append(f"- {key}: {'PASS' if item['passes'] else 'FAIL'} — {item['criterion']}")
    lines.extend(["", f"Overall: **{'PASS' if gates['all_locked_source_gates_pass'] else 'FAIL'}**. This result does not unlock a confirmatory claim.", ""])
    (output / "SOURCE_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(destination), "all_locked_source_gates_pass": gates["all_locked_source_gates_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
