#!/usr/bin/env python3
"""Aggregate the locked VEC source-validation screen and apply its advancement gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("D0_attention", "D1_necessity_kl", "D2_gni", "D3_gni_transport_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/vec_v1_source_screen"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def macro_auprc(labels: np.ndarray, scores: np.ndarray, centres: np.ndarray) -> float:
    values = []
    for centre in sorted(set(centres.tolist())):
        include = centres == centre
        if len(np.unique(labels[include])) == 2:
            values.append(float(average_precision_score(labels[include], scores[include])))
    return float(np.mean(values))


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        means.append(float(values[rng.integers(0, len(values), size=len(values))].mean()))
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    assignments = pd.read_csv(config["fold_assignments"], dtype={"fold": str})
    folds = sorted(assignments.fold.unique().tolist())
    seeds = [int(seed) for seed in config["seeds"]]
    rows = []
    for fold in folds:
        seed_payloads = []
        for seed in seeds:
            path = output / "jobs" / fold / f"seed_{seed}" / "source_validation_arrays.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed source-only VEC job: {path}")
            seed_payloads.append(torch.load(path, map_location="cpu", weights_only=False))
        labels = seed_payloads[0]["labels"].numpy().astype(int)
        centres = np.asarray(seed_payloads[0]["centres"], dtype=object)
        for payload in seed_payloads[1:]:
            if not np.array_equal(labels, payload["labels"].numpy()) or list(centres) != list(payload["centres"]):
                raise AssertionError(f"Seed payloads are not aligned for fold={fold}")
        for candidate in CANDIDATES:
            probabilities = np.mean([payload["probabilities"][candidate].numpy() for payload in seed_payloads], axis=0)
            loss_delta = np.mean([payload["loss_difference_vs_attention"][candidate].numpy() for payload in seed_payloads], axis=0)
            rho = np.nanmean(np.stack([payload["rank_correlation_vs_attention"][candidate].numpy() for payload in seed_payloads]), axis=0)
            mean_delta, ci_low, ci_high = bootstrap_mean(loss_delta, args.bootstrap_replicates, 20265000 + len(rows))
            rows.append(
                {
                    "fold": fold,
                    "candidate": candidate,
                    "n_validation_cases": int(len(labels)),
                    "macro_auprc": macro_auprc(labels, probabilities, centres),
                    "pooled_auprc": float(average_precision_score(labels, probabilities)),
                    "replacement_loss_delta_vs_attention": mean_delta,
                    "replacement_loss_delta_ci95_low": ci_low,
                    "replacement_loss_delta_ci95_high": ci_high,
                    "mean_rank_correlation_vs_attention": float(np.nanmean(rho)),
                    "mean_selected_visual_atoms": float(seed_payloads[0]["budget"]),
                }
            )
    per_fold = pd.DataFrame(rows)
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(analysis / "per_fold_source_validation.csv", index=False)
    gate_rows = []
    for candidate in CANDIDATES[1:]:
        part = per_fold[per_fold.candidate == candidate].copy()
        attention = per_fold[per_fold.candidate == "D0_attention"][["fold", "macro_auprc", "mean_selected_visual_atoms"]].rename(
            columns={"macro_auprc": "attention_macro_auprc", "mean_selected_visual_atoms": "attention_atoms"}
        )
        part = part.merge(attention, on="fold", validate="one_to_one")
        mechanism_pass_folds = int((part.replacement_loss_delta_ci95_low > 0.0).sum())
        median_macro_delta = float(np.median(part.macro_auprc - part.attention_macro_auprc))
        mean_rho = float(part.mean_rank_correlation_vs_attention.mean())
        cost_ok = bool((part.mean_selected_visual_atoms <= part.attention_atoms + 1e-12).all())
        decision = bool(mechanism_pass_folds >= 4 and median_macro_delta >= -0.02 and mean_rho < 0.90 and cost_ok)
        gate_rows.append(
            {
                "candidate": candidate,
                "mechanism_pass_folds_of_5": mechanism_pass_folds,
                "median_macro_auprc_delta_vs_attention": median_macro_delta,
                "mean_rank_correlation_vs_attention": mean_rho,
                "cost_not_higher": cost_ok,
                "advancement_gate_passed": decision,
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(analysis / "advancement_gate.csv", index=False)
    qualified = gate[gate.advancement_gate_passed].candidate.tolist()
    selected = None
    if qualified:
        summary = per_fold[per_fold.candidate.isin(qualified)].groupby("candidate", as_index=False).macro_auprc.mean()
        selected = str(summary.sort_values(["macro_auprc", "candidate"], ascending=[False, True]).iloc[0].candidate)
    decision = {
        "status": "complete",
        "test_lock": "ready_to_open_once" if selected else "closed",
        "selected_candidate": selected,
        "advancement_rule": "four_of_five positive paired-bootstrap lower bounds; median macro-AUPRC delta >= -0.02; mean rank rho < 0.90; no higher cost",
        "held_out_test_labels_opened": False,
        "candidates": gate_rows,
    }
    (analysis / "source_screen_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
