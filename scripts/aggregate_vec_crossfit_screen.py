#!/usr/bin/env python3
"""Apply the VEC source-validation advancement gate to D4 cross-fitted scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("D0_attention", "D4_crossfit_necessity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/vec_v2_crossfit_screen"))
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
    rng = np.random.default_rng(seed)
    sampled = np.asarray([values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(replicates)])
    return float(values.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    folds = sorted(pd.read_csv(config["fold_assignments"], dtype={"fold": str}).fold.unique().tolist())
    seeds = [int(seed) for seed in config["seeds"]]
    rows = []
    for fold in folds:
        payloads = [torch.load(output / "jobs" / fold / f"seed_{seed}" / "source_validation_arrays.pt", map_location="cpu", weights_only=False) for seed in seeds]
        labels = payloads[0]["labels"].numpy().astype(int)
        centres = np.asarray(payloads[0]["centres"], dtype=object)
        for payload in payloads[1:]:
            if not np.array_equal(labels, payload["labels"].numpy()) or list(centres) != list(payload["centres"]):
                raise AssertionError(f"Unaligned cross-fit seeds for fold={fold}")
        for candidate in CANDIDATES:
            probability = np.mean([item["probabilities"][candidate].numpy() for item in payloads], axis=0)
            delta = np.mean([item["loss_difference_vs_attention"][candidate].numpy() for item in payloads], axis=0)
            rho = np.nanmean(np.stack([item["rank_correlation_vs_attention"][candidate].numpy() for item in payloads]), axis=0)
            mean, low, high = bootstrap_mean(delta, args.bootstrap_replicates, 20266000 + len(rows))
            rows.append(
                {
                    "fold": fold,
                    "candidate": candidate,
                    "n_validation_cases": len(labels),
                    "macro_auprc": macro_auprc(labels, probability, centres),
                    "replacement_loss_delta_vs_attention": mean,
                    "replacement_loss_delta_ci95_low": low,
                    "replacement_loss_delta_ci95_high": high,
                    "mean_rank_correlation_vs_attention": float(np.nanmean(rho)),
                    "mean_selected_visual_atoms": float(payloads[0]["budget"]),
                }
            )
    per_fold = pd.DataFrame(rows)
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(analysis / "per_fold_source_validation.csv", index=False)
    base = per_fold[per_fold.candidate == "D0_attention"][["fold", "macro_auprc", "mean_selected_visual_atoms"]].rename(
        columns={"macro_auprc": "attention_macro_auprc", "mean_selected_visual_atoms": "attention_atoms"}
    )
    candidate = per_fold[per_fold.candidate == "D4_crossfit_necessity"].merge(base, on="fold", validate="one_to_one")
    mechanism_pass_folds = int((candidate.replacement_loss_delta_ci95_low > 0.0).sum())
    median_macro_delta = float(np.median(candidate.macro_auprc - candidate.attention_macro_auprc))
    mean_rho = float(candidate.mean_rank_correlation_vs_attention.mean())
    cost_ok = bool((candidate.mean_selected_visual_atoms <= candidate.attention_atoms + 1e-12).all())
    qualified = bool(mechanism_pass_folds >= 4 and median_macro_delta >= -0.02 and mean_rho < 0.90 and cost_ok)
    decision = {
        "status": "complete",
        "test_lock": "ready_to_open_once" if qualified else "closed",
        "selected_candidate": "D4_crossfit_necessity" if qualified else None,
        "held_out_test_labels_opened": False,
        "mechanism_pass_folds_of_5": mechanism_pass_folds,
        "median_macro_auprc_delta_vs_attention": median_macro_delta,
        "mean_rank_correlation_vs_attention": mean_rho,
        "cost_not_higher": cost_ok,
        "advancement_gate_passed": qualified,
    }
    (analysis / "source_screen_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
