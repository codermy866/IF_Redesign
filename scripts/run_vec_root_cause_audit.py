#!/usr/bin/env python3
"""Run a source-only diagnostic audit of VEC measurement and ranking signal.

This is not a candidate search and it never opens an outer held-out partition.
It quantifies whether label-free verifier quantities recover observed-label
replacement loss on the fold-specific source-validation cases only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cervix_cogalign.cesl import SLOT_IDS, SourceOnlyDonorPool  # noqa: E402
from cervix_cogalign.verifiable_evidence import representation_grounding, top_slot  # noqa: E402
from evaluate_cesl_fold import infer, load_model, metadata_for_donors, replacement_loss  # noqa: E402
from run_vec_crossfit_screen import crossfit_necessity_score  # noqa: E402
from run_vec_source_screen import information_and_conflict, replacement_necessity  # noqa: E402
from train_cesl_backbone import load_fold_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("seed", type=int)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "auto_research/verifiable_evidence_chain_20260905/root_cause_audit"))
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def per_case_spearman(left: np.ndarray, right: np.ndarray, available: np.ndarray) -> np.ndarray:
    result = np.full(len(left), np.nan, dtype=float)
    for row in range(len(left)):
        slots = np.flatnonzero(available[row])
        if len(slots) >= 3:
            correlation = spearmanr(left[row, slots], right[row, slots]).statistic
            result[row] = float(correlation) if np.isfinite(correlation) else np.nan
    return result


def score_diagnostic(name: str, score: np.ndarray, observed_loss: np.ndarray, available: np.ndarray) -> dict:
    valid = available & np.isfinite(score) & np.isfinite(observed_loss)
    flat_score = score[valid]
    flat_loss = observed_loss[valid]
    pooled = spearmanr(flat_score, flat_loss).statistic if len(flat_score) >= 3 else np.nan
    within = per_case_spearman(score, observed_loss, available)
    selected = top_slot(score, available)
    observed_top = top_slot(observed_loss, available)
    selected_loss = observed_loss[np.arange(len(selected)), selected]
    return {
        "name": name,
        "pooled_atom_spearman_to_observed_loss": finite(pooled),
        "mean_within_case_spearman_to_observed_loss": finite(np.nanmean(within)),
        "median_within_case_spearman_to_observed_loss": finite(np.nanmedian(within)),
        "top_atom_matches_observed_top_rate": finite(np.mean(selected == observed_top)),
        "mean_observed_loss_at_selected_top": finite(np.nanmean(selected_loss)),
        "n_valid_atoms": int(valid.sum()),
        "n_cases_with_rankable_atoms": int(np.isfinite(within).sum()),
    }


def donor_diagnostic(pool: SourceOnlyDonorPool, recipients: list[int], available: np.ndarray, *, draws: int, seed: int) -> dict:
    levels: Counter[str] = Counter()
    candidate_counts: list[int] = []
    unique_draw_counts: list[int] = []
    for local_row, recipient in enumerate(recipients):
        for local_slot in np.flatnonzero(available[local_row]):
            slot = int(local_slot + 1)
            candidates, level = pool._candidates(recipient, slot, "main_matched")
            levels[level] += 1
            candidate_counts.append(len(candidates))
            edges = pool.edges([recipient], slot, draws=draws, q_variant="main_matched", seed=seed)
            unique_draw_counts.append(len({edge.donor_index for edge in edges}))
    total = sum(levels.values())
    return {
        "recipient_slot_pairs": int(total),
        "matching_level_fraction": {level: float(count / total) for level, count in sorted(levels.items())},
        "candidate_pool_size": {
            "min": int(min(candidate_counts)),
            "median": finite(np.median(candidate_counts)),
            "mean": finite(np.mean(candidate_counts)),
            "max": int(max(candidate_counts)),
        },
        "unique_donors_in_three_draws": {
            "mean": finite(np.mean(unique_draw_counts)),
            "one_donor_rate": finite(np.mean(np.asarray(unique_draw_counts) == 1)),
            "three_donor_rate": finite(np.mean(np.asarray(unique_draw_counts) == draws)),
        },
    }


def modality_summary(selected: np.ndarray) -> dict:
    kinds = np.asarray(["colposcopy"] * 4 + ["oct"] * 6, dtype=object)
    return {kind: float(np.mean(kinds[selected] == kind)) for kind in ("colposcopy", "oct")}


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir) / "jobs" / args.fold / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "diagnostics.json"
    marker = root / "complete.json"
    if output_path.is_file() and marker.is_file():
        print(f"already complete: {root}")
        return

    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = load_fold_data(config, args.fold)
    train_indices = data["parts"]["train"]["index"].astype(int).tolist()
    validation_indices = data["parts"]["val"]["index"].astype(int).tolist()
    full_masks = data["payload"]["available"].bool().numpy()[np.asarray(validation_indices, dtype=int)]
    visual_available = full_masks[:, 1:]
    labels = data["payload"]["labels"][validation_indices].detach().cpu().numpy().astype(int)
    train_labels = data["payload"]["labels"][train_indices].detach().cpu().numpy().astype(int)
    model = load_model(config, data, args.fold, "selection", args.seed, device)
    donor_pool = SourceOnlyDonorPool(metadata_for_donors(data), train_indices)
    batch_size = int(config["model"]["batch_size"])

    # Construct every verifier component without accessing source-validation labels.
    base_logits, attention_all = infer(model, data, validation_indices, full_masks, device, batch_size)
    necessity_main = replacement_necessity(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="main_matched", seed=args.seed, device=device, batch_size=batch_size * 4,
    )
    necessity_center = replacement_necessity(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="center_only", seed=args.seed, device=device, batch_size=batch_size * 4,
    )
    necessity_pooled = replacement_necessity(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="pooled_unmatched", seed=args.seed, device=device, batch_size=batch_size * 4,
    )
    information, conflict = information_and_conflict(
        model, data, validation_indices, full_masks, base_logits, device=device, batch_size=batch_size
    )
    grounding = representation_grounding(
        data["payload"]["features"][validation_indices, 1:].float().cpu().numpy(), visual_available
    )
    variants = np.stack([necessity_main, necessity_center, necessity_pooled])
    instability = np.std(np.where(np.isfinite(variants), variants, 0.0), axis=0)
    instability[~visual_available] = np.nan
    attention = attention_all[:, 1:]

    # Outcome labels enter only here, after all label-free quantities are fixed.
    observed_full, _ = replacement_loss(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="main_matched", seed=args.seed, target=labels.astype(np.float32),
        target_name="observed_source_validation_label_root_cause_audit", device=device, batch_size=batch_size * 4,
        capture_edges=False,
    )
    observed_loss = observed_full[:, 1:]
    d4_crossfit = crossfit_necessity_score(
        grounding, necessity_main, information, conflict, instability, observed_loss, visual_available, labels, args.seed
    )
    components = {
        "attention": attention,
        "posterior_necessity": necessity_main,
        "information": information,
        "grounding_proxy": grounding,
        "negative_predictive_conflict": -conflict,
        "negative_q_instability": -instability,
        "D4_crossfit_estimate": d4_crossfit,
    }
    score_results = {name: score_diagnostic(name, score, observed_loss, visual_available) for name, score in components.items()}
    d4_valid = visual_available & np.isfinite(d4_crossfit) & np.isfinite(observed_loss)
    d4_r2 = r2_score(observed_loss[d4_valid], d4_crossfit[d4_valid]) if d4_valid.sum() > 1 else np.nan
    attention_top = top_slot(attention, visual_available)
    necessity_top = top_slot(necessity_main, visual_available)
    d4_top = top_slot(d4_crossfit, visual_available)
    q_main_center = per_case_spearman(necessity_main, necessity_center, visual_available)
    q_main_pooled = per_case_spearman(necessity_main, necessity_pooled, visual_available)

    diagnostics = {
        "fold": args.fold,
        "seed": int(args.seed),
        "scope": "source_train/source_validation only; no held-out partition accessed",
        "test_labels_opened": False,
        "sample": {
            "source_train_cases": int(len(train_indices)),
            "source_validation_cases": int(len(validation_indices)),
            "source_train_positive_rate": finite(np.mean(train_labels)),
            "source_validation_positive_rate": finite(np.mean(labels)),
            "mean_available_visual_atoms": finite(visual_available.sum(axis=1).mean()),
            "min_available_visual_atoms": int(visual_available.sum(axis=1).min()),
            "max_available_visual_atoms": int(visual_available.sum(axis=1).max()),
        },
        "donor_distribution": donor_diagnostic(donor_pool, validation_indices, visual_available, draws=args.draws, seed=args.seed),
        "q_rank_agreement": {
            "main_vs_center_only_mean_within_case_spearman": finite(np.nanmean(q_main_center)),
            "main_vs_pooled_unmatched_mean_within_case_spearman": finite(np.nanmean(q_main_pooled)),
        },
        "score_recovery_of_observed_replacement_loss": score_results,
        "crossfit_estimator": {
            "atom_level_out_of_fold_r2": finite(d4_r2),
            "top_atom_agreement_with_attention": finite(np.mean(d4_top == attention_top)),
            "mean_observed_loss_difference_d4_top_minus_attention_top": finite(
                np.nanmean(observed_loss[np.arange(len(d4_top)), d4_top] - observed_loss[np.arange(len(attention_top)), attention_top])
            ),
        },
        "selection": {
            "posterior_necessity_top_agreement_with_attention": finite(np.mean(necessity_top == attention_top)),
            "D4_top_agreement_with_attention": finite(np.mean(d4_top == attention_top)),
            "attention_top_modality": modality_summary(attention_top),
            "posterior_necessity_top_modality": modality_summary(necessity_top),
            "D4_top_modality": modality_summary(d4_top),
        },
        "slot_observed_loss": {
            SLOT_IDS[slot + 1]: finite(np.nanmean(observed_loss[:, slot][visual_available[:, slot]]))
            for slot in range(visual_available.shape[1])
        },
    }
    output_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    marker.write_text(json.dumps({"status": "complete", "test_labels_opened": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"fold": args.fold, "seed": args.seed, "test_labels_opened": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
