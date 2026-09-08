#!/usr/bin/env python3
"""Screen a VEC chain with a cross-fitted necessary first atom and attention follow-up."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cervix_cogalign.cesl import SourceOnlyDonorPool  # noqa: E402
from cervix_cogalign.verifiable_evidence import representation_grounding, top_slot  # noqa: E402
from evaluate_cesl_fold import infer, load_model, metadata_for_donors, replacement_loss  # noqa: E402
from train_cesl_backbone import load_fold_data  # noqa: E402
from run_vec_crossfit_screen import crossfit_necessity_score  # noqa: E402
from run_vec_source_screen import (  # noqa: E402
    fixed_budget_masks,
    information_and_conflict,
    macro_auprc,
    per_case_rank_correlation,
    probabilities,
    replacement_necessity,
)


CANDIDATES = ("D0_attention", "D5_verified_attention_chain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("seed", type=int)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/vec_v3_constrained_chain_screen"))
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def constrained_chain_masks(necessity: np.ndarray, attention: np.ndarray, full_masks: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one verified atom, then preserve information with attention order."""
    if budget < 2:
        raise ValueError("D5 is defined for at least one verified and one attention-ranked atom")
    visual_available = full_masks[:, 1:]
    masks = np.zeros_like(full_masks, dtype=bool)
    masks[:, 0] = full_masks[:, 0]
    priority = attention.copy()
    first = top_slot(necessity, visual_available)
    for row in range(len(full_masks)):
        slots = np.flatnonzero(visual_available[row]).tolist()
        if not slots:
            continue
        primary = int(first[row])
        masks[row, primary + 1] = True
        # Record the verified first step as highest in the chain ordering, but
        # choose all remaining steps from the frozen diagnostic attention order.
        priority[row, primary] = float(np.nanmax(attention[row, slots]) + 1.0)
        remaining = [slot for slot in slots if slot != primary]
        remaining.sort(key=lambda slot: (-float(attention[row, slot]), slot))
        for slot in remaining[: budget - 1]:
            masks[row, slot + 1] = True
    return masks, first, priority


def main() -> None:
    args = parse_args()
    base_config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    root = Path(args.output_dir) / "jobs" / args.fold / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "complete.json"
    arrays_path = root / "source_validation_arrays.pt"
    if marker.is_file() and arrays_path.is_file():
        print(f"already complete: {root}")
        return
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = load_fold_data(base_config, args.fold)
    source_train_indices = data["parts"]["train"]["index"].astype(int).tolist()
    validation_indices = data["parts"]["val"]["index"].astype(int).tolist()
    model = load_model(base_config, data, args.fold, "selection", args.seed, device)
    donor_pool = SourceOnlyDonorPool(metadata_for_donors(data), source_train_indices)
    full_masks = data["payload"]["available"].bool().numpy()[np.asarray(validation_indices, dtype=int)]
    visual_available = full_masks[:, 1:]
    batch_size = int(base_config["model"]["batch_size"])

    # All representation and intervention features are label-free.
    base_logits, full_attention = infer(model, data, validation_indices, full_masks, device, batch_size)
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
    grounding = representation_grounding(data["payload"]["features"][validation_indices, 1:].float().cpu().numpy(), visual_available)
    variants = np.stack([necessity_main, necessity_center, necessity_pooled])
    instability = np.std(np.where(np.isfinite(variants), variants, 0.0), axis=0)
    instability[~visual_available] = np.nan

    # Outcome supervision is case-wise cross-fitted within source validation.
    labels = data["payload"]["labels"][validation_indices].detach().cpu().numpy().astype(int)
    observed_loss, _ = replacement_loss(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="main_matched", seed=args.seed, target=labels.astype(np.float32),
        target_name="observed_source_validation_label_crossfit_chain", device=device, batch_size=batch_size * 4, capture_edges=False,
    )
    necessity_crossfit = crossfit_necessity_score(
        grounding, necessity_main, information, conflict, instability, observed_loss[:, 1:], visual_available, labels, args.seed
    )
    attention = full_attention[:, 1:]
    chain_mask, verified_first, chain_priority = constrained_chain_masks(necessity_crossfit, attention, full_masks, args.budget)
    attention_mask, _ = fixed_budget_masks(attention, full_masks, args.budget)
    policy_masks = {"D0_attention": attention_mask, "D5_verified_attention_chain": chain_mask}
    scores = {"D0_attention": attention, "D5_verified_attention_chain": chain_priority}
    centres = np.asarray([str(data["payload"]["centers"][index]) for index in validation_indices], dtype=object)
    attention_top = top_slot(attention, visual_available) + 1
    payload = {
        "fold": args.fold,
        "seed": args.seed,
        "budget": int(args.budget),
        "labels": torch.as_tensor(labels),
        "centres": centres.tolist(),
        "probabilities": {},
        "loss_difference_vs_attention": {},
        "rank_correlation_vs_attention": {},
    }
    summary = {"fold": args.fold, "seed": args.seed, "budget": int(args.budget), "candidates": {}, "test_labels_opened": False}
    for name in CANDIDATES:
        logits, _ = infer(model, data, validation_indices, policy_masks[name], device, batch_size)
        probability = probabilities(logits)
        if name == "D5_verified_attention_chain":
            selected_top = verified_first + 1
        else:
            selected_top = attention_top
        difference = observed_loss[np.arange(len(selected_top)), selected_top] - observed_loss[np.arange(len(attention_top)), attention_top]
        rho = per_case_rank_correlation(scores[name], attention, visual_available)
        payload["probabilities"][name] = torch.as_tensor(probability)
        payload["loss_difference_vs_attention"][name] = torch.as_tensor(difference)
        payload["rank_correlation_vs_attention"][name] = torch.as_tensor(rho)
        summary["candidates"][name] = {
            "macro_auprc": macro_auprc(labels, probability, centres),
            "mean_observed_replacement_loss_difference_vs_attention": float(np.nanmean(difference)),
            "mean_rank_correlation_vs_attention": float(np.nanmean(rho)),
        }
    torch.save(payload, arrays_path)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "fold": args.fold,
                "seed": args.seed,
                "source_only_crossfit_calibration": True,
                "verified_first_step": True,
                "held_out_test_labels_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
