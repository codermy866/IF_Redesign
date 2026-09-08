#!/usr/bin/env python3
"""Run one source-only VEC candidate-screen job for a fold and random seed.

The script never reads the held-out test partition.  It derives all ranking
scores without labels, then uses source-validation labels only for the frozen
mechanism and discrimination summaries written after scoring is complete.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cervix_cogalign.cesl import SLOT_IDS, SourceOnlyDonorPool, sigmoid  # noqa: E402
from cervix_cogalign.verifiable_evidence import (  # noqa: E402
    binary_symmetric_kl,
    posterior_conflict,
    representation_grounding,
    top_slot,
    verified_evidence_score,
)
from evaluate_cesl_fold import infer, load_model, metadata_for_donors, replacement_loss  # noqa: E402
from train_cesl_backbone import load_fold_data  # noqa: E402


CANDIDATES = ("D0_attention", "D1_necessity_kl", "D2_gni", "D3_gni_transport_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("seed", type=int)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/vec_v1_source_screen"))
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def probabilities(logits: np.ndarray) -> np.ndarray:
    return np.asarray(sigmoid(logits), dtype=float)


def macro_auprc(labels: np.ndarray, scores: np.ndarray, centres: np.ndarray) -> float:
    values = []
    for centre in sorted(set(centres.tolist())):
        include = centres == centre
        if len(np.unique(labels[include])) == 2:
            values.append(float(average_precision_score(labels[include], scores[include])))
    if not values:
        raise ValueError("Source-validation split has no centre with both endpoint classes")
    return float(np.mean(values))


def replacement_necessity(
    model,
    data: dict,
    global_indices: list[int],
    full_masks: np.ndarray,
    base_logits: np.ndarray,
    donor_pool: SourceOnlyDonorPool,
    *,
    draws: int,
    q_variant: str,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Mean symmetric posterior KL under outcome-blind source-only replacements."""
    available = data["payload"]["available"].bool().numpy()
    position = {index: local for local, index in enumerate(global_indices)}
    total = np.zeros((len(global_indices), len(SLOT_IDS)), dtype=float)
    count = np.zeros_like(total, dtype=int)
    operations = []
    for slot in range(1, len(SLOT_IDS)):
        recipients = [index for index in global_indices if available[index, slot]]
        operations.extend(donor_pool.edges(recipients, slot, draws=draws, q_variant=q_variant, seed=seed))
    features = data["payload"]["features"]
    clinical = data["clinical"]
    base_probability = probabilities(base_logits)
    for start in range(0, len(operations), batch_size):
        chunk = operations[start : start + batch_size]
        recipients = [edge.recipient_index for edge in chunk]
        donors = [edge.donor_index for edge in chunk]
        slots = np.asarray([edge.slot_index for edge in chunk], dtype=int)
        local = np.asarray([position[index] for index in recipients], dtype=int)
        replaced = features[recipients].clone()
        replaced[torch.arange(len(chunk)), torch.as_tensor(slots)] = features[donors, torch.as_tensor(slots)]
        observed = torch.as_tensor(full_masks[local], dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits = model(replaced.to(device), clinical[recipients].to(device), observed)["logit"].detach().cpu().numpy()
        values = binary_symmetric_kl(base_probability[local], probabilities(logits))
        for edge, local_index, value in zip(chunk, local, values):
            total[local_index, edge.slot_index] += float(value)
            count[local_index, edge.slot_index] += 1
    result = np.full_like(total, np.nan, dtype=float)
    valid = count > 0
    result[valid] = total[valid] / count[valid]
    return result[:, 1:]


def information_and_conflict(
    model,
    data: dict,
    global_indices: list[int],
    full_masks: np.ndarray,
    base_logits: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score information gain and predictive disagreement without outcomes."""
    available = full_masks[:, 1:]
    removed_masks: list[np.ndarray] = []
    atom_masks: list[np.ndarray] = []
    repeated_indices: list[int] = []
    references: list[tuple[int, int]] = []
    for row, global_index in enumerate(global_indices):
        for local_slot in np.flatnonzero(available[row]):
            slot = int(local_slot + 1)
            remainder = full_masks[row].copy()
            remainder[slot] = False
            atom_only = np.zeros(len(SLOT_IDS), dtype=bool)
            atom_only[0] = bool(full_masks[row, 0])
            atom_only[slot] = True
            removed_masks.append(remainder)
            atom_masks.append(atom_only)
            repeated_indices.append(global_index)
            references.append((row, local_slot))
    information = np.full(available.shape, np.nan, dtype=float)
    conflict = np.full_like(information, np.nan)
    if not repeated_indices:
        return information, conflict
    remainder_logits, _ = infer(model, data, repeated_indices, np.stack(removed_masks), device, batch_size)
    atom_logits, _ = infer(model, data, repeated_indices, np.stack(atom_masks), device, batch_size)
    base_probability = probabilities(base_logits)
    remainder_probability = probabilities(remainder_logits)
    atom_probability = probabilities(atom_logits)
    for position, (row, local_slot) in enumerate(references):
        information[row, local_slot] = float(binary_symmetric_kl(base_probability[row], remainder_probability[position]))
        conflict[row, local_slot] = float(posterior_conflict(atom_probability[position], remainder_probability[position]))
    return information, conflict


def fixed_budget_masks(scores: np.ndarray, full_masks: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-budget masks and return selected visual slot indices."""
    if budget < 1:
        raise ValueError("The VEC screen requires a positive visual budget")
    visual_available = full_masks[:, 1:]
    masks = np.zeros_like(full_masks, dtype=bool)
    masks[:, 0] = full_masks[:, 0]
    selected = np.full((len(full_masks), budget), -1, dtype=int)
    for row in range(len(full_masks)):
        slots = np.flatnonzero(visual_available[row])
        order = sorted(slots.tolist(), key=lambda slot: (-float(np.nan_to_num(scores[row, slot], nan=-np.inf)), slot))
        for count, local_slot in enumerate(order[:budget]):
            masks[row, local_slot + 1] = True
            selected[row, count] = int(local_slot + 1)
    return masks, selected


def per_case_rank_correlation(left: np.ndarray, right: np.ndarray, available: np.ndarray) -> np.ndarray:
    result = np.full(len(left), np.nan, dtype=float)
    for row in range(len(left)):
        slots = np.flatnonzero(available[row])
        if len(slots) >= 3:
            value = spearmanr(left[row, slots], right[row, slots]).statistic
            result[row] = float(value) if np.isfinite(value) else np.nan
    return result


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
    if not source_train_indices or not validation_indices:
        raise AssertionError("A VEC screen fold requires non-empty source train and validation partitions")
    model = load_model(base_config, data, args.fold, "selection", args.seed, device)
    donor_pool = SourceOnlyDonorPool(metadata_for_donors(data), source_train_indices)
    available_all = data["payload"]["available"].bool().numpy()
    full_masks = available_all[np.asarray(validation_indices, dtype=int)]
    visual_available = full_masks[:, 1:]
    batch_size = int(base_config["model"]["batch_size"])

    # Label-free score construction: validation labels are not accessed above this line.
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
    visual_features = data["payload"]["features"][validation_indices, 1:].float().cpu().numpy()
    grounding = representation_grounding(visual_features, visual_available)
    necessity_variants = np.stack([necessity_main, necessity_center, necessity_pooled])
    # Unavailable slots are excluded downstream.  Filling them here avoids a
    # meaningless all-NaN variance warning while retaining an explicit NaN mask.
    instability = np.std(np.where(np.isfinite(necessity_variants), necessity_variants, 0.0), axis=0)
    instability[~visual_available] = np.nan
    attention = full_attention[:, 1:]
    scores = {
        "D0_attention": attention,
        "D1_necessity_kl": necessity_main,
        "D2_gni": verified_evidence_score(grounding, necessity_main, information, conflict, visual_available),
        "D3_gni_transport_gate": verified_evidence_score(
            grounding, necessity_main, information, conflict, visual_available, instability=instability
        ),
    }
    if tuple(scores) != CANDIDATES:
        raise AssertionError("Unexpected VEC candidate registry")
    masks = {}
    selected = {}
    raw_probabilities = {}
    for name, score in scores.items():
        policy_mask, chosen = fixed_budget_masks(score, full_masks, args.budget)
        logits, _ = infer(model, data, validation_indices, policy_mask, device, batch_size)
        masks[name] = policy_mask
        selected[name] = chosen
        raw_probabilities[name] = probabilities(logits)

    # Frozen post-hoc source-validation audits.  No test partition is read.
    labels = data["payload"]["labels"][validation_indices].detach().cpu().numpy().astype(int)
    centres = np.asarray([str(data["payload"]["centers"][index]) for index in validation_indices], dtype=object)
    observed_loss, _ = replacement_loss(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="main_matched", seed=args.seed, target=labels.astype(np.float32),
        target_name="observed_source_validation_label", device=device, batch_size=batch_size * 4, capture_edges=False,
    )
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
    for name, score in scores.items():
        candidate_top = top_slot(score, visual_available) + 1
        candidate_loss = observed_loss[np.arange(len(candidate_top)), candidate_top]
        attention_loss = observed_loss[np.arange(len(attention_top)), attention_top]
        difference = candidate_loss - attention_loss
        rho = per_case_rank_correlation(score, attention, visual_available)
        payload["probabilities"][name] = torch.as_tensor(raw_probabilities[name])
        payload["loss_difference_vs_attention"][name] = torch.as_tensor(difference)
        payload["rank_correlation_vs_attention"][name] = torch.as_tensor(rho)
        summary["candidates"][name] = {
            "macro_auprc": macro_auprc(labels, raw_probabilities[name], centres),
            "pooled_auprc": float(average_precision_score(labels, raw_probabilities[name])),
            "mean_selected_visual_atoms": float(masks[name][:, 1:].sum(axis=1).mean()),
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
                "source_only_policy_scoring": True,
                "labels_used_only_after_scoring_for_source_validation_audit": True,
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
