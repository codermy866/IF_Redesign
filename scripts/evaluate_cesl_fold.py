#!/usr/bin/env python3
"""Generate frozen CESL policy predictions and post-hoc mechanism audits.

No held-out label enters any policy score, donor match, temperature fit, or
threshold choice.  Held-out labels are attached only after policy predictions
are constructed, for retrospective outcome and replacement-loss audits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cervix_cogalign.cesl import (  # noqa: E402
    CESLSetModel,
    SLOT_IDS,
    SLOT_KINDS,
    SourceOnlyDonorPool,
    fit_temperature,
    sigmoid,
    stable_int,
)
from cervix_cogalign.io import read_json, write_jsonl  # noqa: E402
from cervix_cogalign.metrics import select_balanced_accuracy_threshold  # noqa: E402
from train_cesl_backbone import BACKBONES, get_run_root, load_fold_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("seed", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def evaluation_root(config: dict, fold: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "evaluations" / fold / f"seed_{seed}"


def load_model(config: dict, data: dict, fold: str, backbone: str, seed: int, device: torch.device) -> CESLSetModel:
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    checkpoint_path = get_run_root(config, backbone, fold, seed) / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing completed CESL checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = CESLSetModel(
        visual_dim=int(checkpoint["visual_feature_dim"]),
        clinical_dim=int(checkpoint["clinical_feature_dim"]),
        token_dim=int(model_config["token_dim"]),
        transformer_layers=int(model_config["transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.inference_mode()
def infer(
    model: CESLSetModel,
    data: dict,
    global_indices: list[int],
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(global_indices) != len(masks):
        raise ValueError("Inference indices and masks are misaligned")
    logits: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    features = data["payload"]["features"]
    clinical = data["clinical"]
    for start in range(0, len(global_indices), batch_size):
        stop = min(start + batch_size, len(global_indices))
        index = global_indices[start:stop]
        observed = torch.as_tensor(masks[start:stop], dtype=torch.bool, device=device)
        output = model(features[index].to(device), clinical[index].to(device), observed)
        logits.append(output["logit"].detach().cpu().numpy())
        attentions.append(output["attention"].detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(attentions)


def metadata_for_donors(data: dict) -> list[dict]:
    available = data["payload"]["available"].bool().numpy()
    return [
        {
            "id": str(data["ids"][index]),
            "patient_id": str(data["payload"]["patient_ids"][index]),
            "center": str(data["payload"]["centers"][index]),
            "nuisance_stratum": tuple(data["payload"]["nuisance_strata"][index]),
            "available": available[index].tolist(),
        }
        for index in range(len(data["ids"]))
    ]


def replacement_loss(
    model: CESLSetModel,
    data: dict,
    global_indices: list[int],
    full_masks: np.ndarray,
    base_logits: np.ndarray,
    donor_pool: SourceOnlyDonorPool,
    *,
    draws: int,
    q_variant: str,
    seed: int,
    target: np.ndarray,
    target_name: str,
    device: torch.device,
    batch_size: int,
    capture_edges: bool,
) -> tuple[np.ndarray, list[dict]]:
    """Conditional replacement loss for a fixed label-free or observed target.

    ``target`` is a predicted class for policy scoring or the observed label for
    a post-hoc audit.  Donor selection never receives labels in either mode.
    """
    if len(global_indices) != len(target):
        raise ValueError("Counterfactual target is misaligned")
    features = data["payload"]["features"]
    clinical = data["clinical"]
    available = data["payload"]["available"].bool().numpy()
    position = {index: local for local, index in enumerate(global_indices)}
    score_sum = np.zeros((len(global_indices), len(SLOT_IDS)), dtype=np.float64)
    score_count = np.zeros_like(score_sum, dtype=np.int64)
    edge_rows: list[dict] = []
    base_target = torch.as_tensor(target.astype(np.float32))
    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(base_logits, dtype=torch.float32), base_target, reduction="none"
    ).numpy()
    operations = []
    for slot_index in range(1, len(SLOT_IDS)):
        recipients = [index for index in global_indices if available[index, slot_index]]
        for edge in donor_pool.edges(recipients, slot_index, draws=draws, q_variant=q_variant, seed=seed):
            operations.append(edge)
    for start in range(0, len(operations), batch_size):
        chunk = operations[start : start + batch_size]
        recipients = [edge.recipient_index for edge in chunk]
        donors = [edge.donor_index for edge in chunk]
        local = np.asarray([position[index] for index in recipients], dtype=int)
        slots = np.asarray([edge.slot_index for edge in chunk], dtype=int)
        x = features[recipients].clone()
        x[torch.arange(len(chunk)), torch.as_tensor(slots)] = features[donors, torch.as_tensor(slots)]
        observed = torch.as_tensor(full_masks[local], dtype=torch.bool, device=device)
        with torch.inference_mode():
            output = model(x.to(device), clinical[recipients].to(device), observed)
        replacement_logits = output["logit"].detach().cpu()
        replacement_loss_value = torch.nn.functional.binary_cross_entropy_with_logits(
            replacement_logits, torch.as_tensor(target[local], dtype=torch.float32), reduction="none"
        ).numpy()
        delta = replacement_loss_value - base_loss[local]
        for edge, local_index, value in zip(chunk, local, delta):
            score_sum[local_index, edge.slot_index] += float(value)
            score_count[local_index, edge.slot_index] += 1
            if capture_edges:
                edge_rows.append(
                    {
                        "recipient_id": str(data["ids"][edge.recipient_index]),
                        "recipient_patient_id": str(data["payload"]["patient_ids"][edge.recipient_index]),
                        "recipient_center": str(data["payload"]["centers"][edge.recipient_index]),
                        "donor_id": str(data["ids"][edge.donor_index]),
                        "donor_patient_id": str(data["payload"]["patient_ids"][edge.donor_index]),
                        "donor_center": str(data["payload"]["centers"][edge.donor_index]),
                        "slot": SLOT_IDS[edge.slot_index],
                        "draw": edge.draw,
                        "q_variant": edge.q_variant,
                        "matching_level": edge.matching_level,
                        "target_mode": target_name,
                        "replacement_loss_delta": float(value),
                        "donor_label_used_for_matching": False,
                    }
                )
    scores = np.full_like(score_sum, np.nan, dtype=np.float64)
    valid = score_count > 0
    scores[valid] = score_sum[valid] / score_count[valid]
    return scores, edge_rows


def slot_order(scores: np.ndarray, available: np.ndarray, row: int) -> list[int]:
    candidates = [slot for slot in range(1, len(SLOT_IDS)) if available[row, slot]]
    return sorted(candidates, key=lambda slot: (-float(np.nan_to_num(scores[row, slot], nan=-1e9)), slot))


def fixed_budget_masks(
    scores: np.ndarray,
    available: np.ndarray,
    *,
    budget: int,
    policy: str,
    global_indices: list[int],
    seed: int,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    masks = np.zeros_like(available, dtype=bool)
    selected: list[list[int]] = []
    for row, global_index in enumerate(global_indices):
        masks[row, 0] = available[row, 0]
        candidates = slot_order(scores, available, row)
        if policy == "random":
            rng = np.random.default_rng(stable_int("cesl_random_control", global_index, seed=seed))
            rng.shuffle(candidates)
        chosen = candidates[: min(int(budget), len(candidates))]
        masks[row, chosen] = True
        selected.append(chosen)
    return masks, selected, np.zeros(len(global_indices), dtype=bool)


def all_evidence_masks(available: np.ndarray) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    selected = [[slot for slot in range(1, len(SLOT_IDS)) if available[row, slot]] for row in range(len(available))]
    return available.copy(), selected, np.zeros(len(available), dtype=bool)


def adaptive_masks(
    model: CESLSetModel,
    data: dict,
    global_indices: list[int],
    scores: np.ndarray,
    available: np.ndarray,
    *,
    gamma: float,
    epsilon: float | None,
    minimum_visual_atoms: int,
    maximum_visual_atoms: int,
    fixed_order: list[str] | None,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    orders: list[list[int]] = []
    for row in range(len(global_indices)):
        if fixed_order is None:
            order = slot_order(scores, available, row)
        else:
            order = [SLOT_IDS.index(name) for name in fixed_order if available[row, SLOT_IDS.index(name)]]
        orders.append(order[: min(maximum_visual_atoms, len(order))])
    staged_indices: list[int] = []
    staged_masks: list[np.ndarray] = []
    stage_ref: list[tuple[int, int]] = []
    for row, (global_index, order) in enumerate(zip(global_indices, orders)):
        for count in range(1, len(order) + 1):
            mask = np.zeros(len(SLOT_IDS), dtype=bool)
            mask[0] = bool(available[row, 0])
            mask[order[:count]] = True
            staged_indices.append(global_index)
            staged_masks.append(mask)
            stage_ref.append((row, count))
    if not staged_masks:
        raise AssertionError("CESL policy found no visual atoms in a complete-modality cohort")
    logits, _ = infer(model, data, staged_indices, np.stack(staged_masks), device, batch_size)
    margin_by_stage = {(row, count): abs(2.0 * float(sigmoid(logit)) - 1.0) for (row, count), logit in zip(stage_ref, logits)}
    masks = np.zeros_like(available, dtype=bool)
    selected: list[list[int]] = []
    stopped = np.zeros(len(global_indices), dtype=bool)
    for row, order in enumerate(orders):
        selected_count = len(order)
        for count in range(1, len(order) + 1):
            remaining = order[count:]
            residual = max((float(np.nan_to_num(scores[row, slot], nan=-1e9)) for slot in remaining), default=-float("inf"))
            sufficient_margin = margin_by_stage[(row, count)] >= gamma
            sufficient_residual = epsilon is None or residual <= epsilon
            if count >= minimum_visual_atoms and sufficient_margin and sufficient_residual:
                selected_count = count
                stopped[row] = True
                break
        chosen = order[:selected_count]
        masks[row, 0] = available[row, 0]
        masks[row, chosen] = True
        selected.append(chosen)
    return masks, selected, stopped


def record_policy(
    *,
    policy: str,
    fold: str,
    seed: int,
    split: str,
    global_indices: list[int],
    logits: np.ndarray,
    labels: np.ndarray,
    data: dict,
    selected: list[list[int]],
    stopped: np.ndarray,
    temperature: float,
    threshold: float,
) -> list[dict]:
    probabilities = np.asarray(sigmoid(logits / temperature), dtype=float)
    output = []
    for local, global_index in enumerate(global_indices):
        slots = [SLOT_IDS[slot] for slot in selected[local]]
        output.append(
            {
                "id": str(data["ids"][global_index]),
                "patient_id": str(data["payload"]["patient_ids"][global_index]),
                "center": str(data["payload"]["centers"][global_index]),
                "fold": fold,
                "seed": seed,
                "split": split,
                "policy": policy,
                "label": int(labels[local]),
                "logit": float(logits[local]),
                "probability": float(probabilities[local]),
                "threshold": float(threshold),
                "predicted_label": int(probabilities[local] >= threshold),
                "temperature_source_validation": float(temperature),
                "selected_visual_atoms": int(len(slots)),
                "relative_evidence_cost": float(len(slots)),
                "stopping_criterion_met": bool(stopped[local]),
                "selected_slots": json.dumps(slots, ensure_ascii=False),
            }
        )
    return output


def mechanism_rows(
    *,
    fold: str,
    seed: int,
    model_name: str,
    global_indices: list[int],
    available: np.ndarray,
    cevs_predicted: np.ndarray,
    cevs_observed: np.ndarray,
    attention: np.ndarray,
    data: dict,
) -> list[dict]:
    rows = []
    for local, global_index in enumerate(global_indices):
        candidates = [slot for slot in range(1, len(SLOT_IDS)) if available[local, slot]]
        if not candidates:
            continue
        cev_slot = max(candidates, key=lambda slot: float(np.nan_to_num(cevs_predicted[local, slot], nan=-1e9)))
        attention_slot = max(candidates, key=lambda slot: float(attention[local, slot]))
        same_kind = [slot for slot in candidates if SLOT_KINDS[slot] == SLOT_KINDS[cev_slot]]
        random_slot = same_kind[stable_int("matched_random_audit", global_index, cev_slot, seed=seed) % len(same_kind)]
        rows.append(
            {
                "id": str(data["ids"][global_index]),
                "patient_id": str(data["payload"]["patient_ids"][global_index]),
                "center": str(data["payload"]["centers"][global_index]),
                "fold": fold,
                "seed": seed,
                "model": model_name,
                "cev_slot": SLOT_IDS[cev_slot],
                "attention_slot": SLOT_IDS[attention_slot],
                "matched_random_slot": SLOT_IDS[random_slot],
                "cev_predicted_score": float(cevs_predicted[local, cev_slot]),
                "attention_score": float(attention[local, attention_slot]),
                "observed_replacement_loss_cev": float(cevs_observed[local, cev_slot]),
                "observed_replacement_loss_attention": float(cevs_observed[local, attention_slot]),
                "observed_replacement_loss_random": float(cevs_observed[local, random_slot]),
            }
        )
    return rows


def rank_rows(
    *,
    fold: str,
    seed: int,
    model_name: str,
    global_indices: list[int],
    available: np.ndarray,
    cevs: np.ndarray,
    attention: np.ndarray,
    data: dict,
) -> list[dict]:
    rows = []
    centers = np.asarray([str(data["payload"]["centers"][index]) for index in global_indices], dtype=object)
    for center in sorted(set(centers.tolist())):
        include = centers == center
        values = []
        for slot in range(1, len(SLOT_IDS)):
            valid = include & available[:, slot]
            if valid.any():
                values.append((slot, float(np.nanmean(cevs[valid, slot])), float(np.mean(attention[valid, slot]))))
        cev_order = {slot: rank for rank, (slot, _, _) in enumerate(sorted(values, key=lambda item: -item[1]), start=1)}
        attention_order = {slot: rank for rank, (slot, _, _) in enumerate(sorted(values, key=lambda item: -item[2]), start=1)}
        for slot, mean_cev, mean_attention in values:
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "center": center,
                    "slot": SLOT_IDS[slot],
                    "mean_predicted_cev": mean_cev,
                    "mean_attention": mean_attention,
                    "cev_rank": cev_order[slot],
                    "attention_rank": attention_order[slot],
                    "n_cases": int((include & available[:, slot]).sum()),
                }
            )
    return rows


def policy_masks(
    model: CESLSetModel,
    data: dict,
    global_indices: list[int],
    available: np.ndarray,
    full_attention: np.ndarray,
    cevs: np.ndarray,
    *,
    fixed_budget: int,
    config: dict,
    device: torch.device,
    seed: int,
    include_full: bool,
) -> dict[str, tuple[np.ndarray, list[list[int]], np.ndarray]]:
    stopping = config["stopping_rule"]
    output: dict[str, tuple[np.ndarray, list[list[int]], np.ndarray]] = {}
    if include_full:
        output["C0_full_evidence"] = all_evidence_masks(available)
    attention_masks, attention_selected, attention_stopped = fixed_budget_masks(
        full_attention, available, budget=fixed_budget, policy="attention", global_indices=global_indices, seed=seed
    )
    cev_masks, cev_selected, cev_stopped = fixed_budget_masks(
        cevs, available, budget=fixed_budget, policy="cev", global_indices=global_indices, seed=seed
    )
    random_masks, random_selected, random_stopped = fixed_budget_masks(
        cevs, available, budget=fixed_budget, policy="random", global_indices=global_indices, seed=seed
    )
    adaptive = adaptive_masks(
        model,
        data,
        global_indices,
        cevs,
        available,
        gamma=float(stopping["margin_gamma"]),
        epsilon=float(stopping["residual_cev_epsilon"]),
        minimum_visual_atoms=int(stopping["minimum_visual_atoms"]),
        maximum_visual_atoms=int(stopping["maximum_visual_atoms"]),
        fixed_order=None,
        device=device,
        batch_size=int(config["model"]["batch_size"]),
    )
    uncertainty = adaptive_masks(
        model,
        data,
        global_indices,
        cevs,
        available,
        gamma=float(stopping["uncertainty_margin_gamma"]),
        epsilon=None,
        minimum_visual_atoms=int(stopping["minimum_visual_atoms"]),
        maximum_visual_atoms=int(stopping["maximum_visual_atoms"]),
        fixed_order=list(stopping["fixed_order"]),
        device=device,
        batch_size=int(config["model"]["batch_size"]),
    )
    output.update(
        {
            "C1_attention_fixed_budget": (attention_masks, attention_selected, attention_stopped),
            "C2_cev_fixed_budget": (cev_masks, cev_selected, cev_stopped),
            "C3_cev_adaptive": adaptive,
            "random_fixed_budget": (random_masks, random_selected, random_stopped),
            "uncertainty_only": uncertainty,
        }
    )
    return output


def raw_logits_for_policies(
    model: CESLSetModel,
    data: dict,
    global_indices: list[int],
    policies: dict[str, tuple[np.ndarray, list[list[int]], np.ndarray]],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output = {}
    for policy, (masks, _, _) in policies.items():
        logits, _ = infer(model, data, global_indices, masks, device, batch_size)
        output[policy] = logits
    return output


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    root = evaluation_root(config, args.fold, args.seed)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "evaluation_complete.json"
    if marker.is_file() and (root / "test_predictions.csv").is_file():
        print(f"already complete: {root}")
        return
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data = load_fold_data(config, args.fold)
    selection_model = load_model(config, data, args.fold, "selection", args.seed, device)
    css_model = load_model(config, data, args.fold, "css", args.seed, device)
    c0_model = load_model(config, data, args.fold, "c0_full", args.seed, device)
    source_train_indices = data["parts"]["train"]["index"].astype(int).tolist()
    val_indices = data["parts"]["val"]["index"].astype(int).tolist()
    test_indices = data["parts"]["test"]["index"].astype(int).tolist()
    donor_pool = SourceOnlyDonorPool(metadata_for_donors(data), source_train_indices)
    available_all = data["payload"]["available"].bool().numpy()
    val_available = available_all[np.asarray(val_indices, dtype=int)]
    test_available = available_all[np.asarray(test_indices, dtype=int)]
    batch_size = int(config["model"]["batch_size"])
    draws = int(config["counterfactual_distribution"]["draws"])
    main_q = "main_matched"

    # Source-validation policy construction: no held-out labels appear here.
    val_full_logits, val_attention = infer(selection_model, data, val_indices, val_available, device, batch_size)
    val_pseudo = (val_full_logits >= 0.0).astype(np.float32)
    val_cev, val_edges = replacement_loss(
        selection_model,
        data,
        val_indices,
        val_available,
        val_full_logits,
        donor_pool,
        draws=draws,
        q_variant=main_q,
        seed=args.seed,
        target=val_pseudo,
        target_name="predicted_class",
        device=device,
        batch_size=batch_size * 4,
        capture_edges=True,
    )
    provisional_masks, _, _ = adaptive_masks(
        selection_model,
        data,
        val_indices,
        val_cev,
        val_available,
        gamma=float(config["stopping_rule"]["margin_gamma"]),
        epsilon=float(config["stopping_rule"]["residual_cev_epsilon"]),
        minimum_visual_atoms=int(config["stopping_rule"]["minimum_visual_atoms"]),
        maximum_visual_atoms=int(config["stopping_rule"]["maximum_visual_atoms"]),
        fixed_order=None,
        device=device,
        batch_size=batch_size,
    )
    fixed_budget = int(np.clip(np.rint(provisional_masks[:, 1:].sum(axis=1).mean()), 1, int(config["stopping_rule"]["maximum_visual_atoms"])))
    selection_val_policies = policy_masks(
        selection_model,
        data,
        val_indices,
        val_available,
        val_attention,
        val_cev,
        fixed_budget=fixed_budget,
        config=config,
        device=device,
        seed=args.seed,
        include_full=False,
    )
    selection_val_logits = raw_logits_for_policies(selection_model, data, val_indices, selection_val_policies, device, batch_size)
    c0_val_masks, c0_val_selected, c0_val_stopped = all_evidence_masks(val_available)
    c0_val_logits, _ = infer(c0_model, data, val_indices, c0_val_masks, device, batch_size)

    css_val_full_logits, css_val_attention = infer(css_model, data, val_indices, val_available, device, batch_size)
    css_val_pseudo = (css_val_full_logits >= 0.0).astype(np.float32)
    css_val_cev, _ = replacement_loss(
        css_model,
        data,
        val_indices,
        val_available,
        css_val_full_logits,
        donor_pool,
        draws=draws,
        q_variant=main_q,
        seed=args.seed,
        target=css_val_pseudo,
        target_name="predicted_class",
        device=device,
        batch_size=batch_size * 4,
        capture_edges=False,
    )
    css_val_mask, css_val_selected, css_val_stopped = adaptive_masks(
        css_model,
        data,
        val_indices,
        css_val_cev,
        val_available,
        gamma=float(config["stopping_rule"]["margin_gamma"]),
        epsilon=float(config["stopping_rule"]["residual_cev_epsilon"]),
        minimum_visual_atoms=int(config["stopping_rule"]["minimum_visual_atoms"]),
        maximum_visual_atoms=int(config["stopping_rule"]["maximum_visual_atoms"]),
        fixed_order=None,
        device=device,
        batch_size=batch_size,
    )
    css_val_logits, _ = infer(css_model, data, val_indices, css_val_mask, device, batch_size)

    # Held-out prediction policy construction remains label-free through this point.
    test_full_logits, test_attention = infer(selection_model, data, test_indices, test_available, device, batch_size)
    test_pseudo = (test_full_logits >= 0.0).astype(np.float32)
    test_cev, test_edges = replacement_loss(
        selection_model,
        data,
        test_indices,
        test_available,
        test_full_logits,
        donor_pool,
        draws=draws,
        q_variant=main_q,
        seed=args.seed,
        target=test_pseudo,
        target_name="predicted_class",
        device=device,
        batch_size=batch_size * 4,
        capture_edges=True,
    )
    selection_test_policies = policy_masks(
        selection_model,
        data,
        test_indices,
        test_available,
        test_attention,
        test_cev,
        fixed_budget=fixed_budget,
        config=config,
        device=device,
        seed=args.seed,
        include_full=False,
    )
    selection_test_logits = raw_logits_for_policies(selection_model, data, test_indices, selection_test_policies, device, batch_size)
    c0_test_masks, c0_test_selected, c0_test_stopped = all_evidence_masks(test_available)
    c0_test_logits, _ = infer(c0_model, data, test_indices, c0_test_masks, device, batch_size)

    css_test_full_logits, css_test_attention = infer(css_model, data, test_indices, test_available, device, batch_size)
    css_test_pseudo = (css_test_full_logits >= 0.0).astype(np.float32)
    css_test_cev, css_test_edges = replacement_loss(
        css_model,
        data,
        test_indices,
        test_available,
        css_test_full_logits,
        donor_pool,
        draws=draws,
        q_variant=main_q,
        seed=args.seed,
        target=css_test_pseudo,
        target_name="predicted_class",
        device=device,
        batch_size=batch_size * 4,
        capture_edges=True,
    )
    css_test_mask, css_test_selected, css_test_stopped = adaptive_masks(
        css_model,
        data,
        test_indices,
        css_test_cev,
        test_available,
        gamma=float(config["stopping_rule"]["margin_gamma"]),
        epsilon=float(config["stopping_rule"]["residual_cev_epsilon"]),
        minimum_visual_atoms=int(config["stopping_rule"]["minimum_visual_atoms"]),
        maximum_visual_atoms=int(config["stopping_rule"]["maximum_visual_atoms"]),
        fixed_order=None,
        device=device,
        batch_size=batch_size,
    )
    css_test_logits, _ = infer(css_model, data, test_indices, css_test_mask, device, batch_size)

    # Prediction objects are now fixed. Temperature and threshold fit only on source validation labels.
    val_labels = data["payload"]["labels"][val_indices].numpy().astype(int)
    test_labels = data["payload"]["labels"][test_indices].numpy().astype(int)
    policy_pairs = {
        "C0_full_evidence": ((c0_val_logits, c0_val_selected, c0_val_stopped), (c0_test_logits, c0_test_selected, c0_test_stopped)),
        **{
            policy: ((selection_val_logits[policy], *selection_val_policies[policy][1:]), (selection_test_logits[policy], *selection_test_policies[policy][1:]))
            for policy in selection_val_policies
        },
        "C4_cesl_css": ((css_val_logits, css_val_selected, css_val_stopped), (css_test_logits, css_test_selected, css_test_stopped)),
    }
    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    calibration: dict[str, dict] = {}
    for policy, ((val_logits, val_selected, val_stopped), (test_logits, test_selected, test_stopped)) in policy_pairs.items():
        temperature = fit_temperature(val_logits, val_labels)
        val_probability = np.asarray(sigmoid(val_logits / temperature), dtype=float)
        threshold = select_balanced_accuracy_threshold(val_labels, val_probability)
        calibration[policy] = {
            "temperature_source_validation": temperature,
            "threshold_source_validation": threshold,
            "fixed_budget_source_validation": fixed_budget,
        }
        validation_rows.extend(
            record_policy(
                policy=policy,
                fold=args.fold,
                seed=args.seed,
                split="val",
                global_indices=val_indices,
                logits=val_logits,
                labels=val_labels,
                data=data,
                selected=val_selected,
                stopped=val_stopped,
                temperature=temperature,
                threshold=threshold,
            )
        )
        test_rows.extend(
            record_policy(
                policy=policy,
                fold=args.fold,
                seed=args.seed,
                split="test",
                global_indices=test_indices,
                logits=test_logits,
                labels=test_labels,
                data=data,
                selected=test_selected,
                stopped=test_stopped,
                temperature=temperature,
                threshold=threshold,
            )
        )
    pd.DataFrame(validation_rows).to_csv(root / "validation_predictions.csv", index=False)
    pd.DataFrame(test_rows).to_csv(root / "test_predictions.csv", index=False)

    # This is intentionally after prediction serialization: observed labels are audit-only.
    observed_selection_cev, _ = replacement_loss(
        selection_model,
        data,
        test_indices,
        test_available,
        test_full_logits,
        donor_pool,
        draws=draws,
        q_variant=main_q,
        seed=args.seed,
        target=test_labels.astype(np.float32),
        target_name="observed_label_posthoc",
        device=device,
        batch_size=batch_size * 4,
        capture_edges=False,
    )
    observed_css_cev, _ = replacement_loss(
        css_model,
        data,
        test_indices,
        test_available,
        css_test_full_logits,
        donor_pool,
        draws=draws,
        q_variant=main_q,
        seed=args.seed,
        target=test_labels.astype(np.float32),
        target_name="observed_label_posthoc",
        device=device,
        batch_size=batch_size * 4,
        capture_edges=False,
    )
    audits = mechanism_rows(
        fold=args.fold,
        seed=args.seed,
        model_name="selection",
        global_indices=test_indices,
        available=test_available,
        cevs_predicted=test_cev,
        cevs_observed=observed_selection_cev,
        attention=test_attention,
        data=data,
    )
    audits.extend(
        mechanism_rows(
            fold=args.fold,
            seed=args.seed,
            model_name="css",
            global_indices=test_indices,
            available=test_available,
            cevs_predicted=css_test_cev,
            cevs_observed=observed_css_cev,
            attention=css_test_attention,
            data=data,
        )
    )
    pd.DataFrame(audits).to_csv(root / "mechanism_audit.csv", index=False)
    ranks = rank_rows(
        fold=args.fold,
        seed=args.seed,
        model_name="selection",
        global_indices=test_indices,
        available=test_available,
        cevs=test_cev,
        attention=test_attention,
        data=data,
    )
    ranks.extend(
        rank_rows(
            fold=args.fold,
            seed=args.seed,
            model_name="css",
            global_indices=test_indices,
            available=test_available,
            cevs=css_test_cev,
            attention=css_test_attention,
            data=data,
        )
    )
    pd.DataFrame(ranks).to_csv(root / "rank_profiles.csv", index=False)
    pd.DataFrame(val_edges + test_edges + css_test_edges).to_csv(root / "donor_edges.csv.gz", index=False, compression="gzip")

    q_rows = []
    for q_variant in config["counterfactual_distribution"]["variants"]:
        if q_variant == main_q:
            values = css_test_cev
        else:
            values, _ = replacement_loss(
                css_model,
                data,
                test_indices,
                test_available,
                css_test_full_logits,
                donor_pool,
                draws=draws,
                q_variant=q_variant,
                seed=args.seed,
                target=css_test_pseudo,
                target_name="predicted_class",
                device=device,
                batch_size=batch_size * 4,
                capture_edges=False,
            )
        for local, global_index in enumerate(test_indices):
            for slot in range(1, len(SLOT_IDS)):
                if test_available[local, slot]:
                    q_rows.append(
                        {
                            "id": str(data["ids"][global_index]),
                            "patient_id": str(data["payload"]["patient_ids"][global_index]),
                            "center": str(data["payload"]["centers"][global_index]),
                            "fold": args.fold,
                            "seed": args.seed,
                            "model": "css",
                            "q_variant": q_variant,
                            "slot": SLOT_IDS[slot],
                            "predicted_cev": float(values[local, slot]),
                        }
                    )
    pd.DataFrame(q_rows).to_csv(root / "q_sensitivity.csv", index=False)
    trace_rows = []
    for row in test_rows:
        trace_rows.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "center": row["center"],
                "fold": row["fold"],
                "seed": row["seed"],
                "policy": row["policy"],
                "selected_slots": row["selected_slots"],
                "selected_visual_atoms": row["selected_visual_atoms"],
                "stopping_criterion_met": row["stopping_criterion_met"],
            }
        )
    write_jsonl(root / "selection_trace.jsonl", trace_rows)
    (root / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "fold": args.fold,
                "seed": args.seed,
                "policies": sorted(policy_pairs),
                "fixed_budget_source_validation": fixed_budget,
                "held_out_labels_used_for_policy": False,
                "held_out_labels_used_posthoc_for": ["outcome_metrics", "observed_replacement_loss_audit"],
                "donor_label_used_for_matching": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"CESL EVALUATION COMPLETE fold={args.fold} seed={args.seed} policies={len(policy_pairs)}", flush=True)


if __name__ == "__main__":
    main()
