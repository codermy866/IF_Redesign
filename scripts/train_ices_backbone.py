#!/usr/bin/env python3
"""Train one predeclared ICES source-only ablation backbone.

The script consumes source-train labels and source-validation labels only.  It
does not evaluate, calibrate, select, or write a prediction for the held-out
LOCO centre.  Its ``test_labels_opened`` marker is therefore a protocol guard,
not a claim of a new independent test result.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cervix_cogalign.cesl import clinical_matrix  # noqa: E402
from cervix_cogalign.ices import (  # noqa: E402
    ICESSetModel,
    SLOT_IDS,
    adaptive_retention_mask,
    fixed_attention_mask,
    random_fixed_mask,
    random_observation_mask,
    selected_visual_slot_matrix,
    visual_cost,
)
from cervix_cogalign.sequence_evidence import binary_risk, minimal_sufficient_set_objective  # noqa: E402


BACKBONES = (
    "raw_m1_ablation",
    "cluster_m1_m2",
    "cluster_m1_m2_m3",
    "cluster_selected_control",
    "cluster_sufficiency_control",
    "cluster_pure_m3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("backbone", choices=BACKBONES)
    parser.add_argument("seed", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_root(config: dict, backbone: str, fold: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "runs" / backbone / fold / f"seed_{seed}"


def macro_auprc(labels: np.ndarray, probabilities: np.ndarray, centres: np.ndarray) -> float:
    values = []
    for centre in sorted(set(centres.tolist())):
        include = centres == centre
        if len(np.unique(labels[include])) == 2:
            values.append(float(average_precision_score(labels[include], probabilities[include])))
    return float(np.mean(values)) if values else float("nan")


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, draws: int = 500) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = values[rng.integers(0, len(values), size=len(values))].mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def load_fold_data(config: dict, fold: str) -> dict:
    feature_path = Path(config.get("feature_cache", Path(config["output_dir"]) / config["image_encoder"]["feature_file"]))
    if not feature_path.is_file():
        raise FileNotFoundError(f"ICES training requires the path-free feature cache: {feature_path}")
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    if tuple(payload.get("slot_ids", ())) != SLOT_IDS:
        raise AssertionError("ICES feature-cache slot schema differs from the locked protocol")
    ids = [str(value) for value in payload["ids"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("ICES feature cache contains duplicate cases")
    assignments = pd.read_csv(config["fold_assignments"], dtype={"case_id": str, "patient_id": str})
    assignment = assignments[assignments.fold.astype(str) == str(fold)].copy()
    if set(assignment.case_id) != set(ids) or assignment.case_id.duplicated().any():
        raise AssertionError("ICES cache must exactly match a single frozen Formal-v1 fold assignment")
    index_by_id = {case_id: index for index, case_id in enumerate(ids)}
    assignment["index"] = assignment.case_id.map(index_by_id)
    parts = {name: assignment[assignment.split == name].sort_values("index") for name in ("train", "val", "test")}
    held_centres = set(parts["test"].center)
    if set(parts["train"].center) & held_centres or set(parts["val"].center) & held_centres:
        raise AssertionError("Held-out centre entered the ICES source split")
    train_indices = parts["train"]["index"].astype(int).tolist()
    clinical, scaling = clinical_matrix(payload["ages"], payload["hpvs"], payload["tcts"], train_indices)
    return {
        "payload": payload,
        "parts": parts,
        "clinical": torch.from_numpy(clinical),
        "clinical_scaling": scaling,
        "centres": np.asarray(payload["centers"], dtype=object),
        "source_indices": {"train": train_indices, "val": parts["val"]["index"].astype(int).tolist()},
        "held_out_case_count": int(len(parts["test"])),
    }


def create_model(config: dict, data: dict, device: torch.device) -> ICESSetModel:
    settings = config["model"]
    return ICESSetModel(
        visual_dim=int(config["image_encoder"]["feature_dim"]),
        clinical_dim=int(data["clinical"].shape[1]),
        token_dim=int(settings["token_dim"]),
        transformer_layers=int(settings["transformer_layers"]),
        attention_heads=int(settings["attention_heads"]),
        dropout=float(settings["dropout"]),
    ).to(device)


def feature_key(backbone: str, config: dict | None = None) -> str:
    """Resolve a predeclared feature variant without touching outcome data."""
    if config is not None:
        declared = config.get("feature_key_by_backbone", {}).get(backbone)
        if declared:
            return str(declared)
    return "raw_features" if backbone == "raw_m1_ablation" else "cluster_features"


def weighted_bce(logit: torch.Tensor, labels: torch.Tensor, positive_weight: float) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(logit, labels.float(), pos_weight=torch.as_tensor(positive_weight, device=logit.device))


def selected_set_terms(model: ICESSetModel, x: torch.Tensor, clinical: torch.Tensor, available: torch.Tensor, labels: torch.Tensor, config: dict) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Shared selected-set terms for the clean M3 supplementary controls.

    The attention-proposed hard subset is deliberately detached.  This makes
    the supplementary experiment a deletion-regularized *predictor* audit,
    not an assertion that a differentiable selector itself learned causal
    evidence.  Every supplementary M3 arm receives the same selected-set
    classification terms; only the declared regularizer differs.
    """
    settings = config["model"]
    policy = config["policy"]
    full = model(x, clinical, available)
    selected = fixed_attention_mask(full["attention"].detach(), available, int(policy["fixed_visual_budget"]))
    subset = model(x, clinical, selected)
    slots, valid = selected_visual_slot_matrix(selected, int(policy["fixed_visual_budget"]))
    deletion_masks = selected[:, None, :].repeat(1, slots.shape[1], 1)
    for row in range(len(selected)):
        for column in range(slots.shape[1]):
            if valid[row, column]:
                deletion_masks[row, column, slots[row, column]] = False
    batch, width, _ = deletion_masks.shape
    deletion = model(
        x[:, None, :, :].expand(-1, width, -1, -1).reshape(batch * width, x.shape[1], x.shape[2]),
        clinical[:, None, :].expand(-1, width, -1).reshape(batch * width, clinical.shape[1]),
        deletion_masks.reshape(batch * width, -1),
    )["logit"].reshape(batch, width)
    objective = minimal_sufficient_set_objective(
        full_logits=full["logit"], selected_logits=subset["logit"], deletion_logits=deletion,
        labels=labels, selected_units=valid,
        sufficiency_tolerance=float(settings["sufficiency_tolerance"]), deletion_margin=float(settings["deletion_margin"]),
    )
    scalars = {key: float(value.detach().cpu()) for key, value in objective.items()}
    return {"full_logit": full["logit"], "selected_logit": subset["logit"], **objective}, scalars


def m3_set_loss(model: ICESSetModel, x: torch.Tensor, clinical: torch.Tensor, available: torch.Tensor, labels: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Legacy v1 M3 loss retained only to reproduce the completed screen."""
    terms, scalars = selected_set_terms(model, x, clinical, available, labels, config)
    settings = config["model"]
    total = (
        weighted_bce(terms["full_logit"], labels, 1.0)
        + float(settings["subset_classification_weight"]) * weighted_bce(terms["selected_logit"], labels, 1.0)
        + float(settings["sufficiency_weight"]) * terms["sufficiency_loss"]
        + float(settings["minimality_weight"]) * terms["minimality_loss"]
    )
    return total, scalars


@torch.inference_mode()
def source_validation_full_macro_auprc(model: ICESSetModel, data: dict, indices: list[int], device: torch.device, config: dict, backbone: str) -> float:
    """Fast early-stopping score; does not perform a deletion audit."""
    model.eval()
    features = data["payload"][feature_key(backbone, config)]
    available = data["payload"]["available"].bool()
    logits = []
    for start in range(0, len(indices), int(config["model"]["batch_size"])):
        batch = indices[start : start + int(config["model"]["batch_size"])]
        logits.append(model(features[batch].to(device), data["clinical"][batch].to(device), available[batch].to(device))["logit"].cpu())
    probability = torch.sigmoid(torch.cat(logits)).numpy()
    labels = data["payload"]["labels"][indices].numpy().astype(int)
    return macro_auprc(labels, probability, data["centres"][np.asarray(indices, dtype=int)])


@torch.inference_mode()
def evaluate_policy(model: ICESSetModel, data: dict, indices: list[int], device: torch.device, config: dict, policy_name: str, seed: int, backbone: str) -> dict:
    model.eval()
    settings, policy = config["model"], config["policy"]
    features = data["payload"][feature_key(backbone, config)]
    available = data["payload"]["available"].bool()
    labels = data["payload"]["labels"]
    batch_size = int(settings["batch_size"])
    output_logits, output_full_logits, output_masks = [], [], []
    rng = torch.Generator(device=device).manual_seed(seed + 101)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        x = features[batch].to(device)
        c = data["clinical"][batch].to(device)
        full_available = available[batch].to(device)
        full = model(x, c, full_available)
        if policy_name == "full_evidence":
            selected = full_available
            logits = full["logit"]
        elif policy_name == "attention_fixed_cost":
            selected = fixed_attention_mask(full["attention"], full_available, int(policy["fixed_visual_budget"]))
            logits = model(x, c, selected)["logit"]
        elif policy_name == "random_fixed_cost":
            selected = random_fixed_mask(full_available, int(policy["fixed_visual_budget"]), rng)
            logits = model(x, c, selected)["logit"]
        elif policy_name == "adaptive_retention":
            selected = adaptive_retention_mask(
                model, x, c, full_available,
                min_visual_units=int(policy["adaptive_min_visual_units"]),
                max_visual_units=int(policy["adaptive_max_visual_units"]),
                confidence_margin=float(policy["adaptive_confidence_margin"]),
            )
            logits = model(x, c, selected)["logit"]
        else:
            raise ValueError(f"Unknown ICES source-validation policy: {policy_name}")
        output_logits.append(logits.cpu())
        output_full_logits.append(full["logit"].cpu())
        output_masks.append(selected.cpu())
    logits = torch.cat(output_logits)
    full_logits = torch.cat(output_full_logits)
    masks = torch.cat(output_masks)
    y = labels[indices].numpy().astype(int)
    probability = torch.sigmoid(logits).numpy()
    # Source-only observed-label deletion audit; selected clinical token is never deleted.
    slots, valid = selected_visual_slot_matrix(masks, int(policy["adaptive_max_visual_units"]))
    x = features[indices].to(device)
    c = data["clinical"][indices].to(device)
    batch, width = slots.shape
    deletion_masks = masks.to(device)[:, None, :].repeat(1, width, 1)
    for row in range(batch):
        for column in range(width):
            if valid[row, column]:
                deletion_masks[row, column, slots[row, column]] = False
    deletion_logits = model(
        x[:, None, :, :].expand(-1, width, -1, -1).reshape(batch * width, x.shape[1], x.shape[2]),
        c[:, None, :].expand(-1, width, -1).reshape(batch * width, c.shape[1]),
        deletion_masks.reshape(batch * width, -1),
    )["logit"].reshape(batch, width).cpu()
    selected_risk = binary_risk(logits, torch.from_numpy(y).float())
    deletion_risk = functional.binary_cross_entropy_with_logits(deletion_logits, torch.from_numpy(y).float()[:, None].expand_as(deletion_logits), reduction="none")
    delta = deletion_risk - selected_risk[:, None]
    chosen_delta = delta[valid]
    ci_low, ci_high = bootstrap_mean_ci(chosen_delta.numpy(), seed=seed + 701)
    # A predeclared matched-unselected control in the *full* observed case.
    # It evaluates relative model dependence, not clinical causal necessity.
    matched_slots = torch.full_like(slots, 0)
    matched_valid = torch.zeros_like(valid)
    slot_range = torch.arange(masks.shape[1])
    for row in range(batch):
        # Match each deletion on evidence modality (colposcopy versus OCT), not
        # merely on count.  A missing same-modality candidate is recorded as
        # unmatched rather than silently borrowing a different modality.
        remaining = (available[indices[row]].cpu() & ~masks[row] & (slot_range != 0)).clone()
        generator = torch.Generator().manual_seed(seed + 5003 + row)
        for column in range(width):
            if not valid[row, column]:
                continue
            selected_slot = int(slots[row, column])
            same_modality = (slot_range <= 24) if selected_slot <= 24 else (slot_range > 24)
            candidates = torch.nonzero(remaining & same_modality).flatten()
            if not len(candidates):
                continue
            choice = candidates[torch.randint(len(candidates), (1,), generator=generator).item()]
            matched_slots[row, column] = choice
            matched_valid[row, column] = True
            remaining[choice] = False
    full_masks = available[indices].bool()
    selected_full_deletion = full_masks[:, None, :].repeat(1, width, 1)
    matched_full_deletion = full_masks[:, None, :].repeat(1, width, 1)
    for row in range(batch):
        for column in range(width):
            if valid[row, column]:
                selected_full_deletion[row, column, slots[row, column]] = False
            if matched_valid[row, column]:
                matched_full_deletion[row, column, matched_slots[row, column]] = False
    full_features = features[indices].to(device)
    full_clinical = data["clinical"][indices].to(device)
    selected_full_logits = model(
        full_features[:, None, :, :].expand(-1, width, -1, -1).reshape(batch * width, full_features.shape[1], full_features.shape[2]),
        full_clinical[:, None, :].expand(-1, width, -1).reshape(batch * width, full_clinical.shape[1]),
        selected_full_deletion.to(device).reshape(batch * width, -1),
    )["logit"].reshape(batch, width).cpu()
    matched_full_logits = model(
        full_features[:, None, :, :].expand(-1, width, -1, -1).reshape(batch * width, full_features.shape[1], full_features.shape[2]),
        full_clinical[:, None, :].expand(-1, width, -1).reshape(batch * width, full_clinical.shape[1]),
        matched_full_deletion.to(device).reshape(batch * width, -1),
    )["logit"].reshape(batch, width).cpu()
    full_risk = binary_risk(full_logits, torch.from_numpy(y).float())
    selected_full_delta = functional.binary_cross_entropy_with_logits(selected_full_logits, torch.from_numpy(y).float()[:, None].expand_as(selected_full_logits), reduction="none") - full_risk[:, None]
    matched_full_delta = functional.binary_cross_entropy_with_logits(matched_full_logits, torch.from_numpy(y).float()[:, None].expand_as(matched_full_logits), reduction="none") - full_risk[:, None]
    case_contrast = []
    for row in range(batch):
        paired = valid[row] & matched_valid[row]
        if paired.any():
            case_contrast.append(float((selected_full_delta[row, paired].mean() - matched_full_delta[row, paired].mean()).item()))
    case_contrast_array = np.asarray(case_contrast, dtype=float)
    contrast_low, contrast_high = bootstrap_mean_ci(case_contrast_array, seed=seed + 1701)
    return {
        "policy": policy_name,
        "macro_auprc": macro_auprc(y, probability, data["centres"][np.asarray(indices, dtype=int)]),
        "mean_visual_cost": float(visual_cost(masks).mean()),
        "median_visual_cost": float(np.median(visual_cost(masks))),
        "mean_selected_deletion_risk_increase": float(chosen_delta.mean().item()),
        "selected_deletion_risk_increase_ci95": [ci_low, ci_high],
        "mean_full_context_selected_vs_matched_unselected_deletion_delta": float(case_contrast_array.mean()) if len(case_contrast_array) else float("nan"),
        "full_context_selected_vs_matched_unselected_deletion_delta_ci95": [contrast_low, contrast_high],
        "n_selected_deletions": int(valid.sum().item()),
        "n_modality_matched_deletions": int(matched_valid.sum().item()),
        "n_unmatched_selected_deletions": int(valid.sum().item() - matched_valid.sum().item()),
        "global_indices": np.asarray(indices, dtype=np.int64),
        "labels": y,
        "probabilities": probability,
        "visual_cost": visual_cost(masks),
        "selected_deletion_risk_increase": chosen_delta.numpy(),
        "full_context_selected_vs_matched_unselected_case_delta": case_contrast_array,
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(args.seed)
    data = load_fold_data(config, args.fold)
    root = run_root(config, args.backbone, args.fold, args.seed)
    root.mkdir(parents=True, exist_ok=True)
    marker, checkpoint = root / "training_complete.json", root / "checkpoint.pt"
    if marker.is_file() and checkpoint.is_file():
        print(f"already complete: {root}")
        return
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = create_model(config, data, device)
    train_indices, val_indices = data["source_indices"]["train"], data["source_indices"]["val"]
    labels = data["payload"]["labels"]
    positive = int(labels[train_indices].sum())
    negative = int(len(train_indices) - positive)
    if not positive or not negative:
        raise AssertionError("ICES source train requires both endpoint classes")
    positive_weight = negative / positive
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["model"]["learning_rate"]), weight_decay=float(config["model"]["weight_decay"]))
    loader = DataLoader(train_indices, batch_size=int(config["model"]["batch_size"]), shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    best, stale, history = float("-inf"), 0, []
    started = time.time()
    for epoch in range(1, int(config["model"]["max_epochs"]) + 1):
        model.train()
        total, batches, m3_stats = 0.0, 0, []
        for batch in loader:
            index = batch.long()
            x = data["payload"][feature_key(args.backbone, config)][index].to(device)
            c = data["clinical"][index].to(device)
            y = labels[index].to(device)
            available = data["payload"]["available"][index].bool().to(device)
            full = model(x, c, available)
            augmented = model(x, c, random_observation_mask(available, float(config["model"]["visual_keep_probability"])))
            loss = weighted_bce(full["logit"], y, positive_weight) + float(config["model"]["subset_classification_weight"]) * weighted_bce(augmented["logit"], y, positive_weight)
            if args.backbone == "cluster_m1_m2_m3":
                deletion_loss, stats = m3_set_loss(model, x, c, available, y, config)
                loss = loss + deletion_loss
                m3_stats.append(stats)
            elif args.backbone in {"cluster_selected_control", "cluster_sufficiency_control", "cluster_pure_m3"}:
                terms, stats = selected_set_terms(model, x, c, available, y, config)
                selected_full_weight = float(config["model"]["selected_full_weight"])
                selected_subset_weight = float(config["model"]["selected_subset_weight"])
                loss = loss + selected_full_weight * weighted_bce(terms["full_logit"], y, positive_weight)
                loss = loss + selected_subset_weight * weighted_bce(terms["selected_logit"], y, positive_weight)
                if args.backbone in {"cluster_sufficiency_control", "cluster_pure_m3"}:
                    loss = loss + float(config["model"]["sufficiency_weight"]) * terms["sufficiency_loss"]
                if args.backbone == "cluster_pure_m3":
                    loss = loss + float(config["model"]["minimality_weight"]) * terms["minimality_loss"]
                m3_stats.append(stats)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        model.eval()
        score = source_validation_full_macro_auprc(model, data, val_indices, device, config, args.backbone)
        row = {"epoch": epoch, "train_loss": total / max(1, batches), "val_full_macro_auprc": score}
        if m3_stats:
            row["m3_mean_selected_deletion_risk_increase"] = float(np.mean([item["mean_selected_deletion_risk_increase"] for item in m3_stats]))
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best + 1e-7:
            best, stale = score, 0
            torch.save({
                "experiment_id": config["experiment_id"], "fold": args.fold, "backbone": args.backbone, "seed": args.seed,
                "model_config": config["model"], "policy_config": config["policy"], "model_state": model.state_dict(),
                "clinical_scaling": data["clinical_scaling"], "best_source_validation_macro_auprc": best,
                "test_labels_opened": False, "held_out_prediction_written": False,
            }, checkpoint)
        else:
            stale += 1
        if stale >= int(config["model"]["early_stopping_patience"]):
            break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    policies = ("full_evidence", "attention_fixed_cost", "random_fixed_cost", "adaptive_retention")
    reports = {name: evaluate_policy(model, data, val_indices, device, config, name, args.seed, args.backbone) for name in policies}
    serialisable = {name: {key: value for key, value in report.items() if not isinstance(value, np.ndarray)} for name, report in reports.items()}
    np.savez_compressed(root / "source_validation_audits.npz", **{
        f"{name}__{key}": value for name, report in reports.items() for key, value in report.items() if isinstance(value, np.ndarray)
    })
    (root / "source_validation_metrics.json").write_text(json.dumps(serialisable, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(json.dumps({
        "status": "complete", "fold": args.fold, "backbone": args.backbone, "seed": args.seed,
        "best_source_validation_macro_auprc": best, "epochs_completed": len(history),
        "duration_minutes": (time.time() - started) / 60.0,
        "held_out_case_count_not_evaluated": data["held_out_case_count"],
        "test_labels_opened": False, "held_out_prediction_written": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ICES TRAIN COMPLETE fold={args.fold} backbone={args.backbone} seed={args.seed} best={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
