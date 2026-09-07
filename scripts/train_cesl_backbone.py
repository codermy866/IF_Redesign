#!/usr/bin/env python3
"""Train one CESL retrospective backbone without opening held-out predictions."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.cesl import (  # noqa: E402
    CESLSetModel,
    SLOT_IDS,
    clinical_matrix,
    css_loss,
    group_dro_binary_loss,
    random_observation_mask,
)
from cervix_cogalign.io import read_json  # noqa: E402


BACKBONES = ("c0_full", "selection", "css")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("backbone", choices=BACKBONES)
    parser.add_argument("seed", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CentreBalancedBatchSampler(Sampler[list[int]]):
    """Balanced source-centre batches for the CSS-only training objective."""

    def __init__(self, indices: list[int], centre_codes: np.ndarray, batch_size: int, seed: int) -> None:
        self.indices = [int(index) for index in indices]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.by_centre: dict[int, list[int]] = {}
        for index in self.indices:
            self.by_centre.setdefault(int(centre_codes[index]), []).append(index)
        self.centres = sorted(self.by_centre)
        self.epoch = 0
        if not self.centres:
            raise ValueError("Cannot create a centre-balanced sampler without source cases")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        per_centre = math.ceil(self.batch_size / len(self.centres))
        for _ in range(len(self)):
            batch: list[int] = []
            for centre in self.centres:
                choices = rng.choice(self.by_centre[centre], size=per_centre, replace=True)
                batch.extend(int(choice) for choice in choices)
            rng.shuffle(batch)
            yield batch[: self.batch_size]


def macro_auprc(labels: np.ndarray, probabilities: np.ndarray, centres: np.ndarray) -> float:
    values = []
    for centre in sorted(set(centres.tolist())):
        include = centres == centre
        if len(np.unique(labels[include])) == 2:
            values.append(float(average_precision_score(labels[include], probabilities[include])))
    if not values:
        return float("nan")
    return float(np.mean(values))


def get_run_root(config: dict, backbone: str, fold: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "runs" / backbone / fold / f"seed_{seed}"


def load_fold_data(config: dict, fold: str) -> dict:
    feature_path = Path(config["output_dir"]) / config["image_encoder"]["feature_file"]
    if not feature_path.is_file():
        raise FileNotFoundError(f"CESL feature cache is unavailable: {feature_path}")
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    if tuple(payload["slot_ids"]) != SLOT_IDS:
        raise AssertionError("Feature cache slot schema differs from the locked CESL protocol")
    ids = [str(value) for value in payload["ids"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("CESL feature cache has duplicate cases")
    index_by_id = {identifier: index for index, identifier in enumerate(ids)}
    assignments = pd.read_csv(config["fold_assignments"], dtype={"case_id": str, "patient_id": str})
    assignment = assignments[assignments.fold.astype(str) == str(fold)].copy()
    if assignment.case_id.duplicated().any():
        raise AssertionError("A Formal v1 fold assignment contains duplicate cases")
    if set(assignment.case_id) != set(ids):
        missing = sorted(set(ids) - set(assignment.case_id))
        extra = sorted(set(assignment.case_id) - set(ids))
        raise AssertionError(f"CESL atom cohort does not exactly match Formal v1 fold: missing={len(missing)} extra={len(extra)}")
    assignment["index"] = assignment.case_id.map(index_by_id)
    parts = {split: assignment[assignment.split == split].sort_values("index") for split in ("train", "val", "test")}
    if set(parts["train"].index) & set(parts["test"].index):
        raise AssertionError("Train/test assignment rows overlap")
    held = set(parts["test"].center)
    if set(parts["train"].center) & held or set(parts["val"].center) & held:
        raise AssertionError("Held-out centre entered the source split")
    centres = [str(value) for value in payload["centers"]]
    centre_map = {value: index for index, value in enumerate(sorted(set(centres)))}
    centre_codes = np.asarray([centre_map[value] for value in centres], dtype=np.int64)
    stratum_map = {value: index for index, value in enumerate(sorted(set(tuple(item) for item in payload["nuisance_strata"]), key=str))}
    stratum_codes = np.asarray([stratum_map[tuple(value)] for value in payload["nuisance_strata"]], dtype=np.int64)
    train_indices = parts["train"]["index"].astype(int).tolist()
    clinical, scaling = clinical_matrix(payload["ages"], payload["hpvs"], payload["tcts"], train_indices)
    return {
        "payload": payload,
        "parts": parts,
        "ids": ids,
        "centres": np.asarray(centres, dtype=object),
        "centre_codes": centre_codes,
        "stratum_codes": stratum_codes,
        "clinical": torch.from_numpy(clinical),
        "clinical_scaling": scaling,
        "centre_map": centre_map,
    }


@torch.inference_mode()
def validation_score(model: CESLSetModel, data: dict, indices: list[int], device: torch.device, batch_size: int) -> float:
    features = data["payload"]["features"]
    available = data["payload"]["available"].bool()
    labels = data["payload"]["labels"].numpy().astype(int)
    logits = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        output = model(
            features[batch].to(device),
            data["clinical"][batch].to(device),
            available[batch].to(device),
        )
        logits.append(output["logit"].detach().cpu().numpy())
    probability = 1.0 / (1.0 + np.exp(-np.clip(np.concatenate(logits), -40, 40)))
    centres = data["centres"][np.asarray(indices, dtype=int)]
    return macro_auprc(labels[np.asarray(indices, dtype=int)], probability, centres)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    set_seed(args.seed)
    data = load_fold_data(config, args.fold)
    run_root = get_run_root(config, args.backbone, args.fold, args.seed)
    run_root.mkdir(parents=True, exist_ok=True)
    marker = run_root / "training_complete.json"
    checkpoint_path = run_root / "checkpoint.pt"
    if marker.is_file() and checkpoint_path.is_file():
        print(f"already complete: {run_root}")
        return
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = CESLSetModel(
        visual_dim=int(config["image_encoder"]["feature_dim"]),
        clinical_dim=int(data["clinical"].shape[1]),
        token_dim=int(model_config["token_dim"]),
        transformer_layers=int(model_config["transformer_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    train_indices = data["parts"]["train"]["index"].astype(int).tolist()
    val_indices = data["parts"]["val"]["index"].astype(int).tolist()
    labels = data["payload"]["labels"].numpy().astype(int)
    n_positive = int(labels[train_indices].sum())
    n_negative = int(len(train_indices) - n_positive)
    if not n_positive or not n_negative:
        raise AssertionError("A source-train fold must contain both endpoint classes")
    positive_weight = n_negative / n_positive
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    css_enabled = args.backbone == "css"
    dropout_enabled = args.backbone != "c0_full"
    if css_enabled:
        sampler = CentreBalancedBatchSampler(train_indices, data["centre_codes"], int(model_config["batch_size"]), args.seed)
        # The balanced sampler yields global feature-cache indices.  Its dataset
        # must therefore be an identity global-index dataset, not train_indices
        # (whose positional indices are unrelated to the cache indices).
        loader = DataLoader(list(range(len(data["ids"]))), batch_sampler=sampler, num_workers=0)
    else:
        sampler = None
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(train_indices, batch_size=int(model_config["batch_size"]), shuffle=True, generator=generator, num_workers=0)
    features = data["payload"]["features"]
    available = data["payload"]["available"].bool()
    history = []
    best = float("-inf")
    stale = 0
    started = time.time()
    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        batches = 0
        for batch in loader:
            index = batch.long()
            x = features[index].to(device)
            c = data["clinical"][index].to(device)
            y = data["payload"]["labels"][index].to(device)
            mask = available[index].to(device)
            if dropout_enabled:
                mask = random_observation_mask(mask, float(model_config["visual_keep_probability"]))
            output = model(x, c, mask)
            average, worst = group_dro_binary_loss(
                output["logit"], y, torch.as_tensor(data["centre_codes"][index.numpy()], device=device), positive_weight
            )
            loss = average
            if css_enabled:
                centres = torch.as_tensor(data["centre_codes"][index.numpy()], device=device)
                strata = torch.as_tensor(data["stratum_codes"][index.numpy()], device=device)
                loss = (
                    average
                    + float(model_config["css_group_dro_weight"]) * (worst - average)
                    + float(model_config["css_rank_stability_weight"]) * css_loss(output["attention"], centres, strata)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        val_macro_auprc = validation_score(model, data, val_indices, device, int(model_config["batch_size"]))
        row = {"epoch": epoch, "train_loss": total_loss / max(1, batches), "val_macro_auprc": val_macro_auprc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_macro_auprc > best + 1e-7:
            best = val_macro_auprc
            stale = 0
            torch.save(
                {
                    "experiment_id": config["experiment_id"],
                    "fold": args.fold,
                    "backbone": args.backbone,
                    "seed": args.seed,
                    "model_config": model_config,
                    "visual_feature_dim": int(config["image_encoder"]["feature_dim"]),
                    "clinical_feature_dim": int(data["clinical"].shape[1]),
                    "model_state": model.state_dict(),
                    "clinical_scaling": data["clinical_scaling"],
                    "centre_map": data["centre_map"],
                    "best_validation_macro_auprc": best,
                    "test_labels_opened": False,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= int(model_config["early_stopping_patience"]):
            break
    history_path = run_root / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "fold": args.fold,
                "backbone": args.backbone,
                "seed": args.seed,
                "best_validation_macro_auprc": best,
                "epochs_completed": len(history),
                "duration_minutes": (time.time() - started) / 60.0,
                "test_labels_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"CESL TRAIN COMPLETE fold={args.fold} backbone={args.backbone} seed={args.seed} best={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
