#!/usr/bin/env python3
"""Evaluate cross-fitted conditional-replacement necessity on source validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cervix_cogalign.cesl import SourceOnlyDonorPool  # noqa: E402
from cervix_cogalign.verifiable_evidence import representation_grounding, top_slot  # noqa: E402
from evaluate_cesl_fold import infer, load_model, metadata_for_donors, replacement_loss  # noqa: E402
from train_cesl_backbone import load_fold_data  # noqa: E402
from run_vec_source_screen import (  # noqa: E402
    fixed_budget_masks,
    information_and_conflict,
    macro_auprc,
    per_case_rank_correlation,
    probabilities,
    replacement_necessity,
)


CANDIDATES = ("D0_attention", "D4_crossfit_necessity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("seed", type=int)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/vec_v2_crossfit_screen"))
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def crossfit_necessity_score(
    grounding: np.ndarray,
    necessity: np.ndarray,
    information: np.ndarray,
    conflict: np.ndarray,
    instability: np.ndarray,
    target_loss: np.ndarray,
    available: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict observed replacement loss using case-wise cross-fitting only."""
    n_cases, n_slots = available.shape
    score = np.full((n_cases, n_slots), np.nan, dtype=float)
    class_counts = np.bincount(labels.astype(int), minlength=2)
    splitter = (
        StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        if int(class_counts.min()) >= 2
        else KFold(n_splits=2, shuffle=True, random_state=seed)
    )
    components = np.stack(
        [
            grounding,
            np.log1p(np.maximum(necessity, 0.0)),
            np.log1p(np.maximum(information, 0.0)),
            conflict,
            np.log1p(np.maximum(instability, 0.0)),
        ],
        axis=-1,
    )
    for fit_cases, score_cases in splitter.split(np.arange(n_cases), labels):
        fit_rows = [(case, slot) for case in fit_cases for slot in np.flatnonzero(available[case])]
        score_rows = [(case, slot) for case in score_cases for slot in np.flatnonzero(available[case])]
        x_fit = np.stack([components[case, slot] for case, slot in fit_rows])
        y_fit = np.asarray([target_loss[case, slot] for case, slot in fit_rows], dtype=float)
        x_score = np.stack([components[case, slot] for case, slot in score_rows])
        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model.fit(x_fit, y_fit)
        predicted = model.predict(x_score)
        for (case, slot), value in zip(score_rows, predicted):
            score[case, slot] = float(value)
    return score


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

    # Construct only label-free verifier features first.
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
    grounding = representation_grounding(
        data["payload"]["features"][validation_indices, 1:].float().cpu().numpy(), visual_available
    )
    variants = np.stack([necessity_main, necessity_center, necessity_pooled])
    instability = np.std(np.where(np.isfinite(variants), variants, 0.0), axis=0)
    instability[~visual_available] = np.nan

    # Source-validation labels are used only to cross-fit a score on opposite cases.
    labels = data["payload"]["labels"][validation_indices].detach().cpu().numpy().astype(int)
    observed_loss, _ = replacement_loss(
        model, data, validation_indices, full_masks, base_logits, donor_pool,
        draws=args.draws, q_variant="main_matched", seed=args.seed, target=labels.astype(np.float32),
        target_name="observed_source_validation_label_crossfit", device=device, batch_size=batch_size * 4, capture_edges=False,
    )
    observed_visual_loss = observed_loss[:, 1:]
    crossfit_score = crossfit_necessity_score(
        grounding, necessity_main, information, conflict, instability, observed_visual_loss, visual_available, labels, args.seed
    )
    attention = full_attention[:, 1:]
    scores = {"D0_attention": attention, "D4_crossfit_necessity": crossfit_score}
    centres = np.asarray([str(data["payload"]["centers"][index]) for index in validation_indices], dtype=object)
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
    attention_top = top_slot(attention, visual_available) + 1
    for name, score in scores.items():
        mask, _ = fixed_budget_masks(score, full_masks, args.budget)
        logits, _ = infer(model, data, validation_indices, mask, device, batch_size)
        probability = probabilities(logits)
        selected_top = top_slot(score, visual_available) + 1
        difference = observed_loss[np.arange(len(selected_top)), selected_top] - observed_loss[np.arange(len(attention_top)), attention_top]
        rho = per_case_rank_correlation(score, attention, visual_available)
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
