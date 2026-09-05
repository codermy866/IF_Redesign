#!/usr/bin/env python3
"""Fit source-only frozen raw-atom linear controls on each LOCO fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cervix_cogalign.cesl import fit_temperature, sigmoid  # noqa: E402
from cervix_cogalign.io import read_json  # noqa: E402
from cervix_cogalign.metrics import select_balanced_accuracy_threshold  # noqa: E402
from train_cesl_backbone import load_fold_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    return parser.parse_args()


def modality_mean(features: np.ndarray, available: np.ndarray, slots: slice) -> np.ndarray:
    values = features[:, slots, :].astype(np.float32)
    mask = available[:, slots].astype(np.float32)
    denominator = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return (values * mask[:, :, None]).sum(axis=1) / denominator


def records(
    *,
    data: dict,
    indices: list[int],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    fold: str,
    split: str,
    method: str,
    selected_count: np.ndarray,
) -> list[dict]:
    rows = []
    for local, global_index in enumerate(indices):
        rows.append(
            {
                "id": str(data["ids"][global_index]),
                "patient_id": str(data["payload"]["patient_ids"][global_index]),
                "center": str(data["payload"]["centers"][global_index]),
                "fold": fold,
                "seed": "deterministic_linear",
                "split": split,
                "policy": method,
                "label": int(labels[local]),
                "probability": float(probabilities[local]),
                "threshold": float(threshold),
                "predicted_label": int(probabilities[local] >= threshold),
                "selected_visual_atoms": float(selected_count[local]),
                "relative_evidence_cost": float(selected_count[local]),
                "stopping_criterion_met": False,
                "selected_slots": "[]",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(config["output_dir"]) / "linear_controls" / args.fold
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "complete.json"
    if marker.is_file() and (output / "test_predictions.csv").is_file():
        print(f"already complete: {output}")
        return
    data = load_fold_data(config, args.fold)
    features = data["payload"]["features"].numpy()
    available = data["payload"]["available"].bool().numpy()
    clinical = data["clinical"].numpy()
    colpo = modality_mean(features, available, slice(1, 5))
    oct_features = modality_mean(features, available, slice(5, None))
    vectors = {
        "clinical_only_lr": clinical,
        "raw_colposcopy_meanpool_linear": np.concatenate([clinical, colpo], axis=1),
        "raw_oct_meanpool_linear": np.concatenate([clinical, oct_features], axis=1),
        "raw_full_meanpool_linear": np.concatenate([clinical, colpo, oct_features], axis=1),
    }
    selected_count = {
        "clinical_only_lr": np.zeros(len(features)),
        "raw_colposcopy_meanpool_linear": available[:, 1:5].sum(axis=1),
        "raw_oct_meanpool_linear": available[:, 5:].sum(axis=1),
        "raw_full_meanpool_linear": available[:, 1:].sum(axis=1),
    }
    train = data["parts"]["train"]["index"].astype(int).tolist()
    val = data["parts"]["val"]["index"].astype(int).tolist()
    test = data["parts"]["test"]["index"].astype(int).tolist()
    labels = data["payload"]["labels"].numpy().astype(int)
    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    calibration = {}
    for method, vector in vectors.items():
        components = min(128, vector.shape[1], max(2, len(train) - 1))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, random_state=20260905),
            LogisticRegression(C=0.1, max_iter=4000, class_weight="balanced", random_state=20260905),
        )
        model.fit(vector[train], labels[train])
        val_logit = model.decision_function(vector[val])
        temperature = fit_temperature(val_logit, labels[val])
        val_probability = np.asarray(sigmoid(val_logit / temperature), dtype=float)
        threshold = select_balanced_accuracy_threshold(labels[val], val_probability)
        test_logit = model.decision_function(vector[test])
        test_probability = np.asarray(sigmoid(test_logit / temperature), dtype=float)
        calibration[method] = {"temperature_source_validation": temperature, "threshold_source_validation": threshold}
        validation_rows.extend(
            records(
                data=data,
                indices=val,
                labels=labels[val],
                probabilities=val_probability,
                threshold=threshold,
                fold=args.fold,
                split="val",
                method=method,
                selected_count=selected_count[method][val],
            )
        )
        test_rows.extend(
            records(
                data=data,
                indices=test,
                labels=labels[test],
                probabilities=test_probability,
                threshold=threshold,
                fold=args.fold,
                split="test",
                method=method,
                selected_count=selected_count[method][test],
            )
        )
    pd.DataFrame(validation_rows).to_csv(output / "validation_predictions.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "test_predictions.csv", index=False)
    (output / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(json.dumps({"status": "complete", "fold": args.fold, "methods": sorted(vectors)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CESL LINEAR CONTROLS COMPLETE fold={args.fold} methods={len(vectors)}", flush=True)


if __name__ == "__main__":
    main()
