#!/usr/bin/env python3
"""Fit leakage-safe tabular and frozen-feature baselines on every LOCO fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json  # noqa: E402
from cervix_cogalign.metrics import binary_metrics, select_balanced_accuracy_threshold  # noqa: E402
from cervix_cogalign.prompts import normalize_hpv, normalize_tct  # noqa: E402


def tabular_model(columns: list[str]):
    numeric = [column for column in columns if column == "age"]
    categorical = [column for column in columns if column != "age"]
    transformer = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return make_pipeline(
        transformer,
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=20260902),
    )


def image_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=20260902),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/formal_v1.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/formal_v1/controls"))
    args = parser.parse_args()
    config = read_json(args.config)
    manifest = pd.read_csv(config["manifest"])
    manifest["case_id"] = manifest.case_id.astype(str)
    manifest["label"] = manifest.binary_label.astype(int)
    manifest["center"] = manifest.hospital_name.astype(str)
    manifest["hpv_category"] = manifest.hpv.map(normalize_hpv)
    manifest["tct_category"] = manifest.tct.map(normalize_tct)
    manifest = manifest.set_index("case_id", drop=False)
    assignment = pd.read_csv(Path(config["output_dir"]) / "fold_assignments.csv")
    assignment.case_id = assignment.case_id.astype(str)
    patient_key = assignment.drop_duplicates("case_id").set_index("case_id").patient_id
    manifest["patient_id"] = manifest.case_id.map(patient_key)
    if manifest.loc[patient_key.index, "patient_id"].isna().any():
        raise AssertionError("Missing hashed patient identifier")

    payload = torch.load(config["feature_cache"], map_location="cpu", weights_only=False)
    cache = payload.get("features", payload)
    features = {str(key).rsplit("||", 1)[-1]: value for key, value in cache.items()}

    def pooled(ids: list[str], modality: str) -> np.ndarray:
        missing = [identifier for identifier in ids if identifier not in features]
        if missing:
            raise KeyError(f"Missing {len(missing)} cached features: {missing[:3]}")
        return np.stack([features[identifier][modality].float().mean(0).numpy() for identifier in ids])

    predictions = []
    summaries = []
    for fold in config["centres"].values():
        fold_assignment = assignment[assignment.fold == fold]
        parts = {}
        for split in ["train", "val", "test"]:
            ids = fold_assignment[fold_assignment.split == split].case_id.tolist()
            parts[split] = manifest.loc[ids].copy()
        train, val, test = parts["train"], parts["val"], parts["test"]
        models: dict[str, tuple[object, list[str] | str]] = {
            "clinical_only_lr": (tabular_model(["age", "hpv_category", "tct_category"]), ["age", "hpv_category", "tct_category"]),
            "clinical_plus_center_lr": (tabular_model(["age", "hpv_category", "tct_category", "center"]), ["age", "hpv_category", "tct_category", "center"]),
            "frozen_colposcopy_linear": (image_model(), "colpo"),
            "frozen_oct_linear": (image_model(), "oct"),
            "frozen_multimodal_linear": (image_model(), "multimodal"),
        }
        for method, (model, source) in models.items():
            if isinstance(source, list):
                train_x, val_x, test_x = train[source], val[source], test[source]
            elif source == "multimodal":
                train_x = np.concatenate([pooled(train.case_id.tolist(), "colpo"), pooled(train.case_id.tolist(), "oct")], axis=1)
                val_x = np.concatenate([pooled(val.case_id.tolist(), "colpo"), pooled(val.case_id.tolist(), "oct")], axis=1)
                test_x = np.concatenate([pooled(test.case_id.tolist(), "colpo"), pooled(test.case_id.tolist(), "oct")], axis=1)
            else:
                train_x = pooled(train.case_id.tolist(), source)
                val_x = pooled(val.case_id.tolist(), source)
                test_x = pooled(test.case_id.tolist(), source)
            model.fit(train_x, train.label)
            val_probability = model.predict_proba(val_x)[:, 1]
            test_probability = model.predict_proba(test_x)[:, 1]
            threshold = select_balanced_accuracy_threshold(val.label, val_probability)
            metrics = binary_metrics(test.label, test_probability, threshold)
            metrics.update({"fold": fold, "method": method, "validation_threshold": threshold})
            summaries.append(metrics)
            for identifier, patient, center, label, probability in zip(
                test.case_id, test.patient_id, test.center, test.label, test_probability
            ):
                predictions.append(
                    {
                        "id": identifier,
                        "patient_id": patient,
                        "center": center,
                        "fold": fold,
                        "method": method,
                        "label": int(label),
                        "probability": float(probability),
                        "threshold": threshold,
                        "predicted_label": int(probability >= threshold),
                    }
                )
        print(f"completed controls for {fold}", flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "fold_metrics.csv", index=False)
    macro = pd.DataFrame(summaries).groupby("method")[["accuracy", "balanced_accuracy", "auroc", "auprc", "brier", "ece"]].agg(["mean", "std"])
    with (output / "summary.json").open("w", encoding="utf-8") as sink:
        json.dump(
            {method: {f"{metric}_{stat}": float(value) for (metric, stat), value in row.items()} for method, row in macro.iterrows()},
            sink,
            ensure_ascii=False,
            indent=2,
        )
    print(macro.to_string())


if __name__ == "__main__":
    main()
