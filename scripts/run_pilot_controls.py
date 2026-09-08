#!/usr/bin/env python3
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
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json, read_jsonl  # noqa: E402
from cervix_cogalign.prompts import normalize_hpv, normalize_tct  # noqa: E402


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
    }


def fit_tabular(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> np.ndarray:
    numeric = [column for column in columns if column == "age"]
    categorical = [column for column in columns if column != "age"]
    transformer = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = make_pipeline(transformer, LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    model.fit(train[columns], train.label)
    return model.predict_proba(val[columns])[:, 1]


def fit_embedding(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    model.fit(train_x, train_y)
    return model.predict_proba(val_x)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure data-side signal with fixed simple controls")
    parser.add_argument("--config", default=str(ROOT / "configs/pilot_v0.json"))
    parser.add_argument(
        "--feature-cache",
        default=str(ROOT.parent.parent / "exp_infofusion_2026/paper_revision/cache/patch_features_final_1897.pt"),
    )
    parser.add_argument("--output", default=str(ROOT / "results/pilot_v0/controls"))
    args = parser.parse_args()
    config = read_json(args.config)
    manifest = pd.read_csv(config["manifest"]).set_index("case_id", drop=False)

    parts = {}
    for split in ["train", "val"]:
        ids = [row["id"] for row in read_jsonl(ROOT / f"results/pilot_v0/baseline/{split}.jsonl")]
        part = manifest.loc[ids].copy()
        part["label"] = part.binary_label.astype(int)
        part["center"] = part.hospital_name.astype(str)
        part["hpv_category"] = part.hpv.map(normalize_hpv)
        part["tct_category"] = part.tct.map(normalize_tct)
        parts[split] = part
    train, val = parts["train"], parts["val"]
    predictions: dict[str, np.ndarray] = {
        "majority": np.full(len(val), float(train.label.mean())),
        "center_only": fit_tabular(train, val, ["center"]),
        "clinical_only": fit_tabular(train, val, ["age", "hpv_category", "tct_category"]),
        "clinical_plus_center": fit_tabular(
            train, val, ["age", "hpv_category", "tct_category", "center"]
        ),
    }

    cache = torch.load(args.feature_cache, map_location="cpu")["features"]
    feature_by_case = {key.rsplit("||", 1)[-1]: value for key, value in cache.items()}

    def pooled(frame: pd.DataFrame, modality: str) -> np.ndarray:
        return np.stack(
            [feature_by_case[case_id][modality].float().mean(dim=0).numpy() for case_id in frame.case_id]
        )

    train_colpo, val_colpo = pooled(train, "colpo"), pooled(val, "colpo")
    train_oct, val_oct = pooled(train, "oct"), pooled(val, "oct")
    predictions["frozen_colposcopy_mean"] = fit_embedding(train_colpo, train.label.to_numpy(), val_colpo)
    predictions["frozen_oct_mean"] = fit_embedding(train_oct, train.label.to_numpy(), val_oct)
    predictions["frozen_concat_mean"] = fit_embedding(
        np.concatenate([train_colpo, train_oct], axis=1),
        train.label.to_numpy(),
        np.concatenate([val_colpo, val_oct], axis=1),
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.DataFrame(
        {
            "case_id": val.case_id.to_numpy(),
            "patient_id": [row["patient_id"] for row in read_jsonl(ROOT / "results/pilot_v0/baseline/val.jsonl")],
            "label": val.label.to_numpy(),
            **predictions,
        }
    )
    pred_frame.to_csv(output / "validation_predictions.csv", index=False)
    report = {
        name: metrics(val.label.to_numpy(), probability)
        for name, probability in predictions.items()
    }
    report["interpretation"] = (
        "Frozen embedding controls test whether the selected data contain separable signal; they do not test the "
        "cognition-alignment mechanism and reuse no old performance result."
    )
    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
