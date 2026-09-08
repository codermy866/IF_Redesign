#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_jsonl  # noqa: E402
from cervix_cogalign.parsing import parse_prediction  # noqa: E402
from cervix_cogalign.rewards import cognition_reward, format_reward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate patient-clustered cervical predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame.dropna(subset=["predicted_label"]).copy()
    if valid.empty:
        return {}
    y = valid["label"].astype(int).to_numpy()
    pred = valid["predicted_label"].astype(int).to_numpy()
    probability = valid["probability"].fillna(valid["predicted_label"]).astype(float).to_numpy()
    result = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, probability),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = roc_auc_score(y, probability)
        result["auprc"] = average_precision_score(y, probability)
    return {key: float(value) for key, value in result.items()}


def clustered_intervals(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, list[float]]:
    patients = frame["patient_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        pieces = [frame[frame["patient_id"].astype(str) == patient] for patient in sampled]
        metrics = point_metrics(pd.concat(pieces, ignore_index=True))
        for key, value in metrics.items():
            draws.setdefault(key, []).append(value)
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in draws.items()
        if values
    }


def main() -> None:
    args = parse_args()
    parsed = []
    for row in read_jsonl(args.predictions):
        label, probability = parse_prediction(row.get("completion", ""))
        parsed.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "center": row["center"],
                "split": row["split"],
                "label": int(row["label"]),
                "predicted_label": label,
                "probability": probability,
                "format_reward": format_reward(row.get("completion", "")),
                "cognition_reward": cognition_reward(row.get("completion", "")),
                "latency_seconds": row.get("latency_seconds"),
                "completion": row.get("completion", ""),
            }
        )
    frame = pd.DataFrame(parsed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "patient_level_predictions.csv", index=False)
    summary = {
        "n": int(len(frame)),
        "parse_coverage": float(frame.predicted_label.notna().mean()),
        "probability_coverage": float(frame.probability.notna().mean()),
        "format_pass_rate": float(frame.format_reward.mean()),
        "mean_cognition_reward": float(frame.cognition_reward.mean()),
        "metrics": point_metrics(frame),
        "clustered_95ci": clustered_intervals(frame, args.bootstrap, args.seed),
        "warning": "Balanced pilot enrichment means these values are feasibility signals, not prevalence-valid clinical performance.",
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
