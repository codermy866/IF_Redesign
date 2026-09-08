#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def calculate(frame: pd.DataFrame, suffix: str) -> dict[str, float]:
    y = frame.label.astype(int).to_numpy()
    pred = frame[f"predicted_label_{suffix}"].astype(int).to_numpy()
    prob = frame[f"probability_{suffix}"].fillna(frame[f"predicted_label_{suffix}"]).to_numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired, patient-clustered comparison of two prediction files")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    columns = ["id", "patient_id", "label", "predicted_label", "probability"]
    ref = pd.read_csv(args.reference)[columns]
    cand = pd.read_csv(args.candidate)[columns]
    frame = ref.merge(cand, on=["id", "patient_id", "label"], suffixes=("_ref", "_cand"), validate="one_to_one")
    frame = frame.dropna(subset=["predicted_label_ref", "predicted_label_cand"])
    ref_correct = frame.predicted_label_ref.astype(int) == frame.label.astype(int)
    cand_correct = frame.predicted_label_cand.astype(int) == frame.label.astype(int)
    ref_only = int((ref_correct & ~cand_correct).sum())
    cand_only = int((~ref_correct & cand_correct).sum())
    discordant = ref_only + cand_only
    mcnemar_p = float(binomtest(min(ref_only, cand_only), discordant, 0.5).pvalue) if discordant else 1.0

    patients = frame.patient_id.unique()
    rng = np.random.default_rng(args.seed)
    deltas: dict[str, list[float]] = {"accuracy": [], "auroc": [], "auprc": []}
    for _ in range(args.bootstrap):
        sampled = rng.choice(patients, len(patients), replace=True)
        draw = pd.concat([frame[frame.patient_id == patient] for patient in sampled], ignore_index=True)
        if draw.label.nunique() < 2:
            continue
        a, b = calculate(draw, "ref"), calculate(draw, "cand")
        for metric in deltas:
            deltas[metric].append(b[metric] - a[metric])
    report = {
        "n_paired": int(len(frame)),
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "reference_metrics": calculate(frame, "ref"),
        "candidate_metrics": calculate(frame, "cand"),
        "candidate_minus_reference": {
            metric: {
                "point": calculate(frame, "cand")[metric] - calculate(frame, "ref")[metric],
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            }
            for metric, values in deltas.items()
        },
        "mcnemar": {
            "candidate_only_correct": cand_only,
            "reference_only_correct": ref_only,
            "exact_p": mcnemar_p,
        },
        "warning": "Exploratory balanced pilot; confidence intervals and p-values are descriptive, not confirmatory.",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
