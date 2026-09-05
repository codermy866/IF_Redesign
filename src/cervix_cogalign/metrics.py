from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        include = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if include.any():
            value += include.mean() * abs(float(y[include].mean()) - float(probability[include].mean()))
    return float(value)


def binary_metrics(y, probability, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": expected_calibration_error(y, probability),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, probability))
        result["auprc"] = float(average_precision_score(y, probability))
    return result


def fixed_prediction_metrics(y, probability, prediction) -> dict[str, float]:
    """Metrics for predictions produced with fold-specific frozen thresholds."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": expected_calibration_error(y, probability),
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, probability))
        result["auprc"] = float(average_precision_score(y, probability))
    return result


def select_balanced_accuracy_threshold(y, probability) -> float:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    scored = [
        (balanced_accuracy_score(y, probability >= threshold), -abs(threshold - 0.5), threshold)
        for threshold in candidates
    ]
    return float(max(scored)[2])
