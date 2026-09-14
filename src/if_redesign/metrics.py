"""Metrics for diagnosis, lesion sites, and evidence assimilation."""
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def evidence_metrics(assimilation):
    values = np.asarray(assimilation, dtype=float)
    return {"mean_A": float(values.mean()), "median_A": float(np.median(values)), "proportion_A_positive": float((values > 0).mean())}
