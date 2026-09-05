from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import fixed_prediction_metrics


REPORTED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "auroc",
    "auprc",
    "brier",
    "ece",
)


def valid_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["label", "probability", "predicted_label", "patient_id", "center"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table lacks columns: {sorted(missing)}")
    clean = frame.dropna(subset=["label", "probability", "predicted_label"]).copy()
    clean["label"] = clean.label.astype(int)
    clean["predicted_label"] = clean.predicted_label.astype(int)
    clean["patient_id"] = clean.patient_id.astype(str)
    clean["center"] = clean.center.astype(str)
    return clean


def point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    clean = valid_predictions(frame)
    if clean.empty:
        return {}
    return fixed_prediction_metrics(clean.label, clean.probability, clean.predicted_label)


def macro_centre_metrics(frame: pd.DataFrame) -> dict[str, float]:
    by_centre = [point_metrics(part) for _, part in valid_predictions(frame).groupby("center")]
    return {
        metric: float(np.nanmean([row.get(metric, np.nan) for row in by_centre]))
        for metric in REPORTED_METRICS
    }


def stratified_cluster_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Sample patients within centre and outcome strata, preserving test prevalence."""
    pieces: list[np.ndarray] = []
    for _, stratum in frame.groupby(["center", "label"], sort=True):
        patients = stratum.patient_id.unique()
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        patient_rows = {patient: stratum.index[stratum.patient_id == patient].to_numpy() for patient in patients}
        for patient in sampled_patients:
            pieces.append(patient_rows[patient])
    return np.concatenate(pieces)


def _cluster_strata(frame: pd.DataFrame) -> list[list[np.ndarray]]:
    strata: list[list[np.ndarray]] = []
    for _, stratum in frame.groupby(["center", "label"], sort=True):
        strata.append(
            [stratum.index[stratum.patient_id == patient].to_numpy() for patient in stratum.patient_id.unique()]
        )
    return strata


def _draw_indices(strata: list[list[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for patients in strata:
        selected = rng.integers(0, len(patients), size=len(patients))
        pieces.extend(patients[index] for index in selected)
    return np.concatenate(pieces)


def _array_metrics(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = y.astype(np.int8, copy=False)
    prediction = prediction.astype(np.int8, copy=False)
    positive = y == 1
    negative = ~positive
    tp = np.count_nonzero(prediction[positive] == 1)
    fn = np.count_nonzero(prediction[positive] == 0)
    tn = np.count_nonzero(prediction[negative] == 0)
    fp = np.count_nonzero(prediction[negative] == 1)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1_denominator = 2 * tp + fp + fn
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability)) + 1]
    counts = np.diff(np.r_[starts, len(y)])
    positives_by_score = np.add.reduceat(sorted_y, starts)
    negatives_by_score = counts - positives_by_score
    negatives_before = np.r_[0, np.cumsum(negatives_by_score)[:-1]]
    auroc = (
        float(np.sum(positives_by_score * (negatives_before + 0.5 * negatives_by_score)))
        / (n_positive * n_negative)
        if n_positive and n_negative
        else np.nan
    )
    descending_order = order[::-1]
    descending_probability = probability[descending_order]
    descending_y = y[descending_order]
    endpoints = np.r_[np.flatnonzero(np.diff(descending_probability)), len(y) - 1]
    true_positives = np.cumsum(descending_y)[endpoints]
    retrieved = endpoints + 1
    recall = true_positives / n_positive if n_positive else np.full_like(true_positives, np.nan)
    precision = true_positives / retrieved
    auprc = (
        float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        if n_positive and n_negative
        else np.nan
    )
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        include = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9 else probability < edges[index + 1]
        )
        if include.any():
            ece += include.mean() * abs(float(y[include].mean()) - float(probability[include].mean()))
    return {
        "accuracy": float(np.mean(y == prediction)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(2 * tp / f1_denominator) if f1_denominator else 0.0,
        "auroc": float(auroc),
        "auprc": auprc,
        "brier": float(np.mean((probability - y) ** 2)),
        "ece": float(ece),
    }


def _macro_array_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    centre: np.ndarray,
) -> dict[str, float]:
    summaries = [
        _array_metrics(y[centre == value], probability[centre == value], prediction[centre == value])
        for value in np.unique(centre)
    ]
    return {
        metric: float(np.nanmean([summary[metric] for summary in summaries]))
        for metric in REPORTED_METRICS
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, float]]]:
    clean = valid_predictions(frame).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    y = clean.label.to_numpy(dtype=np.int8)
    probability = clean.probability.to_numpy(dtype=float)
    prediction = clean.predicted_label.to_numpy(dtype=np.int8)
    centre = clean.center.to_numpy(dtype=str)
    strata = _cluster_strata(clean)
    draws = {
        "pooled": {metric: [] for metric in REPORTED_METRICS},
        "macro_centre": {metric: [] for metric in REPORTED_METRICS},
    }
    for _ in range(replicates):
        indices = _draw_indices(strata, rng)
        summaries = {
            "pooled": _array_metrics(y[indices], probability[indices], prediction[indices]),
            "macro_centre": _macro_array_metrics(
                y[indices], probability[indices], prediction[indices], centre[indices]
            ),
        }
        for scale, summary in summaries.items():
            for metric in REPORTED_METRICS:
                value = summary.get(metric, np.nan)
                if np.isfinite(value):
                    draws[scale][metric].append(value)
    intervals: dict[str, dict[str, list[float]]] = {}
    standard_errors: dict[str, dict[str, float]] = {}
    for scale, values_by_metric in draws.items():
        intervals[scale] = {}
        standard_errors[scale] = {}
        for metric, values in values_by_metric.items():
            intervals[scale][metric] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            standard_errors[scale][metric] = float(np.std(values, ddof=1))
    return intervals, standard_errors


def paired_frame(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    columns = ["id", "patient_id", "center", "label", "probability", "predicted_label"]
    ref = reference[columns].copy()
    cand = candidate[columns].copy()
    paired = ref.merge(
        cand,
        on=["id", "patient_id", "center", "label"],
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    return paired.dropna(
        subset=[
            "probability_reference",
            "predicted_label_reference",
            "probability_candidate",
            "predicted_label_candidate",
        ]
    ).reset_index(drop=True)


def _side(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            f"probability_{suffix}": "probability",
            f"predicted_label_{suffix}": "predicted_label",
        }
    )[["id", "patient_id", "center", "label", "probability", "predicted_label"]]


def paired_bootstrap_delta(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    replicates: int,
    seed: int,
    metrics: Iterable[str] = REPORTED_METRICS,
) -> dict:
    paired = paired_frame(reference, candidate)
    reference_clean = _side(paired, "reference")
    candidate_clean = _side(paired, "candidate")
    requested = tuple(metrics)
    reference_point = point_metrics(reference_clean)
    candidate_point = point_metrics(candidate_clean)
    reference_macro = macro_centre_metrics(reference_clean)
    candidate_macro = macro_centre_metrics(candidate_clean)
    draws = {
        "pooled": {metric: [] for metric in requested},
        "macro_centre": {metric: [] for metric in requested},
    }
    rng = np.random.default_rng(seed)
    y = reference_clean.label.to_numpy(dtype=np.int8)
    centre = reference_clean.center.to_numpy(dtype=str)
    reference_probability = reference_clean.probability.to_numpy(dtype=float)
    reference_prediction = reference_clean.predicted_label.to_numpy(dtype=np.int8)
    candidate_probability = candidate_clean.probability.to_numpy(dtype=float)
    candidate_prediction = candidate_clean.predicted_label.to_numpy(dtype=np.int8)
    strata = _cluster_strata(reference_clean)
    for _ in range(replicates):
        indices = _draw_indices(strata, rng)
        summaries = {
            "pooled": (
                _array_metrics(y[indices], reference_probability[indices], reference_prediction[indices]),
                _array_metrics(y[indices], candidate_probability[indices], candidate_prediction[indices]),
            ),
            "macro_centre": (
                _macro_array_metrics(
                    y[indices], reference_probability[indices], reference_prediction[indices], centre[indices]
                ),
                _macro_array_metrics(
                    y[indices], candidate_probability[indices], candidate_prediction[indices], centre[indices]
                ),
            ),
        }
        for scale, (ref_metrics, cand_metrics) in summaries.items():
            for metric in requested:
                delta = cand_metrics.get(metric, np.nan) - ref_metrics.get(metric, np.nan)
                if np.isfinite(delta):
                    draws[scale][metric].append(delta)
    result = {
        "n_paired": int(len(paired)),
        "reference_score_coverage": float(len(reference_clean) / len(reference)),
        "candidate_score_coverage": float(len(candidate_clean) / len(candidate)),
        "reference": {"pooled": reference_point, "macro_centre": reference_macro},
        "candidate": {"pooled": candidate_point, "macro_centre": candidate_macro},
        "candidate_minus_reference": {},
    }
    for scale, values_by_metric in draws.items():
        ref_point = reference_point if scale == "pooled" else reference_macro
        cand_point = candidate_point if scale == "pooled" else candidate_macro
        result["candidate_minus_reference"][scale] = {}
        for metric, values in values_by_metric.items():
            array = np.asarray(values)
            less_equal = (np.count_nonzero(array <= 0.0) + 1) / (len(array) + 1)
            greater_equal = (np.count_nonzero(array >= 0.0) + 1) / (len(array) + 1)
            result["candidate_minus_reference"][scale][metric] = {
                "point": float(cand_point[metric] - ref_point[metric]),
                "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
                "bootstrap_p_two_sided": float(min(1.0, 2.0 * min(less_equal, greater_equal))),
            }
    return result


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()
