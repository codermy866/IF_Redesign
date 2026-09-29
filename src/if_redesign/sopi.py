"""Stagewise Orthogonal Proper-Score Innovation (SOPI).

All functions in this module operate on held-out/cross-fitted predictions.
They deliberately separate posterior movement from outcome-aligned
information gain.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScoreName = Literal["brier", "log"]
MovementName = Literal["absolute", "squared", "kl"]


def _prepare(
    y: ArrayLike,
    q0: ArrayLike,
    q1: ArrayLike,
    *,
    eps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y_arr = np.asarray(y, dtype=np.float64)
    q0_arr = np.asarray(q0, dtype=np.float64)
    q1_arr = np.asarray(q1, dtype=np.float64)

    if y_arr.shape != q0_arr.shape or y_arr.shape != q1_arr.shape:
        raise ValueError("y, q0, and q1 must have identical shapes")
    if np.any((y_arr < 0) | (y_arr > 1)):
        raise ValueError("y must lie in [0, 1]")
    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")

    q0_arr = np.clip(q0_arr, eps, 1.0 - eps)
    q1_arr = np.clip(q1_arr, eps, 1.0 - eps)
    return y_arr, q0_arr, q1_arr


def brier_contributions(
    y: ArrayLike,
    q0: ArrayLike,
    q1: ArrayLike,
    *,
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Per-sample Brier innovation: loss before minus loss after."""
    y_arr, q0_arr, q1_arr = _prepare(y, q0, q1, eps=eps)
    return (y_arr - q0_arr) ** 2 - (y_arr - q1_arr) ** 2


def log_contributions(
    y: ArrayLike,
    q0: ArrayLike,
    q1: ArrayLike,
    *,
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Per-sample log-score advancement."""
    y_arr, q0_arr, q1_arr = _prepare(y, q0, q1, eps=eps)
    return (
        y_arr * np.log(q1_arr / q0_arr)
        + (1.0 - y_arr) * np.log((1.0 - q1_arr) / (1.0 - q0_arr))
    )


def sopi_contributions(
    y: ArrayLike,
    q0: ArrayLike,
    q1: ArrayLike,
    *,
    score: ScoreName = "log",
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Return outcome-aligned innovation for each held-out prediction."""
    if score == "brier":
        return brier_contributions(y, q0, q1, eps=eps)
    if score == "log":
        return log_contributions(y, q0, q1, eps=eps)
    raise ValueError(f"unknown score: {score}")


def estimate_sopi(
    y: ArrayLike,
    q0: ArrayLike,
    q1: ArrayLike,
    *,
    score: ScoreName = "log",
    eps: float = 1e-8,
) -> float:
    """Estimate cohort-level SOPI from held-out/cross-fitted predictions."""
    return float(np.mean(sopi_contributions(y, q0, q1, score=score, eps=eps)))


def posterior_movement(
    q0: ArrayLike,
    q1: ArrayLike,
    *,
    metric: MovementName = "absolute",
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Outcome-agnostic posterior movement used as a diagnostic comparator."""
    q0_arr = np.clip(np.asarray(q0, dtype=np.float64), eps, 1.0 - eps)
    q1_arr = np.clip(np.asarray(q1, dtype=np.float64), eps, 1.0 - eps)
    if q0_arr.shape != q1_arr.shape:
        raise ValueError("q0 and q1 must have identical shapes")

    if metric == "absolute":
        return np.abs(q1_arr - q0_arr)
    if metric == "squared":
        return (q1_arr - q0_arr) ** 2
    if metric == "kl":
        return (
            q1_arr * np.log(q1_arr / q0_arr)
            + (1.0 - q1_arr) * np.log((1.0 - q1_arr) / (1.0 - q0_arr))
        )
    raise ValueError(f"unknown movement metric: {metric}")


def stagewise_sopi(
    y: ArrayLike,
    stage_predictions: ArrayLike,
    *,
    score: ScoreName = "log",
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Return per-stage innovation for nested evidence states.

    stage_predictions has shape (n_samples, n_stages) and contains the
    predictive probability after each evidence state F_0, ..., F_T.
    The returned array has shape (n_samples, n_stages - 1).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    stages = np.asarray(stage_predictions, dtype=np.float64)

    if stages.ndim != 2:
        raise ValueError("stage_predictions must be a 2D array")
    if stages.shape[0] != y_arr.shape[0]:
        raise ValueError("stage_predictions and y must contain the same samples")
    if stages.shape[1] < 2:
        raise ValueError("at least two evidence stages are required")

    parts = [
        sopi_contributions(
            y_arr,
            stages[:, t],
            stages[:, t + 1],
            score=score,
            eps=eps,
        )
        for t in range(stages.shape[1] - 1)
    ]
    return np.stack(parts, axis=1)
