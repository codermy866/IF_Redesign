"""Patient-specific Evidence Advantage and selective evidence updating.

This module contains the reusable, data-agnostic implementation used by the
Stage105 experiments. Labels are used only to build retrospective advantage
targets or to select a threshold on a dedicated development split. Forward
inference requires only pre-update and candidate post-update predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

LossName = Literal["log", "brier"]


def sigmoid(logit: np.ndarray) -> np.ndarray:
    logit = np.asarray(logit, dtype=float)
    out = np.empty_like(logit)
    pos = logit >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logit[pos]))
    exp_z = np.exp(logit[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def per_patient_loss(
    y: np.ndarray,
    logit: np.ndarray,
    loss: LossName = "log",
) -> np.ndarray:
    """Return one binary proper-score loss per patient."""
    y = np.asarray(y, dtype=float)
    logit = np.asarray(logit, dtype=float)
    if y.shape != logit.shape:
        raise ValueError("y and logit must have identical shapes")

    if loss == "log":
        return np.logaddexp(0.0, logit) - y * logit
    if loss == "brier":
        return (sigmoid(logit) - y) ** 2
    raise ValueError(f"unsupported loss: {loss}")


def evidence_advantage(
    y: np.ndarray,
    clinical_logit: np.ndarray,
    updated_logit: np.ndarray,
    loss: LossName = "log",
) -> np.ndarray:
    """Patient-specific advantage of accepting the candidate evidence update.

    A_i = ell(y_i, p_i^clinical) - ell(y_i, p_i^updated).

    Positive values mean the candidate update lowers predictive loss for that
    patient; negative values identify retrospective negative fusion.
    """
    before = per_patient_loss(y, clinical_logit, loss=loss)
    after = per_patient_loss(y, updated_logit, loss=loss)
    return before - after


def advantage_features(
    clinical_logit: np.ndarray,
    updated_logit: np.ndarray,
) -> np.ndarray:
    """Low-dimensional features for the low-variance benefit-sign estimator."""
    z0 = np.asarray(clinical_logit, dtype=float)
    z1 = np.asarray(updated_logit, dtype=float)
    if z0.shape != z1.shape:
        raise ValueError("clinical_logit and updated_logit must match")

    delta = z1 - z0
    p0 = sigmoid(z0)
    p1 = sigmoid(z1)
    confidence_gain = np.abs(p1 - 0.5) - np.abs(p0 - 0.5)
    return np.column_stack(
        [
            z0,
            delta,
            np.abs(delta),
            p0,
            p1,
            confidence_gain,
            np.abs(z0),
            np.abs(z1),
        ]
    )


@dataclass(frozen=True)
class SelectiveUpdateEvaluation:
    n: int
    coverage: float
    benefit_auc: float | None
    decision_regret: float
    always_update_regret: float
    realized_gain_over_clinical: float
    harmful_update_population: float
    harm_rate_selected: float


def selective_logits(
    clinical_logit: np.ndarray,
    updated_logit: np.ndarray,
    advantage_score: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept the candidate update only when its predicted advantage is high."""
    z0 = np.asarray(clinical_logit, dtype=float)
    z1 = np.asarray(updated_logit, dtype=float)
    score = np.asarray(advantage_score, dtype=float)
    if not (z0.shape == z1.shape == score.shape):
        raise ValueError("all arrays must have identical shapes")
    accept = score > float(threshold)
    return np.where(accept, z1, z0), accept


def evaluate_selective_update(
    y: np.ndarray,
    clinical_logit: np.ndarray,
    updated_logit: np.ndarray,
    advantage_score: np.ndarray,
    threshold: float = 0.0,
    loss: LossName = "log",
) -> SelectiveUpdateEvaluation:
    """Retrospective evaluation of an already-trained selective policy."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    z0 = np.asarray(clinical_logit, dtype=float)
    z1 = np.asarray(updated_logit, dtype=float)
    score = np.asarray(advantage_score, dtype=float)

    final_logit, accept = selective_logits(z0, z1, score, threshold)
    l0 = per_patient_loss(y, z0, loss)
    l1 = per_patient_loss(y, z1, loss)
    lf = per_patient_loss(y, final_logit, loss)
    oracle = np.minimum(l0, l1)
    advantage = l0 - l1

    target = advantage > 0
    auc = None
    if np.unique(target).size == 2:
        auc = float(roc_auc_score(target, score))

    selected_harm = float(np.mean(advantage[accept] < 0)) if np.any(accept) else 0.0
    return SelectiveUpdateEvaluation(
        n=int(y.size),
        coverage=float(np.mean(accept)),
        benefit_auc=auc,
        decision_regret=float(np.mean(lf - oracle)),
        always_update_regret=float(np.mean(l1 - oracle)),
        realized_gain_over_clinical=float(np.mean(l0 - lf)),
        harmful_update_population=float(np.mean(lf > l0 + 1e-12)),
        harm_rate_selected=selected_harm,
    )


class AdvantageSignEstimator:
    """Low-variance reference estimator for P(A_i > 0 | pre/post state).

    This is a reference implementation, not a required architectural choice.
    It was introduced after high-dimensional advantage regression showed
    appreciable seed variance in small calibration cohorts.
    """

    def __init__(
        self,
        cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        cv: int = 5,
    ) -> None:
        self.cs = tuple(float(x) for x in cs)
        self.cv = int(cv)
        self.pipeline_: Pipeline | None = None
        self.loss_: LossName | None = None

    def fit(
        self,
        y: np.ndarray,
        clinical_logit: np.ndarray,
        updated_logit: np.ndarray,
        loss: LossName = "log",
    ) -> "AdvantageSignEstimator":
        advantage = evidence_advantage(y, clinical_logit, updated_logit, loss)
        target = (advantage > 0).astype(int)
        counts = np.bincount(target, minlength=2)
        cv = min(self.cv, int(counts.min()))
        if cv < 2:
            raise ValueError("at least two samples per benefit class are required")

        self.pipeline_ = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                (
                    "classifier",
                    LogisticRegressionCV(
                        Cs=list(self.cs),
                        cv=cv,
                        scoring="roc_auc",
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(advantage_features(clinical_logit, updated_logit), target)
        self.loss_ = loss
        return self

    def _check_fitted(self) -> Pipeline:
        if self.pipeline_ is None:
            raise RuntimeError("fit must be called before inference")
        return self.pipeline_

    def predict_score(
        self,
        clinical_logit: np.ndarray,
        updated_logit: np.ndarray,
    ) -> np.ndarray:
        pipe = self._check_fitted()
        return np.asarray(
            pipe.decision_function(advantage_features(clinical_logit, updated_logit)),
            dtype=float,
        )

    def select_threshold(
        self,
        y: np.ndarray,
        clinical_logit: np.ndarray,
        updated_logit: np.ndarray,
        *,
        n_quantiles: int = 41,
    ) -> float:
        """Choose a threshold only on an independent selection split.

        The objective is minimum empirical decision regret. Final evaluation
        must use a different held-out split.
        """
        if self.loss_ is None:
            raise RuntimeError("fit must be called before threshold selection")
        score = self.predict_score(clinical_logit, updated_logit)
        candidates = np.unique(
            np.concatenate(
                [
                    [float(score.min()) - 1e-9, 0.0, float(score.max()) + 1e-9],
                    np.quantile(score, np.linspace(0.0, 1.0, n_quantiles)),
                ]
            )
        )

        best: tuple[float, float, float] | None = None
        for threshold in candidates:
            ev = evaluate_selective_update(
                y,
                clinical_logit,
                updated_logit,
                score,
                threshold=float(threshold),
                loss=self.loss_,
            )
            item = (ev.decision_regret, float(threshold), ev.coverage)
            if best is None:
                best = item
                continue
            if item[0] < best[0] - 1e-12:
                best = item
            elif abs(item[0] - best[0]) <= 1e-12:
                if abs(item[2] - 0.5) < abs(best[2] - 0.5):
                    best = item

        assert best is not None
        return best[1]
