"""Evidence Admission Intervention for multimodal residual-value analysis.

The core IF_Redesign method is Patient-Specific Evidence Advantage. This
module implements the exploratory extension used to quantify how admitting
one modality changes the remaining predictive value of a downstream modality.

The intervention is on information admission into a fixed prediction
function, not on the disease-generating process. These utilities therefore
must not be interpreted as identifying biological causal effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .evidence_advantage import per_patient_loss

AdmissionState = Literal["C", "CO", "CK", "COK"]
LossName = Literal["log", "brier"]


@dataclass(frozen=True)
class EvidenceAdmissionEffects:
    """Patient-level proper-score effects under four information states."""

    oct_advantage: np.ndarray
    colposcopy_advantage: np.ndarray
    residual_colposcopy_advantage: np.ndarray
    oct_to_colposcopy_redundancy: np.ndarray
    joint_advantage: np.ndarray


def admission_logits(
    clinical_logit: np.ndarray,
    oct_update: np.ndarray,
    colposcopy_update: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the four evidence-admission states from additive updates."""
    zc = np.asarray(clinical_logit, dtype=float)
    do = np.asarray(oct_update, dtype=float)
    dk = np.asarray(colposcopy_update, dtype=float)
    if not (zc.shape == do.shape == dk.shape):
        raise ValueError("clinical_logit, oct_update and colposcopy_update must match")
    return {
        "C": zc,
        "CO": zc + do,
        "CK": zc + dk,
        "COK": zc + do + dk,
    }


def evidence_admission_effects(
    y: np.ndarray,
    logits: dict[str, np.ndarray],
    loss: LossName = "log",
) -> EvidenceAdmissionEffects:
    """Compute OCT value, residual colposcopy value and cross-modal redundancy.

    A^O      = L(C) - L(C+O)
    A^K      = L(C) - L(C+K)
    A^{K|O}  = L(C+O) - L(C+O+K)
    R^{O->K} = A^K - A^{K|O}
    """
    required = {"C", "CO", "CK", "COK"}
    if set(logits) != required:
        raise ValueError(f"logits must contain exactly {sorted(required)}")

    y = np.asarray(y, dtype=float)
    lc = per_patient_loss(y, logits["C"], loss)
    lco = per_patient_loss(y, logits["CO"], loss)
    lck = per_patient_loss(y, logits["CK"], loss)
    lcok = per_patient_loss(y, logits["COK"], loss)

    a_o = lc - lco
    a_k = lc - lck
    a_k_given_o = lco - lcok
    redundancy = a_k - a_k_given_o
    joint = lc - lcok

    return EvidenceAdmissionEffects(
        oct_advantage=a_o,
        colposcopy_advantage=a_k,
        residual_colposcopy_advantage=a_k_given_o,
        oct_to_colposcopy_redundancy=redundancy,
        joint_advantage=joint,
    )


def mean_reduction_fraction(effects: EvidenceAdmissionEffects) -> float:
    """Fraction of mean colposcopy advantage reduced after OCT admission."""
    denom = float(np.mean(effects.colposcopy_advantage))
    if abs(denom) < 1e-12:
        raise ZeroDivisionError("mean colposcopy advantage is effectively zero")
    return float(np.mean(effects.oct_to_colposcopy_redundancy) / denom)


class EvidenceAdmissionLogistic:
    """Shared additive logistic reference model for four admission states.

    Inputs are prepared feature blocks:
      C: clinical features
      O: frozen OCT representation
      K: frozen colposcopy representation

    One classifier is fitted to all four masked states so state comparisons
    use the same prediction function rather than four unrelated models.
    """

    def __init__(self, c: float = 1.0, random_state: int = 0) -> None:
        self.c = float(c)
        self.random_state = int(random_state)
        self.model_: LogisticRegression | None = None

    @staticmethod
    def _state_design(
        clinical: np.ndarray,
        oct_features: np.ndarray,
        colposcopy_features: np.ndarray,
        state: AdmissionState,
    ) -> np.ndarray:
        c = np.asarray(clinical, dtype=float)
        o = np.asarray(oct_features, dtype=float)
        k = np.asarray(colposcopy_features, dtype=float)
        if not (len(c) == len(o) == len(k)):
            raise ValueError("all feature blocks must contain the same patients")
        zo = np.zeros_like(o)
        zk = np.zeros_like(k)
        if state == "C":
            return np.concatenate([c, zo, zk], axis=1)
        if state == "CO":
            return np.concatenate([c, o, zk], axis=1)
        if state == "CK":
            return np.concatenate([c, zo, k], axis=1)
        if state == "COK":
            return np.concatenate([c, o, k], axis=1)
        raise ValueError(f"unknown state: {state}")

    def fit(
        self,
        clinical: np.ndarray,
        oct_features: np.ndarray,
        colposcopy_features: np.ndarray,
        y: np.ndarray,
    ) -> "EvidenceAdmissionLogistic":
        y = np.asarray(y, dtype=int)
        designs = [
            self._state_design(clinical, oct_features, colposcopy_features, s)
            for s in ("C", "CO", "CK", "COK")
        ]
        x = np.concatenate(designs, axis=0)
        yy = np.tile(y, 4)
        self.model_ = LogisticRegression(
            C=self.c,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=self.random_state,
        )
        self.model_.fit(x, yy)
        return self

    def _check_fitted(self) -> LogisticRegression:
        if self.model_ is None:
            raise RuntimeError("fit must be called before inference")
        return self.model_

    def decision_function(
        self,
        clinical: np.ndarray,
        oct_features: np.ndarray,
        colposcopy_features: np.ndarray,
        state: AdmissionState,
    ) -> np.ndarray:
        model = self._check_fitted()
        x = self._state_design(clinical, oct_features, colposcopy_features, state)
        return np.asarray(model.decision_function(x), dtype=float)

    def all_states(
        self,
        clinical: np.ndarray,
        oct_features: np.ndarray,
        colposcopy_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            s: self.decision_function(clinical, oct_features, colposcopy_features, s)
            for s in ("C", "CO", "CK", "COK")
        }


def matched_replacement_indices(
    clinical_features: np.ndarray,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Nearest non-self donors for wrong-patient replacement controls.

    Matching never uses pathology. If groups are supplied, donors are
    constrained to the same group, such as acquisition centre.
    """
    x = np.asarray(clinical_features, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("at least two patients are required")

    if groups is None:
        groups = np.zeros(n, dtype=int)
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("groups must have one value per patient")

    donor = np.empty(n, dtype=int)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if len(ids) < 2:
            raise ValueError("each matching group needs at least two patients")
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(x[ids])
        _, local = nn.kneighbors(x[ids])
        donor[ids] = ids[local[:, 1]]

    if np.any(donor == np.arange(n)):
        raise RuntimeError("self-donor detected")
    return donor


def correspondence_loss_gain(
    y: np.ndarray,
    real_logit: np.ndarray,
    replaced_logit: np.ndarray,
    loss: LossName = "log",
) -> np.ndarray:
    """Loss(replaced evidence) minus Loss(real evidence), per patient.

    Positive values indicate that matched wrong-patient replacement worsens
    predictive loss. The replaced logit must be recomputed by the complete
    frozen prediction path after replacement; reusing the recipient's original
    gate or routing decision can create a spurious correspondence effect.
    """
    real = per_patient_loss(y, real_logit, loss)
    replaced = per_patient_loss(y, replaced_logit, loss)
    return replaced - real
