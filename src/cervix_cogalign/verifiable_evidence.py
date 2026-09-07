"""Atom-level utilities for a verifiable counterfactual evidence screen.

These functions operate on frozen multimodal representations.  They deliberately
do not turn representation compatibility into a semantic clinical claim: a
clinician-verified natural-language evidence step needs separate annotations.
"""
from __future__ import annotations

import numpy as np


def binary_symmetric_kl(reference: np.ndarray, altered: np.ndarray) -> np.ndarray:
    """Symmetric KL divergence between aligned Bernoulli posteriors."""
    p = np.clip(np.asarray(reference, dtype=float), 1e-6, 1.0 - 1e-6)
    q = np.clip(np.asarray(altered, dtype=float), 1e-6, 1.0 - 1e-6)
    kl_pq = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    kl_qp = q * np.log(q / p) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p))
    return 0.5 * (kl_pq + kl_qp)


def within_case_rank(values: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Scale observed visual atoms to [0, 1] by within-case ordinal rank."""
    values = np.asarray(values, dtype=float)
    available = np.asarray(available, dtype=bool)
    if values.shape != available.shape:
        raise ValueError("Values and availability must have the same shape")
    result = np.full(values.shape, np.nan, dtype=float)
    for row in range(len(values)):
        slots = np.flatnonzero(available[row])
        if not len(slots):
            continue
        finite = slots[np.isfinite(values[row, slots])]
        if not len(finite):
            continue
        if len(finite) == 1:
            result[row, finite[0]] = 1.0
            continue
        order = np.argsort(values[row, finite], kind="mergesort")
        ranks = np.empty(len(finite), dtype=float)
        ranks[order] = np.arange(len(finite), dtype=float)
        result[row, finite] = ranks / float(len(finite) - 1)
    return result


def representation_grounding(features: np.ndarray, visual_available: np.ndarray) -> np.ndarray:
    """Cross-atom representation compatibility in [0, 1].

    The score measures whether one visual atom is compatible with the remaining
    visual representation of its own case.  It is an observable representation
    grounding proxy, not proof of clinical semantic grounding.
    """
    features = np.asarray(features, dtype=np.float32)
    visual_available = np.asarray(visual_available, dtype=bool)
    if features.ndim != 3 or features.shape[:2] != visual_available.shape:
        raise ValueError("Expected BxKxD features and matching BxK availability")
    result = np.full(visual_available.shape, np.nan, dtype=float)
    for row in range(features.shape[0]):
        slots = np.flatnonzero(visual_available[row])
        if len(slots) == 1:
            result[row, slots[0]] = 0.5
            continue
        for slot in slots:
            others = slots[slots != slot]
            context = features[row, others].mean(axis=0)
            atom = features[row, slot]
            denom = float(np.linalg.norm(atom) * np.linalg.norm(context))
            cosine = float(np.dot(atom, context) / denom) if denom > 1e-8 else 0.0
            result[row, slot] = np.clip(0.5 * (cosine + 1.0), 0.0, 1.0)
    return result


def posterior_conflict(atom_only: np.ndarray, remainder: np.ndarray) -> np.ndarray:
    """Posterior disagreement risk for an atom and the remaining evidence.

    A value is high only if atom-only and remainder posteriors fall on opposite
    sides of 0.5 with high margins.  This is a predictive conflict proxy, not a
    semantic contradiction annotation.
    """
    atom_only = np.asarray(atom_only, dtype=float)
    remainder = np.asarray(remainder, dtype=float)
    if atom_only.shape != remainder.shape:
        raise ValueError("Atom-only and remainder arrays must be aligned")
    atom_margin = np.abs(2.0 * atom_only - 1.0)
    remainder_margin = np.abs(2.0 * remainder - 1.0)
    disagreement = (atom_only - 0.5) * (remainder - 0.5) < 0.0
    return np.where(disagreement, atom_margin * remainder_margin, 0.0)


def verified_evidence_score(
    grounding: np.ndarray,
    necessity: np.ndarray,
    information: np.ndarray,
    conflict: np.ndarray,
    visual_available: np.ndarray,
    *,
    instability: np.ndarray | None = None,
) -> np.ndarray:
    """Combine rank-normalised verifier components without learned outcome tuning."""
    visual_available = np.asarray(visual_available, dtype=bool)
    g = within_case_rank(grounding, visual_available)
    n = within_case_rank(necessity, visual_available)
    i = within_case_rank(information, visual_available)
    c = np.nan_to_num(np.asarray(conflict, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    score = np.power(np.clip(g, 1e-6, 1.0), 1.0 / 3.0)
    score *= np.power(np.clip(n, 1e-6, 1.0), 1.0 / 3.0)
    score *= np.power(np.clip(i, 1e-6, 1.0), 1.0 / 3.0)
    score *= np.clip(1.0 - c, 0.0, 1.0)
    if instability is not None:
        u = within_case_rank(np.asarray(instability, dtype=float), visual_available)
        score *= np.clip(1.0 - np.nan_to_num(u, nan=1.0), 0.0, 1.0)
    score[~visual_available] = np.nan
    return score


def top_slot(scores: np.ndarray, visual_available: np.ndarray) -> np.ndarray:
    """Return the top visual slot index per row; unavailable rows receive -1."""
    scores = np.asarray(scores, dtype=float)
    visual_available = np.asarray(visual_available, dtype=bool)
    if scores.shape != visual_available.shape:
        raise ValueError("Scores and availability must have the same shape")
    selected = np.full(len(scores), -1, dtype=int)
    for row in range(len(scores)):
        slots = np.flatnonzero(visual_available[row])
        if len(slots):
            selected[row] = int(slots[np.argmax(np.nan_to_num(scores[row, slots], nan=-np.inf))])
    return selected
