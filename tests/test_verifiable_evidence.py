from __future__ import annotations

import numpy as np

from cervix_cogalign.verifiable_evidence import (
    binary_symmetric_kl,
    posterior_conflict,
    representation_grounding,
    top_slot,
    verified_evidence_score,
)


def test_symmetric_kl_is_zero_for_identical_posteriors_and_positive_otherwise():
    same = binary_symmetric_kl(np.array([0.2, 0.8]), np.array([0.2, 0.8]))
    changed = binary_symmetric_kl(np.array([0.2]), np.array([0.8]))
    assert np.allclose(same, 0.0)
    assert changed[0] > 0.0


def test_grounding_and_conflict_respect_observed_atoms():
    features = np.array([[[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]]])
    available = np.array([[True, True, False]])
    grounding = representation_grounding(features, available)
    assert grounding[0, 0] > 0.9
    assert grounding[0, 2] != grounding[0, 2]
    conflict = posterior_conflict(np.array([[0.9, 0.2]]), np.array([[0.8, 0.9]]))
    assert conflict[0, 0] == 0.0
    assert conflict[0, 1] > 0.0


def test_verified_score_and_top_slot_are_deterministic():
    available = np.array([[True, True, True]])
    score = verified_evidence_score(
        np.array([[0.8, 0.5, 0.9]]),
        np.array([[0.7, 0.4, 0.95]]),
        np.array([[0.6, 0.3, 0.85]]),
        np.array([[0.0, 0.0, 0.0]]),
        available,
    )
    assert top_slot(score, available).tolist() == [2]
