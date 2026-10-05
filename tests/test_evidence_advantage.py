import numpy as np

from if_redesign.evidence_advantage import (
    advantage_features,
    evidence_advantage,
    evaluate_selective_update,
    selective_logits,
)


def test_evidence_advantage_sign_tracks_patient_specific_help():
    y = np.array([1.0, 0.0])
    clinical = np.array([0.0, 0.0])
    updated = np.array([2.0, 2.0])

    advantage = evidence_advantage(y, clinical, updated, loss="log")

    assert advantage[0] > 0
    assert advantage[1] < 0


def test_true_advantage_sign_recovers_oracle_selective_policy():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    clinical = np.array([0.0, 0.0, -0.4, 0.4])
    updated = np.array([2.0, 2.0, 0.8, -0.8])
    advantage = evidence_advantage(y, clinical, updated)

    ev = evaluate_selective_update(
        y,
        clinical,
        updated,
        advantage,
        threshold=0.0,
    )

    assert ev.decision_regret < 1e-12
    assert 0.0 < ev.coverage < 1.0


def test_selective_logits_respects_threshold():
    clinical = np.array([-1.0, 1.0, 0.0])
    updated = np.array([0.5, 2.0, -0.5])
    score = np.array([0.2, -0.1, 0.0])

    final, accept = selective_logits(clinical, updated, score, threshold=0.0)

    np.testing.assert_array_equal(accept, np.array([True, False, False]))
    np.testing.assert_allclose(final, np.array([0.5, 1.0, 0.0]))


def test_advantage_features_have_expected_shape_and_are_finite():
    clinical = np.array([-1.0, 0.0, 1.0])
    updated = np.array([-0.5, 0.2, 1.5])
    x = advantage_features(clinical, updated)

    assert x.shape == (3, 8)
    assert np.isfinite(x).all()
