import numpy as np

from if_redesign.sopi import (
    brier_contributions,
    estimate_sopi,
    log_contributions,
    posterior_movement,
    stagewise_sopi,
)


def test_brier_bayes_identity():
    p0 = np.array([0.2, 0.4, 0.7])
    p1 = np.array([0.3, 0.6, 0.8])
    expected = (
        p1 * brier_contributions(np.ones_like(p1), p0, p1)
        + (1 - p1) * brier_contributions(np.zeros_like(p1), p0, p1)
    )
    assert np.allclose(expected, (p1 - p0) ** 2)


def test_log_bayes_identity_matches_bernoulli_kl():
    p0 = np.array([0.2, 0.4, 0.7])
    p1 = np.array([0.3, 0.6, 0.8])
    expected = (
        p1 * log_contributions(np.ones_like(p1), p0, p1)
        + (1 - p1) * log_contributions(np.zeros_like(p1), p0, p1)
    )
    kl = p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))
    assert np.allclose(expected, kl)


def test_movement_can_be_large_while_innovation_is_harmful():
    y = np.array([0.0])
    q0 = np.array([0.2])
    q1 = np.array([0.8])
    assert posterior_movement(q0, q1, metric="squared")[0] > 0
    assert estimate_sopi(y, q0, q1, score="brier") < 0
    assert estimate_sopi(y, q0, q1, score="log") < 0


def test_stagewise_additivity_telescopes():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    stages = np.array([
        [0.2, 0.3, 0.1],
        [0.3, 0.5, 0.8],
        [0.4, 0.7, 0.9],
        [0.6, 0.4, 0.2],
    ])
    parts = stagewise_sopi(y, stages, score="log")
    direct = log_contributions(y, stages[:, 0], stages[:, -1])
    assert np.allclose(parts.sum(axis=1), direct)
