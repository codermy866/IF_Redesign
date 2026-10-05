import numpy as np

from if_redesign.evidence_admission import (
    EvidenceAdmissionLogistic,
    admission_logits,
    correspondence_loss_gain,
    evidence_admission_effects,
    matched_replacement_indices,
    mean_reduction_fraction,
)


def test_admission_logits_are_additive():
    c = np.array([0.0, 1.0])
    o = np.array([0.5, -0.2])
    k = np.array([0.1, 0.3])
    z = admission_logits(c, o, k)
    np.testing.assert_allclose(z["C"], c)
    np.testing.assert_allclose(z["CO"], c + o)
    np.testing.assert_allclose(z["CK"], c + k)
    np.testing.assert_allclose(z["COK"], c + o + k)


def test_redundancy_definition_matches_difference_of_advantages():
    y = np.array([1.0, 0.0, 1.0])
    z = {
        "C": np.array([0.0, 0.0, 0.0]),
        "CO": np.array([1.0, -0.7, 0.6]),
        "CK": np.array([0.8, -0.5, 0.4]),
        "COK": np.array([1.2, -0.8, 0.7]),
    }
    eff = evidence_admission_effects(y, z)
    np.testing.assert_allclose(
        eff.oct_to_colposcopy_redundancy,
        eff.colposcopy_advantage - eff.residual_colposcopy_advantage,
    )


def test_mean_reduction_fraction_is_one_when_oct_removes_all_colpo_value():
    y = np.array([1.0, 0.0])
    z = {
        "C": np.array([0.0, 0.0]),
        "CO": np.array([1.0, -1.0]),
        "CK": np.array([1.0, -1.0]),
        "COK": np.array([1.0, -1.0]),
    }
    eff = evidence_admission_effects(y, z)
    assert np.isclose(mean_reduction_fraction(eff), 1.0)


def test_matched_replacement_excludes_self_and_respects_groups():
    x = np.array([[0.0], [0.1], [10.0], [10.2]])
    groups = np.array([0, 0, 1, 1])
    donor = matched_replacement_indices(x, groups)
    assert np.all(donor != np.arange(4))
    assert np.all(groups[donor] == groups)


def test_correspondence_gain_positive_when_replacement_is_worse():
    y = np.array([1.0, 0.0])
    real = np.array([2.0, -2.0])
    wrong = np.array([-1.0, 1.0])
    gain = correspondence_loss_gain(y, real, wrong)
    assert np.all(gain > 0)


def test_shared_admission_model_outputs_all_four_states():
    rng = np.random.default_rng(3)
    n = 30
    c = rng.normal(size=(n, 2))
    o = rng.normal(size=(n, 3))
    k = rng.normal(size=(n, 4))
    y = (c[:, 0] + 0.5 * o[:, 0] + 0.4 * k[:, 0] > 0).astype(int)
    model = EvidenceAdmissionLogistic(c=1.0, random_state=3).fit(c, o, k, y)
    out = model.all_states(c, o, k)
    assert set(out) == {"C", "CO", "CK", "COK"}
    assert all(v.shape == (n,) for v in out.values())
