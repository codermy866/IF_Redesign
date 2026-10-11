import json

import numpy as np
import pytest
from scipy.special import expit

from if_redesign.cfv_jev import (
    REPRESENTATIONS, ResidualHead, center_balanced_nll, encode_evidence,
    extract_interventions, fit_residual,
)
from if_redesign.cfv_jev.training import _objective_and_gradient


def test_laplace_uses_valid_counts_signed_mean_and_clipped_magnitudes():
    delta = np.full((1, 20), 99.)
    mask = np.zeros((1, 20), dtype=bool)
    delta[0, :2] = [3., -1.]
    mask[0, :2] = True
    x = encode_evidence(delta, mask)
    assert x.shape == (1, 26)
    assert x[0, 0] == 1.  # Mean is not clipped; invalid slots are excluded.
    expected = (1 - np.exp(-np.geomspace(.5, 16, 6))) / 2
    np.testing.assert_allclose(x[0, 1:7], expected)
    np.testing.assert_allclose(x[0, 7:13], expected)
    np.testing.assert_array_equal(x[0, 13:], np.zeros(13))


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_all_encodings_keep_modality_permutation_and_zero_evidence(representation):
    rng = np.random.default_rng(4)
    delta = rng.normal(size=(5, 20))
    mask = rng.uniform(size=delta.shape) > .3
    order = np.r_[rng.permutation(12), 12 + rng.permutation(8)]
    actual = encode_evidence(delta, mask, representation)
    np.testing.assert_allclose(actual, encode_evidence(delta[:, order], mask[:, order], representation),
                               atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(encode_evidence(delta, np.zeros_like(mask), representation), np.zeros((5, 26)))
    np.testing.assert_array_equal(encode_evidence(np.zeros_like(delta), mask, representation), np.zeros((5, 26)))


def test_raw_quantiles_exclude_missing_slots():
    delta = np.full((1, 20), -100.)
    mask = np.zeros((1, 20), dtype=bool)
    delta[0, [0, 3]] = [2., 4.]
    mask[0, [0, 3]] = True
    x = encode_evidence(delta, mask, "RAW")
    assert x[0, 0] == 3.
    np.testing.assert_allclose(x[0, 1:13], np.linspace(2, 4, 12))


def test_interventions_rerun_full_nonlinear_model_at_same_clinical_prior():
    calls = []

    def scorer(views, c):
        calls.append((views, c))
        return .4 * c + sum(views) + views[0] * views[12]

    views = [20.] * 20
    views[0], views[12] = 2., 3.
    mask = np.zeros(20, dtype=bool)
    mask[[0, 12]] = True
    result = extract_interventions(views, mask, 1.7, scorer, visual_scale=.5, neutral_view=0.)
    assert len(calls) == 4  # null, full, one recomputation per valid slot
    assert all(len(v) == 20 and c == 1.7 for v, c in calls)
    assert all(v[1] == 0 for v, _ in calls)
    assert result.base_logit == pytest.approx(1.7 + .5 * (2 + 3 + 6))
    assert result.delta[0] == pytest.approx(.5 * (2 + 6))
    assert result.delta[12] == pytest.approx(.5 * (3 + 6))
    np.testing.assert_array_equal(result.delta[~mask], np.zeros(18))
    assert result.evidence().shape == (26,)
    calls.clear()
    empty = extract_interventions(views, np.zeros(20, bool), 1.7, scorer, visual_scale=.5, neutral_view=0.)
    assert len(calls) == 1
    assert empty.base_logit == 1.7
    np.testing.assert_array_equal(empty.evidence(), np.zeros(26))


@pytest.mark.parametrize("risk", [False, True])
def test_center_objective_gradient_and_equal_center_weight(risk):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(7, 26))
    base = rng.normal(size=7)
    y = np.array([0, 1, 1, 0, 0, 1, 0])
    groups = [np.array([0, 1]), np.arange(2, 7)]
    w = rng.normal(0, 2, 26)
    args = (x, base, y, groups, .03, risk, .25)
    value, gradient = _objective_and_gradient(w, *args)
    eps = 1e-6
    numerical = np.empty(26)
    for j in range(26):
        direction = np.zeros(26)
        direction[j] = eps
        numerical[j] = (_objective_and_gradient(w + direction, *args)[0] -
                        _objective_and_gradient(w - direction, *args)[0]) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)
    # Duplicating all observations in just one center cannot increase its weight.
    indices = np.r_[groups[0], groups[0], groups[0], groups[1]]
    repeated_groups = [np.arange(6), np.arange(6, 11)]
    repeated = _objective_and_gradient(w, x[indices], base[indices], y[indices],
                                      repeated_groups, .03, risk, .25)
    assert repeated[0] == pytest.approx(value)
    np.testing.assert_allclose(repeated[1], gradient, atol=1e-14)


def test_source_training_scaler_selection_and_json_zero_anchor():
    rng = np.random.default_rng(37)
    x = rng.normal(0, .1, (100, 26))
    x[:, -1] = 0
    base = rng.normal(size=100)
    y = rng.binomial(1, expit(base + 2 * x[:, 0]))
    groups = np.where(np.arange(100) < 30, "small", "large")
    selection = rng.normal(size=(40, 26))
    selection_base = rng.normal(size=40)
    selection_y = rng.binomial(1, expit(selection_base))
    selection_groups = ["one"] * 10 + ["two"] * 30  # Lists must work too.
    head = fit_residual(x, base, y, groups, selection, selection_base, selection_y, selection_groups)
    np.testing.assert_allclose(head.std, np.maximum(x.std(axis=0), .005))
    assert head.shrinkage == .25
    assert head.metadata["max_abs_gradient"] < 1e-3
    menu = head.metadata["selection_menu"]
    assert head.metadata["selection_center_NLL"] == min(entry["NLL"] for entry in menu)
    assert head.metadata["parameters"] == 26
    restored = ResidualHead.from_dict(json.loads(json.dumps(head.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(restored.predict_logit(selection_base, selection), head.predict_logit(selection_base, selection))
    np.testing.assert_array_equal(head.predict_logit(base, np.zeros_like(x)), base)
    assert center_balanced_nll(head.predict_logit(selection_base, selection), selection_y,
                               selection_groups) == pytest.approx(head.metadata["selection_center_NLL"])


def test_nonfinite_and_misaligned_inputs_fail_before_prediction():
    with pytest.raises(ValueError, match="finite"):
        encode_evidence(np.full((1, 20), np.nan), np.ones((1, 20), bool))
    with pytest.raises(ValueError, match="valid_mask"):
        encode_evidence(np.zeros((1, 20)), np.ones((1, 20)) * 2)
    with pytest.raises(ValueError, match="scalar"):
        extract_interventions([0] * 20, [False] * 20, 0, lambda v, c: [1], neutral_view=0)
    with pytest.raises(ValueError, match="binary"):
        center_balanced_nll([0, 1], [0, 2], ["one", "two"])
    with pytest.raises(ValueError, match="nonmissing"):
        center_balanced_nll([0, 1], [0, 1], ["one", None])


def test_materially_unconverged_residual_is_rejected(monkeypatch):
    from types import SimpleNamespace
    import if_redesign.cfv_jev.training as training

    def failed_fit(*args, **kwargs):
        return SimpleNamespace(fun=.5, x=np.zeros(26), jac=np.ones(26),
                               success=False, message="test convergence failure")

    monkeypatch.setattr(training, "minimize", failed_fit)
    x = np.zeros((2, 26))
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_residual(x, [0, 0], [0, 1], ["a", "a"], x, [0, 0], [0, 1], ["a", "a"])
