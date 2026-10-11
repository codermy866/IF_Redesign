import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from if_redesign.cfv_jev import extract_interventions
from if_redesign.cfv_jev.readout_training import (
    FittedRawReadout, _features, _raw_objective_and_gradient, fit_raw_readout,
)
from if_redesign.cfv_jev.training import center_balanced_nll


def test_raw_gradient_neutral_contrast_and_center_equal_weight():
    rng = np.random.default_rng(9)
    x, xn = rng.normal(size=(7, 33)), rng.normal(size=(7, 33))
    c = rng.normal(size=7)
    y = np.array([0, 1, 0, 1, 0, 1, 1])
    groups = [np.arange(2), np.arange(2, 7)]
    w = rng.normal(0, .1, 140)
    args = (x, xn, c, y, groups, .01)
    value, gradient = _raw_objective_and_gradient(w, *args)
    eps = 1e-6
    numerical = np.empty(140)
    for j in range(140):
        direction = np.zeros(140)
        direction[j] = eps
        numerical[j] = (_raw_objective_and_gradient(w + direction, *args)[0] -
                        _raw_objective_and_gradient(w - direction, *args)[0]) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-5)
    # Repeating just one center does not change its weight.
    rows = np.r_[groups[0], groups[0], groups[0], groups[1]]
    repeated = _raw_objective_and_gradient(w, x[rows], xn[rows], c[rows], y[rows],
                                         [np.arange(6), np.arange(6, 11)], .01)
    assert repeated[0] == pytest.approx(value)
    np.testing.assert_allclose(repeated[1], gradient, atol=1e-14)
    # Identical real/null features cancel inside training for every weight.
    neutral_value, neutral_gradient = _raw_objective_and_gradient(w, xn, xn, c, y, groups, .01)
    baseline = center_balanced_nll(c, y, np.array(["a", "a", "b", "b", "b", "b", "b"]))
    assert neutral_value == pytest.approx(baseline + .005 * (w @ w))
    np.testing.assert_allclose(neutral_gradient, .01 * w, atol=1e-15)


@pytest.fixture(scope="module")
def fitted_readout():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(162)
        train = rng.normal(0, .5, (64, 21, 2))
        train[:, 0, 1] = np.linspace(-2, 2, len(train))
        selection = rng.normal(0, .6, (24, 21, 2))
        selection[:, 0, 1] += 1
        neutral = np.zeros((21, 2))
        c_train = rng.normal(0, .3, len(train))
        c_selection = rng.normal(0, .3, len(selection))
        y_train = (train[:, 0, 1] - train[:, 0, 0] > 0).astype(int)
        y_selection = (selection[:, 0, 1] - selection[:, 0, 0] > 0).astype(int)
        train_groups = np.array(["small"] * 16 + ["large"] * 48)
        selection_groups = np.array(["small"] * 6 + ["large"] * 18)
        fit = fit_raw_readout(train, neutral, c_train, y_train, train_groups,
                             selection, neutral[None], c_selection, y_selection, selection_groups)
        yield fit, train, selection, neutral, c_selection, y_selection, selection_groups
    finally:
        torch.set_num_threads(previous_threads)


def test_source_scaler_menu_convergence_and_json(fitted_readout):
    fit, train, selection, neutral, c, y, groups = fitted_readout
    features = _features(train)
    np.testing.assert_allclose(fit.readout.feature_mean, features.mean(axis=0))
    std = features.std(axis=0)
    std[std == 0] = 1
    np.testing.assert_allclose(fit.readout.feature_scale, std)
    assert fit.metadata["parameters"] == 140
    assert fit.metadata["training_anchor"] == "clinical_plus_real_minus_neutral"
    assert all(item["optimizer_success"] for item in fit.metadata["optimizer_fits"])
    assert len(fit.metadata["selection_menu"]) == 8
    assert fit.metadata["selection_center_NLL"] == min(item["NLL"] for item in fit.metadata["selection_menu"])
    logits = c + fit.readout.effective_scale * (
        fit.readout.score_value(selection, c) - fit.readout.score_value(np.broadcast_to(neutral, selection.shape), c)
    )
    assert center_balanced_nll(logits, y, groups) == pytest.approx(fit.metadata["selection_center_NLL"])
    restored = FittedRawReadout.from_dict(json.loads(json.dumps(fit.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(restored.readout.score_value(selection, c), fit.readout.score_value(selection, c))
    assert restored.metadata == fit.metadata


def test_selected_raw_scorer_retains_exact_clinical_empty_anchor(fitted_readout):
    fit, _, _, neutral, _, _, _ = fitted_readout

    def scorer(views, clinical):
        # Synthetic frozen option logits; no image model or patient data.
        logits = neutral.copy()
        logits[0, 1] = sum(views)
        logits[1:, 1] = views
        return fit.readout.score_value(logits, clinical)

    empty = extract_interventions([1.] * 20, np.zeros(20, bool), -.4, scorer,
                                  visual_scale=fit.readout.effective_scale, neutral_view=0.)
    assert empty.base_logit == -.4
    np.testing.assert_array_equal(empty.delta, np.zeros(20))
    np.testing.assert_array_equal(empty.evidence(), np.zeros(26))


def test_raw_optimizer_failure_and_data_boundary(monkeypatch):
    import if_redesign.cfv_jev.readout_training as module

    def failed(*args, **kwargs):
        np.testing.assert_array_equal(args[1], np.random.default_rng(15217).normal(0, .01, 140))
        assert kwargs["options"] == dict(maxiter=1000, maxls=100, ftol=1e-11, gtol=1e-7)
        return SimpleNamespace(success=False, message="synthetic nonconvergence")

    monkeypatch.setattr(module, "minimize", failed)
    logits = np.zeros((4, 21, 2))
    args = (logits, np.zeros((21, 2)), np.zeros(4), [0, 1, 0, 1], ["a"] * 4) * 2
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_raw_readout(*args)
    bad_args = list(args)
    bad_args[1] = np.zeros((3, 21, 2))
    with pytest.raises(ValueError, match="one shared row"):
        fit_raw_readout(*bad_args)
    bad_args = list(args)
    bad_args[3] = [0, 1, 2, 0]
    with pytest.raises(ValueError, match="binary"):
        fit_raw_readout(*bad_args)
    assert not any("outer" in name or "validation" in name for name in inspect.signature(fit_raw_readout).parameters)
