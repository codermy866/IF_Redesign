"""Synthetic invariants for matched losses, common repair and portable fitting."""
import numpy as np
import pytest
import torch

from if_redesign.pale_ardr import anchor_kl, dro_update, fit, penalty, repair_project
from if_redesign.rime.training import train_head


def test_ardr_anchor_gradient_and_active_centre():
    a = torch.tensor([1., -1., 2., -2.], dtype=torch.double, requires_grad=True)
    z = torch.tensor([-1., 1., 2., -2.], dtype=torch.double, requires_grad=True)
    y = torch.tensor([1., 0., 1., 0.], dtype=torch.double)
    groups = torch.tensor([0, 0, 1, 1])
    loss = penalty(z, a, y, groups, 'A5_ARDR', dict(gamma=3., epsilon=0.))
    loss.backward()
    assert a.grad is None
    assert z.grad[:2].abs().min() > 0
    torch.testing.assert_close(z.grad[2:], torch.zeros(2, dtype=torch.double))


@pytest.mark.parametrize('method', ['A2_ANCHOR_KL', 'A4_GLOBAL_RISK', 'A5_ARDR'])
def test_no_penalty_at_anchor(method):
    a = torch.tensor([1., -1., 2., -2.], dtype=torch.double)
    y = torch.tensor([1., 0., 1., 0.], dtype=torch.double)
    groups = torch.tensor([0, 0, 1, 1])
    assert penalty(a, a, y, groups, method, dict(gamma=3., epsilon=.01)) == 0


def test_relative_risk_weights_centres_equally_not_patients():
    # Unequal centre sizes: only the singleton centre deteriorates by one NLL.
    y = torch.ones(4, dtype=torch.double)
    a = torch.zeros(4, dtype=torch.double)
    singleton_z = -torch.log(torch.expm1(torch.log(torch.tensor(2.)) + 1.)).double()
    z = torch.cat([singleton_z.reshape(1), torch.zeros(3, dtype=torch.double)])
    groups = torch.tensor([0, 1, 1, 1])
    cv = dict(gamma=1., epsilon=0.)
    torch.testing.assert_close(penalty(z, a, y, groups, 'A5_ARDR', cv),
                               torch.tensor(.5, dtype=torch.double), atol=1e-6, rtol=0)
    torch.testing.assert_close(penalty(z, a, y, groups, 'A4_GLOBAL_RISK', cv),
                               torch.tensor(1/16, dtype=torch.double), atol=1e-6, rtol=0)


def test_kl_direction_probability_reference_and_extremes():
    a = torch.tensor([.2, -.5], dtype=torch.double, requires_grad=True)
    z = torch.tensor([.8, -.9], dtype=torch.double, requires_grad=True)
    p, q = a.sigmoid(), z.sigmoid()
    expected = (p*torch.log(p/q) + (1-p)*torch.log((1-p)/(1-q))).mean()
    torch.testing.assert_close(anchor_kl(a, z), expected, atol=1e-14, rtol=0)
    anchor_kl(a, z).backward()
    assert a.grad is None and torch.isfinite(z.grad).all()
    extreme = anchor_kl(torch.tensor([1000., -1000.]), torch.tensor([-1000., 1000.]))
    assert torch.isfinite(extreme) and extreme > 0


def test_group_dro_matches_exponentiated_reference_and_detaches_risk():
    weights = torch.tensor([.25, .75], dtype=torch.double)
    risks = torch.tensor([.2, 1.], dtype=torch.double, requires_grad=True)
    expected = weights * torch.exp(.1 * risks.detach())
    expected /= expected.sum()
    result = dro_update(weights, risks, .1)
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad
    extreme = dro_update(weights, torch.tensor([10000., -10000.]), 1.)
    assert torch.isfinite(extreme).all() and extreme.sum() == 1


def test_repair_gray_missingness_bounds_and_no_target_fitting():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(8, 2, 4, 16))
    valid = np.ones(raw.shape[:-1], dtype=bool)
    state = dict(basis=np.eye(16)[:, :8], scale=np.ones(8), fit_centering=np.arange(16)*.1)
    before = {k: v.copy() for k, v in state.items()}
    neutral, _ = repair_project(np.zeros_like(raw), valid, state)
    assert np.array_equal(neutral, np.zeros_like(neutral))
    h, _ = repair_project(raw, valid, state)
    assert abs(h).max() <= 1
    changed = raw.copy(); changed[-1] += 1e6
    transformed, _ = repair_project(changed, valid, state)
    np.testing.assert_array_equal(h[:-1], transformed[:-1])
    valid[0, 0, 0] = False
    masked, _ = repair_project(raw, valid, state)
    assert np.array_equal(masked[0, 0, 0], np.zeros(8))
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def synthetic_inputs():
    rng = np.random.default_rng(17)
    return dict(h=torch.tensor(rng.normal(size=(8, 2, 4, 8))*.1, dtype=torch.double),
                k=torch.zeros(8, 1, 1, 1, dtype=torch.double),
                valid=torch.ones(8, 2, 4, dtype=torch.bool),
                b=torch.tensor(rng.normal(size=(8, 34)), dtype=torch.double),
                anchor=torch.zeros(8, dtype=torch.double),
                coordinate_scale=torch.ones(56, dtype=torch.double))


def test_baseline_fixed_epoch_exact_legacy_loss_parity():
    inp = synthetic_inputs(); y = np.tile([0, 1], 4); groups = np.repeat(['a', 'b'], 4)
    legacy, _ = train_head(inp | dict(indices=np.arange(8)), None, y, groups,
                           'mlp', 'laplace', .3, .003, 17, fixed_epochs=2)
    new, _ = fit(inp, None, y, groups, None, .003, 17, 'A0_PALE', {}, fixed_epochs=2)
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], atol=1e-12, rtol=1e-12)


def test_six_losses_keep_forward_capacity_and_no_visual_anchor_fallback():
    inp = synthetic_inputs(); y = np.tile([0, 1], 4); groups = np.repeat(['a', 'b'], 4)
    methods = [('A0_PALE', {}), ('A1_STRONG_L2', dict(lambda_multiplier=2.)),
               ('A2_ANCHOR_KL', dict(gamma=1.)), ('A3_GROUP_DRO', dict(eta=.01)),
               ('A4_GLOBAL_RISK', dict(gamma=1., epsilon=0.)),
               ('A5_ARDR', dict(gamma=1., epsilon=0.))]
    signatures = []
    for name, candidate in methods:
        model, metadata = fit(inp, inp, y, groups, y, .003, 17, name, candidate,
                              max_epochs=2, patience=2)
        assert torch.isfinite(model(**inp)).all()
        empty = inp | dict(valid=torch.zeros_like(inp['valid']))
        torch.testing.assert_close(model(**empty), inp['anchor'], atol=0, rtol=0)
        signatures.append((metadata['parameters'],
                           tuple((k, tuple(v.shape)) for k, v in model.state_dict().items())))
    assert all(signature == signatures[0] for signature in signatures)
