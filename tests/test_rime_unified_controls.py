"""Synthetic tests for fairness, conditional scoring and auxiliary semantics."""
import numpy as np
import torch
from if_redesign.rime import RIMEHead, prepare_inputs, coordinate_scale
from if_redesign.rime.scorers import ScoreControl, site_auxiliary
from if_redesign.rime.controls_training import train_control


def inputs():
    rng = np.random.default_rng(43)
    h = rng.normal(scale=.2, size=(8, 2, 4, 8))
    h[:, :, 3] = h[:, :, 0]
    valid = np.ones(h.shape[:-1], dtype=bool)
    scale = coordinate_scale(h, valid, np.arange(4))
    return prepare_inputs(h, valid, rng.normal(size=(8, 34)),
                          rng.normal(size=8)*.2, .8,
                          fitted_scale=scale, deduplicate=True)


def prediction(head, data):
    return head(**{k: v for k, v in data.items() if k != 'indices'})


def test_capacity_matched_local_control_is_context_independent():
    for dim in [26, 34, 42]:
        local = ScoreControl(dim, .3, 'measure', 'laplace', 'local_capacity_matched').double()
        conditional = ScoreControl(dim, .3, 'measure', 'laplace').double()
        assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in conditional.parameters())
        h = torch.randn(3, 2, 4, 8, dtype=torch.double)
        b = torch.randn(3, dim, dtype=torch.double)
        torch.testing.assert_close(local.scores(h, b), local.scores(h, b + 10))


def test_shared_atom_inputs_and_empty_site_lse_fallback():
    data = inputs()
    assert (data['valid'].sum(-1) == 3).all()
    for pool, read in [('attention', 'moment'), ('attention', 'laplace'),
                       ('measure', 'moment'), ('measure', 'laplace'), ('measure', 'lse')]:
        head = RIMEHead(34, .3, pool, read).double()
        with torch.no_grad():
            head.beta.fill_(.1)
        assert torch.isfinite(prediction(head, data)).all()
        empty = dict(data, valid=torch.zeros_like(data['valid']))
        torch.testing.assert_close(prediction(head, empty), data['anchor'])


def test_site_loss_uses_equal_patient_weight_and_reaches_selector():
    data = inputs()
    head = ScoreControl(34, .3, 'measure', 'laplace').double()
    targets = torch.linspace(-.2, .2, 16, dtype=torch.double).reshape(8, 2)
    value = site_auxiliary(head, data, targets, .1)
    q = head.scores(data['h'], data['b'])
    means = (q * data['valid']).sum(-1)/data['valid'].sum(-1)
    torch.testing.assert_close(value, .1*((means-targets)**2).mean(1).mean())
    value.backward()
    assert head.a.grad.abs().max() > 0


def test_patient_auxiliary_and_diversity_training_do_not_change_anchor():
    from if_redesign.rime.training import train_head
    data = inputs()
    anchor = data['anchor'].clone()
    y = np.tile([0, 1], 4)
    groups = np.repeat(['a', 'b'], 4)
    head, metadata = train_control(data, data, y, groups, .3, .003, 17,
                                  max_epochs=2, patience=2, auxiliary=torch.zeros(8, 2),
                                  aux_weight=.1, auxiliary_mode='patient')
    assert metadata['auxiliary_mode'] == 'patient'
    assert torch.isfinite(prediction(head, data)).all()
    _, diversity = train_head(data, data, y, groups, 'attention', 'laplace',
                               .3, .003, 17, max_epochs=2, patience=2,
                               attention_diversity_weight=.1)
    assert diversity['attention_diversity_weight'] == .1
    torch.testing.assert_close(data['anchor'], anchor)
