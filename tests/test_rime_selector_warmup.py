"""Two-phase head training is a controlled optimization ablation, not a measured clinical gain."""
import numpy as np
import torch

from if_redesign.rime.readout import RIMEHead
from if_redesign.rime.training import train_head


def _fake_inputs():
    rng = np.random.default_rng(203)
    n = 12
    h = torch.as_tensor(rng.normal(size=(n, 2, 3, 8)) * .15, dtype=torch.double)
    delta = h[..., :, None, :] - h[..., None, :, :]
    kernel = torch.exp(-delta.square().sum(-1) / .3)
    clinical = torch.as_tensor(rng.normal(size=n) * .2, dtype=torch.double)
    return dict(h=h, k=kernel, valid=torch.ones((n,2,3),dtype=torch.bool),
                b=torch.as_tensor(rng.normal(size=(n,34)),dtype=torch.double),
                anchor=clinical, coordinate_scale=torch.ones(56,dtype=torch.double),
                indices=np.arange(n))


def _prediction(head, row):
    return head(**{key: val for key, val in row.items() if key != 'indices'})


def test_initial_output_gate_blocks_selector_gradient():
    inputs = _fake_inputs()
    head = RIMEHead(34, pooling='measure', readout='laplace').double()
    z = _prediction(head, inputs)
    torch.testing.assert_close(z, inputs['anchor'])
    z.sum().backward()
    for p in (head.a, head.U, head.V, head.g):
        assert p.grad is None or torch.all(p.grad == 0)


def test_staged_training_preserves_capacity_and_finite_inference():
    inputs = _fake_inputs()
    labels = np.tile([0, 1], 6)
    center = np.repeat(['a', 'b', 'c'],4)
    legacy, old = train_head(inputs,inputs,labels,center,'measure','laplace',
                             tau=.3,lam=.003,seed=17,max_epochs=4,patience=4)
    baseline, new0 = train_head(inputs,inputs,labels,center,'measure','laplace',
                                tau=.3,lam=.003,seed=17,max_epochs=4,patience=4,
                                selector_warmup_epochs=0)
    for a, b in zip(legacy.parameters(),baseline.parameters()):
        torch.testing.assert_close(a,b)
    assert old['parameters'] == new0['parameters']
    stage, new = train_head(inputs,inputs,labels,center,'measure','laplace',
                            tau=.3,lam=.003,seed=17,max_epochs=5,patience=1,
                            selector_warmup_epochs=2)
    assert new['selector_warmup_epochs']==2
    assert new['epochs_run']>=3, 'selector phase must execute'
    assert new['parameters']==old['parameters']
    assert all(p.requires_grad for p in (stage.a,stage.U,stage.V,stage.g))
    assert torch.isfinite(_prediction(stage,inputs)).all()


def test_stage_restores_only_initially_trainable_params():
    inputs = _fake_inputs()
    labels = np.tile([0,1],6)
    groups = np.repeat(['a','b','c'],4)
    stage, state = train_head(inputs,inputs,labels,groups,'mean','moment',
                              tau=.3,lam=.003,seed=17,max_epochs=3,patience=2,
                              selector_warmup_epochs=1)
    assert not any(p.requires_grad for p in (stage.a,stage.U,stage.V))
    assert stage.g.requires_grad
    assert torch.isfinite(_prediction(stage,inputs)).all()
