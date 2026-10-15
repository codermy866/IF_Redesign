"""Check the portable P4 path against its explicit mathematical definition."""
import numpy as np
import torch
from if_redesign.pale_matched import make_p4_head
from if_redesign.pale_matched.semantic import semantic_features, semantic_state


def test_p4_matches_explicit_formula_and_missing_fallback():
    torch.manual_seed(29)
    model = make_p4_head().double()
    with torch.no_grad():
        model.g.copy_(torch.linspace(-.2, .2, 56))
        model.beta.copy_(torch.linspace(-.1, .3, 56))
    h = torch.randn(3, 4, 5, 8, dtype=torch.double) * .2
    valid = torch.ones(3, 4, 5, dtype=torch.bool)
    valid[0, 1] = False
    valid[1, 0, :3] = False
    valid[2] = False
    scale = torch.linspace(.3, 1., 56, dtype=torch.double)
    anchor = torch.tensor([-.5, .3, 1.], dtype=torch.double)
    # Independent, scalar-coordinate construction; avoids shared encoder logic.
    coordinates = []
    for d in range(8):
        x = h[..., d]
        coordinates.append(x)
        for sign in (1., -1.):
            for t in (.5, 2., 8.):
                coordinates.append(1 - torch.exp(-t * (sign * x).clamp_min(0)))
    phi = torch.stack(coordinates, -1)
    v = (phi * valid[..., None]).sum(-2) / valid.sum(-1).clamp_min(1)[..., None] / scale
    expected = []
    for i in range(3):
        exists = valid[i].any(-1)
        if exists.any():
            vv = v[i, exists]
            expected.append(anchor[i] + ((vv @ model.g).softmax(0)[:, None] * vv).sum(0) @ model.beta)
        else:
            expected.append(anchor[i])
    actual = model(h, None, valid, torch.empty(3, 0), anchor, scale)
    torch.testing.assert_close(actual, torch.stack(expected))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 112
    assert not hasattr(model, 'local_mlp')


def test_semantic_pca26_uses_fixed_source_state_and_gray_zero():
    rng = np.random.default_rng(43)
    source = rng.normal(size=(40, 35))
    state = semantic_state(source, np.ones((40, 20), dtype=bool))
    target = rng.normal(size=(2, 35))
    mask = np.ones((2, 20), dtype=bool)
    mask[1] = False
    output = semantic_features(target, mask, state)
    assert output.shape == (2, 26)
    np.testing.assert_allclose(output[0], np.clip(target[0] @ state['basis'] / state['std'], -16, 16))
    assert not output[1].any()
    assert not semantic_features(np.zeros((1, 35)), mask[:1], state).any()
