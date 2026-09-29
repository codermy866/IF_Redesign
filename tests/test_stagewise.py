import torch

from if_redesign.stagewise import PriorAnchoredUpdater, masked_mean_site_evidence


def test_masked_mean_site_evidence():
    x = torch.tensor([[1.0, 2.0, 100.0]])
    m = torch.tensor([[1.0, 1.0, 0.0]])
    out = masked_mean_site_evidence(x, m, dim=1)
    assert torch.allclose(out, torch.tensor([1.5]))


def test_prior_anchored_update_identity_and_direction():
    updater = PriorAnchoredUpdater(beta_init=1.0, learnable_beta=False)
    q0 = torch.tensor([0.2, 0.5, 0.8])
    zero = updater(q0, torch.zeros_like(q0))
    assert torch.allclose(zero, q0, atol=1e-6)

    evidence = torch.tensor([1.0, 1.0, 1.0])
    up = updater(q0, evidence)
    down = updater(q0, -evidence)
    assert torch.all(up > q0)
    assert torch.all(down < q0)
