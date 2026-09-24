import torch

from if_redesign.pcgrad import gradient_cosine, symmetric_pcgrad


def test_nonconflicting_gradients_are_summed_unchanged():
    left = [torch.tensor([1.0, 0.0])]
    right = [torch.tensor([1.0, 1.0])]
    assert torch.equal(symmetric_pcgrad(left, right)[0], torch.tensor([2.0, 1.0]))


def test_symmetric_projection_removes_pairwise_conflict():
    left = [torch.tensor([1.0, 0.0])]
    right = [torch.tensor([-1.0, 1.0])]
    assert torch.allclose(symmetric_pcgrad(left, right)[0], torch.tensor([0.5, 1.5]))
    assert gradient_cosine(left, right) < 0
