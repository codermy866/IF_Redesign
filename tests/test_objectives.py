import torch

from if_redesign.objectives import evidence_assimilation_loss, evidence_utility


def test_evidence_utility_is_positive_when_true_class_probability_improves():
    labels = torch.tensor([1.0, 0.0])
    before = torch.tensor([0.4, 0.6])
    after = torch.tensor([0.8, 0.2])
    assert torch.all(evidence_utility(labels, before, after) > 0)


def test_assimilation_loss_is_finite_and_differentiable():
    before = torch.tensor([0.0], requires_grad=True)
    after = torch.tensor([0.5], requires_grad=True)
    loss, utility = evidence_assimilation_loss(before, after, torch.tensor([1.0]), torch.tensor([0.2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert utility.item() > 0
    assert after.grad is not None
