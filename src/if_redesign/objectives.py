"""Clinical supervision losses and evidence utility."""
from torch.nn import functional as F


def evidence_utility(label, probability_before, probability_after, epsilon=1e-6):
    """Return A = NLL_before - NLL_after for each patient."""
    label = label.float()
    before = probability_before.clamp(epsilon, 1 - epsilon)
    after = probability_after.clamp(epsilon, 1 - epsilon)
    nll_before = -(label * before.log() + (1 - label) * (1 - before).log())
    nll_after = -(label * after.log() + (1 - label) * (1 - after).log())
    return nll_before - nll_after


def evidence_assimilation_loss(logits_before, logits_after, label, potential_gain, direction_margin=0.0, gap_delta=0.05, gap_weight=0.25):
    """Train risk predictions before and after adding an OCT evidence item."""
    label = label.float()
    before = F.binary_cross_entropy_with_logits(logits_before, label, reduction="none")
    after = F.binary_cross_entropy_with_logits(logits_after, label, reduction="none")
    assimilation = before - after
    direction = F.relu(float(direction_margin) - assimilation).mean()
    gap = F.relu(potential_gain.float() - assimilation - float(gap_delta)).mean()
    loss = (before.mean() + after.mean()) / 2.0 + direction + float(gap_weight) * gap
    return loss, assimilation
