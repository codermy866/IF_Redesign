"""Symmetric two-task PCGrad used by the locked G2 method."""
from collections.abc import Sequence

import torch


def gradient_dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]):
    if len(left) != len(right) or not left:
        raise ValueError("Gradient lists must be non-empty and have equal length")
    if any(a.shape != b.shape for a, b in zip(left, right)):
        raise ValueError("Corresponding gradients must have equal shapes")
    return sum(torch.sum(a.detach() * b.detach()) for a, b in zip(left, right))


def gradient_cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor], eps=1e-12):
    dot = gradient_dot(left, right)
    left_norm = torch.sqrt(sum(torch.sum(value.detach() ** 2) for value in left))
    right_norm = torch.sqrt(sum(torch.sum(value.detach() ** 2) for value in right))
    return dot / (left_norm * right_norm).clamp_min(float(eps))


def symmetric_pcgrad(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor], eps=1e-12):
    """Project both task gradients when their global dot product is negative."""
    dot = gradient_dot(left, right)
    if float(dot) >= 0.0:
        return [a + b for a, b in zip(left, right)]
    left_norm_sq = sum(torch.sum(value.detach() ** 2) for value in left).clamp_min(float(eps))
    right_norm_sq = sum(torch.sum(value.detach() ** 2) for value in right).clamp_min(float(eps))
    return [(a - dot / right_norm_sq * b) + (b - dot / left_norm_sq * a)
            for a, b in zip(left, right)]
