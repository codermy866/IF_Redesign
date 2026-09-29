"""Stage-ordered prior-anchored fusion.

The new modality is represented as an evidence update to an existing
probabilistic belief rather than as a symmetric input to an unconstrained
joint predictor.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_mean_site_evidence(
    evidence: Tensor,
    valid_mask: Tensor | None = None,
    *,
    dim: int = -1,
) -> Tensor:
    """Aggregate site-level evidence while ignoring unavailable sites."""
    if valid_mask is None:
        return evidence.mean(dim=dim)

    mask = valid_mask.to(device=evidence.device, dtype=evidence.dtype)
    if mask.shape != evidence.shape:
        raise ValueError("valid_mask must have the same shape as evidence")

    count = mask.sum(dim=dim).clamp_min(1.0)
    return (evidence * mask).sum(dim=dim) / count


class PriorAnchoredUpdater(nn.Module):
    """Update a pre-existing probability in log-odds space.

    q1 = sigmoid(logit(q0) + beta * evidence), beta >= 0.
    """

    def __init__(
        self,
        beta_init: float = 1.0,
        *,
        learnable_beta: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if beta_init < 0:
            raise ValueError("beta_init must be non-negative")
        if not 0 < eps < 0.5:
            raise ValueError("eps must lie in (0, 0.5)")

        self.eps = float(eps)
        self.learnable_beta = bool(learnable_beta)

        if self.learnable_beta:
            if beta_init == 0:
                raw = torch.tensor(-20.0)
            else:
                raw = torch.tensor(math.log(math.expm1(beta_init)))
            self.raw_beta = nn.Parameter(raw)
        else:
            self.register_buffer("_fixed_beta", torch.tensor(float(beta_init)))

    @property
    def beta(self) -> Tensor:
        if self.learnable_beta:
            return F.softplus(self.raw_beta)
        return self._fixed_beta

    def forward(self, q0: Tensor, evidence: Tensor) -> Tensor:
        if q0.shape != evidence.shape:
            raise ValueError("q0 and evidence must have the same shape")

        q0 = q0.to(dtype=evidence.dtype).clamp(self.eps, 1.0 - self.eps)
        prior_logit = torch.logit(q0)
        return torch.sigmoid(prior_logit + self.beta * evidence)
