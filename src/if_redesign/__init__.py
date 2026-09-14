"""Core components for Supervision-Matched Adaptation."""

from .allocation import AllocationResult, derive_allocation
from .labels import build_site_targets, parse_positive_sites
from .objectives import evidence_assimilation_loss, evidence_utility
from .sampling import build_training_schedule

__all__ = [
    "AllocationResult",
    "build_site_targets",
    "build_training_schedule",
    "derive_allocation",
    "evidence_assimilation_loss",
    "evidence_utility",
    "parse_positive_sites",
]
