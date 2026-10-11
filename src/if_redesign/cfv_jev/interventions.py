"""Recompute the same frozen patient predictor after neutralizing each slot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .evidence import N_VIEWS, encode_evidence, valid_mask


def neutral_image():
    """Create the fixed RGB-128, 224 x 224 image without loading a model."""
    from PIL import Image
    return Image.new("RGB", (224, 224), (128, 128, 128))


@dataclass(frozen=True)
class InterventionResult:
    clinical_logit: float
    base_logit: float
    full_score: float
    neutral_score: float
    delta: np.ndarray
    valid_mask: np.ndarray

    def evidence(self, representation="LAP"):
        return encode_evidence(self.delta[None], self.valid_mask[None], representation)[0]


def extract_interventions(
    views: Sequence, mask, clinical_logit: float, scorer: Callable,
    *, visual_scale: float = 1.0, neutral_view=None,
):
    """Extract signed changes without pathology labels or fitting parameters.

    ``scorer(views, clinical_logit)`` returns an unanchored scalar h(X,c).
    It must be deterministic and frozen, with a fresh language prefix for
    each changed input. Every call keeps all twenty ordered slots and the
    same clinical logit. Invalid slots are neutralized before any score.
    The base logit is c + visual_scale * (h(X,c) - h(neutral,c)); valid slot j
    contributes base(X,c) - base(X with slot j neutral,c).

    The default neutral view is a gray image. Non-image demonstration
    predictors may explicitly supply their equivalent ``neutral_view``.
    """
    if len(views) != N_VIEWS:
        raise ValueError("Exactly twenty ordered view slots are required")
    valid = valid_mask(mask, (N_VIEWS,))
    c, scale = float(clinical_logit), float(visual_scale)
    if not np.isfinite(c) or not np.isfinite(scale) or scale < 0:
        raise ValueError("clinical_logit must be finite; visual_scale must be finite and nonnegative")
    neutral = neutral_image() if neutral_view is None else neutral_view
    null_views = (neutral,) * N_VIEWS
    full_views = tuple(view if keep else neutral for view, keep in zip(views, valid))

    def score(input_views):
        value = np.asarray(scorer(input_views, c), dtype=float)
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError("scorer must return one finite scalar")
        return float(value)

    null_score = score(null_views)
    full_score = score(full_views) if valid.any() else null_score
    base = c + scale * (full_score - null_score)
    delta = np.zeros(N_VIEWS)
    for j in np.flatnonzero(valid):
        replaced = list(full_views)
        replaced[j] = neutral
        delta[j] = scale * (full_score - score(tuple(replaced)))
    if not np.isfinite(base) or not np.isfinite(delta).all():
        raise ValueError("Intervention logits must remain finite")
    delta.setflags(write=False)
    valid.setflags(write=False)
    return InterventionResult(c, float(base), full_score, null_score, delta, valid)
