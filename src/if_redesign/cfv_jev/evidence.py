"""Label-free, equal-capacity summaries of fixed-slot neutral replacements."""
from __future__ import annotations

import numpy as np

REPRESENTATIONS = ("LAP", "RAW", "POWER", "FOURIER", "DUP")
LAPLACE_GRID = tuple(np.geomspace(.5, 16, 6).tolist())
N_VIEWS = 20
N_FEATURES = 26


def valid_mask(values, shape):
    mask = np.asarray(values)
    if mask.shape != shape or not np.isin(mask, [False, True]).all():
        raise ValueError(f"valid_mask must have shape {shape} and contain only booleans or 0/1")
    return mask.astype(bool)


def encode_evidence(delta, mask, representation="LAP"):
    """Encode N x 20 signed logit changes into N x 26 coordinates.

    The first twelve slots are OCT; the final eight are colposcopy. Each
    modality contributes its unbounded signed mean and twelve coordinates.
    Magnitudes in LAP/POWER/DUP are clipped to [0, 1], Fourier inputs to
    [-1, 1]. RAW retains signed quantiles. Actual valid counts determine the
    averages. Empty modalities are exactly zero. No labels are accepted.
    """
    values = np.asarray(delta, dtype=float)
    if values.ndim != 2 or values.shape[1] != N_VIEWS or not np.isfinite(values).all():
        raise ValueError("delta must be finite with shape (N, 20)")
    valid = valid_mask(mask, values.shape)
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    blocks = []
    for sl in (slice(0, 12), slice(12, 20)):
        v, m = values[:, sl], valid[:, sl]
        count = np.maximum(m.sum(axis=1), 1)
        mean = (v * m).sum(axis=1) / count
        pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
        if representation == "LAP":
            extra = np.column_stack([
                (-np.expm1(-t * pos) * m).sum(axis=1) / count for t in LAPLACE_GRID
            ] + [(-np.expm1(-t * neg) * m).sum(axis=1) / count for t in LAPLACE_GRID])
        elif representation == "POWER":
            extra = np.column_stack([(pos**k * m).sum(axis=1) / count for k in range(1, 7)] +
                                    [(neg**k * m).sum(axis=1) / count for k in range(1, 7)])
        elif representation == "FOURIER":
            clipped = np.clip(v, -1, 1)
            extra = np.column_stack([
                ((np.cos(np.pi * k * clipped) - 1) * m).sum(axis=1) / count for k in range(1, 7)
            ] + [(np.sin(np.pi * k * clipped) * m).sum(axis=1) / count for k in range(1, 7)])
        elif representation == "RAW":
            extra = np.zeros((len(v), 12))
            for i in range(len(v)):
                if m[i].any():
                    extra[i] = np.quantile(v[i, m[i]], np.linspace(0, 1, 12))
        else:
            absolute = ((pos + neg) * m).sum(axis=1) / count
            extra = np.column_stack([mean] * 6 + [absolute] * 6)
        blocks.append(np.column_stack([mean, extra]))
    encoded = np.concatenate(blocks, axis=1)
    if not np.isfinite(encoded).all():
        raise ValueError("Evidence encoding produced nonfinite coordinates")
    return encoded
