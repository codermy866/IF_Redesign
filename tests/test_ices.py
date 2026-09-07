from __future__ import annotations

import torch

from cervix_cogalign.ices import (
    ICESSetModel,
    N_SLOTS,
    adaptive_retention_mask,
    fixed_attention_mask,
    random_fixed_mask,
)


def _model() -> ICESSetModel:
    return ICESSetModel(visual_dim=8, clinical_dim=4, token_dim=16, transformer_layers=1, attention_heads=4, dropout=0.0).eval()


def test_ices_fixed_and_random_controls_match_visual_cost_and_keep_clinical():
    model = _model()
    x, clinical = torch.randn(3, N_SLOTS, 8), torch.randn(3, 4)
    available = torch.ones(3, N_SLOTS, dtype=torch.bool)
    attention = model(x, clinical, available)["attention"]
    fixed = fixed_attention_mask(attention, available, 3)
    random = random_fixed_mask(available, 3, torch.Generator().manual_seed(7))
    assert fixed[:, 0].all() and random[:, 0].all()
    assert fixed[:, 1:].sum(dim=1).tolist() == [3, 3, 3]
    assert random[:, 1:].sum(dim=1).tolist() == [3, 3, 3]


def test_ices_adaptive_retention_is_bounded_and_keeps_clinical():
    model = _model()
    x, clinical = torch.randn(2, N_SLOTS, 8), torch.randn(2, 4)
    available = torch.ones(2, N_SLOTS, dtype=torch.bool)
    selected = adaptive_retention_mask(model, x, clinical, available, min_visual_units=1, max_visual_units=4, confidence_margin=0.2)
    assert selected[:, 0].all()
    assert ((selected[:, 1:].sum(dim=1) >= 1) & (selected[:, 1:].sum(dim=1) <= 4)).all()
