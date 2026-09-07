"""Shared source-only primitives for ICES-v1 ablations.

ICES represents a case with one clinical token, acquired colposcopy-view
tokens, and twelve parsed OCT C/S position tokens.  The latter are either a
ten-frame mean (M1) or the matched fifth-frame control.  This module supports
offline evidence retention and *within-case* deletion audits only: it neither
localises pathology nor implements a clinical acquisition policy.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


MAX_COLPOSCOPY_VIEWS = 24
OCT_POSITION_CLUSTERS = 12
CLINICAL_INDEX = 0
COLPOSCOPY_INDICES = tuple(range(1, 1 + MAX_COLPOSCOPY_VIEWS))
OCT_INDICES = tuple(range(1 + MAX_COLPOSCOPY_VIEWS, 1 + MAX_COLPOSCOPY_VIEWS + OCT_POSITION_CLUSTERS))
VISUAL_INDICES = COLPOSCOPY_INDICES + OCT_INDICES
N_SLOTS = 1 + MAX_COLPOSCOPY_VIEWS + OCT_POSITION_CLUSTERS
SLOT_IDS = (
    "clinical",
    *(f"colposcopy_view_{index:02d}" for index in range(MAX_COLPOSCOPY_VIEWS)),
    *(f"oct_position_{index:02d}" for index in range(OCT_POSITION_CLUSTERS)),
)
SLOT_KIND_CODES = torch.tensor([0] + [1] * MAX_COLPOSCOPY_VIEWS + [2] * OCT_POSITION_CLUSTERS, dtype=torch.long)


def masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over observed evidence, failing closed for malformed cases."""
    if values.shape != mask.shape:
        raise ValueError("ICES importance values and observed mask must share a shape")
    if (~mask).all(dim=1).any():
        raise ValueError("ICES mask contains a case without an observed evidence token")
    return torch.softmax(values.masked_fill(~mask, float("-inf")), dim=1)


class ICESSetModel(nn.Module):
    """Small set model shared across the predeclared M1--M3 comparisons."""

    def __init__(
        self,
        *,
        visual_dim: int,
        clinical_dim: int,
        token_dim: int = 128,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.clinical_dim = int(clinical_dim)
        self.token_dim = int(token_dim)
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.visual_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(self.clinical_dim, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.kind_embedding = nn.Embedding(3, self.token_dim)
        self.position_embedding = nn.Embedding(N_SLOTS, self.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=attention_heads,
            dim_feedforward=4 * self.token_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.importance_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
            nn.Linear(self.token_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.token_dim + 1, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, 1),
        )

    def forward(
        self, visual_features: torch.Tensor, clinical_features: torch.Tensor, observed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if visual_features.ndim != 3 or visual_features.shape[1] != N_SLOTS:
            raise ValueError(f"Expected Bx{N_SLOTS}xD visual features, got {tuple(visual_features.shape)}")
        if observed.dtype != torch.bool or observed.shape != visual_features.shape[:2]:
            raise ValueError("ICES observed must be a BxN_SLOTS boolean mask")
        tokens = self.visual_encoder(visual_features.float())
        tokens = tokens.clone()
        tokens[:, CLINICAL_INDEX, :] = self.clinical_encoder(clinical_features.float())
        slots = torch.arange(N_SLOTS, device=tokens.device)
        kinds = SLOT_KIND_CODES.to(tokens.device)
        tokens = tokens + self.position_embedding(slots)[None] + self.kind_embedding(kinds)[None]
        encoded = self.transformer(tokens, src_key_padding_mask=~observed)
        importance_logits = self.importance_head(encoded).squeeze(-1)
        attention = masked_softmax(importance_logits, observed)
        pooled = (attention.unsqueeze(-1) * encoded).sum(dim=1)
        observed_fraction = observed.float().mean(dim=1, keepdim=True)
        logit = self.classifier(torch.cat([pooled, observed_fraction], dim=1)).squeeze(-1)
        return {"logit": logit, "attention": attention, "importance_logits": importance_logits}


def random_observation_mask(available: torch.Tensor, keep_probability: float) -> torch.Tensor:
    """Source-training observation masking; clinical evidence is always retained."""
    if available.dtype != torch.bool or not 0.0 < keep_probability <= 1.0:
        raise ValueError("ICES availability must be boolean and keep_probability in (0, 1]")
    batch = available.shape[0]
    rate = torch.empty((batch, 1), device=available.device).uniform_(max(0.20, keep_probability - 0.35), keep_probability)
    retained = available & (torch.rand_like(available.float()) <= rate)
    retained[:, CLINICAL_INDEX] = available[:, CLINICAL_INDEX]
    return retained


def fixed_attention_mask(attention: torch.Tensor, available: torch.Tensor, visual_budget: int) -> torch.Tensor:
    """Keep clinical evidence plus the top available visual tokens per case."""
    if visual_budget < 1:
        raise ValueError("ICES visual budget must be positive")
    if attention.shape != available.shape:
        raise ValueError("ICES attention and availability must share a shape")
    visual_available = available.clone()
    visual_available[:, CLINICAL_INDEX] = False
    scores = attention.masked_fill(~visual_available, float("-inf"))
    count = visual_available.sum(dim=1)
    selected = torch.zeros_like(available)
    selected[:, CLINICAL_INDEX] = available[:, CLINICAL_INDEX]
    if int(count.min().item()) < 1:
        raise ValueError("ICES case lacks visual evidence")
    for row in range(len(selected)):
        k = min(int(visual_budget), int(count[row].item()))
        chosen = torch.topk(scores[row], k=k).indices
        selected[row, chosen] = True
    return selected


def random_fixed_mask(available: torch.Tensor, visual_budget: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Label-free random matched-cost control for post-hoc source validation."""
    if visual_budget < 1:
        raise ValueError("ICES visual budget must be positive")
    selected = torch.zeros_like(available)
    selected[:, CLINICAL_INDEX] = available[:, CLINICAL_INDEX]
    for row in range(len(selected)):
        candidates = torch.nonzero(available[row] & (torch.arange(N_SLOTS, device=available.device) != CLINICAL_INDEX)).flatten()
        if not len(candidates):
            raise ValueError("ICES case lacks visual evidence")
        order = torch.randperm(len(candidates), device=available.device, generator=generator)
        selected[row, candidates[order[: min(int(visual_budget), len(candidates))]]] = True
    return selected


@torch.inference_mode()
def adaptive_retention_mask(
    model: ICESSetModel,
    visual_features: torch.Tensor,
    clinical_features: torch.Tensor,
    available: torch.Tensor,
    *,
    min_visual_units: int,
    max_visual_units: int,
    confidence_margin: float,
) -> torch.Tensor:
    """Offline set-conditioned retention mask.

    Candidate order is estimated from all currently available case evidence,
    then each increasingly large retained set is re-scored.  Consequently it
    is useful for retrospective evidence-sufficiency analysis, but is *not*
    an acquisition-time policy or a claim that later images were never viewed.
    """
    if not 1 <= min_visual_units <= max_visual_units:
        raise ValueError("ICES adaptive visual-unit bounds are invalid")
    full = model(visual_features, clinical_features, available)
    ranked = fixed_attention_mask(full["attention"], available, max_visual_units)
    result = torch.zeros_like(available)
    result[:, CLINICAL_INDEX] = available[:, CLINICAL_INDEX]
    for row in range(len(result)):
        candidates = torch.nonzero(ranked[row] & (torch.arange(N_SLOTS, device=ranked.device) != CLINICAL_INDEX)).flatten()
        for count in range(1, len(candidates) + 1):
            candidate = result[row : row + 1].clone()
            candidate[0, candidates[:count]] = True
            output = model(visual_features[row : row + 1], clinical_features[row : row + 1], candidate)
            probability = torch.sigmoid(output["logit"])[0]
            margin = torch.abs(2.0 * probability - 1.0)
            if count >= min_visual_units and float(margin) >= confidence_margin:
                result[row] = candidate[0]
                break
        else:
            result[row, candidates] = True
    return result


def selected_visual_slot_matrix(mask: torch.Tensor, max_visual_units: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-width selected visual slot indices and a validity mask."""
    if max_visual_units < 1:
        raise ValueError("ICES max_visual_units must be positive")
    output = torch.full((len(mask), max_visual_units), CLINICAL_INDEX, dtype=torch.long, device=mask.device)
    valid = torch.zeros((len(mask), max_visual_units), dtype=torch.bool, device=mask.device)
    for row in range(len(mask)):
        slots = torch.nonzero(mask[row] & (torch.arange(N_SLOTS, device=mask.device) != CLINICAL_INDEX)).flatten()
        k = min(len(slots), max_visual_units)
        output[row, :k] = slots[:k]
        valid[row, :k] = True
    return output, valid


def visual_cost(mask: torch.Tensor) -> np.ndarray:
    """Number of retained visual evidence units per case."""
    return mask[:, 1:].sum(dim=1).detach().cpu().numpy().astype(int)
