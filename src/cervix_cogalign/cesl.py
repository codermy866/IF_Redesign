"""Shared components for the locked CESL-v2 retrospective study.

The module intentionally separates (1) label-free policy scores used at
held-out inference from (2) observed-label counterfactual audits calculated
only after predictions have been frozen.  It is not a clinical acquisition
system and it does not infer lesions or regions of interest.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from scipy.optimize import minimize_scalar

from cervix_cogalign.prompts import normalize_hpv, normalize_tct


SLOT_IDS = (
    "clinical",
    "colposcopy_00",
    "colposcopy_01",
    "colposcopy_02",
    "colposcopy_03",
    "oct_00",
    "oct_01",
    "oct_02",
    "oct_03",
    "oct_04",
    "oct_05",
)
SLOT_TO_INDEX = {name: index for index, name in enumerate(SLOT_IDS)}
VISUAL_SLOT_INDICES = tuple(range(1, len(SLOT_IDS)))
SLOT_KINDS = ("clinical", "colposcopy", "colposcopy", "colposcopy", "colposcopy", "oct", "oct", "oct", "oct", "oct", "oct")
KIND_TO_CODE = {"clinical": 0, "colposcopy": 1, "oct": 2}
SLOT_KIND_CODES = torch.tensor([KIND_TO_CODE[kind] for kind in SLOT_KINDS], dtype=torch.long)
HPV_VOCAB = ("unknown", "negative", "hpv16_18", "other_high_risk")
TCT_VOCAB = ("unknown", "NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC", "cancer_suspected")


def stable_int(*parts: object, seed: int = 0) -> int:
    payload = "\x1f".join(map(str, (*parts, seed))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def hpv_group(value: Any) -> str:
    normalized = normalize_hpv(value)
    if normalized == "阴性":
        return "negative"
    if normalized.startswith("HPV16/18"):
        return "hpv16_18"
    if normalized.startswith("其他/"):
        return "other_high_risk"
    return "unknown"


def tct_group(value: Any) -> str:
    normalized = normalize_tct(value)
    if normalized in TCT_VOCAB:
        return normalized
    return "unknown"


def age_bin(value: Any) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(age):
        return "missing"
    if age < 35:
        return "lt35"
    if age < 45:
        return "35to44"
    if age < 55:
        return "45to54"
    return "ge55"


def nuisance_stratum(age: Any, hpv: Any, tct: Any) -> tuple[str, str, bool]:
    """Declared non-outcome matching variables for q; never includes labels."""
    return (age_bin(age), hpv_group(hpv), tct_group(tct) != "unknown")


def clinical_matrix(
    ages: Sequence[Any], hpvs: Sequence[Any], tcts: Sequence[Any], fit_indices: Sequence[int]
) -> tuple[np.ndarray, dict[str, float]]:
    """Create a fixed clinical design matrix with source-train age scaling only."""
    parsed_age = np.asarray([float(x) if _finite_number(x) else np.nan for x in ages], dtype=np.float32)
    fit = parsed_age[np.asarray(fit_indices, dtype=int)]
    finite = fit[np.isfinite(fit)]
    mean = float(finite.mean()) if len(finite) else 0.0
    scale = float(finite.std()) if len(finite) > 1 else 1.0
    if scale < 1e-6:
        scale = 1.0
    result = np.zeros((len(ages), 2 + len(HPV_VOCAB) + len(TCT_VOCAB)), dtype=np.float32)
    safe_age = np.where(np.isfinite(parsed_age), parsed_age, mean)
    result[:, 0] = (safe_age - mean) / scale
    result[:, 1] = ~np.isfinite(parsed_age)
    hpv_index = {value: index for index, value in enumerate(HPV_VOCAB)}
    tct_index = {value: index for index, value in enumerate(TCT_VOCAB)}
    for row, (hpv, tct) in enumerate(zip(hpvs, tcts)):
        result[row, 2 + hpv_index[hpv_group(hpv)]] = 1.0
        result[row, 2 + len(HPV_VOCAB) + tct_index[tct_group(tct)]] = 1.0
    return result, {"age_mean_source_train": mean, "age_std_source_train": scale}


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, float("-inf"))
    # Every valid CESL policy preserves at least one observed atom.  The guard
    # makes malformed masks fail closed rather than silently produce NaNs.
    if (~mask).all(dim=1).any():
        raise ValueError("CESL mask contains a case with no observed evidence atom")
    return torch.softmax(masked, dim=1)


class CESLSetModel(nn.Module):
    """Small raw-atom set model used identically across CESL policy controls."""

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
        self.kind_embedding = nn.Embedding(len(KIND_TO_CODE), self.token_dim)
        self.position_embedding = nn.Embedding(len(SLOT_IDS), self.token_dim)
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
        if visual_features.ndim != 3 or visual_features.shape[1] != len(SLOT_IDS):
            raise ValueError(f"Expected Bx{len(SLOT_IDS)}xD visual features, got {tuple(visual_features.shape)}")
        if observed.shape != visual_features.shape[:2]:
            raise ValueError("Observed-mask shape does not match visual features")
        visual_tokens = self.visual_encoder(visual_features.float())
        tokens = visual_tokens.clone()
        tokens[:, 0, :] = self.clinical_encoder(clinical_features.float())
        device = tokens.device
        slots = torch.arange(len(SLOT_IDS), device=device)
        kinds = SLOT_KIND_CODES.to(device)
        tokens = tokens + self.position_embedding(slots)[None, :, :] + self.kind_embedding(kinds)[None, :, :]
        encoded = self.transformer(tokens, src_key_padding_mask=~observed)
        importance_logits = self.importance_head(encoded).squeeze(-1)
        attention = masked_softmax(importance_logits, observed)
        pooled = torch.sum(attention.unsqueeze(-1) * encoded, dim=1)
        observed_fraction = observed.float().mean(dim=1, keepdim=True)
        logit = self.classifier(torch.cat([pooled, observed_fraction], dim=1)).squeeze(-1)
        return {
            "logit": logit,
            "attention": attention,
            "importance_logits": importance_logits,
            "encoded": encoded,
        }


def random_observation_mask(available: torch.Tensor, keep_probability: float) -> torch.Tensor:
    """Stochastic source-train masking for subset-inference robustness."""
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("keep_probability must be in (0, 1]")
    batch = available.shape[0]
    rate = torch.empty((batch, 1), device=available.device).uniform_(max(0.20, keep_probability - 0.35), keep_probability)
    keep = torch.rand_like(available.float()) <= rate
    observed = available & keep
    observed[:, 0] = available[:, 0]
    return observed


def css_loss(attention: torch.Tensor, centres: torch.Tensor, strata: torch.Tensor) -> torch.Tensor:
    """Variance of conditional centre mean rank distributions.

    It aligns model attention distributions after declared, label-free nuisance
    stratification.  It is not a claim that centre effects have been removed.
    """
    penalties: list[torch.Tensor] = []
    for stratum in torch.unique(strata):
        include = strata == stratum
        centre_means: list[torch.Tensor] = []
        for centre in torch.unique(centres[include]):
            part = attention[include & (centres == centre)]
            if len(part):
                centre_means.append(part.mean(dim=0))
        if len(centre_means) >= 2:
            penalties.append(torch.stack(centre_means, dim=0).var(dim=0, unbiased=False).mean())
    if not penalties:
        return attention.new_zeros(())
    return torch.stack(penalties).mean()


def group_dro_binary_loss(logits: torch.Tensor, labels: torch.Tensor, centres: torch.Tensor, positive_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.full_like(labels.float(), float(positive_weight))
    weights = torch.where(labels.float() > 0, weights, torch.ones_like(weights))
    individual = functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none") * weights
    per_centre = [individual[centres == centre].mean() for centre in torch.unique(centres)]
    return individual.mean(), torch.stack(per_centre).max()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one scalar temperature on source-validation logits only."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(logits) != len(labels) or not len(logits):
        raise ValueError("Temperature fitting requires non-empty aligned logits and labels")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = np.clip(logits / temperature, -40.0, 40.0)
        # Stable binary cross entropy from logits.
        return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

    fitted = minimize_scalar(objective, bounds=(math.log(0.05), math.log(10.0)), method="bounded")
    return float(math.exp(float(fitted.x)))


def sigmoid(values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(values, torch.Tensor):
        return torch.sigmoid(values)
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


@dataclass(frozen=True)
class DonorEdge:
    recipient_index: int
    donor_index: int
    slot_index: int
    draw: int
    q_variant: str
    matching_level: str


class SourceOnlyDonorPool:
    """Deterministic donor q with no outcome column in matching logic."""

    def __init__(self, metadata: Sequence[dict[str, Any]], source_train_indices: Sequence[int]) -> None:
        self.metadata = metadata
        self.source_train_indices = tuple(int(index) for index in source_train_indices)

    def _available(self, index: int, slot_index: int) -> bool:
        available = self.metadata[index]["available"]
        return bool(available[slot_index])

    def _candidates(self, recipient_index: int, slot_index: int, q_variant: str) -> tuple[list[int], str]:
        recipient = self.metadata[recipient_index]
        base = [
            index
            for index in self.source_train_indices
            if index != recipient_index and self._available(index, slot_index)
        ]
        if not base:
            raise RuntimeError(f"No source donor is available for slot {slot_index}")
        if q_variant == "pooled_unmatched":
            return base, "pooled_unmatched"
        same_centre = [index for index in base if self.metadata[index]["center"] == recipient["center"]]
        if q_variant == "center_only":
            return (same_centre, "same_center") if len(same_centre) >= 3 else (base, "transport_pool")
        if q_variant != "main_matched":
            raise ValueError(f"Unknown q variant: {q_variant}")
        target = recipient["nuisance_stratum"]
        same_centre_matched = [index for index in same_centre if self.metadata[index]["nuisance_stratum"] == target]
        if len(same_centre_matched) >= 3:
            return same_centre_matched, "same_center_stratum"
        if len(same_centre) >= 3:
            return same_centre, "same_center"
        transport_matched = [index for index in base if self.metadata[index]["nuisance_stratum"] == target]
        if len(transport_matched) >= 3:
            return transport_matched, "transport_stratum"
        return base, "transport_pool"

    def edges(
        self,
        recipient_indices: Iterable[int],
        slot_index: int,
        *,
        draws: int,
        q_variant: str,
        seed: int,
    ) -> list[DonorEdge]:
        output: list[DonorEdge] = []
        for recipient_index in recipient_indices:
            candidates, level = self._candidates(int(recipient_index), slot_index, q_variant)
            ordered = sorted(candidates)
            for draw in range(draws):
                choice = stable_int("cesl_donor", recipient_index, slot_index, q_variant, draw, seed=seed) % len(ordered)
                output.append(
                    DonorEdge(
                        recipient_index=int(recipient_index),
                        donor_index=int(ordered[choice]),
                        slot_index=int(slot_index),
                        draw=int(draw),
                        q_variant=q_variant,
                        matching_level=level,
                    )
                )
        return output
