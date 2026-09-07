"""Case-intrinsic sequence evidence primitives for ICES.

The module deliberately treats a parsed OCT C/S group as a reproducible scanner
position cluster, not as a clinician-confirmed lesion or anatomical coordinate.
Its counterfactual operator is evidence deletion from a selected set, rather
than cross-patient feature replacement.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as functional


_OCT_POSITION_PATTERN = re.compile(r"_C(?P<c>\d+)_S(?P<s>\d+)_(?P<frame>\d+)\.[^.]+$", re.IGNORECASE)


@dataclass(frozen=True)
class OCTPositionCluster:
    """One reproducible within-case OCT scanner-position evidence unit."""

    c_index: int
    s_index: int
    frame_paths: tuple[str, ...]

    @property
    def unit_id(self) -> str:
        return f"oct_C{self.c_index:02d}_S{self.s_index:02d}"


def parse_oct_position_frame(path: str | Path) -> tuple[int, int, int]:
    """Parse machine filename fields C, S, and within-position frame number."""
    match = _OCT_POSITION_PATTERN.search(Path(path).name)
    if not match:
        raise ValueError(f"OCT filename does not expose the required C/S/frame fields: {Path(path).name}")
    return tuple(int(match.group(name)) for name in ("c", "s", "frame"))


def cluster_oct_position_frames(
    paths: Iterable[str | Path],
    *,
    expected_positions: int = 12,
    expected_frames_per_position: int = 10,
) -> tuple[OCTPositionCluster, ...]:
    """Group raw OCT frames by their C/S scanner-position fields.

    The function fails closed when a case does not expose exactly the declared
    position/frame structure. This prevents silent conversion of arbitrary file
    ordering into a claimed sequence evidence representation.
    """
    grouped: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for raw_path in paths:
        path = str(raw_path)
        c_index, s_index, frame_index = parse_oct_position_frame(path)
        grouped[(c_index, s_index)].append((frame_index, path))
    if len(grouped) != expected_positions:
        raise ValueError(f"Expected {expected_positions} OCT C/S positions, found {len(grouped)}")
    clusters: list[OCTPositionCluster] = []
    for (c_index, s_index), members in sorted(grouped.items()):
        frame_indices = [frame for frame, _ in members]
        if len(members) != expected_frames_per_position or len(set(frame_indices)) != expected_frames_per_position:
            raise ValueError(
                f"OCT C/S position {(c_index, s_index)} must contain {expected_frames_per_position} unique frames; "
                f"found {len(members)} frames and {len(set(frame_indices))} unique indices"
            )
        clusters.append(
            OCTPositionCluster(c_index=c_index, s_index=s_index, frame_paths=tuple(path for _, path in sorted(members)))
        )
    return tuple(clusters)


def binary_risk(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-case observed-label binary risk for source training or post-hoc audit."""
    if logits.shape != labels.shape:
        raise ValueError("Logits and labels must be aligned")
    return functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")


def minimal_sufficient_set_objective(
    *,
    full_logits: torch.Tensor,
    selected_logits: torch.Tensor,
    deletion_logits: torch.Tensor,
    labels: torch.Tensor,
    selected_units: torch.Tensor,
    sufficiency_tolerance: float,
    deletion_margin: float,
) -> dict[str, torch.Tensor]:
    """Return set-level sufficiency and deletion-counterfactual minimality losses.

    ``deletion_logits[b, k]`` is the prediction after removing selected evidence
    unit ``k`` from case ``b``. The only intervention is ``do(unobserve e_k)``
    within the same case; no cross-patient donor is used. The returned losses
    make no lesion-causality claim.
    """
    if full_logits.ndim != 1 or selected_logits.shape != full_logits.shape or labels.shape != full_logits.shape:
        raise ValueError("Full, selected, and label tensors must all be aligned one-dimensional case vectors")
    if deletion_logits.ndim != 2 or deletion_logits.shape[0] != len(full_logits):
        raise ValueError("Deletion logits must be BxK")
    if selected_units.shape != deletion_logits.shape or selected_units.dtype != torch.bool:
        raise ValueError("selected_units must be a boolean BxK mask aligned to deletion logits")
    if not 0.0 <= sufficiency_tolerance or not 0.0 <= deletion_margin:
        raise ValueError("Sufficiency tolerance and deletion margin must be non-negative")

    full_risk = binary_risk(full_logits, labels)
    selected_risk = binary_risk(selected_logits, labels)
    sufficiency = torch.relu(selected_risk - full_risk - float(sufficiency_tolerance)).mean()
    deletion_risk = functional.binary_cross_entropy_with_logits(
        deletion_logits, labels.float().unsqueeze(1).expand_as(deletion_logits), reduction="none"
    )
    deletion_delta = deletion_risk - selected_risk.unsqueeze(1)
    failure_to_be_necessary = torch.relu(float(deletion_margin) - deletion_delta)
    denominator = selected_units.float().sum().clamp_min(1.0)
    minimality = (failure_to_be_necessary * selected_units.float()).sum() / denominator
    return {
        "selected_risk": selected_risk.mean(),
        "sufficiency_loss": sufficiency,
        "minimality_loss": minimality,
        "mean_selected_deletion_risk_increase": (deletion_delta * selected_units.float()).sum() / denominator,
    }
