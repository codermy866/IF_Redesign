from __future__ import annotations

import pytest
import torch

from cervix_cogalign.sequence_evidence import cluster_oct_position_frames, minimal_sufficient_set_objective


def _paths() -> list[str]:
    return [f"case_C{c}_S{c}_{frame}.png" for c in range(1, 3) for frame in range(1, 3)]


def test_oct_cluster_builder_groups_c_s_fields_and_orders_frames():
    clusters = cluster_oct_position_frames(_paths()[::-1], expected_positions=2, expected_frames_per_position=2)
    assert [cluster.unit_id for cluster in clusters] == ["oct_C01_S01", "oct_C02_S02"]
    assert [path.rsplit("_", 1)[-1] for path in clusters[0].frame_paths] == ["1.png", "2.png"]


def test_oct_cluster_builder_fails_closed_for_incomplete_position():
    with pytest.raises(ValueError, match="must contain"):
        cluster_oct_position_frames(_paths()[:-1], expected_positions=2, expected_frames_per_position=2)


def test_minimal_sufficient_objective_rewards_sufficient_and_necessary_selected_units():
    labels = torch.tensor([1.0])
    full = torch.tensor([3.0])
    selected = torch.tensor([2.8])
    deletion = torch.tensor([[0.1, 0.4]])
    outcome = minimal_sufficient_set_objective(
        full_logits=full,
        selected_logits=selected,
        deletion_logits=deletion,
        labels=labels,
        selected_units=torch.tensor([[True, True]]),
        sufficiency_tolerance=0.10,
        deletion_margin=0.30,
    )
    assert outcome["sufficiency_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert outcome["minimality_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert outcome["mean_selected_deletion_risk_increase"].item() > 0.4
