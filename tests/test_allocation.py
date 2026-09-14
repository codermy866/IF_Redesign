import pandas as pd

from if_redesign.allocation import derive_allocation


def test_allocation_respects_rank_and_exposure_budgets():
    spectral = pd.DataFrame([
        {"task": task, "rank": rank, "cumulative_spectral_energy": energy}
        for task in ("D", "S", "A")
        for rank, energy in ((2, 0.80), (4, 0.90), (8, 0.97), (16, 0.995))
    ])
    variance = pd.DataFrame([
        {"task": task, "normalized_gradient_variance": value}
        for task, value in (("D", 0.8), ("S", 0.9), ("A", 0.85))
    ])
    result = derive_allocation(spectral, variance)
    assert sum(result.ranks.values()) == 16
    assert min(result.ranks.values()) >= 1
    assert abs(sum(result.task_exposure.values()) - 1.0) < 1e-9
    assert min(result.task_exposure.values()) > 0
