"""Source-derived capacity and exposure allocation."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


TASKS = ("D", "S", "A")


@dataclass(frozen=True)
class AllocationResult:
    ranks: dict
    task_exposure: dict
    alpha: float
    objective: float
    normalized_gradient_variance: dict

    def to_dict(self):
        return asdict(self)


def residual_spectral_bias(spectral_energy, task, rank):
    rows = spectral_energy[spectral_energy["task"] == task].groupby("rank")["cumulative_spectral_energy"].mean().sort_index()
    if len(rows) < 2:
        raise ValueError(f"At least two spectral ranks are required for task {task}")
    ranks = rows.index.to_numpy(float)
    residual = 1.0 - rows.to_numpy(float)
    if rank < ranks.min():
        slope = (residual[1] - residual[0]) / (ranks[1] - ranks[0])
        return float(min(1.0, residual[0] + slope * (rank - ranks[0])))
    return float(np.interp(rank, ranks, residual))


def derive_allocation(spectral_energy, gradient_variance, total_rank=16, total_steps=120, reference_ranks=None):
    """Minimize the supervision-matched source objective over integer ranks."""
    reference_ranks = reference_ranks or {"D": 8, "S": 4, "A": 4}
    variance = gradient_variance.groupby("task")["normalized_gradient_variance"].mean().to_dict()
    missing = set(TASKS).difference(variance)
    if missing:
        raise ValueError(f"Missing gradient variance for tasks: {sorted(missing)}")
    alpha = float(np.median([
        residual_spectral_bias(spectral_energy, task, reference_ranks[task])
        * total_steps / (3.0 * variance[task] * reference_ranks[task])
        for task in TASKS
    ]))
    best = None
    for r_d in range(1, total_rank - 1):
        for r_s in range(1, total_rank - r_d):
            ranks = {"D": r_d, "S": r_s, "A": total_rank - r_d - r_s}
            weights = {task: np.sqrt(alpha * variance[task] * ranks[task] / (3.0 * total_steps)) for task in TASKS}
            scale = sum(weights.values())
            exposure = {task: float(weights[task] / scale) for task in TASKS}
            objective = sum((
                residual_spectral_bias(spectral_energy, task, ranks[task])
                + alpha * variance[task] * ranks[task] / (total_steps * exposure[task])
            ) / 3.0 for task in TASKS)
            candidate = (float(objective), ranks, exposure)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return AllocationResult(
        ranks=best[1], task_exposure=best[2], alpha=alpha, objective=best[0],
        normalized_gradient_variance={task: float(variance[task]) for task in TASKS},
    )
