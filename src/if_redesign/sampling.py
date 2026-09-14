"""Deterministic task exposure and positive-site sampling."""
import hashlib
import random

import numpy as np


TASK_NAMES = ("diag", "site", "assim")


def _shuffle(rows, key):
    rows = list(rows)
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    random.Random(seed).shuffle(rows)
    return rows


def _task_sequence(steps, proportions):
    values = np.asarray([float(proportions[name]) for name in TASK_NAMES])
    if np.any(values <= 0) or not np.isclose(values.sum(), 1.0, atol=1e-6):
        raise ValueError("Task exposure must be positive and sum to one")
    raw = values * steps
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: steps - int(counts.sum())]:
        counts[index] += 1
    remaining = dict(zip(TASK_NAMES, counts.tolist()))
    assigned = {name: 0 for name in TASK_NAMES}
    sequence = []
    for position in range(steps):
        candidates = [name for name in TASK_NAMES if remaining[name] > 0]
        name = max(
            candidates,
            key=lambda item: ((position + 1) * proportions[item] - assigned[item], -TASK_NAMES.index(item)),
        )
        sequence.append(name)
        assigned[name] += 1
        remaining[name] -= 1
    return sequence


def build_training_schedule(rows, steps, task_exposure, site_positive_fraction, seed=0):
    """Build a schedule while balancing only the training site task."""
    if not 0 < site_positive_fraction < 1:
        raise ValueError("site_positive_fraction must lie strictly between zero and one")
    pools = {name: [row for row in rows if row["task"] == name] for name in TASK_NAMES}
    for name, pool in pools.items():
        if not pool:
            raise ValueError(f"No rows for task {name}")
    shuffled = {name: _shuffle(pool, f"{seed}|{name}") for name, pool in pools.items()}
    positive = _shuffle([row for row in pools["site"] if int(row["label"]) == 1], f"{seed}|positive")
    negative = _shuffle([row for row in pools["site"] if int(row["label"]) == 0], f"{seed}|negative")
    if not positive or not negative:
        raise ValueError("Site rows must contain positive and negative examples")
    sequence = _task_sequence(int(steps), task_exposure)
    site_steps = sequence.count("site")
    positive_count = int(round(site_steps * site_positive_fraction))
    positive_positions = set(np.linspace(0, site_steps - 1, positive_count, dtype=int).tolist())
    counters = {"diag": 0, "assim": 0, "positive": 0, "negative": 0}
    site_index = 0
    schedule = []
    for task in sequence:
        if task in {"diag", "assim"}:
            pool = shuffled[task]
            schedule.append(pool[counters[task] % len(pool)])
            counters[task] += 1
            continue
        kind = "positive" if site_index in positive_positions else "negative"
        pool = positive if kind == "positive" else negative
        schedule.append(pool[counters[kind] % len(pool)])
        counters[kind] += 1
        site_index += 1
    return schedule
