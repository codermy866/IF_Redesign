#!/usr/bin/env python
"""Evaluate Evidence Admission Intervention from held-out state logits.

Required CSV columns:
    y, z_c, z_co, z_ck, z_cok

Optional columns:
    group, oct_advantage_score
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from if_redesign.evidence_admission import (
    evidence_admission_effects,
    mean_reduction_fraction,
)


def mean_ci(values, groups, n_boot, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.zeros(len(values), dtype=int)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    boot = []
    for _ in range(n_boot):
        idx = []
        for g in unique:
            ids = np.flatnonzero(groups == g)
            idx.extend(rng.choice(ids, size=len(ids), replace=True))
        boot.append(float(np.mean(values[np.asarray(idx)])))
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1092026105)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"y", "z_c", "z_co", "z_ck", "z_cok"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    logits = {
        "C": df["z_c"].to_numpy(),
        "CO": df["z_co"].to_numpy(),
        "CK": df["z_ck"].to_numpy(),
        "COK": df["z_cok"].to_numpy(),
    }
    effects = evidence_admission_effects(df["y"].to_numpy(), logits)
    group = df["group"].to_numpy() if "group" in df.columns else None

    out = {
        "n": int(len(df)),
        "oct_advantage_mean": float(np.mean(effects.oct_advantage)),
        "oct_advantage_ci95": mean_ci(
            effects.oct_advantage, group, args.bootstrap, args.seed + 1
        ),
        "colposcopy_advantage_mean": float(np.mean(effects.colposcopy_advantage)),
        "residual_colposcopy_advantage_mean": float(
            np.mean(effects.residual_colposcopy_advantage)
        ),
        "redundancy_mean": float(np.mean(effects.oct_to_colposcopy_redundancy)),
        "redundancy_ci95": mean_ci(
            effects.oct_to_colposcopy_redundancy,
            group,
            args.bootstrap,
            args.seed + 2,
        ),
        "colposcopy_advantage_reduction_fraction": mean_reduction_fraction(effects),
    }

    if "oct_advantage_score" in df.columns:
        score = df["oct_advantage_score"].to_numpy()
        out["spearman_score_vs_residual_colposcopy"] = float(
            spearmanr(score, effects.residual_colposcopy_advantage).statistic
        )
        out["spearman_score_vs_redundancy"] = float(
            spearmanr(score, effects.oct_to_colposcopy_redundancy).statistic
        )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
