#!/usr/bin/env python
"""Reference train/validation/test runner for Stage109-style analysis.

Input is an NPZ containing:
    c_train, o_train, k_train, y_train
    c_val,   o_val,   k_val,   y_val
    c_test,  o_test,  k_test,  y_test

Optional:
    group_test

Feature extraction and clinical encoding are intentionally outside this script
so private raw records and image paths never need to be exposed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from if_redesign import (
    AdvantageSignEstimator,
    EvidenceAdmissionLogistic,
    evidence_admission_effects,
    mean_reduction_fraction,
    per_patient_loss,
    sigmoid,
)


def diagnostic_metrics(y, logit):
    p = sigmoid(logit)
    out = {
        "n": int(len(y)),
        "nll": float(np.mean(per_patient_loss(y, logit, "log"))),
        "brier": float(np.mean((p - y) ** 2)),
    }
    if np.unique(y).size == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--c-grid", default="0.01,0.1,1.0")
    parser.add_argument("--seed", type=int, default=1092026105)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    required = [
        "c_train", "o_train", "k_train", "y_train",
        "c_val", "o_val", "k_val", "y_val",
        "c_test", "o_test", "k_test", "y_test",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"missing NPZ arrays: {missing}")

    c_grid = [float(x) for x in args.c_grid.split(",")]
    best = None
    for c in c_grid:
        model = EvidenceAdmissionLogistic(c=c, random_state=args.seed).fit(
            data["c_train"], data["o_train"], data["k_train"], data["y_train"]
        )
        val = model.all_states(data["c_val"], data["o_val"], data["k_val"])
        val_nll = float(np.mean([
            np.mean(per_patient_loss(data["y_val"], val[s], "log"))
            for s in ("C", "CO", "CK", "COK")
        ]))
        candidate = (val_nll, c, model, val)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    val_nll, best_c, model, val = best
    test = model.all_states(data["c_test"], data["o_test"], data["k_test"])

    advantage_estimator = AdvantageSignEstimator().fit(
        data["y_val"], val["C"], val["CO"], loss="log"
    )
    score = advantage_estimator.predict_score(test["C"], test["CO"])

    effects = evidence_admission_effects(data["y_test"], test, loss="log")
    benefit_target = effects.oct_advantage > 0
    benefit_auc = (
        float(roc_auc_score(benefit_target, score))
        if np.unique(benefit_target).size == 2
        else None
    )

    out = {
        "best_C": best_c,
        "validation_mean_state_nll": val_nll,
        "state_metrics": {
            s: diagnostic_metrics(data["y_test"], test[s])
            for s in ("C", "CO", "CK", "COK")
        },
        "effects": {
            "A_O_mean": float(np.mean(effects.oct_advantage)),
            "A_K_mean": float(np.mean(effects.colposcopy_advantage)),
            "A_K_given_O_mean": float(
                np.mean(effects.residual_colposcopy_advantage)
            ),
            "R_O_to_K_mean": float(
                np.mean(effects.oct_to_colposcopy_redundancy)
            ),
            "colposcopy_advantage_reduction_fraction": mean_reduction_fraction(
                effects
            ),
        },
        "link_to_evidence_advantage": {
            "benefit_auc": benefit_auc,
            "spearman_score_vs_residual_colposcopy": float(
                spearmanr(
                    score, effects.residual_colposcopy_advantage
                ).statistic
            ),
            "spearman_score_vs_redundancy": float(
                spearmanr(
                    score, effects.oct_to_colposcopy_redundancy
                ).statistic
            ),
        },
        "claim_boundary": (
            "Information-admission analysis only; no biological causal effect "
            "and no colposcopy replacement claim."
        ),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
