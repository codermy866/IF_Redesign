"""CPU-only CFV-JEV example using entirely synthetic views and labels."""
from __future__ import annotations

import json
import numpy as np
from scipy.special import expit

from if_redesign.cfv_jev import center_balanced_nll, extract_interventions, fit_residual


def toy_frozen_predictor(views, clinical_logit):
    """An intentionally imperfect nonlinear visual predictor with no labels."""
    values = np.asarray(views)
    return float(.2 * clinical_logit + .2 * values[:12].sum() +
                 .4 * values[12:].sum() + .1 * values[0] * values[12])


def synthetic_partition(rng, n):
    views = rng.normal(0, .35, (n, 20))
    mask = rng.uniform(size=(n, 20)) > .15
    mask[0] = False  # Exercise the exact clinical fallback.
    c = rng.normal(0, .7, n)
    truth = c + .12 * (views[:, :12] * mask[:, :12]).sum(1) + .7 * (views[:, 12:] * mask[:, 12:]).sum(1)
    labels = rng.binomial(1, expit(truth))
    groups = np.where(np.arange(n) % 4 == 0, "small_source", "large_source")
    inputs = [extract_interventions(v, m, prior, toy_frozen_predictor, neutral_view=0.)
              for v, m, prior in zip(views, mask, c)]
    base = np.asarray([result.base_logit for result in inputs])
    evidence = np.asarray([result.evidence() for result in inputs])
    assert base[0] == c[0] and np.count_nonzero(evidence[0]) == 0
    return evidence, base, labels, groups


def main():
    rng = np.random.default_rng(187)
    training = synthetic_partition(rng, 240)
    selection = synthetic_partition(rng, 80)
    heldout = synthetic_partition(rng, 120)
    head = fit_residual(*training, *selection)
    restored = type(head).from_dict(json.loads(json.dumps(head.to_dict(), allow_nan=False)))
    x, base, labels, groups = heldout
    logits = restored.predict_logit(base, x)
    assert logits[0] == base[0]
    print(json.dumps({
        "data": "synthetic demonstration only", "representation": head.representation,
        "parameters": 26, "shrinkage": head.shrinkage,
        "selected_l2": head.metadata["l2"], "optimizer_success": head.metadata["optimizer_success"],
        "heldout_base_center_NLL": center_balanced_nll(base, labels, groups),
        "heldout_corrected_center_NLL": center_balanced_nll(logits, labels, groups),
        "zero_evidence_clinical_anchor": True,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
