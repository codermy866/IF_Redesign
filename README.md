# IF_Redesign: Stagewise Multimodal Information Innovation

This repository contains the current core implementation for our Information Fusion study on **stage-ordered multimodal information fusion**.

## Current method

The current method is built around two components:

1. **Prior-anchored belief updating**  
   A new modality does not reconstruct the prediction from scratch. It updates a pre-existing belief:

   `logit(q1) = logit(q0) + beta * r`, with `beta >= 0`.

2. **Stagewise Orthogonal Proper-Score Innovation (SOPI)**  
   Modality value is measured by the reduction in outcome-aligned predictive risk:

   `I_l(F1:F0) = E[l(Y,p0) - l(Y,p1)]`.

   Brier and log-score realizations are implemented. At the Bayes predictors, the log-score form reduces to conditional mutual information `I(Y; O | C)`.

The central distinction is:

> **Posterior movement is not task-relevant information.**

A prediction can move substantially after a new modality is observed while becoming less correct. SOPI evaluates whether the update is useful for the realized outcome.

## Repository layout

```text
src/if_redesign/stagewise.py   Prior-anchored fusion operator and site evidence pooling
src/if_redesign/sopi.py        Brier/log SOPI, posterior movement, stagewise accounting
scripts/evaluate_sopi.py       CLI for held-out prediction tables
docs/METHOD.md                 Mathematical method summary
docs/EXPERIMENT_STATUS.md      Aggregate experimental status
tests/test_stagewise.py        Prior-anchor mathematical tests
tests/test_sopi.py             SOPI identities and falsification tests

src/if_redesign/*.py           Earlier G2-PCGrad reference implementation
configs/g2_pcgrad.json         Legacy/reference configuration
```

## Minimal usage

```python
import numpy as np
import torch

from if_redesign import PriorAnchoredUpdater, estimate_sopi

q0 = torch.tensor([0.30, 0.70])
oct_evidence = torch.tensor([0.50, -0.25])

updater = PriorAnchoredUpdater(beta_init=1.0, learnable_beta=False)
q1 = updater(q0, oct_evidence)

y = np.array([1, 0])
V = estimate_sopi(y, q0.numpy(), q1.detach().numpy(), score="log")
J = estimate_sopi(y, q0.numpy(), q1.detach().numpy(), score="brier")
```

## Evaluation protocol

SOPI must be computed from **held-out or cross-fitted predictions**. In the cervical study, both `q0` and `q1` are generated under patient-level leave-one-centre-out evaluation. Pathology labels are required only for retrospective innovation estimation, not for forward inference.

## Current experimental status

The current study has completed:
- four-centre patient-level LOCO evaluation;
- Qwen3-VL and Phi-3.5-Vision cross-family experiments;
- prior-feature versus explicit prior-anchor factorial analysis;
- Brier/log proper-score innovation analysis;
- within-centre patient-permutation falsification;
- synthetic nuisance-rate verification;
- finite-sample bias/coverage analysis;
- negative tests for semantic recognition, hard orthogonality, robustness proxies, committor dynamics, and latent steering.

Private clinical data, patient identifiers, model weights, OOF prediction tables, and manuscript files are intentionally excluded from this public repository.

## Legacy reference

The earlier G2-PCGrad phenotype-fusion implementation is retained because it serves as a specialized OCT reference baseline. It is no longer the primary methodological contribution of the repository.
