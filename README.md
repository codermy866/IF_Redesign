# IF_Redesign: Patient-Specific Evidence Advantage for Selective Multimodal Updating

This repository contains the current core implementation for our Information Fusion study on **patient-specific evidence value and selective multimodal decision making**.

## Scientific question

Conventional multimodal fusion often assumes that an available modality should contribute to every prediction. Our current formulation asks a different question:

> **Should newly available evidence be allowed to modify the existing prediction for this patient?**

For the cervical study, the existing belief is built from clinical evidence (age / HPV / cytology), and OCT supplies a candidate update before downstream colposcopy/biopsy.

## Current method

### 1. Prior-anchored candidate update

Let `z0` denote the clinical logit and `delta(O)` the OCT-derived update:

```
z1 = z0 + delta(O)
p0 = sigmoid(z0)
p1 = sigmoid(z1)
```

The OCT branch proposes a candidate update rather than reconstructing the prediction from scratch.

### 2. Patient-Specific Evidence Advantage

For a proper scoring loss `ell`, define the retrospective patient-level consequence of accepting the update as

```
A_i = ell(y_i, p0_i) - ell(y_i, p1_i)
```

- `A_i > 0`: accepting OCT lowers predictive loss.
- `A_i ~= 0`: OCT is approximately redundant.
- `A_i < 0`: accepting OCT causes negative fusion for that patient.

The core methodological shift is therefore from **modality weighting** to **decision-consequence estimation**.

### 3. Advantage-guided selective updating

A forward estimator predicts whether the candidate update is beneficial. The final prediction is

```
z_final = z1   if score_advantage > threshold
          z0   otherwise
```

Labels are used only to create retrospective advantage supervision on development data. Final inference uses no pathology label.

## Leakage-safe evaluation protocol

The current Stage105 protocol separates development roles:

```
calibration split -> fit advantage / benefit estimator
selection split   -> choose acceptance threshold
validation split  -> final held-out policy evaluation
```

The OCT/clinical prediction backbone is frozen while the advantage estimator is evaluated. The outer held-out centre and Wuhan remain outside the Stage105 screening analysis.

## Current Stage105 result

Across four centres and two seeds (8 fold-seed cells), the independent-threshold experiment currently shows:

- benefit discrimination AUC above chance in **8/8** cells;
- mean benefit AUC **0.723** (range **0.638-0.801**);
- positive realized gain over the clinical prediction in **7/8** cells;
- selected-update coverage ranging from **0.264 to 0.969**;
- selected-policy decision regret lower than always-updating in **4/8** cells.

Interpretation:

> **Patient-specific benefit of accepting an OCT update is predictably non-random, but a single transferred threshold policy is not yet uniformly stable across centres/seeds.**

Accordingly, **Evidence Advantage is retained as the core mathematical object**, while threshold/risk-control calibration remains an active experimental component rather than a claimed guarantee.

## Repository layout

```text
src/if_redesign/evidence_advantage.py   Evidence Advantage, benefit estimator, selective update
scripts/evaluate_evidence_advantage.py  Held-out evaluation CLI
configs/stage105_advantage_v1.json      Public Stage105 protocol
results/STAGE105_AGGREGATE.json         Aggregate, non-patient-level Stage105 results
docs/METHOD.md                          Current mathematical method
docs/EXPERIMENT_STATUS.md               Stagewise experimental evidence
tests/test_evidence_advantage.py        Mathematical and policy sanity tests

src/if_redesign/sopi.py                 Earlier proper-score innovation analysis
src/if_redesign/stagewise.py            Earlier prior-anchored updater
src/if_redesign/*.py                    G2-PCGrad and legacy/reference components
```

## Minimal usage

```python
import numpy as np

from if_redesign import (
    AdvantageSignEstimator,
    evidence_advantage,
    evaluate_selective_update,
)

y_cal = np.array([1, 0, 1, 0])
z0_cal = np.array([0.1, -0.2, -0.3, 0.4])
z1_cal = np.array([0.8, -0.6, 0.2, -0.1])

estimator = AdvantageSignEstimator().fit(y_cal, z0_cal, z1_cal)
score = estimator.predict_score(z0_cal, z1_cal)

advantage = evidence_advantage(y_cal, z0_cal, z1_cal)
evaluation = evaluate_selective_update(
    y_cal, z0_cal, z1_cal, score, threshold=0.0
)
```

## Clinical interpretation boundary

The current paper studies whether **OCT should modify an existing clinical belief before colposcopy**. It does **not** claim that OCT replaces colposcopy. The incremental value of subsequent colposcopy is reserved for exploratory/future analysis.

## Public-repository policy

Private clinical data, patient identifiers, raw OCT images, model weights, private OOF prediction tables, and manuscript files are intentionally excluded. The public repository contains reusable methodology, protocol definitions, aggregate results, and tests only.

## Legacy reference

SOPI and the earlier G2-PCGrad phenotype-fusion implementation are retained as important precursor analyses and reference baselines. They are no longer the final primary methodological contribution.
