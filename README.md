# IF_Redesign: Patient-Specific Evidence Advantage for Trustworthy Multimodal Updating

This repository contains the current reproducible method package for the Information Fusion study on **patient-specific evidence value, selective multimodal updating, and exploratory downstream evidence redundancy**.

## Scientific question

Conventional multimodal fusion often assumes that an available modality should contribute to every prediction. We instead ask:

> **Should newly available OCT evidence be allowed to modify the existing clinical belief for this patient?**

The core shift is:

```text
available modality -> fuse it
```

to:

```text
clinical belief -> candidate OCT update -> estimate patient-specific advantage -> accept/reject
```

## Core method

### 1. Clinical-reference-anchored candidate update

```text
z0 = g_C(C)
z1 = z0 + Delta(O)
```

OCT proposes an update to the existing clinical belief rather than replacing the clinical predictor.

### 2. Patient-Specific Evidence Advantage

For proper-score loss ell:

```text
A_i = ell(y_i, p0_i) - ell(y_i, p1_i)
```

- A_i > 0: accepting OCT lowers predictive loss.
- A_i ~= 0: OCT is approximately redundant.
- A_i < 0: accepting OCT is harmful fusion for that patient.

The forward model estimates whether the candidate OCT update is beneficial before pathology is observed.

### 3. Advantage-guided selective evidence updating

```text
S_i = 1[score_advantage > tau]
z_final = z0 + S_i * Delta(O)
```

Stage105 supports the learnability of patient-specific OCT benefit. A single transferred threshold is not uniformly dominant, so threshold/risk-control calibration is treated as an operating-policy problem rather than a universal guarantee.

## Exploratory Evidence Admission Intervention

Colposcopy is **not** part of the core IF model. Stage109 is a separate mechanism analysis using an aligned clinical + OCT + colposcopy cohort.

A single shared predictor is evaluated under four evidence states:

```text
C
C + O
C + K
C + O + K
```

and defines:

```text
A^O      = L(C)   - L(C+O)
A^K      = L(C)   - L(C+K)
A^{K|O}  = L(C+O) - L(C+O+K)
R^{O->K} = A^K - A^{K|O}
```

Positive R^{O->K} means that colposcopy has less incremental predictive value after OCT has already been admitted. This is an **information-admission intervention**, not a biological causal-effect claim.

## Current aggregate evidence

### Stage105: Direct Evidence Advantage

- four centres x two seeds;
- benefit AUC > 0.5 in **8/8** fold-seed cells;
- mean benefit AUC **0.723** (range **0.638-0.801**);
- positive realized gain over clinical in **7/8** cells;
- decision regret lower than unconditional updating in **4/8** cells.

Interpretation: patient-specific OCT benefit is learnable, while universal threshold superiority is not yet supported.

### Stage109: Evidence Admission Intervention

Held-out aligned-cohort test: **N=275, CIN2+=32**.

- OCT advantage: **+0.02543**, 95% CI **[0.01037, 0.04162]**;
- colposcopy advantage before OCT: **+0.03726**;
- residual colposcopy advantage after OCT: **+0.02415**;
- OCT-induced colposcopy redundancy: **+0.01311**, 95% CI **[0.00722, 0.01900]**;
- mean reduction in colposcopy incremental advantage: **35.2%**;
- held-out OCT-benefit AUC: **0.891**;
- predicted OCT advantage vs residual colposcopy value: rho **-0.144**, 95% CI **[-0.267, -0.019]**;
- predicted OCT advantage vs redundancy: rho **+0.379**, 95% CI **[0.256, 0.496]**.

The residual colposcopy contribution remains positive. Therefore the repository does **not** claim that OCT replaces colposcopy.

## Repository layout

```text
src/if_redesign/evidence_advantage.py
    Patient-Specific Evidence Advantage and selective updating

src/if_redesign/evidence_admission.py
    Evidence Admission Intervention, residual value, matched replacement controls

scripts/evaluate_evidence_advantage.py
scripts/evaluate_evidence_admission.py
    Data-agnostic held-out evaluation CLIs

configs/stage105_advantage_v1.json
configs/stage109_evidence_admission_v1.json
    Frozen public protocols

results/STAGE105_AGGREGATE.json
results/STAGE109_EAI_AGGREGATE.json
    Aggregate results only; no patient-level predictions

docs/SCIENTIFIC_STORY.md
docs/METHOD.md
docs/EXPERIMENT_MAP.md
docs/EXPERIMENT_STATUS.md
    Scientific narrative, mathematical method and stagewise evidence

tests/test_evidence_advantage.py
tests/test_evidence_admission.py
    Mathematical and implementation sanity tests
```

## Methodological validity controls

Stage106/107 are retained as claim-boundary audits:

- scalar objective repairs were not independently confirmed;
- wrong-patient modality replacement must recompute the complete frozen policy;
- reusing the recipient's original gate can create a spurious correspondence effect;
- no finite-sample or universal new-centre safety guarantee is claimed.

## Clinical positioning

The core IF paper positions OCT as **pre-colposcopy evidence**:

```text
Age / HPV / cytology -> OCT evidence assessment -> downstream colposcopy / biopsy
```

The core question is whether OCT should modify the existing clinical belief. Stage109 only explores whether accepting OCT changes the *residual predictive value* of subsequent colposcopy.

## Public-repository policy

Private clinical records, patient identifiers, raw OCT/colposcopy images, private prediction tables, model weights and manuscript files are intentionally excluded. The repository contains reusable methodology, protocol definitions, aggregate results and tests only.

## Legacy/reference components

SOPI, G2-PCGrad and earlier fusion utilities remain in the repository as precursor analyses and reference baselines. They are not the final primary methodological contribution.