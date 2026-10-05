# Experimental Status

This file records the aggregate evidence supporting the current IF_Redesign method. No patient-level data are exposed.

## Stage102: patient-specific OCT information

Patient-matched / shuffled analyses showed a positive pooled OCT signal beyond the clinical reference. However fixed correspondence objectives were not uniformly robust across centres.

**Decision:** use as motivation that patient-specific OCT information exists; do not retain fixed correspondence loss as the final method.

## Stage103: discrete Evidence-State routing

Discrete {clinical, OCT, fusion} routing produced centre-dependent state utility and unstable regret / negative-fusion behaviour.

**Decision:** NO-GO as final core.

## Stage104: absolute action-risk / JEV-like formulation

Absolute conditional risks for clinical, OCT and fusion were estimated and subtracted to form relative value.

The apparent negative-fusion reduction was largely due to policy collapse toward clinical-only selection (~97-100% in decisive folds).

**Decision:** NO-GO; motivates direct relative advantage instead of subtracting noisy absolute risks.

## Stage105: Direct Evidence Advantage

```text
A_i = loss_i(clinical) - loss_i(clinical + OCT update)
```

Four centres x two seeds under independent calibration -> threshold selection -> validation:

- benefit AUC > 0.5 in **8/8** cells;
- mean benefit AUC **0.723**;
- range **0.638-0.801**;
- positive realized gain over clinical in **7/8** cells;
- selected-update coverage **0.264-0.969**;
- decision regret lower than unconditional updating in **4/8** cells.

**Decision:** Evidence Advantage and benefit prediction are retained as the core. Universal threshold superiority and finite-sample risk guarantees are not claimed.

## Stage106: objective-repair audit

Risk-magnitude-weighted sign, unweighted sign, ridge advantage and related scalar repairs were compared on reused exploratory source data. Confidence intervals did not establish stable independent benefit.

**Decision:** NO-GO as a new contribution. These are reviewer controls, not the main method.

## Stage107: policy-correspondence null audit

A synthetic null showed that reusing the recipient's original gate after wrong-patient OCT replacement can create a positive apparent correspondence effect even when Y is conditionally independent of OCT.

Recomputing the complete policy under replacement restored the expected null.

**Decision:** any replacement control must recompute the entire frozen prediction path. This is a validity constraint, not an efficacy claim.

## Stage109: Evidence Admission Intervention with colposcopy

Role: exploratory mechanism / Discussion extension only.

Aligned clinical + OCT + colposcopy cohort; Wuda/Wuhan remained locked. A single shared predictor was evaluated under C, C+O, C+K and C+O+K on an independent patient-level test split.

Held-out test: N=275, CIN2+=32.

| State | NLL | Brier | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| C | 0.2840 | 0.0838 | 0.8438 | 0.4467 |
| C+O | 0.2585 | 0.0754 | 0.8746 | 0.5801 |
| C+K | 0.2467 | 0.0718 | 0.8768 | 0.5876 |
| C+O+K | 0.2344 | 0.0672 | 0.8798 | 0.6471 |

Effects:

- A^O = **+0.02543**, 95% CI **[0.01037, 0.04162]**;
- A^K = **+0.03726**, 95% CI **[0.01170, 0.06242]**;
- A^{K|O} = **+0.02415**, 95% CI **[0.00003, 0.04779]**;
- R^{O->K} = **+0.01311**, 95% CI **[0.00722, 0.01900]**;
- mean reduction in colposcopy incremental advantage: **35.2%**;
- OCT-benefit sign AUC on held-out test: **0.891**;
- predicted OCT advantage vs residual colposcopy value: rho **-0.144**, 95% CI **[-0.267, -0.019]**;
- predicted OCT advantage vs redundancy: rho **+0.379**, 95% CI **[0.256, 0.496]**.

Wrong-patient replacement did not establish a robust pooled patient-correspondence effect, and cross-centre transport screening was heterogeneous.

**Decision:** GO as exploratory evidence-admission / residual-value analysis only. Do not claim biological causality or colposcopy replacement.

## Frozen paper-level status

```text
Core scientific question                     FROZEN
Patient-Specific Evidence Advantage           FROZEN / RETAIN
Benefit prediction                            SUPPORTED
Advantage-guided selective update             CORE POLICY FORM
Universal threshold dominance                 NOT SUPPORTED
Biological causal-effect claim                NOT CLAIMED
Evidence Admission Intervention               GO EXPLORATORY
OCT replacement of colposcopy                 NOT SUPPORTED
```

## Public-repository policy

Raw clinical records, patient identifiers, images, model weights, private prediction arrays and manuscript files are excluded. Only reusable code, public protocol definitions, aggregate results and tests are stored here.