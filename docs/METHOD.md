# Current Method: Patient-Specific Evidence Advantage Learning

## Scientific target

Let `C` denote evidence already available for a patient, `O` a newly available modality, and `Y` the diagnostic target. The central problem is not simply whether `(C,O)` improves average prediction, but whether accepting `O` is beneficial for the **current patient**.

The method therefore separates two questions:

1. What candidate change would OCT make to the current clinical belief?
2. Should that candidate change be accepted?

This converts multimodal fusion from unconditional feature combination into a patient-specific evidence decision.

## 1. Clinical belief and OCT-induced candidate update

The clinical pathway produces a pre-OCT logit

```
z_i^0 = e_i = g_C(C_i)
p_i^0 = sigmoid(z_i^0)
```

A patient-level OCT representation `h_i` is obtained from the OCT sequence. The OCT branch predicts an additive evidence update

```
Delta_i = Delta_theta(h_i)
z_i^1 = z_i^0 + Delta_i
p_i^1 = sigmoid(z_i^1)
```

The additive structure explicitly represents OCT as an update to an existing belief rather than a replacement predictor.

## 2. Patient-Specific Evidence Advantage

For a proper scoring loss `ell`, define

```
A_i = ell(Y_i, p_i^0) - ell(Y_i, p_i^1)
```

This is the realized predictive consequence of accepting the OCT update.

- `A_i > 0`: the OCT update lowers predictive risk.
- `A_i < 0`: the OCT update increases predictive risk (negative fusion).

The conditional target is

```
A(C,O) = E[A_i | C_i=C, O_i=O]
```

or, for the binary accept/reject decision,

```
B_i = 1[A_i > 0].
```

The current low-variance reference implementation predicts the benefit sign using only quantities available before pathology at inference time: the clinical logit, the proposed OCT update, pre/post probabilities, and confidence change.

## 3. Leakage-safe advantage estimation

Stage105 keeps the diagnostic backbone frozen and separates patients into three source-development roles:

```
calibration: fit the advantage/benefit estimator
selection:   choose the acceptance threshold
validation:  evaluate the frozen policy
```

The pathology label is required only to construct `A_i` or `B_i` on calibration/selection data. It is never required for forward inference on validation/test patients.

This separation is important because directly learning three absolute action risks and subtracting them was empirically unstable and collapsed toward the clinical-only action.

## 4. Advantage-guided selective evidence update

Let `s_i` be the forward advantage score. For a threshold `tau`,

```
S_i = 1[s_i > tau]
z_i^final = z_i^0 + S_i * Delta_i
p_i^final = sigmoid(z_i^final)
```

The policy therefore accepts OCT only when its predicted patient-specific advantage exceeds the selected threshold.

Primary policy metrics include:

- benefit AUC: discrimination of `A_i > 0`;
- update coverage: fraction of patients accepting OCT;
- decision regret relative to the per-patient oracle between `z0` and `z1`;
- negative/harmful update frequency;
- realized gain over the clinical belief;
- standard AUROC/AUPRC/NLL/Brier for diagnostic performance.

## 5. Why this is not ordinary gating

A conventional gate learns a weight or routing score and is usually supervised only through the final prediction objective. Evidence Advantage introduces an explicit outcome-aligned decision consequence:

```
A_i = loss_before - loss_after.
```

The learned score is therefore tied to whether accepting a modality is predicted to help or harm the current patient, rather than to an uninterpreted fusion weight.

## 6. Relation to SOPI

Stagewise Orthogonal Proper-Score Innovation (SOPI) remains a useful population-level retrospective estimand:

```
E[A_i] = E[ell(Y,p0) - ell(Y,p1)].
```

The current method moves from the population average to the patient-specific conditional decision problem. In this sense, Evidence Advantage is the individualized decision object, while SOPI remains a population-level information accounting tool.

## Current cervical instantiation

```
age / HPV / cytology -> clinical belief z0
OCT sequence         -> candidate update Delta
Evidence Advantage   -> accept or reject OCT update
pathology            -> retrospective supervision/evaluation
```

OCT is positioned as **pre-colposcopy evidence**. The current method does not claim replacement of colposcopy.

## Current methodological boundary

Stage105 supports the predictability of patient-specific benefit across all tested fold-seed cells, but the transferred decision threshold is not uniformly superior to unconditional updating. Therefore:

- the Evidence Advantage object is retained as the core method;
- the low-variance benefit estimator is retained as the current reference implementation;
- threshold/risk-control calibration is not yet claimed as a universal guarantee.
