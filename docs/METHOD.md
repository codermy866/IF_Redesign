# Method: Patient-Specific Evidence Advantage Learning

## 1. Problem formulation

Let C_i denote clinical evidence already available for patient i, O_i the OCT evidence, and Y_i the pathology-referenced binary endpoint. The core problem is:

> **Should O_i be allowed to modify the current clinical belief for patient i?**

The paper therefore treats multimodal fusion as a patient-specific evidence decision rather than unconditional feature combination.

## 2. Clinical belief and OCT-induced candidate update

The clinical predictor establishes:

```text
z_i^0 = e_i = g_C(C_i)
p_i^0 = sigmoid(z_i^0)
```

Patient-level OCT representation h_i proposes an additive update:

```text
Delta_i = Delta_theta(h_i)
z_i^1 = z_i^0 + Delta_i
p_i^1 = sigmoid(z_i^1)
```

The additive form is important: OCT changes an existing belief rather than reconstructing the diagnosis independently.

## 3. Patient-Specific Evidence Advantage

For proper-score loss ell:

```text
A_i = ell(Y_i, p_i^0) - ell(Y_i, p_i^1)
```

A_i is the realized patient-level predictive consequence of admitting OCT.

- A_i > 0: beneficial update;
- A_i < 0: harmful update / negative fusion.

The forward object is the conditional value:

```text
V(C,O) = E[A_i | C_i=C, O_i=O]
```

or the benefit sign B_i = 1[A_i > 0]. Pathology is used to construct training/evaluation targets only; it is unavailable at inference.

## 4. Advantage estimation and selective updating

Stage105 uses leakage-separated development roles:

```text
calibration -> fit benefit/advantage estimator
selection   -> choose operating threshold
validation  -> evaluate frozen policy
```

Let s_i be the forward advantage score. For threshold tau:

```text
S_i = 1[s_i > tau]
z_i^final = z_i^0 + S_i * Delta_i
p_i^final = sigmoid(z_i^final)
```

Primary evaluation includes diagnostic AUROC/AUPRC/NLL/Brier together with benefit AUC, update coverage, decision regret, realized gain and harmful-update frequency.

## 5. Why this is not ordinary gating

Ordinary gates learn a fusion weight or route primarily through the final diagnostic objective. Evidence Advantage explicitly supervises the **decision consequence of admitting a modality**:

```text
A_i = loss_before - loss_after
```

This directly targets help-versus-harm at the patient level.

## 6. Validation boundaries from Stage106/107

Two controls constrain the final claims.

First, scalar risk-magnitude or sign-objective repairs did not independently establish a stronger contribution; they are not promoted as new theory.

Second, wrong-patient replacement is valid only when the complete frozen policy is recomputed after replacing the modality. Reusing the recipient's original gate can produce apparent correspondence even under a null model.

Accordingly:

- Evidence Advantage is not described as a biological causal effect;
- no mismatch-based causal claim is made without a valid full-policy replacement test;
- source-centre performance is not presented as a universal transport guarantee.

## 7. Exploratory Evidence Admission Intervention

Stage109 is a separate Discussion-level mechanism analysis and does not alter the core method.

Let K_i denote colposcopy evidence. A single shared additive prediction function is evaluated under binary admission indicators m_O and m_K:

```text
z(C,O,K) = z_C(C) + m_O Delta_O(O) + m_K Delta_K(K)
```

giving the states C, C+O, C+K and C+O+K.

The held-out patient-level effects are:

```text
A_i^O      = L_i(C)   - L_i(C+O)
A_i^K      = L_i(C)   - L_i(C+K)
A_i^{K|O}  = L_i(C+O) - L_i(C+O+K)
R_i^{O->K} = A_i^K - A_i^{K|O}
```

R_i^{O->K} measures how much of the original incremental predictive value of colposcopy becomes redundant after OCT is already admitted.

This is an intervention on **information availability/admission**, not on the disease-generating process.

## 8. Clinical interpretation boundary

The current method supports a pre-colposcopy role for OCT but does not claim clinical replacement of colposcopy. Stage109 shows partial predictive redundancy while the residual colposcopy advantage remains positive.

## 9. Frozen contribution hierarchy

1. Patient-Specific Evidence Advantage.
2. Advantage-Guided Selective Evidence Updating.
3. Exploratory Evidence Admission Intervention for downstream residual-value analysis.

The first two are the core Method. The third is a mechanism/Discussion extension.