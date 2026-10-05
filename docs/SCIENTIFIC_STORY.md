# Scientific Story

## Core problem

Multimodal diagnosis often treats modality availability as permission to fuse.
The IF_Redesign paper instead asks:

> **For the current patient, should newly available OCT evidence be allowed to modify the existing clinical belief?**

The distinction is:

```text
available modality != beneficial evidence
```

The methodological target is therefore patient-specific evidence utility,
not merely a better attention or gating weight.

## Core method

### Clinical belief

```text
z_i^0 = g_C(C_i)
p_i^0 = sigmoid(z_i^0)
```

### Candidate OCT update

```text
Delta_i = Delta_theta(O_i)
z_i^1 = z_i^0 + Delta_i
```

### Patient-Specific Evidence Advantage

```text
A_i = ell(Y_i, p_i^0) - ell(Y_i, p_i^1)
```

Positive A_i means that accepting the OCT update lowers proper-score loss for
that patient; negative A_i is patient-level harmful fusion.

The forward task is to estimate this benefit before pathology is observed.

### Selective evidence updating

```text
S_i = 1[score_i > tau]
z_i^final = z_i^0 + S_i Delta_i
```

Stage105 supports the learnability of patient-specific OCT benefit, while also
showing that a single transferred threshold is not uniformly dominant. The
threshold/risk-control layer is therefore an operating-policy problem, not a
new causal guarantee.

## Why the failed stages matter

- Stage102: OCT contains patient-specific information, but fixed correspondence
  objectives are not uniformly transportable.
- Stage103: discrete {clinical, OCT, fusion} state routing is unstable.
- Stage104: absolute action-risk estimation collapses toward clinical-only routing.
- Stage105: direct relative Evidence Advantage avoids that absolute-risk
  decomposition and yields non-random patient-specific benefit prediction.
- Stage106: scalar objective repairs do not independently justify a new
  mathematical contribution.
- Stage107: a mismatch control is invalid if the original recipient gate is
  reused; the complete frozen policy must be recomputed after replacement.

These failures narrow the final claim rather than being hidden.

## Exploratory clinical-mechanism extension

Colposcopy is **not** part of the core IF model. In a separate aligned cohort,
Stage109 asks:

> **After OCT is admitted, how much incremental predictive value remains in subsequent colposcopy?**

Using the same prediction function under four evidence states:

```text
C
C + O
C + K
C + O + K
```

define:

```text
A^O      = L(C)   - L(C+O)
A^K      = L(C)   - L(C+K)
A^{K|O}  = L(C+O) - L(C+O+K)
R^{O->K} = A^K - A^{K|O}
```

Positive R^{O->K} is **evidence redundancy**: colposcopy has less incremental
predictive value after OCT is already known.

This is an intervention on information admission, not on disease biology.
Accordingly, the permitted interpretation is partial predictive redundancy,
not a biological causal effect and not replacement of colposcopy.

## Final contribution hierarchy

1. **Patient-Specific Evidence Advantage** as the patient-level decision
   consequence of admitting OCT evidence.
2. **Advantage-Guided Selective Evidence Updating** as the corresponding
   pre-colposcopy decision rule.
3. **Evidence Admission Intervention** as an exploratory mechanism analysis
   showing how OCT changes the residual value of downstream colposcopy.

The third item strengthens the clinical and multimodal-fusion interpretation
without changing the core method.