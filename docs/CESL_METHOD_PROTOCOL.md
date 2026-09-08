# CESL: Counterfactual Evidence Sufficiency Learning

Status: exploratory method proposal, 2026-09-03. This is not part of the
locked Formal v1a analysis and must not be reported as a completed result.

## Question

For CIN2+ prediction, can a policy stop after acquiring a small set of
clinical, colposcopy, and OCT evidence atoms while retaining diagnostic
performance and robustness under pre-specified, matched evidence
replacements? The target is not universal fusion; it is a small sufficient
evidence set under a declared intervention distribution.

## Claim boundary

The current data have no clinician-reviewed lesion ROI or image-level
morphology labels. Consequently, CESL may evaluate *conditional replacement
robustness* under a documented donor distribution, but may not call an atom a
causal lesion or claim clinical causal necessity. Literature novelty and
clinical validity remain unverified.

## Atoms and current data constraint

The Formal v1 panels expose only three coarse atoms per case:

- clinical block: age, HPV, and TCT;
- one colposcopy contact sheet;
- one OCT contact sheet.

They can support a coarse modality-level CESL feasibility study only. The
claim "fewest B-scans" requires a new raw-frame dataset that preserves the
selected colposcopy frames and OCT slices as individually addressable atoms;
it cannot be inferred from the existing contact sheets.

## Definitions

Let E be the available atoms and S a selected subset. For the observed label y,
the supervised evaluation risk is R_theta(S) = -log p_theta(y | S).

For a selected atom k in S, define replacement necessity:

    D_k(S) = E_{e'_k ~ q_k(. | pa_k)} [R_theta((S \ {k}) union {e'_k}) - R_theta(S)].

A positive D_k means that the specified replacement distribution degrades
performance on average. It is a conditional robustness measure, not causal
identification.

The proposed stopping criterion needs a different quantity for an unobserved
atom j. Define its expected acquisition value:

    A_j(S) = R_theta(S) - E_{e_j ~ q_j(. | pa_j)}[R_theta(S union {e_j})].

At deployment y is unknown, so a learned value head A_phi(j | S), trained only
on source-centre data, estimates this quantity. The policy stops when

    m_theta(S) >= gamma and max_{j not in S} A_phi(j | S) <= epsilon,

where m_theta is the predicted-class probability margin. A selected set is
therefore operationally sufficient only relative to gamma, epsilon, the atom
dictionary, and q; it is not an absolute property of a patient.

The training objective is a constrained selective-acquisition objective:

    min_{theta, phi, pi} E[R_theta(S_tau) + lambda C(S_tau)]

subject to a pre-specified counterfactual-margin retention rate on held-out
centres. The policy and value head must be fitted without held-out-centre
labels.

## Intervention distribution q

All donor selection is source-centre-only during development and must preserve
an audit row containing recipient, donor, atom type, matching variables, seed,
and image hashes.

- Clinical: donors matched within centre and pre-specified age/measurement
  availability strata; unknown HPV/TCT remain unknown rather than imputed.
- Colposcopy/OCT: first implement whole-panel replacement only. Raw-frame CESL
  later matches centre, acquisition availability, and panel/frame position.
- Stress-test donors that intentionally alter diagnostic content may use an
  outcome-stratified donor pool, but are explicitly labelled stress tests, not
  plausible patient counterfactuals.

## Confirmatory-style predictions for a future CESL study

1. At a matched acquisition-cost budget, CESL preserves held-out macro-centre
   AUPRC and sensitivity relative to full evidence.
2. Dropping/replacing high-D atoms causes a larger loss than matched random or
   attention-ranked low-D atoms.
3. The held-out-centre rank stability of D exceeds that of the specified
   attention/gradient baseline under the same atoms and q.
4. The policy's selected atom count decreases without a pre-specified excess
   in calibration error or counterfactual-margin failures.

Failure of prediction 2 or 3 falsifies the proposed mechanism under the
declared protocol. A null result must remain reportable.

## Required comparator and ablation matrix

| Family | Comparator | Purpose |
|---|---|---|
| Efficiency | full evidence fusion | Performance and cost reference. |
| Selection | random atoms, matched cardinality | Rules out gains from simply using fewer inputs. |
| Selection | attention/gradient top-k, matched cardinality | Tests whether counterfactual value adds beyond attribution. |
| Stopping | uncertainty-only policy | Tests whether CESL is more than confidence thresholding. |
| Counterfactual | centre-matched q vs unconditioned q | Tests sensitivity to intervention plausibility. |
| Granularity | coarse panels vs raw-frame atoms | Tests whether the method benefits from real sequence-level acquisition. |

## Admission criteria before CESL training

1. Freeze the raw-frame atom builder and train/validation/test split audit.
2. Freeze q, matching variables, cost schedule, gamma, epsilon, and all
   success/failure rules before held-out predictions are opened.
3. Confirm that no test-centre label or donor is used to train pi or A_phi.
4. Run all comparator policies with identical backbone, source data, and
   acquisition budget.
5. Keep the existing unreviewed proxy ROI path out of CESL causal claims.
