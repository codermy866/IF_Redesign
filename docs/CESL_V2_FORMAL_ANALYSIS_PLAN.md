# CESL-v2 Formal Retrospective Analysis Plan

Status: locally locked on 2026-09-05 before any CESL held-out prediction is
generated. This is a new exploratory follow-on study, not a revision of the
completed Formal v1 primary analysis.

## Objective and unit of analysis

The endpoint is patient-level CIN2+ prediction. The unit of analysis and
bootstrap resampling unit is one patient/case. The cohort, five outer
leave-one-centre-out folds, and source train/validation partitions are reused
without modification from Formal v1. No individual raw frame is treated as an
independent patient replicate.

The study asks whether a model can retain a smaller set of already available
raw-frame evidence atoms while preserving diagnostic ranking and whether the
retained evidence is more robust to declared, conditional replacement than
attention or random alternatives. It is an **offline evidence-retention**
experiment. The data do not support a claim that images could have been
avoided prospectively, that scanning time is reduced, or that an atom denotes
a lesion.

## Frozen representation and leakage controls

- Atom dictionary: one clinical atom, up to four colposcopy frames, and up to
  six OCT B-scans, using the existing deterministic raw-atom manifest.
- Image representation: frozen local ImageNet ResNet-50 features. The visual
  encoder is never fit on a held-out centre.
- Train, validation, calibration, threshold fitting, fixed-budget matching,
  and donor-pool construction use source-centre data only.
- Held-out labels are never used by a selection policy. They are used only
  after predictions for outcome metrics and the explicitly labelled
  post-hoc replacement audit.
- Every donor edge records recipient, donor, atom slot, q variant, matching
  level, seed, and a declaration that donor labels were not used for matching.

## Method conditions

The backbone is held fixed for a comparison family. C1, C2, C3, the random
control, and the uncertainty-only control share an identical no-CSS backbone.
C4 differs only through its declared source-centre CSS objective. C0 is the
static full-evidence reference.

| Condition | Counterfactual valuation | Stopping / retention rule | CSS |
|---|---:|---|---:|
| C0 | No | All available atoms | No |
| C1 | Attention ranking | Source-validation matched fixed budget | No |
| C2 | Predicted-class CEV | Source-validation matched fixed budget | No |
| C3 | Predicted-class CEV | Margin plus residual-CEV stopping | No |
| C4 | Predicted-class CEV | Margin plus residual-CEV stopping | Yes |
| Random | No | Random matched fixed budget | No |
| Uncertainty-only | No | Fixed order plus margin stopping | No |

`Predicted-class CEV` is label-free at selection time: the model's own
predicted class defines the replacement loss. The observed-label risk change
is reported later as a mechanism audit, not fed into a held-out decision.

## Primary and secondary analyses

Primary estimand: three-seed ensemble macro-centre AUPRC on the five held-out
centre test sets. C4 is compared with C0, C1, C2, C3, Random, and
Uncertainty-only using paired, patient-clustered bootstrap differences and
Holm adjustment for this primary comparison family.

Secondary analyses:

1. AUROC, balanced accuracy, sensitivity, specificity, Brier score, ECE,
   calibration slope/intercept, selected visual-atom count, and relative
   evidence cost.
2. CEV mechanism audit: observed-label replacement loss of high-CEV,
   attention-ranked, and type-matched random atoms under the declared donor
   distribution.
3. M3 audit: pairwise held-out-centre rank stability, stopping-rate stability,
   and worst-centre performance for C4 versus C3.
4. q sensitivity: matched source donors, centre-only donors, and pooled
   unmatched source donors. This is a sensitivity analysis, not a search for
   the most favourable donor distribution.
5. Availability stress tests: removal of clinical evidence, either visual
   modality, and predeclared reduced-frame conditions. These are model-input
   stress tests, not clinical missingness prevalence estimates.
6. Calibration and net-benefit curves use source-only temperature scaling and
   source-only thresholds. They are retrospective analytical outputs and not
   deployment recommendations.

## Explicitly unavailable experiments

Temporal validation cannot be validly started because the frozen manifest has
no structured acquisition date. A sixth independent hospital, prospective
silent deployment, and reader study all require new data and, where
applicable, clinical governance. No pseudo ROI, mask, polygon, or bbox is
used as an alternative.

## Interpretation rules

The three modules are independently supported only if their respective
falsification rule is met. A lower atom count alone is insufficient. A null,
negative, or unstable result remains part of the record. Any future clinical
or prospective claim requires separate protocol, data, and human review.
