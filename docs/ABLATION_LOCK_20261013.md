# RIME-JEV surgical module revisions and orthogonal ablation lock — 2026-10-13

**Status: proposed optimization and experiments; no new clinical result has been measured.** Applies to `paper/final-confirmation-20261013`. The original `main` default remains joint training. No private GPUs, data paths or predictions are bundled.

## Current model dependency audit

1. **Frozen JEV raw OCT to 8D PCA**: the first 8 source-weighted variance directions need not preserve CIN2+ discrimination; a bag-level label cannot be treated as a frame pathology label.
2. **Conditional kernel QP**: the score `q=tanh(a'h+<hU,bV>)` is zero initially (a=0; hU and bV small); the kernel QP initially optimizes geometry/diversity, not clinical relevance. Exact-feature deduplication establishes a numerical invariance only with frozen preprocessing and context.
3. **Signed Laplace readout**: fixed scales (0.5,2,8) may saturate and have correlated channels; a 56D Moment/Legendre control is already implemented. Laplace here is a set of empirical exponential responses, *not a fitted Laplace distribution*.
4. **Anchor residual**: `z=anchor+r @ beta` with `beta=0`; therefore the first backward pass supplies exactly zero prediction-loss gradient to `a,U,V,g`. Later gradients exist after beta moves, but selector learning may lag the small-data optimization. A source-only two-stage ablation is now exposed.
5. **Historical CFV**: view-removal code is separate from the RIME head. It must not be described as an already joint-trained RIME+CFV contribution.

## Code change: optimizer-only control

`train_head(..., selector_warmup_epochs=0)` is the unchanged joint training; `selector_warmup_epochs=15` freezes ONLY the initially trainable `a,U,V,g` while fitting residual `beta` for 15 epochs, then unfreezes them. Both conditions keep the same trainable parameter budget, architecture, maximum update budget and per-source model-selection protocol. Early stopping must not terminate before the released selector trains. The new regression test is `tests/test_rime_selector_warmup.py`. **The test source was committed but its execution on the private training environment is not yet verified.**

## Stage 1: four-cell core factorial, same patient/split/feature/anchor

| Id | Pooler | Readout | Selector optimization | Result |
|:--|:--|:--|:--|:--|
| AM | Attention | 56D zero-anchored Moment | Joint | Pending |
| AL | Attention | 56D signed Laplace | Joint | Pending |
| MM | Conditional kernel measure | 56D Moment | Joint | Pending |
| ML | Conditional kernel measure | 56D signed Laplace | Joint | Pending |

Mandatory comparators: unchanged patient anchor; fixed 8D preprocessed raw-frame JEV; clinical-only, OCT-only, colposcopy-only and full multimodal baseline with identical test patients. Record all centre-wise CIN2+ AUROC, AUPRC, NLL, Brier and CIN3+ sensitivity with denominators.

At 56D equal-capacity training, pre-specify three hypotheses:
- Measure increment at fixed Laplace: `D_measure = AUROC(ML)-AUROC(AL)`.
- Laplace increment at fixed measure: `D_laplace = AUROC(ML)-AUROC(MM)`.
- Factor interaction: `I = [AUROC(ML)-AUROC(MM)]-[AUROC(AL)-AUROC(AM)]`.

A negative main effect cannot be hidden by reporting ML alone. Use centre-stratified **paired patient bootstrap** 95% CI for the differences; report five held-out centre results and pooled/centre-equal aggregates, with each estimand defined. AUPRC comparability requires reporting endpoint prevalence.

## Stage 2: optimizer and representation, one axis at a time

1. **Optimizer effect**: ML-joint vs ML-warmup (source-chosen warmup among 0/5/15), SAME 8D PCA, 56D Laplace, anchor, parameter count and epochs. Warmup is not a separate representation innovation; report as optimization ablation.
2. **Geometry vs disease signal**: ML q=0 geometry-only vs q(h) with b removed vs q(h,b) full; preserve measure and readout. A q=0 baseline says whether learned conditional evidence is important.
3. **Representation loss**: compare source-only 8D PCA against 8D source-supervised patient-BAG projection; the latter must not assign patient histology to individual OCT frames as local ground truth. Keep downstream dimension/capacity unchanged. Higher 16D/32D PCA variants are diagnostic only until generic-dimensional code and matched 112D/224D controls are implemented. Do not compare larger representation with 56D readout and attribute improvement to the projection.
4. **Laplace saturation**: report feature activation histograms, saturation fractions, collinearity and `beta` residual logit magnitudes. Compare fixed temperatures to source-selected temperature families with pre-specified search, never choose scales on outer test targets.
5. **Anchor preservation**: source-select correction shrinkage alpha from {0,0.25,0.5,1} at training/inner-source folds; alpha=0 must recover the full three-modal anchor. Do not improve test AUROC through sign flips or test-driven correction selection.

## Stage 3: CFV admission experiment (NOT yet implemented in latest RIME branch)

Define source-patient out-of-fold **predictive** advantage of site s: `A_is = loss(y_i, z_i^(minus s))-loss(y_i,z_i(full))`. Recompute entire anchor/context and affected views; do not reuse fixed-context head-only deletion. Train an inference-time scorer `q_theta(h_s,b)` against cross-fitted source advantage, then use the unchanged kernel-measure optimization. At inference, the scorer sees features/context only, never the outcome. Compare:
- ML with standard patient-BCE selector (baseline),
- ML with q=0 (geometry),
- ML with source-advantage guided selector,
- ML with shuffled-advantage control within source centre (same supervision budget).

Only after these controlled comparisons may CFV count as a trainable contribution. Counterfactual intervention measures **model sensitivity/predictive value**, not pathology location or a causal biological effect.

## Decision and compute policy

1. Finish and freeze the outer-fold source training/selection plan, data validation and patient global deduplication first.
2. Cache identical frozen JEV raw-frame hidden features and source-fitted anchors for all *within-fold* four factorial cells; never share source-fitted PCA/scales across an outer held-out fold.
3. Run the four core cells, then stage-wise ML variants. Schedule 5 outer centres x 3 seeds; stop spending 27B GPU-hours on CFV recomputation until the base ML versus AL/MM is interpretable. Reuse head-scale experiments locally where feasible.
4. Track design/failure logs even when a QP becomes singular; no post hoc removal of nonconvergent seeds. Allow predefined source-only stable fallback; flag deviations.
5. Report old exploratory results distinctly from the latest RIME. No predictive gain or >0.8 AUROC can be claimed until real LOCO outcomes are computed.

## Paper endpoint

Final method claim can have a compact structure:
- **Representation of patient evidence** (frozen JEV + patient-bag projection)
- **Conditional replication-invariant evidence measure** (new core algorithm)
- **Signed multiscale Laplace decision readout**
- **Counterfactual decision-value verification**, or (only if stage 3 passes) a separately ablated **advantage-guided evidence selection objective**.

These are linked by a single problem: *extract pathology-relevant support from correlated, heterogeneous, multimodal cervical evidence without allowing repeated/weak support to dominate the patient decision*.
