# Missing-modality fusion diagnostic experiment v1

Locked before new-model validation inference, 2026-09-06. Exploratory development;
human_review_pending. Motivation: user-supplied npj Digital Medicine accepted PDF,
DOI 10.1038/s41746-026-02783-3. This is NOT an implementation of NAIM/ODST.

## Evidence and questions

All 15 cross-fitted gain runs and 90 teachers completed. OOF mean macro-AP@4
0.5866, seen 0.5828, attention 0.5965, entropy 0.6164, lesion 0.6123.
First-seed OOF improvement did not replicate. We test whether weak/unstable
diagnostic integration, independently of selection, warrants a revised method.

The reference motivates unimodal pretraining, explicit availability masks, fusion
comparisons and progressive test-time modality removal. Sections 4.6 and 4.7 of
the supplied article disagree on freezing versus fine-tuning and fusion naming.
We explicitly test both, without presenting either as a faithful reproduction.
Its survival objective and endpoints do not apply to binary CIN2+.

## Locked data boundary

Existing 984 distinct patients, five existing LOCO source partitions. Cache audit:
984/984 have all three input modalities and all 12 OCT positions. Feature-level
unknown clinical values remain unknown; their presence does not make a complete
clinical record. No additional patients from 3000_nums are silently introduced.
Only SOURCE TRAIN is split into fit/internal calibration (same helper/seed+100 as
the cross-fitting reference). Age imputation/scaling fit only on fit patients.
External source validation is for reporting, never fitting/checkpoint selection.
No outer hospital inference. This cohort has historical exploratory reuse.

## Predeclared factorial and controls

Five folds shiyan/enshi/wuhan/jingzhou/xiangyang, seeds 20260905/06/07: 15 runs.
Each run trains three unimodal branches once, then four fusion heads (45+60
training stages). Fixed ImageNet ResNet50 cache and available slots unchanged.
Branches: clinical 14->64; each visual token 2048->64 with LayerNorm/GELU,
masked learned attention pooling, OCT positional embeddings. No new backbone.
Each branch trained with unweighted binary CE (proper probability objective).
Fusion: concatenate three masked 64-d latents plus explicit 3-bit availability,
64-unit GELU bottleneck and scalar logit. The four arms share pretrained branch
initialization, initial fusion weights, fit/cal cases, minibatch order and seeds:

* frozen_full: frozen branches; 1.5 * full-input CE.
* tune_full: branches fine-tuned at 1/10 head LR; 1.5 * full-input CE.
* frozen_missing: frozen branches; full CE + 0.5 * missing-pattern CE.
* tune_missing: fine-tuned branches; full CE + 0.5 * missing-pattern CE.

Missing patterns sampled uniformly among the seven NONEMPTY subsets of T/C/O,
independently of labels/center. Same sampled masks for the two missing arms.
Whole-modality removal, not first/last-frame substitution. Frozen means latent
encoder protection; it is an established comparator, not claimed novelty.
Controls: each pretrained unimodal predictor, available-modality mean probability
(late fusion), and existing fresh ICES diagnostic references as a legacy benchmark.
Legacy architecture/loss differs; do not interpret that contrast as a single-factor
causal ablation. No parameter sweep. No point label or gain model used here.

AdamW head LR 0.001, weight decay 0.0001, batch 64, max 80 epochs, patience 12.
Unimodal checkpoint: internal-cal CE. All four fusion checkpoints: same equally
weighted internal-cal CE over seven availability patterns, including full input.
No outer/source-reporting-validation-based early stopping. All-missing inputs
raise an explicit error: no clinical prediction without any evidence.

## Endpoints and reporting

Primary endpoints: source-validation complete-input macro-AP and equally weighted
seven-pattern macro-AP. Secondary pooled AUROC, unweighted CE, Brier, sensitivity
and specificity at an INTERNAL CAL threshold targeting 90% sensitivity for each
pattern (report calibration sample sizes; no guaranteed clinical sensitivity).
Progressive modality removal 0,25,50,75,100% uses nested random patient masks,
same masks across models/seeds, 20 fixed mask replicates, never outcome-dependent.
Report actual removed fractions, complete-input penalty, per-modality dependence,
seed spread, all centers, all arms. These are synthetic availability interventions,
not scanner-style interventions, lesion-preserving perturbations or clinical causes.
No claims of survival benefits, prospective utility or validated CoE reasoning.

Patient-level paired uncertainty; shared patients across source folds must not be
counted as independent repeats. Seeds do not multiply sample size. Primary
contrasts are missing-full within each encoder strategy and frozen-tune within
each training strategy; report interaction. Any promising result needs locked
independent validation. No automatic repeated tuning until a desired p-value.

## Decision rule and stopping

Complete the finite matrix, verify artifacts, report all outcomes. A candidate
remains exploratory unless complete-input performance is retained and robustness
improvement repeats across seeds/folds. If not, state that the new structural
hypothesis failed; do not rename it a successful innovation. Subsequent candidate
modules need a new protocol, not silent expansion of this screen.
