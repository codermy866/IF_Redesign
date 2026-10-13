# IF 2026 final confirmation protocol (locked design, not locked results)

Date: 2026-10-13. Target submission: 2026-11-15. This document is a source-level protocol, **not evidence that the private multicentre runs have completed**. Source repository release does not contain patient records, fitted anchor, result CSVs, data loader, or job scheduler.

## Scientific question and candidate identity

**Question:** In patient-level cervical triage, 12 scanner positions with up to 10 correlated OCT frames can create pseudo-replication. Can evidence be aggregated without treating exact repetitions as independent support, preserve the magnitude and sign of heterogeneous evidence, and improve held-out *patient decisions* beyond an unchanged three-modality clinical/native/semantic anchor?

Working candidate: **RIME-JEV**, the frozen JEV decision backbone plus exact-atom conditional kernel measure, signed multiscale Laplace coordinates and a residual decision readout. The historical **CFV-JEV** branch supplies a **counterfactual decision-change verification protocol**. It is *not* implemented as a trainable part of the current RIME branch. Only call a trainable CFV component part of the final model if it is integrated, separately ablated, and improves source-selected external evaluation. Do not call the model an exact reproduction of the original Visual Jev implementation: shared-question Visual Jev and the released 27B medical JEV package have different upstream backbones/protocols.

### Inputs and labels

- Clinical: age, HPV, TCT; colposcopy images: up to eight valid views; OCT: up to 12 positions, reference acquisition 10 ordered frames per position. Explicit masks distinguish unavailable images from neutral gray slots.
- Complete anchor (caller-owned, independently source-fitted): clinical logit + native patient decision contrast + 26-dimensional semantic predictor. Its original OCT protocol uses four-frame montages [0,3,6,9]. New RIME local evidence uses **independent raw frames**. Never conflate their preprocessing protocols.
- Main outcome: **patient CIN2+**; safety secondary: **CIN3+**. Reference provenance must be reported by histopathology/excision versus adjudicated fallback; stratify by provenance to test label incorporation bias.
- At most one record per globally deduplicated patient across centres and datasets. All preprocessing, clinical priors, projection matrices, coordinate scales, model selection and calibration belong to source data only.

## Locked method / ablation grid

The crucial capacity-matched **2 x 2** grid (equal 56 coordinate dimensions and equal trainable parameter count):

| Local pooler | Moment / Legendre | Signed Laplace |
| --- | --- | --- |
| Ordinary learned attention | A/M | A/L |
| Exact-atom conditional kernel measure | M/M | **M/L = RIME** |

Interpretation: M/L > A/L demonstrates the measure's incremental benefit when readout is held constant; M/L > M/M demonstrates the Laplace incremental benefit when pooler is fixed; the 2x2 interaction is diagnostic, not required to be positive to claim two distinct main effects.

Mandatory clinical comparators, with matched cohort and outer folds: clinical-only; OCT-only; colposcopy-only; clinical+OCT+colposcopy fusion backbone; complete unchanged anchor; frozen native JEV; patient-only adapted JEV; dual-task adapted JEV; mean MIL; MLP pooler with at least as many selector coefficients; geometry-only kernel; identity-kernel QP; GMP; semantic-head refit; historical CFV-JEV as a clearly labelled separately implemented baseline. CAPR-MIL is conditional on obtaining and fairly adapting its code; do not call the local wrapper a full published CAPR-MIL reproduction. Five credible recent **external** methods must be reproduced or marked unavailable, with year, backbone and budget disclosed. Source-matched experimental controls do not substitute for five external papers.

## Evaluation (no target peeking)

1. Audit global patient IDs, centre labels, reference-label provenance, OCT validity and counts, colposcopy missingness, class prevalence, corrupted paths, and scanner/view counts. Save anonymized aggregate audit only.
2. Freeze five-centre **leave-one-centre-out** outer folds. Within each outer source set, separate training and selection patients. Fit clinical model, JEV adapter (if used), semantic anchor, PCA projection, kernel bandwidth and per-coordinate scales exclusively on source-training data. Select `tau` in {0.1,0.3,1.0}, `lambda` in {0.003,0.03,0.3}, epochs and seed ensemble by predefined source-only inner-CV rules. Refit on the permitted source-training data; never use held-out labels, including for threshold or model choice.
3. Freeze seeds {17,29,43}, same source-split assignments and input manifests across variants. Preserve failed/nonconvergent QPs and denominator of runs. Expected hardware is 2x A6000; save software versions, hash of manifest, hash of git SHA, checkpoint, frozen preprocessing, and all per-patient predictions.
4. Primary report: pooled out-of-centre AUROC, AUPRC, paired AUROC differences versus anchor and each factorial control with patient-level 95% bootstrap CI stratified by centre. Report individual centres and centre-weighted averages. Also report NLL, Brier, calibration (slope/intercept and ECE, clearly defined) and F1 at a **source-locked threshold**. For CIN3+ report sensitivity, missed cases, and counts separately; do not claim safety from AUROC alone.
5. Counterfactual verification: for each valid site, recompute the **complete** patient input with that site's OCT neutralized, including the original native/semantic anchor, context, local representation and pooling, not just RIME head deletion. Define signed logit change as `delta_site=z(full)-z(without site)`. Report decision threshold flips, prevalence of negative/misleading evidence, difference from fixed-context approximation, and whether aggregate decision changes concentrate on independently reviewed abnormal positions. Site reread labels are imaging phenotypes, **not** pathology localization ground truth. Report per modality (OCT/colposcopy) interventions separately.
6. Mechanism stress tests: exact-copy duplication (expect numerical invariance **only with frozen anchor/context/preprocessing**), permutation within position, missing sites, near-duplicate frames with increasing perturbation (no exact invariance guarantee), injected noisy/outlier frames, and centre shift. Compare M/L versus A/L and M/M; report prediction drift and decision-flip rate.
7. Include a **histopathology-only** sensitivity analysis because adjudicated OCT-clinical labels may be correlated with the imaging predictors. If an independent locked external three-modal cohort is unavailable, describe LOCO as internal-external cross-validation, **not** independent external confirmation.

## Pre-registered decision rule for 2026-11-15

- Retain measure as contribution only if its result beats the matched attention baseline reproducibly on outer-fold point estimates with supportive uncertainty and invariance/robustness evidence; mathematical exact-copy invariance alone is not clinical superiority.
- Retain Laplace as contribution only if it beats the 56-dimensional moment comparator under matched selection, with a plausible heterogeneous-evidence/centre-shift signature. Never describe these functions as fitting a Laplace probability distribution or calibrated uncertainty.
- Retain **trainable CFV** as the third innovation only after it is implemented in the current branch and is superior to the same RIME model without CFV under the same protocol. Otherwise use counterfactual change as a decisive **verification analysis**, not a learned component or an invented improvement.
- Clinical safety claims require explicit CIN3+ evidence, denominators and uncertainty. A pooled positive AUROC delta without CIN3 assessment cannot justify a safety claim.
- After source selection, freeze all decisions before opening test labels. Do not repeatedly change the story/model based on outer test results. If a pre-specified gate fails, present it honestly and select the previously locked simpler candidate using source-only evidence.

## Deliverables and reporting guardrails

- `audit.csv`: aggregate cohort/QC/provenance and centre counts (no private IDs).
- `results/predictions.csv`: patient-aligned, deidentified ID, centre, model, CIN2+ target, CIN3+ target (if valid), and **held-out probability**; source thresholds stored separately.
- `results/primary.csv`, `results/paired_deltas.csv`, `results/counterfactual.csv`, `results/robustness.csv` and per-centre fold run logs.
- Every table is marked **planned** until sourced from held-out numerical outputs. Synthetic demonstrations and CPU unit tests cannot populate paper performance tables.
