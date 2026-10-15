# Low-AUROC recovery plan — 2026-10-13 (analysis and source-only candidates)

## Ground truth on what has been measured
The earlier **984-patient/five-centre development** report records SEMANTIC_PCA26 AUROC 0.737427, JEV_NATIVE 0.715702, historical LAPLACE 0.675017, and historical LAPLACE+CFV 0.641752. These are not performance figures for the 2026-10-13 RIME implementation. A prior 893-patient/four-centre analysis exceeded 0.8 under a different protocol and **must not** be pooled or compared as the same benchmark. Historical RIME extraction progress was 49,019/118,080 frames early on October 13; verify present completeness on the GPU server before attributing any patient-level performance to complete RIME.

## Phase 0: high-priority failure triage (before new training)

Run `scripts/diagnose_low_auc.py` on five-centre out-of-fold **real** patient predictions. It rejects unequal patient sets, duplicated patient/model entries, and inconsistent labels or centres. Inspect the pooled/centre-specific AUROC, prediction orientation (diagnostic only; no test-time sign reversal), AUPRC, NLL, per-centre difference against anchor, threshold flip count, optional missingness / histopathology strata, and logit residual size. Check OCT frame completion, valid masks, reread annotation eligibility, center-specific prevalence and label provenance.

On source data only, check a 20-image all-gray parity case and compare `AUROC(y,decision[true]-decision[false])` with the same orientation used during clinical training. Check source inference preprocessing parity between 4-frame montage patient anchor and independent raw-frame local task. Do not use outer targets to decide the sign or tune preprocessing.

## Phase 1: recover the strongest 3-modality anchor, not just OCT

1. Source-only clinical prior (age/HPV/TCT) and clinical+colpo/OCT readouts; use identical outer LOCO partitions, patient IDs, source selection, and endpoint provenance.
2. Compare frozen released JEV, patient-only LoRA and dual-task LoRA as **distinct backbone adaptation choices**. Report vision encoder unchanged and source clinical/semantic calibration. If adapted 27B is prohibitively costly, prioritize a low-cost source-trained clinical+colposcopy+OCT anchor rather than retraining until an upgrade is seen.
3. Measure image-feature class separability using source-only linear probes, nearest-neighbor label consistency and within-vs-between-centre variance; test OCT independent frame protocols. A weak image probe indicates representation bottleneck; a strong probe with poor readout indicates pooling/training bottleneck.

## Phase 2: targeted RIME fixes with matched controls

**F1 — representation bottleneck.** Evaluate 8/16/32 source-only PCA coordinates with the same downstream capacity limits, plus frozen supervised projection trained on source patients only as a diagnostic; do not use test distribution to fit PCA. The current first 8 high-variance directions can mainly encode scanner/site differences.

**F2 — selector learning bottleneck.** `RIMEHead` initializes `beta=0`, `g=0`, `a=0` and low-magnitude `U,V`. At initialization the patient BCE gradient toward the measure selector and site attention is zero because it passes through `beta`. Evaluate source-selected two-phase learning: (i) train the residual with fixed attention/measure, (ii) unfreeze selector and site attention. Compare with ordinary joint training, same optimizer budget and equivalent selection. Compare kernel-QP to attention and geometry-only QP: the QP with initially zero conditional scores optimizes diversity, **not disease relevance**.

**F3 — Laplace/readout calibration.** At each source fold quantify per-scale coordinate variance, saturation, residual logit magnitude and rank flips. Ablate temperatures [0.5,2,8] against source-selected smaller-scale alternatives; compare to equal-56D Legendre/moment readout. Do not present this as learning a Laplace *probability distribution*. Use source-fitted residual shrinkage (including zero correction) and centre-wise negative-log-likelihood safeguards. Retain the complete original anchor for no-OCT patients.

**F4 — decision-value supervision (conditional next candidate).** First prove phase F1-F3 have stable source-fold benefit. If they do, compute OOF source-only site-level decision advantages using a frozen reference: `a_is=loss(y_i,z_i(without site s))-loss(y_i,z_i(full))`, with full patient-context recomputation. Regress/rank an in-model score `q_is` toward these patient-specific *predictive* advantages, then solve the unchanged simplex kernel QP. Compare with q learned only through patient BCE and q=0 geometry-only. No causal or pathology-localization interpretation is allowed. Avoid applying outcome-derived advantages at inference.

## Fast stop/go, no test-set optimization

- Prefer full anchor to a worse residual model unless **source-only** held-internally-out-centre validation supports improvement. Preserve anchor as alpha=0 reference; use source-selected alpha grid, not test AUROC peak.
- Continue Laplace only if same-patient, same-capacity source comparisons favor it with a defensible centre distribution; otherwise present it as a negative ablation, or keep it only if it has an independently validated robustness signature.
- Keep CFV as counterfactual **audit** until a trainable counterpart has a direct positive incremental result against RIME without CFV.
- The final pre-registered outer five-centre LOCO test is evaluated once after choices are frozen. Do not iterate on outer-fold outcomes and then rebrand them as independent confirmation.

## Reproduction command and artifacts

```bash
python scripts/diagnose_low_auc.py /private/heldout_predictions.csv \
  --anchor anchor --out /private/analysis/auc_triage
```

Required columns: patient_id, center, model, y_cin2, prob. Optional: valid_oct_sites, has_colpo, histology_only, reference_type.

Outputs: summary.csv, by_center.csv, paired_diagnostics.csv, by_subgroup.csv, audit.json. The CLI operates on already-frozen predictions, uses labels for **diagnosis only**, and cannot be used to fit or select models. Local synthetic 85-patient smoke data passed; real clinical outputs are not yet available in the public repository.
