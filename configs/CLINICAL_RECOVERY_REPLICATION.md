# Source clinical recovery: paired data-only replication

Locked after provenance audit and before recovered-input model training/inference,
2026-09-06. Follow-up to MISSING_MODALITY_FUSION_PROTOCOL.md; same architecture,
hyperparameters, seeds, patient fit/cal/source-val partitions, early stopping,
seven-pattern comparisons and full image cache. Another 15 runs: 45 branches and
60 fusion heads. The ONLY changed input is HPV/TCT source strings from the
original 3000_nums.xlsx, exact OCT acquisition prefix join, all 984 cases matched
uniquely. No pathology column, OCT reread, report or spreadsheet color is used.

Reason: old cache has HPV='1' in 605 and TCT='1' in 610 cases. Legacy normalizers
explicitly treat '1' as unknown, despite original workbook containing genotype
and cytology values. With the SAME normalizers applied to recovered text, HPV
unknown decreases 821->254 and TCT unknown 743->136. This is reproducible
information loss at the data interface, not a reason to invent missing labels.
Two old HPV-negative records disagree with nonnegative original source strings;
recorded in audit.json, clinical verification remains pending.

Source sidecar results/clinical_source_recovery_v1/clinical_sidecar.npz preserves
row order/IDs and stores raw strings. Original cache/manifests/checkpoints are
never overwritten. Result namespace missing_modality_fusion_restored_v1.
Same normalizer deliberately retained to isolate source-string recovery; residual
ambiguities remain (e.g. '其他', numeric atypical HPV codes, composite TCT). The
historical name 'other_high_risk' is a model vocabulary label, NOT clinical
adjudication that every numeric HPV genotype is high risk. Do not use it as a
medical categorization in a manuscript without clinical review.

Primary repair contrasts: same arm restored minus original at complete-input and
seven-pattern macro-AP, paired across seed/patient. Clinical-only branch change,
identical visual-branch checkpoints, restored-missing patterns without T unchanged
for simple late fusion are negative controls. Fusion changes without T can occur
through training, so are not required invariant. Report all arms, seed variability,
CE/Brier and no success-only filtering. Previous gain-module estimates conditioned
on degraded clinical inputs may need re-estimation, but are not silently relabeled
invalid or replaced by this repair experiment. No gain or CoE-module efficacy is
established by better clinical fields alone. Outer hospital evaluation still closed.
