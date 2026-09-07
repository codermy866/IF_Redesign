# Cross-fitted diagnostic gain on recovered clinical inputs

2026-09-06, exploratory finite replication, no hyperparameter selection.
Use CROSS_FITTED_GAIN_PROTOCOL.md and CROSS_FITTED_GAIN_MATH.md unchanged except
HPV/TCT strings restored from the exact-match clinical sidecar documented in
CLINICAL_RECOVERY_REPLICATION.md. Five existing source splits x three seeds;
90 new diagnostic teachers/reference models, same 8 selection policies and
budgets0..6, full12 comparison, signed gains and same patient-nested fitting.

Rationale locked before restored-gain validation: 2x2 repair controls showed
clinical single-modality AP rising .3798->.5875 but no clear full multimodal gain.
Previously all gains were conditioned on heavily degraded clinical context, so
the question is whether gain selection depends on faithful clinical inputs.
Do NOT assume it will improve. Exact same source fit/cal/gain partitions and
random initialization seeds permit a paired comparison with original-input runs.
No use of repaired fusion pilot validation to tune this gain model or teachers.

Every teacher's age scaling/calibration excludes its OOF patients; selector
clinical scaling uses only source train. No old weights initialize new teachers.
Same Ridge alpha10, 16dim fixed random projection, same gain states/actions.
Context-aware offline reranking, not acquiring unseen evidence or clinical causes.
Labels from OCT reread are used solely as the same lesion-score training control
and validation localization endpoint; no evaluation ground-truth image filtering.

Output results/cross_fitted_gain_restored_v1. Preserve old results and report all
seeds/folds/policies. No outer hospital inference or LLM synthesis. Stop after
finite matrix, aggregate all results. Do not keep tuning until a desired effect.
This replication is needed to assess whether previous module failures survive a
verified input repair; repair itself is not an innovation module.
