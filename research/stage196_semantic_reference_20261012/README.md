# Original Stage196 PCA26 reference

The complete original experiment entry point is `run_factorial.py`.
`semantic_state` fits a 26-column basis and source scales; `semantic_features`
applies it. The inner source-centre CV and final source refit are in the same
file. Input is the cached gray-subtracted patient-level JEV hidden vector.
`native_worker.py` and `qwen27b/interventions.py` show its image extraction.
`prepare_cache.py` documents the historical cache reuse. The 32-dimensional
`qwen27b/semantic_readout.py` is a separate experiment, not PCA26.

These originals are source archives, not a turnkey data release. Their runtime
expects the original private Stage196 location, source locks, image cache index,
cohort metadata, `DESIGN_LOCK.json`, verification gates and model bundle. They
will not run in this public `research/` location without the caller's private
layout. For feature-only review, the exact two PCA26 functions are importable
from `if_redesign.pale_matched.semantic` without loading the original campaign.
No private files, fitted projection, predictions or model bundle are included.

The audit adapter replaces two server-specific input paths with the environment
variables `IF_AUTHORITATIVE_LABELS` and `IF_WEAK_LABEL_WORKBOOK`. Algorithmic
feature, fitting and prediction functions are unchanged.
