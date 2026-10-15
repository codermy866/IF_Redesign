# Matched PALE/P4 execution source snapshot

This directory preserves the latest lightweight head, source-only training,
selection/refit, baseline adaptation, bootstrap-statistic and missing-observation
execution code. `code/` preserves the actual head dependencies. The primary
configuration remains P3; P4 is a prespecified no-frame-adapter comparison.
The portable, cache-free P4 entry point is `if_redesign.pale_matched.make_p4_head`.

These are private-environment adapters, not a self-contained clinical dataset.
To review the whole pipeline, start with `heads.py`, `training.py`, `data.py`,
`worker.py`, `preflight.py`, then `recent_models.py` / `recent_training.py`.
`robustness.py` holds the anchor fixed; `native_robustness.py` and
`native_summary.py` separately recompute the affected native context. `report.py`
and `recent_report.py` contain statistical calculations, not published results.

To execute on an authorized private installation, place this snapshot at its
original `03_new_experiment_design/201_PALE_MATCHED_VALIDATION_20261014` location.
`runtime.py` intentionally resolves that private directory layout. It requires
Stage196/197/199/200 caches and source partitions, the root `scripts/rime_*.py`,
private cohort/index manifests, alias salt and immutable execution manifests.
Copy `protocol_defaults.json` to private `CONFIG.json`, and `recent_defaults.json`
to `RECENT_CONFIG.json`; remove `caller_required_fields`, replace placeholder
centre names and supply the original authorized deadline. Generate private
`EXPERIMENT_MANIFEST.json` and `RECENT_MANIFEST.json` dependency hashes rather
than reusing host-specific paths. No expired server deadline is silently reset.

Author source is deliberately not vendored. `vendor_revisions.json` records four
exact upstream Git commits; `python fetch_vendor.py` retrieves them. Preserve
upstream attribution and licenses. Recent author adapters additionally require
`timm`, `einops`, and `torch-geometric` in a compatible environment. Native JEV
inference requires the separately installed released model and the repository's
JEV dependency environment. No training is started by importing the portable
package or by running the synthetic example.

The source excludes patient records, model weights, cached features, trained
states, logs, actual results, figures, manuscripts and the manuscript-export
finalizer. The orchestration sources require an independently prepared private
plan; they are not evidence that all queued comparisons have completed.
