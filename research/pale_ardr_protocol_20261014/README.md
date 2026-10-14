# Fixed PALE/ARDR source-only execution adapters

These source snapshots run one shared forward model under six training losses.
All arms use the same FR1 source-centering/gray-reanchoring correction,
source-fitted readout scales, frozen patient anchor, source splits, seeds,
initialization and prediction-head capacity. ARDR is an optional validation
comparison and is not enabled as the default main method.

The portable API is `src/if_redesign/pale_ardr`. The bundled `code` directory
preserves the exact head dependencies used by this trial. These clinical
adapters are not a standalone example: they require caller-owned frozen
Stage197/199 caches, checkpoints, a patient loader `scripts/rime_dataset.py`,
and separately provided private manifests. Copy this directory under the
caller's `03_new_experiment_design` root so `runtime.py` can resolve the
existing private dependency tree. Do not execute it directly in this public
tree or against a published clinical cohort without a separately frozen plan.

Copy `protocol_defaults.json` to a private `CONFIG.json`, remove the
`caller_required_fields` documentation key, and supply the actual centre names,
pilot outer/held-source centres and unchanged campaign deadline. Required
private files are `INPUT_LOCK.json` (including `cohort_path`, `cohort_sha256`
and `legacy_expected_metrics`) and `EXPERIMENT_MANIFEST.json` with
`execution_files` mapping dependency paths to SHA256 digests. Verify those
digests before execution; this public snapshot does not generate a clinical
input lock or authorize changing an existing one.

Execution order is `test_preflight.py`, `audit.py`, then `controller.py`.
The controller checks mathematical/gradient and legacy-reproduction gates,
runs the fixed source-only pilot, selects candidates entirely within each
outer source partition, and queues outer refits under the fixed deadline.
It resumes already completed work instead of refitting it. `summarize.py`
reports every seed, centre and arm, paired centre/label-stratified bootstrap,
anchor-risk changes and CIN3+ known/unknown denominators. `collect_evidence.py`
adds measured fitting cost and active-penalty diagnostics after completion.

The public audit differs only by taking historical expected scores from the
caller-owned lock and deriving patient/endpoint counts from the loader.
Reporting prose refers to historical scores without embedding study results.
The defaults omit server timestamps and centre names; all statistical and
optimization rules are retained. No patient records, fitted model state,
cache, prediction aliases or alias salt, clinical result table, execution log
or manuscript accompanies these sources. Creating bootstrap reports does not
make the repeatedly used cohort an independent confirmation cohort.
