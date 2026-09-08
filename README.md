# IF Redesign core code

This repository contains reusable core code for evidence-centric multimodal
diagnosis research: Diagnostic Evidence Alignment (DEA), Counterfactual
Evidence Sufficiency Learning (CESL), case-intrinsic evidence sets (ICES),
cross-fitted diagnostic gain learning, missing-modality fusion screens,
counterfactual visual re-inspection utilities, and bootstrap-based analysis.

No data, image files, patient identifiers, trained weights, logs, result tables, figures, or manuscript material are included.

## Installation

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Run the unit tests with:

```bash
pytest -q
```

## De-identified data contract

Provide your own de-identified files at the paths declared in `configs/cesl_v2_example.json`.

- The manifest CSV contains one row per case, with at least `case_id`, `age`, `hpv`, and `tct` columns.
- The atom manifest is JSON Lines. Each row contains `id`, `patient_id`, `center`, `label`, and `atoms`; visual atoms have a slot identifier such as `colposcopy_00` or `oct_00` and an image path.
- The fold-assignment CSV contains `case_id`, `patient_id`, `center`, `fold`, and `split`, where `split` is `train`, `val`, or `test`.
- Configure `folds` with the leave-one-centre-out fold labels used by your assignment file.

Keep all input data outside version control. The provided `.gitignore` excludes inputs, feature caches, checkpoints, logs, and outputs.

## Run order

For the CESL atom-level pipeline, first extract frozen raw-atom features, then run
the resumable pipeline:

```bash
python scripts/run_cesl_feature_queue.py --config configs/cesl_v2_example.json --gpus 0 1
python scripts/run_cesl_formal_pipeline.py --config configs/cesl_v2_example.json --gpus 0 1
```

For the ICES/cross-fitted gain/missing-modality pipeline, adapt
`configs/ices_v1_exploratory.json` to your de-identified manifest, fold table,
and feature-cache locations, then run the relevant queue script:

```bash
python scripts/run_ices_feature_queue.py --config configs/ices_v1_exploratory.json --gpus 0 1
python scripts/run_ices_training_queue.py --config configs/ices_v1_exploratory.json --gpus 0 1
python scripts/launch_cross_fitted_gain.py
python scripts/launch_missing_modality_fusion.py
```

The feature extractor uses the torchvision ImageNet ResNet-50 weights and may download them on first use. For CPU-only validation, pass `--device cpu` directly to `scripts/extract_cesl_atom_features.py`.

## Diagnostic Evidence Alignment (DEA)

Read `docs/DIAGNOSTIC_EVIDENCE_ALIGNMENT.md` and `configs/dea_v1.json` before running
any DEA job. The method trains adapter-only Qwen2.5-VL / LLaVA-Med arms (`zero_shot`,
`sft`, `alignment`, `alignment_dpo`, plus the 3B `alignment_no_rank` ablation) under
five-centre LOCO with cross-fitted teacher utility, pairwise ranking alignment, and
visual DPO preference optimization.

Typical order:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/build_dea_dataset.py --config configs/dea_v1.json
.venv/bin/python scripts/prepare_dea_models.py
PYTHONPATH=src:scripts .venv/bin/python scripts/launch_dea_campaign.py --prepare
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src:scripts .venv/bin/python scripts/launch_dea_campaign.py
PYTHONPATH=src:scripts .venv/bin/python scripts/summarize_dea.py --config configs/dea_v1.json
```

Provide your own de-identified fold assignments, clinical sidecar, site labels,
cluster manifest, and cross-fitted gain targets at the paths declared in
`configs/dea_v1.json`. No data, predictions, adapters, or result tables are
stored in this repository.

## Method components

- `src/cervix_cogalign/evidence_alignment.py`: DEA prompts, chain targets, pairwise rank loss, DPO loss, and LoRA-only guardrails.
- `scripts/run_dea.py`: adapter-only SFT, alignment, DPO, and frozen evaluation for DEA arms.
- `scripts/build_dea_dataset.py`: label-independent fixed-budget site selection and preference-pair construction.
- `scripts/launch_dea_campaign.py`: resumable dual-GPU campaign queue without killing existing jobs.
- `scripts/summarize_dea.py`: patient-level metrics, ranking audits, and paired ablation summaries.
- `src/cervix_cogalign/cesl.py`: evidence atoms, outcome-blind source-only donor pools, conditional counterfactual evidence values, adaptive sufficiency stopping, and centre/stability regularisation.
- `src/cervix_cogalign/ices.py`: set-transformer evidence modeling, fixed/adaptive evidence retention masks, modality dropout, and visual-cost accounting.
- `src/cervix_cogalign/sequence_evidence.py`: OCT position-cluster parsing and within-case deletion-counterfactual sufficiency/minimality losses.
- `src/cervix_cogalign/verifiable_evidence.py`: representation grounding, necessity, information, conflict, and verifier-style evidence scoring utilities.
- `src/cervix_cogalign/counterfactual.py`: deterministic proxy ROI and finite image-perturbation helpers for engineering audits.
- `scripts/train_cesl_backbone.py`: full-evidence, selection-only, and CSS backbone training.
- `scripts/train_ices_backbone.py`: source-only ICES backbone training and predeclared module ablations.
- `scripts/run_cross_fitted_gain.py`: nested diagnostic teachers, matched out-of-fold gain labels, and frozen policy evaluation.
- `scripts/run_missing_modality_fusion.py`: unimodal, late-fusion, frozen-fusion, trainable-fusion, and modality-dropout stress tests.
- `scripts/run_csf_screen.py` and `scripts/run_csf_memory.py`: counterfactual sufficiency frontier screens and expected gain memory baselines.
- `scripts/evaluate_cesl_fold.py`: fixed-policy held-out inference and post-freeze mechanism audits.
- `scripts/run_cesl_availability_stress.py`: missing-modality and reduced-evidence stress tests.
- `scripts/run_cesl_statistics.py`: paired, patient-clustered bootstrap inference and multiplicity adjustment.

This code is for research use. Validate all data processing, endpoint definitions, and governance requirements before any clinical or operational use.
