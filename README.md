# CESL core code

This repository contains the reusable research code for Counterfactual Evidence Sufficiency Learning (CESL): feature extraction, source-only counterfactual donor construction, backbone training, frozen held-out inference, availability stress tests, and bootstrap-based analysis.

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

First extract frozen raw-atom features, then run the resumable pipeline:

```bash
python scripts/run_cesl_feature_queue.py --config configs/cesl_v2_example.json --gpus 0 1
python scripts/run_cesl_formal_pipeline.py --config configs/cesl_v2_example.json --gpus 0 1
```

The feature extractor uses the torchvision ImageNet ResNet-50 weights and may download them on first use. For CPU-only validation, pass `--device cpu` directly to `scripts/extract_cesl_atom_features.py`.

## Method components

- `src/cervix_cogalign/cesl.py`: evidence atoms, outcome-blind source-only donor pools, conditional counterfactual evidence values, adaptive sufficiency stopping, and centre/stability regularisation.
- `scripts/train_cesl_backbone.py`: full-evidence, selection-only, and CSS backbone training.
- `scripts/evaluate_cesl_fold.py`: fixed-policy held-out inference and post-freeze mechanism audits.
- `scripts/run_cesl_availability_stress.py`: missing-modality and reduced-evidence stress tests.
- `scripts/run_cesl_statistics.py`: paired, patient-clustered bootstrap inference and multiplicity adjustment.

This code is for research use. Validate all data processing, endpoint definitions, and governance requirements before any clinical or operational use.
