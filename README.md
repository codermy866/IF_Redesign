# IF_Redesign: CFV-JEV

The latest method in this repository is **CFV-JEV**: a clinical-anchored
patient predictor corrected using the signed effects of replacing individual
image views with neutral images. The implementation includes the frozen
backbone training policy, RAW patient-readout fitting, complete-input
interventions, 26-dimensional evidence encodings, and source-only residual
fitting. Synthetic examples run on a CPU.

The full-image frozen-protocol evaluation is ongoing. This code release is
an implementation reference; evidence for improved performance and independent
confirmation remains pending. See [implementation status](docs/EXPERIMENT_STATUS.md)
and [method details](docs/METHOD.md).

## Install and run the synthetic example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/demo_cfv_jev.py
python -m pytest -q
```

The example generates synthetic views and labels in memory. It runs the
intervention, source fitting, parameter serialization, and held-out prediction
chain, including the exact zero-evidence clinical fallback. Its scores are
demonstration output.

## Current modules

| Module | Function |
| --- | --- |
| `cfv_jev/backbone_training.py` | Source-only language-adapter training; frozen modality policy and budget controls |
| `cfv_jev/visual_jev.py` | Fresh 20-slot / 21-question inference; portable 140-parameter RAW patient readout |
| `cfv_jev/readout_training.py` | Source-center-balanced RAW readout fitting and source-selected update scale |
| `cfv_jev/interventions.py` | Full-input neutral replacement for each valid view, with a fixed clinical logit |
| `cfv_jev/evidence.py` | LAP, RAW, POWER, FOURIER, and DUP evidence encodings with identical 26-coordinate capacity |
| `cfv_jev/training.py` | Bias-free residual fitting, source-only standard deviations, fixed shrinkage, soft center-risk penalty |
| `configs/cfv_jev.json` | Current method defaults |

The shared numerical interface is:

```python
from if_redesign.cfv_jev import extract_interventions, fit_residual

# scorer(views, clinical_logit) returns the same frozen, unanchored readout.
result = extract_interventions(
    ordered_views, valid_mask, clinical_logit, scorer,
    visual_scale=source_selected_scale,
)
evidence = result.evidence("LAP")  # 26 coordinates

# Each tuple is (evidence, base_logits, labels, source_center_ids).
head = fit_residual(*source_training_arrays, *source_selection_arrays)
final_logit = head.predict_logit([result.base_logit], evidence[None])[0]
```

The ordered input has twelve OCT slots followed by eight colposcopy slots.
Each replacement preserves the slot and recomputes the full language prefix.
Missing slots are neutralized before scoring and excluded from aggregation.
Evidence extraction accepts no pathology labels. Labels enter source training
and source selection only; the caller must maintain patient-level separation
and source provenance.

## Connect a real Visual-JEV model

Install the separately maintained [Visual Jev upstream implementation](https://github.com/guanxuyu-sv/Visual-Jev)
and compatible model dependencies; optional dependencies here can be installed
with `python -m pip install -e '.[visual]'`. The upstream shared-context
question interface is described in its [original paper](https://arxiv.org/abs/2609.25845).
Third-party source and checkpoint licenses apply to their respective materials.

Supply a compatible Qwen3-VL execution model initialized with the selected
language adapter, the source-fitted clinical prior, and fitted readout/head
parameters. A bare pretrained backbone is insufficient to reproduce the
trained method. A generic inference connection is:

```python
from if_redesign.cfv_jev import RawPatientReadout, VisualJEVAdapter

readout = RawPatientReadout.from_dict(readout_parameters)
adapter = VisualJEVAdapter(selected_model)
result = extract_interventions(
    ordered_views, valid_mask, clinical_logit, adapter.scorer(readout),
    visual_scale=readout.effective_scale,
)
probability = head.predict_proba([result.base_logit], result.evidence()[None])[0]
```

`RawPatientReadout` and `ResidualHead` provide JSON-compatible `to_dict` and
`from_dict` methods. The RAW readout is fitted with its paired all-neutral
reference; only the visual update is scaled, and the clinical logit is added
once. The 26-parameter correction has no intercept or feature mean subtraction,
and uses shrinkage `0.25` during both fitting and prediction.

Fit the RAW readout using separate source-training and source-selection
tuples, each containing `(option_logits, neutral_option_logits, clinical_logits,
labels, center_ids)`:

```python
from if_redesign.cfv_jev import fit_raw_readout

fitted = fit_raw_readout(*source_raw_training_arrays, *source_raw_selection_arrays)
readout = fitted.readout
```

Option logits have shape `(N, 21, 2)`. Neutral logits may be per patient or
a shared `(21, 2)` array when neutral inputs are identical. Scaling parameters
are fitted on source training only; source selection chooses L2 and the
visual update scale. Evaluation data enter neither stage.

For source-only backbone training, `BackboneTrainingPolicy` and
`train_source_backbone` implement the frozen 512-step schedule. They preserve
all twenty slots while sampling 50% full input, 40% neutral OCT, and 10% neutral
colposcopy. Patient-answer supervision uses whole-vocabulary cross entropy;
view questions receive no patient pathology labels. The vision encoder stays
frozen and source-selection NLL chooses among trained checkpoints every sixteen
steps. Step zero remains a diagnostic reference. Caller-supplied model loading
and source data assembly remain environment-specific.

```python
from if_redesign.cfv_jev import (
    BackboneTrainingPolicy, configure_language_lora, train_source_backbone,
)

policy = BackboneTrainingPolicy(seed=17)
model.backbone = configure_language_lora(
    model.backbone, policy, released_adapter=upstream_answer_adapter_path,
)
# Each source item is TrainingPatient(key, twenty_views, binary_label).
training_result = train_source_backbone(model, source_train, source_selection, policy=policy)
```

`configure_language_lora` also supports fresh initialization for explicitly
different training runs. The reference method initializes from the upstream
answer adapter before its source training.

For matched content/budget controls, `BackboneTrainingPolicy(modality="O")`
or `modality="K"` permanently neutralizes the other modality while retaining
the same augmentation and budget. These controls can include all-neutral
training inputs; their counts are reported. LAP without the soft risk penalty
is obtained through `fit_residual(..., risk_penalty=False)`, and equal-capacity
representation controls through `encode_evidence(..., representation=...)`.

## Scope and earlier components

This tree contains core code, configuration, documentation, and synthetic
tests. Patient records, images, fitted weights, experiment outputs, and
manuscript materials remain outside the release. The empirical Laplace
coordinates summarize predictor sensitivity; they do not establish a
clinical causal effect or a calibrated uncertainty distribution. The
center-risk term is a soft penalty, without an every-center safety guarantee.

Existing SOPI, stagewise prior-anchoring, and G2-PCGrad modules remain as
historical reference implementations. They are separate from the current
CFV-JEV computation chain.
