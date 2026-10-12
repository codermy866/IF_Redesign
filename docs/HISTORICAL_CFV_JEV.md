# CFV-JEV implementation

CFV-JEV combines a frozen patient predictor, view-replacement sensitivity,
signed empirical evidence coordinates, and a clinical-anchored residual.
The public package provides reusable computation; data loading, model weights,
and the private experiment scheduler are not included.

## Fixed predictor and view replacement

Let `c` be the clinical-reference logit and `h(X,c)` a fixed multimodal
readout. Its baseline-referenced logit is

```text
b(X,c) = c + visual_scale * [h(X,c) - h(X_empty,c)]
delta_j = b(X,c) - b(X_without_j,c)
```

`X_empty` contains only neutral images. `X_without_j` replaces view `j` with
an RGB `(128,128,128)` image while keeping its slot, image dimensions, prompt,
and question count. The reference layout has 12 OCT and 8 colposcopy slots;
missing slots remain neutral and are masked from evidence aggregation.
`extract_interventions` accepts the views, their validity mask, a clinical
logit, and a frozen scorer. `visual_scale` carries the source-selected base
coefficient divided by its temperature.

The predictor stays frozen. Each replacement reruns the language prefix and
patient readout. Deterministic frozen vision encodings may be cached after
comparison with direct pixel execution; language caches cannot be reused
across changed image inputs. Extraction accepts no outcome labels.

## Evidence coordinates

`encode_evidence` takes sensitivities and a matching validity mask. For each
modality, it uses the valid-view signed mean and six coordinates for each
sign. With `u_j = clip(delta_j,0,1)`, `v_j = clip(-delta_j,0,1)`, and
`t = geomspace(0.5,16,6)`, these additional coordinates are

```text
mean_valid(1 - exp(-t * u_j))
mean_valid(1 - exp(-t * v_j))
```

This gives 13 coordinates per modality and 26 total. Aggregation uses the
actual valid-view count; an empty modality contributes zeros. `RAW`,
`FOURIER`, `POWER`, and `DUP` encodings provide the same output dimension.
The Laplace coordinates are empirical transforms of model sensitivities,
not a fitted Laplace probability distribution or calibrated uncertainty.

## Residual fitting and inference

`fit_raw_readout` fits the 140-parameter baseline RAW patient readout from
real and paired all-neutral source-training option logits. It standardizes
33 coordinates (32 modality descriptors plus the global option-logit
difference), applies four exact-GELU hidden units and their clinical-logit
interaction, and trains `c+r(real,c)-r(neutral,c)` with center-equal BCE plus
L2. Separate source-selection inputs choose L2 from `{0.01,0.1}`, base rho
from `{0.5,1}`, and temperature from `{1,1.5}`. The returned
`RawPatientReadout` reports unanchored `r`; intervention extraction performs
the neutral subtraction and clinical anchoring exactly once.

`fit_residual` takes separate source-training and source-selection arrays.
Training standard deviations scale the coordinates with a floor of `0.005`.
Scaled values are clipped to `[-16,16]`, without mean subtraction. A
26-parameter linear residual has no intercept:

```text
z = b + 0.25 * dot(clip(x / std_train,-16,16), w)
p = sigmoid(z)
```

`ResidualHead.predict_logit` applies this frozen transformation. Training and
inference use the same `0.25` coefficient. The objective is centre-equal
binary cross-entropy plus L2. The risk-regularized version adds the centre
mean of `max(NLL(z)-NLL(b)-0.001,0)`. L2 is selected from `{0.003,0.03,0.3}`
using source-selection centre-equal NLL. Held-out evaluation data do not
enter fitting or selection.

With no valid visual evidence, `x=0`; the neutral baseline gives `b=c`, so
the final logit is exactly `c`. The risk term is a soft penalty and does not
guarantee that every centre improves. View replacement measures model
decision sensitivity and does not establish a clinical causal effect.

## Backbone provenance and historical modules

The full-image integration uses the external Visual Jev shared-context
question interface with a Qwen3-VL backbone. See the
[original paper](https://arxiv.org/abs/2609.25845) and
[upstream source](https://github.com/guanxuyu-sv/Visual-Jev).
Install and license the upstream software and checkpoints separately.
`RawPatientReadout` and `VisualJEVAdapter` connect the public core to that
interface; patient data loading remains the caller's responsibility.

`stagewise.py` and `sopi.py` are historical stagewise-fusion and retrospective
proper-score utilities. The G2-PCGrad modules and configuration are also
retained as historical references. They are not required components of the
current CFV-JEV residual.
