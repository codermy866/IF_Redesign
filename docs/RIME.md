# RIME-JEV implementation

This module exposes the current numerical method through caller-owned arrays.
It includes no cohort, trained parameters, experiment outputs, or manuscript.
The kernel QP and hierarchical readout match their experimental sources.
The CPU training loop preserves its objective and optimizer; its optional
CAPR-MIL import is replaced by an explicit caller-supplied block factory.

## Frozen frame features

```python
from if_redesign.rime.frame_features import FrozenFrameJEV

model = FrozenFrameJEV("/path/to/licensed/jev27b-bundle", device=0)
hidden_contrast, binary_logit_contrast, parity = model.extract(
    "/path/to/caller-owned/oct-frame.png", parity=True,
)
```

Use the separately installed `jev27b` extra described in the root README.
There is no automatic model download. The loader verifies the released
adapter tensors. The frame extractor consumes image paths and no labels.
It keeps the existing reread prompt with one frame in image slot zero and
19 neutral gray views. Hidden contrasts subtract the all-gray task reference.
The vision encoder processes the canonical 20-image batch; using a smaller
vision batch can change BF16 rounding. Optional parity checks compare cached
encoder-output inference with direct-pixel inference at tolerance 1e-5.

## Fit source preprocessing

```python
from if_redesign.rime import fit_projection, project, source_bandwidth

projection = fit_projection(hidden, valid, source_train_indices)
h = project(hidden, valid, projection)
sigma = source_bandwidth(h, valid, source_train_indices)
```

`hidden` has shape `[patients, positions, frames, hidden_dimension]` and must
already subtract the gray reference. `valid` is its Boolean observation mask.
Zero placeholders never count as observations. Basis fitting uses centered
source covariance with equal patient influence and equal valid-position
influence within each patient. The eight PCA coordinates keep deterministic
loading signs. Inference retains the gray origin rather than subtracting the
fitting mean; source scales have floor 0.005 and coordinates are clipped to
[-3,3] and divided by three. Bandwidth is the median of patient-wise medians
of nonzero within-position distances, with floor 0.001. Fit every basis,
bandwidth, clinical preprocessing state, and coordinate scale again within
each inner source-training partition.

## Measure and readout

The conditional score is `tanh(a' h + (h' U) (b' V))`, with rank-two context
interaction. Exact duplicate coordinates are merged within each position.
The simplex QP minimizes `0.5 w' K w - tau q' w`, where K is the Gaussian
kernel. No entropy or instance-level diagonal regularizer is added.
`measure.py` provides a float64 active-set solve, KKT checks, an independent
SLSQP reference, and implicit derivatives with respect to conditional scores.
Kernel and preprocessing parameters stay frozen during head training.

Each coordinate contributes its mean and positive/negative Laplace responses
at temperatures 0.5, 2, and 8, giving 56 position coordinates. The comparator
uses zero-anchored Legendre degrees one through seven with the same size.
Source coordinate scales are fitted on the original observations before
deduplication or changing the pooler. Shared position attention produces the
patient residual. The prediction is `anchor + residual @ beta` with beta
initialized to zero. Empty OCT evidence returns the original anchor exactly.

```python
from if_redesign.rime import prepare_inputs, train_head

train = prepare_inputs(
    h, valid, context, anchor_logits, sigma,
    source_indices=source_train_indices, indices=source_train_indices,
    pooling="measure", readout="laplace",
)
selection = prepare_inputs(
    h, valid, context, anchor_logits, sigma,
    fitted_scale=train["coordinate_scale"].cpu().numpy(),
    indices=source_selection_indices,
    pooling="measure", readout="laplace",
)
head, metadata = train_head(
    train, selection, source_targets, centers,
    "measure", "laplace", tau=0.3, lam=0.003, seed=17,
)
```

Source targets are a patient-aligned array indexed by each input dictionary's
`indices`; only training and source-selection targets may be available.
Use disjoint source-training and source-selection patients. Outer targets
must remain unavailable during fitting and selection. At evaluation, reuse
the selected head, frozen source projection, bandwidth, context preprocessing,
and coordinate scale with the same pooling/readout settings.

`prepare_inputs` merges exact duplicates for measure/geometry modes. Calling
`RIMEHead` directly requires the same preparation; the solver alone is not
a deduplication API. Reuse frozen scales when testing copies: recomputing
scales changes the experiment. Distinct highly correlated frames are not
automatically identical atoms. Singular or nonconvergent QPs raise errors
rather than introduce a silent objective regularizer.

## Anchor and fair comparisons

The complete anchor contains a source-fitted clinical prior, source-selected
native patient decision contrast, and a 26-coordinate semantic predictor.
Context contains those semantic coordinates plus source-preprocessed clinical
variables. Both are supplied by the caller. The original anchor uses
four-frame OCT montages [0,3,6,9], whereas the new local branch encodes each
raw frame independently. Keep that distinction when integrating the APIs:
the existing single-frame patient adapter is a different input protocol.
This package does not recreate a private fitted anchor or its predictions.

The equal-capacity factorial compares attention/moment, attention/Laplace,
measure/moment, and measure/Laplace. Other exposed poolers are mean, an MLP
with at least the selector's effective coefficient count, geometry-only QP,
identity-kernel QP, dual-ridge GMP, and semantic-head refitting. For GMP, tau
is a ridge coefficient; for attention it is a temperature; for QP it scales
conditional scores. The refit control accepts `semantic_x` and
`initial_semantic_weight`, penalizes effective coefficients, and preserves the
complete initial anchor. Identity pooling can retain or merge copies through
the explicit `deduplicate` argument. `atoms(..., kind="cluster")` exposes
the complete-linkage medoid control.

CAPR-MIL is an optional adapted comparator, not a bundled RIME dependency.
For `pooling="caprmil"`, supply `caprmil_factory` to `train_head`; it must
return a compatible separately licensed block of width eight. The block sees
only available frames, with no artificial spatial coordinates. This API does
not reproduce the original paper's full benchmark or training protocol.

Training uses centre-equal BCE, weight-0.1 within-centre positive/negative
pair loss, and L2 on effective trainable coefficients. Adam uses learning
rate 0.003, gradient clipping at one, maximum 300 epochs, and patience 30.
Epoch zero is a genuine source-selection candidate. Select at most nine
tau/lambda combinations using source-only inner CV and fixed seeds
17/29/43; ensemble mean probabilities determine pooled AUROC, then NLL,
stronger lambda, and lower tau. Refit for the median source-selected epoch
without outer validation (`validation=None`, `fixed_epochs=...`). The public
trainer fits one head; multi-fold scheduling and patient-aligned ensembles
are caller responsibilities. Retain all seeds and negative outcomes.

Replication tests establish a numerical property conditional on fixed
context, not improved diagnosis. Deleting evidence requires recomputing the
actual patient anchor/context, conditional scores, measure, and position
attention. Fixed-context head deletion is only an approximation. Patient-level
performance, clinical safety, and independent confirmation are not established
by this implementation or its synthetic example.
