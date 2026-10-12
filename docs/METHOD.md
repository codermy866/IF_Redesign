# Native Qwen task adaptation and patient readouts

## Upstream and restricted adaptation

The caller supplies the pinned JEV 27B release and its native Qwen vision
encoder. `NativeJEV` loads NF4 double quantization with BF16 computation,
the released `adapter_vllm`, binary false/true verbalizers and released bias.
It checks equality of every loaded adapter tensor. `MedicalQwen` permits only
language q/v LoRA A/B tensors in layers 48-63 to receive gradients. Vision and
base weights stay frozen.

The policy uses 512 AdamW updates (learning rate 3e-5, weight decay 0.01),
gradient clipping at 1, and the following loss:

```text
binary CE(logits + released bias, task target)
  + 0.01 * mean_parameter_squared_distance_from_released_adapter
```

The published general-purpose temperature is not applied to medically adapted
logits. Patient-only training uses 512 pathology updates. Dual-task training
alternates 256 pathology and 256 position-reread updates. Both sample source
centers and classes uniformly. Checkpoints 128/256/512 are selected by source
selection AUROC, then center AUROC and earlier step. Adapter and final patient
readout are fitted sequentially; this is not joint end-to-end readout training.

## Inputs and task semantics

Prompts have 20 image slots: 12 OCT and 8 colposcopy. Neutral slots are RGB
gray. Raw OCT frames retain aspect ratio with shorter edge 256 and center
crop 224. Colposcopy images are resized to 224 square. Patient training samples
one of ten frames per position; inference uses index 4. Position-reread uses
four frames from one position in the first four slots, with other slots gray.
Two fixed bags use indices [0,3,6,9] and [1,2,5,8]. File order is not verified
temporal or anatomical geometry. Local targets apply to bags, not each frame.

The prompts distinguish structural imaging abnormality from patient CIN2+
pathology. No pathological position target or cross-modality pixel alignment
is assumed. Labels are separate from the image manifest and unavailable during
inference feature extraction.

## Current coordinates and prediction

`extract_patient_features` computes the patient hidden vector minus its
all-gray reference; patient decision contrast A; colposcopy-only contrast K;
and two local-task contrasts per scanner position against a separate gray
reread reference. The 32-coordinate readout contains 16 semantic coordinates,
14 local coordinates and A/K.

The semantic basis is learned from source-centered hidden vectors. Inference
does not subtract their mean, so gray evidence stays zero. Local coordinates
are signed mean, absolute mean, standard deviation, second/third/absolute-third
moments, six positive/negative moment or normalized log-sum-exp coordinates,
and positive/negative maxima. FOCAL temperatures are 0.5/1/2. RELIABLE_FOCAL
divides the averaged local response by
`1 + abs(response0-response1)/source_disagreement_Q90`. This discount is a
candidate heuristic; disagreement may also reflect genuine heterogeneous
findings and does not estimate calibrated uncertainty.

Local Q90 scales have floor 0.05 and normalized values are clipped to [-8,8].
Final source coordinate standard deviations have floor 0.005; coordinates
are clipped to [-16,16]. The prediction is

```text
z = clinical_logit + 0.25 * dot(coordinates, w)
```

The head has 32 coefficients and no intercept. No visual evidence gives
exact clinical fallback. This linear residual is unbounded: shrinkage does
not provide a hard safety guarantee.

## Ranking and selection

WITHIN constructs positive/negative patient pairs inside each source center.
ALL_PAIRS also includes ordered cross-center blocks. Each nonempty block has
equal weight, with at most 2048 pairs and RNG seed 195. The objective is average
pair-logistic loss plus 0.1 center-equal BCE plus 0.5 * lambda * squared weight
norm. Lambda in {0.01,0.1,1,10} is selected by source-selection pooled AUROC,
then lower NLL and stronger lambda.

Three geometries and two losses yield six readouts for each of three upstream
conditions: frozen released, patient-only adapted and dual-task adapted.
Patient-level source splits and multi-fold selection/locking remain caller
responsibilities. No held-out outcomes are accepted for fitting or selection.

## Completed numerical references

`InterventionJEV` retains released and bare-LM branches. Only the frozen
released branch uses its published temperature. Cached vision replacement is
checked against direct-pixel replacement with a direct fallback; language
prefixes are recomputed. `stack_interventions(..., branch=0)` chooses the
released branch and branch 1 the bare branch. Supply the same fixed input views
used to derive a reference: raw-frame adaptation and historical montage
preprocessing are different input protocols.

`semantic_readout.py` contains completed 32-coordinate semantic/evidence
families, including CENTERED32__ROBUST. It uses within-center ranking and mean
or log-sum-exp center risk, with source-center CV for lambda.
`evidence_readout.py` retains bounded evidence variants and scalar/PCA controls.
PCA_LINEAR is an uncentered 18-coordinate linear residual with coefficient
scale 1, distinct from the current 32-coordinate shrinkage-0.25 readout.
The three numerical readout modules are unchanged from their experimental
sources. Reproducing private predictions also requires the original partitions,
clinical priors, preprocessing and fitted parameters; these are not released.

Previous CFV-JEV equations remain in [historical documentation](HISTORICAL_CFV_JEV.md).
