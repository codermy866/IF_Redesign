# IF_Redesign: RIME-JEV evidence measure learning

The latest module is **RIME-JEV**: independent raw-frame JEV hidden contrasts,
source-fitted eight-coordinate projection, exact-atom kernel measure pooling,
signed 56-coordinate Laplace readout, and a patient residual over a complete
caller-supplied clinical/native/semantic anchor. The local evidence branch
merges exact copies within each scanner position before solving its conditional
simplex QP. Ordinary attention combines scanner positions.

Run its data-free CPU example after installation:

```bash
python scripts/demo_rime.py
```

See [RIME API and implementation boundaries](docs/RIME.md) and
[defaults](configs/rime.json). The implementation is an experimental candidate;
the new patient-level evaluation is still underway. Replication invariance is
conditional on frozen features, scales, and patient context. It does not assert
that an independently recomputed upstream anchor is invariant.

## Native Qwen reference implementations

The repository also provides source-only medical task adaptation on the native
Qwen multimodal backbone loaded from the pinned **autotrust/JEV-27B-VL**
release. It includes raw OCT frame preparation, patient/position task training,
label-free feature extraction, and clinical-anchored patient readouts.

This is an implementation release of an ongoing research baseline. It is not
an assertion of a final method, demonstrated algorithmic novelty, or completed
independent confirmation. See [method details](docs/METHOD.md) and
[implementation status](docs/EXPERIMENT_STATUS.md).

## Install and run a CPU example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/demo_qwen27b_readout.py
python -m pytest -q
```

The example creates synthetic features and labels in memory, fits all six
current readouts on source data, selects using a separate source partition,
and predicts on synthetic held-out observations. It checks exact clinical
fallback when visual evidence is absent. Its metrics are demonstration output.

## RIME modules

| Module | Function |
| --- | --- |
| `rime/frame_features.py` | Independent-frame frozen JEV extraction with direct-pixel parity check |
| `rime/preprocessing.py` | Source projection/scaling, exact atom merging, and portable tensor inputs |
| `rime/measure.py` | Simplex kernel QP, independent numerical reference, and implicit score gradients |
| `rime/readout.py` | Equal-size Laplace/moment coordinates and hierarchical residual controls |
| `rime/training.py` | Centre-balanced BCE/ranking training and source epoch selection |
| `scripts/demo_rime.py` | Synthetic four-cell factorial example with held-out targets excluded |

## Native Qwen modules

| Module | Function |
| --- | --- |
| `qwen27b/native.py` | Local released-model loading, binary head and loaded-adapter equality checks |
| `qwen27b/adapter.py` | Native vision features, 20 image slots, raw-frame transforms, restricted medical LoRA |
| `qwen27b/training.py` | Source-center/class-balanced 512-update patient or alternating dual-task training |
| `qwen27b/features.py` | Patient hidden contrast, local-task contrasts and colposcopy-only contrast |
| `qwen27b/readout.py` | Current 32-coordinate moment/focal readouts and within/all-center pair losses |
| `qwen27b/interventions.py` | Frozen released/bare model view replacements for reference readouts |
| `qwen27b/semantic_readout.py` | Completed 32-coordinate semantic/evidence reference families |
| `qwen27b/evidence_readout.py` | Completed evidence controls, including the 18-coordinate PCA linear reference |
| `configs/qwen27b_adaptation.json` | Current adaptation and readout defaults |

The three numerical readout modules preserve the experimental source code.
Private paths, manifests and experiment scheduling are replaced by explicit
caller inputs. Previously published CFV-JEV, SOPI and G2-PCGrad code remains
available as historical reference.

## Connect the actual 27B bundle

CUDA training uses the separately installed optional dependencies:

```bash
python -m pip install -e '.[jev27b,test]'
```

Obtain the separately licensed release from
[ModelScope](https://modelscope.cn/models/autotrust/JEV-27B-VL).
The recorded base identity is `Qwen/Qwen3.8-27B`; the loading release revision
is `fee8c1d19d60216f8e9bcbb348141c0aa87ee2cc` and the base revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Supply a local bundle with the
backbone/processor files, `adapter_vllm/adapter_config.json`,
`adapter_vllm/adapter_model.safetensors`, `adapter_vllm/decision_head.json`,
and `calibration.json`. No weights are bundled or automatically downloaded.
The loader verifies released versus loaded adapter tensors; the caller must
also obtain the backbone from the recorded revision.

```python
from if_redesign.qwen27b import MedicalQwen, train_source_adapter

model = MedicalQwen("/path/to/licensed/jev27b-bundle", device=0)
result = train_source_adapter(
    model, image_manifest, separate_training_targets,
    source_train_keys, source_selection_keys,
    outer_center=heldout_center,
    variant="DUAL_TASK",  # or PATIENT_ONLY
    output_dir="outputs/source-adapter",
    seed=17,
)
```

The image manifest maps caller-owned keys to `oct_sites` (string positions
1-12, ten ordered raw frame paths each) and `colpo_paths` (up to eight).
Targets always contain `center` and binary patient pathology `y`.
Patient-only training needs no weak-label metadata. Dual-task training also
uses twelve weak `reread` labels (`-1` unresolved) and `reread_valid`; unusable
weak targets and missing image positions are omitted from that task's sampler.
Source training and source selection must use disjoint patients and exclude
the held-out center. The source-selected checkpoint is restored on return.

Only released language attention q/v LoRA tensors in layers 48-63 are
continued; the pinned architecture has 16 eligible tensors and 3,014,656
trainable parameters. Native vision and base weights remain frozen. The
FROZEN control uses the released model without medical updates. Patient-only
and dual-task runs share 512 updates and 20 image slots; prompt token counts
differ, so this is not an exact FLOPs match.

Missing OCT positions retain neutral slots. Short frame lists cannot cause an
indexing error: patient inference wraps the frame index over available paths,
while a reread bag retains gray slots for unavailable selected indices.
The reference data protocol still uses ten ordered frames per position.
Audit image readability before building a manifest; invalid image files are
not silently treated as valid observations.

## Fit the current patient readout

```python
from if_redesign.qwen27b import extract_patient_features, stack_features
from if_redesign.qwen27b import readout

train_rows = [extract_patient_features(model, image_manifest[k])
              for k in source_train_keys]
selection_rows = [extract_patient_features(model, image_manifest[k])
                  for k in source_selection_keys]
train = stack_features(train_rows, source_train_clinical_logits)
selection = stack_features(selection_rows, source_selection_clinical_logits)
state, heads, scores = readout.train_select(
    train, source_train_y, source_train_centers,
    selection, source_selection_y, source_selection_centers,
)

# Freeze the state/head before extracting/evaluating held-out data.
head = heads[source_selected_family]
heldout = stack_features(heldout_rows, heldout_clinical_logits)
heldout_logits = readout.predict(head, heldout, state)
```

Feature extraction receives image paths, not outcome labels. Clinical logits
must come from a separately source-fitted clinical prior. Projection bases and
scales use source training only. Source selection chooses regularization;
held-out outcomes enter neither operation. Cache features separately for each
adapter and preprocessing identity. Patient loading, cache management and
multi-fold scheduling remain caller responsibilities.

Local reread labels describe imaging phenotype, not frame pathology or CIN2+
localization. Repeat-bag disagreement is an experimental discount, not a
calibrated uncertainty estimate. Clinical anchoring has exact zero-evidence
fallback but no hard safety bound. No clinical causal interpretation is claimed.

Patient records, images, fitted weights, predictions, study results,
manuscripts and figures remain outside this release.
