# Latest method code map — 2026-10-15

This is a code snapshot for method review, not a declaration of a newly selected
winner or completed independent validation. The locked matched study still uses
P3 as its primary arm. P4 is the simpler candidate discussed with the author.
No clinical performance table accompanies this release.

| Component | Code |
| --- | --- |
| Portable P4 and matched alternatives | `src/if_redesign/pale_matched/heads.py` |
| P4 constructor and fixed specification | `src/if_redesign/pale_matched/__init__.py` |
| Independent, data-free P4 example | `scripts/demo_p4.py` |
| Original PCA26 fit and transform | `research/stage196_semantic_reference_20261012/run_factorial.py`, functions `semantic_state`, `semantic_features` |
| Importable exact PCA26 functions | `src/if_redesign/pale_matched/semantic.py` |
| Original patient-level native hidden contrasts | `research/stage196_semantic_reference_20261012/native_worker.py`, `qwen27b/interventions.py` |
| Independent OCT frame extraction | `scripts/rime_frame_worker.py` |
| Frame PCA and source-only anchor construction | `scripts/rime_dataset.py` |
| Corrected source anchor adapter | `research/pale_input_preparation_20261013/anchor_adapter.py` |
| Shared FR1 fold transformation | `research/pale_input_preparation_20261013/fr1_fold_data.py` |
| Portable FR1 transformation | `src/if_redesign/pale_ardr/preprocessing.py` |
| Exact matched training/selection workers | `research/pale_matched_validation_20261014/` |
| Optional ARDR loss comparison | `src/if_redesign/pale_ardr/`, `research/pale_ardr_protocol_20261014/` |

## P4 prediction path

Frozen released JEV image features and a caller-owned clinical/native/semantic
patient anchor are retained. Independently extracted 5120-dimensional frame
contrasts are mapped to eight source-fitted FR1 coordinates. For each coordinate,
the readout stacks the identity response and `1-exp(-t*relu(sign*h))` for signs
+/- and t=0.5,2,8. This yields 56 frame coordinates. Masked within-site means,
fixed source-fitted coordinate scales, shared site attention and a linear
residual produce `anchor + sum_site(alpha * v) @ beta`.

P4 has no frame MLP or its outer tanh, no kernel-QP frame weighting, no CFV
value estimator and no ARDR loss by default. Its effective trainable local head
contains two 56-dimensional vectors (112 parameters); inherited inactive selector
parameters remain frozen. Empty local evidence returns the supplied anchor.
This is conditional on a fixed anchor; it is not a full-native missing-input
invariance claim. Exact frame duplication can change the within-site mean.

PCA26 uses centered source hidden features to fit the basis but does not subtract
a mean during transformation. Coordinates are scaled and clipped to [-16,16].
They are learned numerical axes, not 26 named clinical semantic attributes.
The separate legacy `qwen27b/semantic_readout.py` contains 32-dimensional
alternatives and must not be mistaken for the PCA26 implementation.

## Run without private data

```bash
python -m pip install -e '.[test]'
python scripts/demo_p4.py
python -m pytest -q
```

The example uses synthetic features and checks initialization, an optimizer step
and exact anchor fallback. Private experiment adapters need the caller's data,
source partitions, frozen weights, caches, locks and manifests; installing this
repository does not reconstruct the clinical experiment. Source provenance is
listed in `docs/LATEST_CODE_SOURCES.json`. Do not use the outer outcomes to
silently replace the primary method or omit failed comparisons.
