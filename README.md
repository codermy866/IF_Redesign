# Granularity-Selective Phenotype Fusion

This repository contains the core implementation of the final cervical OCT information-fusion
method, **G2-PCGrad**. It combines a frozen clinical risk score with OCT phenotype evidence learned
from site-level secondary-reading labels.

The method separates two supervision resolutions:

- `z`: local OCT lesion semantics. Numbers in the secondary-reading field identify positive OCT
  positions; unavailable positions are masked rather than treated as negatives.
- `y`: patient-level CIN2+ pathology used to optimize the fused patient prediction.

Both losses update the same trainable OCT representation. When their shared gradients conflict,
G2 applies symmetric two-task PCGrad. Projection is applied only when the global gradient dot
product is negative. The site head receives only the scaled local gradient, and the patient fusion
head receives only the patient-level gradient.

## Locked method structure

```text
frozen clinical Qwen3-VL risk ─────────────────────────┐
                                                      ├─ patient fusion loss y
OCT fixed ConvNeXt prefix → trainable final block     │
                             ├─ site head → loss z    │
                             └─ masked mean phenotype ┘

shared final-block gradients: symmetric PCGrad(lambda_z * g_z, g_y)
final predictor: nonnegative two-feature logistic fusion fit on inner validation only
```

The clinical branch is represented by precomputed logits, preserving the frozen clinical model.

## Repository layout

```text
configs/g2_pcgrad.json          Locked method hyperparameters
src/if_redesign/labels.py       Secondary-reading labels and missing-site masks
src/if_redesign/modeling.py     ConvNeXt feature tail and training-time fusion head
src/if_redesign/pcgrad.py       Symmetric two-task PCGrad
src/if_redesign/training.py     Patient-bag G2 optimization step
src/if_redesign/fusion.py       Masked pooling and inner-validation logistic fusion
src/if_redesign/metrics.py      Patient/site binary metrics
tests/                          Mathematical and routing tests
```

## Installation and validation

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## Core usage

```python
import torch
from if_redesign import ConvNeXtFeatureTail, G2PCGradTrainer, PatientFusionHead

encoder = ConvNeXtFeatureTail.from_torchvision().cuda()
patient_head = PatientFusionHead().cuda()
optimizer = torch.optim.AdamW([
    {"params": encoder.parameters(), "lr": 1e-4},
    {"params": patient_head.parameters(), "lr": 3e-4},
], weight_decay=0.01)
trainer = G2PCGradTrainer(encoder, patient_head, optimizer, lambda_z=0.1)

statistics = trainer.step(
    fixed_features=fixed_convnext_prefix_maps,
    clinical_risk=source_standardized_clinical_logits,
    patient_labels=cin2plus_labels,
    site_labels=oct_reread_site_labels,
    site_valid_mask=available_oct_site_mask,
)
```

After OCT training, pool site probabilities with `masked_mean_site_probability` and fit
`fit_nonnegative_fusion` using the source inner-validation partition. Apply that transformation
unchanged to held-out or external patients.

Patient identifiers, clinical records, OCT images, cached features, model weights, predictions,
experimental results, manuscripts, and figures are intentionally excluded.
