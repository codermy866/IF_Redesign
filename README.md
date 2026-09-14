# Supervision-Matched Adaptation for Cervical OCT

This repository contains the core implementation of Supervision-Matched Adaptation (SMA) for
cervical OCT vision-language models. The method uses lesion positions recorded in the clinical
second-reading field as direct site-level supervision.

SMA contains four components:

1. parse second-reading position numbers into site-level lesion targets;
2. control positive lesion-site exposure during training while retaining natural validation prevalence;
3. route diagnosis, site localization, and evidence-assimilation supervision through separate LoRA
   subspaces under a fixed total rank budget;
4. derive task ranks and task exposure by minimizing

   `J(r,q) = sum_k lambda_k [B_k(r_k) + alpha * nu_k * r_k / (T * q_k)]`.

`B_k(r_k)` is residual gradient spectral energy, `nu_k` is normalized gradient variance, `r_k` is
task-specific LoRA rank, and `q_k` is the task's training exposure.

## Repository layout

```text
configs/sma_qwen3vl_2b.json       Method configuration
src/if_redesign/labels.py         Second-reading lesion-site targets
src/if_redesign/sampling.py       Task and positive-site exposure schedules
src/if_redesign/allocation.py     Source-derived rank/exposure allocation
src/if_redesign/modeling.py       Task-routed LoRA construction
src/if_redesign/objectives.py     Diagnosis, localization, and assimilation losses
src/if_redesign/metrics.py        Site, diagnosis, and evidence-utility metrics
scripts/derive_allocation.py      Allocation command-line entry point
scripts/train_sma.py              Qwen3-VL training and evaluation entry point
tests/                            Unit tests for the core method
```

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

## Input format

Training and validation data are JSON Lines files. Each row has a `task` field.

```json
{"task":"diag","patient_key":"P001","image":"/path/frame.png","prompt":"...","label":1}
{"task":"site","patient_key":"P001","site":3,"image":"/path/site3.png","prompt":"...","label":1}
{"task":"assim","patient_key":"P001","image":"/path/frame.png","baseline_prompt":"...","prompt":"...","label":1,"p_gain":0.2}
```

Generate site labels directly from a second-reading entry:

```python
from if_redesign.labels import build_site_targets

targets = build_site_targets("2, 5, 9", number_of_sites=12)
```

## Derive a supervision-matched allocation

```bash
derive-sma-allocation \
  --spectral-energy supervision_spectral_energy.csv \
  --gradient-variance gradient_variance.csv \
  --output allocation.json
```

## Train

```bash
train-sma \
  --config configs/sma_qwen3vl_2b.json \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --train-jsonl /path/to/training_tasks.jsonl \
  --validation-jsonl /path/to/validation_tasks.jsonl \
  --output /path/to/output \
  --device cuda:0
```

Training-site sampling may be balanced, but validation data must retain its natural clinical
distribution. Patient identifiers, clinical data, OCT images, checkpoints, and experiment results are
not included in this repository.
