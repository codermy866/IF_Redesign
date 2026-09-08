# Diagnostic Evidence Alignment v1

This is an additive, exploratory extension. Existing attention, cross-fitted
utility, ICES, SFT and missing-modality results remain intact.

## Scientific question and mathematical objectives

Can a frozen-vision, LoRA-tuned VLM learn to express and rank diagnostically
useful OCT evidence, and distinguish original positive-site evidence from
within-patient negative-site replacements?

For a source-training patient i and site k, reuse the cross-fitted teacher
utility at a COMMON pre-OCT context:

    G_ik = CE(y_i, f^(-i)(C_i)) - CE(y_i, f^(-i)(C_i, e_ik)).

The existing C includes clinical information AND colposcopy. Thus this is
privileged multimodal teacher supervision for an OCT+clinical student, not a
claim that those two predictors see identical inputs. The teacher's fit AND
calibration sets exclude i. Later acquisition steps are not pooled together.

Define the student's value as a language-model log odds, requiring no new head:

    a_theta(e_k | E,C) = log p_theta(high | E,C,k) - log p_theta(low | E,C,k)
    L_rank = mean_(G_i > G_j + delta) softplus(-(a_i-a_j))
    L_alignment = L_language + lambda L_rank.

Each update samples a valid pair on a deterministic schedule. Importance is
never supplied as model input. Binary diagnostic probabilities use normalized
positive/negative completion likelihoods, not parsed self-reported confidence.

The preference stage uses standard reference-based DPO:

    L_DPO = -log sigmoid(beta * [(log pi(chosen)-log pi(rejected))
                                 -(log pi_ref(chosen)-log pi_ref(rejected))])
    L_stage2 = L_DPO + eta L_language(original factual case).

Reference completion log probabilities are cached with the exact frozen
starting alignment adapter in eval mode BEFORE any DPO updates. LoRA dropout is
zero. The factual anchor retains original patient-level diagnostic supervision.

## What the local data actually support

- Cohort feature cache: 984 unique patients, 37 evidence slots, 2048-d features.
- OCT evidence consists of 12 scanner-site clusters, 10 raw frames per cluster.
- The user's OCT re-read annotation marks positive sites; unlisted sites on a
  resolved annotation are negative. Missing/unresolved sheets remain unknown.
- A site label is neither pixel segmentation, per-frame lesion truth nor CIN2+
  pathology. No fabricated morphology is used as a target.
- Each site is displayed as frames 1, 5, 10 in a three-frame strip. Existing
  feature cache references are retained for traceability/teacher controls; the
  VLM consumes actual pixels through its own frozen pretrained vision encoder.
  ResNet vectors are not inserted into an unrelated VLM embedding space.
- Four sites per patient are chosen without outcomes, using a patient-hash and
  seed. All primary VLM arms see the identical images. This is a fixed candidate
  budget feasibility experiment; it is not an all-120-frame clinical study.
- Pathology belongs in the supervision/evaluation sidecar only. HPV, TCT and age
  are the explicit input allowlist. No ID, hospital, pathology, utility score or
  site re-read label appears in the clinical prompt.
- CoE descriptions explicitly state that specific morphology is unverified.
  This weak-supervision stage is not an expert-reviewed clinical reasoning corpus.

## Counterfactual validity and shortcut controls

The primary DPO pair is a high-positive-utility, reread-positive site versus an
unselected reread-negative site from the SAME patient. Exactly the same text,
same slot identifiers, same number of images and same two possible answers
are used for the original and replacement, with opposite preferences. A
text-only model therefore cannot satisfy both examples. The donor identity
and intervention type are never passed into these prompts.

Acquisition SESSION and clinical metadata are preserved, but raw pixel
background and acquisition POSITION are not identical. Replacement is a
site-level intervention, not validated causal removal of a lesion. Primary
DPO teaches site-finding support, not a newly invented patient pathology label.

Deletion/citation pairs are also retained in the dataset as a distinct audit.
They are NOT primary DPO training: explicit deletion prompts permit textual
shortcuts. Do not equate citation-format compliance with visual grounding.

Deleting one informative unit need not change a correct diagnosis: other OCT
sites and clinical information may remain sufficient. Consequently, unchanged
diagnoses are measured, not unconditionally penalized. Counterfactual AUROC
against original patient labels is explicitly an inherited-label robustness
analysis, not accuracy against known counterfactual disease ground truth.

## Locked comparison matrix

| Arm | Language supervision | Utility supervision | Preference stage |
|---|---|---|---|
| zero_shot | none | none | none |
| sft | weak site CoE + diagnosis; importance unassessed | none | none |
| alignment_no_rank (3B ablation) | CoE + utility-derived importance + diagnosis | verbal target only | none |
| alignment | same aligned target | pairwise log-odds ranking | none |
| alignment_dpo | initialize exact alignment adapter | inherited | balanced visual DPO + factual anchor |

Backbones: pinned Qwen2.5-VL-3B, Qwen2.5-VL-7B, official Microsoft LLaVA-Med
Mistral-7B. Five existing LOCO folds x three seeds x four primary arms = 180
primary cells (135 trained adapters). The 3B no-ranking ablation adds 15 cells.
All checkpoints are retained. Fixed final epoch selection is declared in
advance; source validation and held-center test are saved separately. No
automatic tuning against held-center metrics or cherry-picking of seeds.

LLaVA-Med uses a strict, checked key conversion into the installed Transformers
implementation, including CLIP padding and a zero image-placeholder row. Native
versus converted numerical parity remains a separate verification requirement;
until checked, report it as a converted-checkpoint implementation. Each new
backbone must pass a real backward smoke test before full training.

The existing teacher attention/utility baselines are exported separately with
their original multimodal inputs. They provide historical/contextual baselines;
their AUROCs are not an input-matched causal estimate of the new VLM's benefit.

## Evaluation and allowed claims

- Original-image AUROC, AUPRC, F1 (fixed threshold 0.5), per-center metrics.
- Spearman ranking versus held-out frozen-teacher CE gain, with valid counts.
  These are model-based targets, not human diagnostic utility truth.
- High/low-score deletion, within-patient normal-site replacement; probability
  changes, prediction consistency, structured-citation validity and parse rate.
- Three-seed patient ensembles before center-stratified paired bootstrap;
  AUROC/AUPRC/F1 confidence intervals and paired ablation differences. Source
  folds overlap and are not treated as independent patient replicates.
- Do not call these results center-robust causal discovery, clinical validation,
  morphology faithfulness, reduced acquisition time, or publication-ready before
  the corresponding evidence is available. The existing cohort is exploratory.

## Running and tracking

Use the existing `.venv`; the verified environment has Python 3.11,
PyTorch 2.3.1+cu121, Transformers 4.57.6, PEFT 0.17.1.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/build_dea_dataset.py
.venv/bin/python scripts/prepare_dea_models.py
PYTHONPATH=src:scripts .venv/bin/python scripts/launch_dea_campaign.py
PYTHONPATH=src:scripts .venv/bin/python scripts/summarize_dea.py
```

`configs/dea_v1.json` is the reproducible configuration. Private data, adapters,
immutable download status, dataset audits, reference log probabilities, logs,
metrics and queue state are under `results/dea_v1`. Queue processes obtain an
exclusive file lock, wait for pre-existing GPU memory use to drop below the
configured threshold, and never kill existing jobs. Failed training stops that
cell and preserves its artifacts. A partial checkpoint requires an explicitly
named retry output, preventing silent overwrite.

The queue also completes the one missing restored-clinical OOF run before
building its dependent data. Dataset v2 audit is required before any full run.
CPU preparation and model downloads can proceed while GPUs remain busy.

## Reference and adaptation boundaries

The supplied `2603.20698v3 (1).pdf` is *Clinical Cognition Alignment for
Gastrointestinal Diagnosis with Multimodal LLMs*, arXiv:2603.20698v3, 3 July 2026.
Its Sections 3.2 and 4.4 use human-refined reasoning and lesion masks; Section
5.1 reports eight L20 GPUs. Those resources cannot be assumed here. Its latent
factor argument requires assumptions not established by our site annotations;
we do not inherit a causal guarantee by reusing SFT/preference optimization.

Official implementation references consulted:
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- https://github.com/microsoft/LLaVA-Med/blob/main/README.md
- https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b
