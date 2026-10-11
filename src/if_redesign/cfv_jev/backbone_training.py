"""Source-only patient-answer training with the frozen modality policy.

This is a thin training loop over caller-supplied models, examples, preparation,
and source-selection scoring. It does not provide cohort assembly, pretrained
weights, clinical-prior fitting, or the upstream Visual-JEV implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from copy import deepcopy
from typing import Callable, Sequence
import math
import random

import numpy as np


@dataclass(frozen=True)
class BackboneTrainingPolicy:
    """Exact frozen defaults; shortened steps are useful for synthetic checks."""
    name: str = "colpo_focus"
    modality: str = "OK"
    steps: int = 512
    seed: int = 17
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (.9, .95)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    accumulation_patients: int = 8
    selection_interval: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = .05
    language_projection_names: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    def __post_init__(self):
        if self.name != "colpo_focus" or self.modality not in ("OK", "O", "K"):
            raise ValueError("Use the frozen colpo_focus policy and modality OK, O, or K")
        if any(not isinstance(v, int) or v <= 0 for v in (
            self.steps, self.accumulation_patients, self.selection_interval,
        )):
            raise ValueError("Steps, accumulation_patients, and selection_interval must be positive integers")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

    def to_dict(self):
        return {
            **asdict(self),
            "augmentation_probabilities": {"full": .5, "drop_oct": .4, "drop_colpo": .1},
            "image_slots": 20, "training_questions": 1, "inference_questions": 21,
            "supervision": "patient pathology answer only",
            "step_zero": "source selection reference, excluded from selectable checkpoints",
            "checkpoint_selection": "source selection NLL every 16 steps and final step",
        }

    def scheduled_learning_rate(self, step):
        """The zero-based warmup/cosine schedule used before each optimizer step."""
        if not isinstance(step, int) or not 0 <= step < self.steps:
            raise ValueError("step must lie in [0, steps)")
        warmup = min(100, self.steps // 10)
        scale = min(1., (step + 1) / max(1, warmup))
        progress = max(0., (step - warmup) / max(1, self.steps - warmup))
        return self.learning_rate * scale * .5 * (1 + math.cos(math.pi * min(1., progress)))

    def apply_views(self, views, rng, neutral_factory):
        """Keep all slots; apply the same augmentation even to O/K controls.

O controls can become all neutral in the 40% drop-OCT branch, and K controls
in the 10% drop-colposcopy branch. These are budget-matched controls, rather
than individually optimized single-modality training strategies.
"""
        result = self.evaluation_views(views, neutral_factory)
        draw = rng.random()
        if draw < .5:
            branch = "full"
        elif draw < .9:
            branch = "drop_oct"
            result[:12] = [neutral_factory() for _ in range(12)]
        else:
            branch = "drop_colpo"
            result[12:] = [neutral_factory() for _ in range(8)]
        return result, branch

    def evaluation_views(self, views, neutral_factory):
        """Apply only the permanent O/K input control, without augmentation."""
        if len(views) != 20:
            raise ValueError("Exactly twenty ordered image slots are required")
        result = list(views)
        if self.modality == "O":
            result[12:] = [neutral_factory() for _ in range(8)]
        elif self.modality == "K":
            result[:12] = [neutral_factory() for _ in range(12)]
        return result


@dataclass(frozen=True)
class TrainingPatient:
    """Generic training input; keys are used only to verify source separation."""
    key: str
    views: Sequence
    label: int


def configure_language_lora(backbone, policy=None, *, released_adapter=None):
    """Configure language-only LoRA without supplying or distributing weights.

If a caller supplies a released adapter path, it is loaded as the trainable
initialization. Otherwise a fresh adapter targets all seven language projection
types. The initialized model is returned; callers attach it to their VDM.
"""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    policy = policy or BackboneTrainingPolicy()
    backbone.requires_grad_(False)
    if released_adapter is not None:
        configured = PeftModel.from_pretrained(backbone, str(released_adapter), is_trainable=True)
    else:
        targets = [name for name, module in backbone.named_modules()
                   if isinstance(module, torch.nn.Linear) and "language_model" in name
                   and name.rsplit(".", 1)[-1] in policy.language_projection_names]
        if not targets:
            raise ValueError("No supported language-model linear projections were found")
        configured = get_peft_model(backbone, LoraConfig(
            r=policy.lora_rank, lora_alpha=policy.lora_alpha, lora_dropout=policy.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets,
        ))
    if any(("visual" in name or "vision" in name) and p.requires_grad
           for name, p in configured.named_parameters()):
        raise ValueError("The vision encoder must remain frozen")
    if hasattr(configured, "gradient_checkpointing_enable"):
        configured.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(configured, "enable_input_require_grads"):
        configured.enable_input_require_grads()
    return configured


def patient_answer_loss(model, prepared, target):
    """Use the patient answer token with whole-vocabulary cross entropy.

The view questions are never assigned patient pathology labels. Candidate
order can be reversed by the trainer, with the target index reversed with it.
"""
    import torch
    if target not in (0, 1) or len(prepared.suffix_ids) != 1:
        raise ValueError("Training requires one binary patient-answer question")
    full = torch.cat([prepared.prefix_ids, prepared.suffix_ids[0]])[None].to(model.device_str)
    output = model.forward_train(
        prepared.pixel_values.to(model.device_str, model.dtype),
        prepared.image_grid_thw.to(model.device_str), full,
        prepared.position_ids[0][:, None].to(model.device_str),
        torch.ones_like(full), torch.tensor([full.shape[1] - 1]), ["choice"], [2],
    )
    token = model.option_token_ids[torch.tensor([target], device=model.device_str)]
    return torch.nn.functional.cross_entropy(output["lm_logits"].float(), token)


@dataclass(frozen=True)
class BackboneTrainingResult:
    selected_step: int
    selection_nll: float
    baseline_selection_nll: float
    trace: tuple[dict, ...]
    mask_counts: dict
    selected_parameters: dict


def source_selection_nll(model, patients, policy=None, *, bindings=None):
    """Evaluate pooled NLL on caller-supplied source-selection patients only.

Input augmentation is disabled, while permanent O/K content controls remain.
Each patient receives a fresh prefix and the twenty-one independent questions.
No training labels are attached to individual view questions.
"""
    import torch
    from scipy.special import expit
    from sklearn.metrics import log_loss
    from .visual_jev import VisualJEVAdapter, prepare_patient
    policy = policy or BackboneTrainingPolicy()
    if not patients or any(p.label not in (0, 1) for p in patients):
        raise ValueError("Nonempty source-selection patients must have binary labels")
    model.eval()
    if hasattr(model, "keep_full_lm_logits"):
        model.keep_full_lm_logits = False
    probabilities = []
    with torch.inference_mode():
        for patient in patients:
            views = policy.evaluation_views(patient.views, VisualJEVAdapter.neutral_view)
            prepared = prepare_patient(model, views, bindings=bindings)
            logits = model.run_prefix_share_batch(prepared)["lm_option_logits"][:, :2].float()
            if logits.shape != (21, 2) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Source selection requires twenty-one finite binary option logits")
            difference = float((logits[0, 1] - logits[0, 0]).cpu())
            probabilities.append(float(expit(difference)))
    return float(log_loss([p.label for p in patients], probabilities, labels=[0, 1]))


def train_source_backbone(
    model, source_train: Sequence[TrainingPatient], source_selection: Sequence[TrainingPatient],
    *, prepare_training: Callable | None = None, selection_nll: Callable | None = None,
    policy: BackboneTrainingPolicy | None = None,
    neutral_factory: Callable | None = None,
    loss_callback: Callable = patient_answer_loss,
    bindings=None,
):
    """Train, select, and restore only source-selected trainable parameters.

``prepare_training(model, views, question)`` must prepare a single independent
patient question. ``selection_nll(model, source_selection)`` must evaluate
unaugmented source-selection inputs; it receives no validation/outer patients.
For O/K controls this callback applies ``policy.evaluation_views`` to keep
the permanently absent modality neutral. The original checkpoint criterion
is pooled source-selection patient NLL, rather than a center-balanced score.
The caller is responsible for trustworthy source splits and for constructing
the upstream execution interface. This function never launches cohort jobs.
"""
    import torch
    from .visual_jev import VisualJEVAdapter, patient_questions, prepare_patient
    policy = policy or BackboneTrainingPolicy()
    neutral_factory = neutral_factory or VisualJEVAdapter.neutral_view
    if prepare_training is None:
        prepare_training = lambda model, views, question: prepare_patient(
            model, views, [question], bindings=bindings,
        )
    if selection_nll is None:
        selection_nll = lambda model, patients: source_selection_nll(
            model, patients, policy, bindings=bindings,
        )
    for examples in (source_train, source_selection):
        if not examples or any(p.label not in (0, 1) or len(p.views) != 20 for p in examples):
            raise ValueError("Nonempty source inputs require binary labels and twenty image slots")
        if len({p.key for p in examples}) != len(examples):
            raise ValueError("Patient keys must be unique within each source split")
    if {p.key for p in source_train} & {p.key for p in source_selection}:
        raise ValueError("Source train and source selection patient keys must be disjoint")
    named_parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not named_parameters:
        raise ValueError("The supplied model must have trainable language-adapter parameters")
    if any("vision" in name or "visual" in name for name in named_parameters):
        raise ValueError("The vision encoder must remain frozen")
    torch.manual_seed(policy.seed)
    rng = random.Random(policy.seed)
    optimizer = torch.optim.AdamW(list(named_parameters.values()), lr=policy.learning_rate,
                                 betas=policy.betas, weight_decay=policy.weight_decay)

    def evaluate():
        model.eval()
        if hasattr(model, "keep_full_lm_logits"):
            model.keep_full_lm_logits = False
        with torch.inference_mode():
            score = float(selection_nll(model, source_selection))
        if not np.isfinite(score) or score < 0:
            raise ValueError("Source-selection NLL must be finite and nonnegative")
        return score

    baseline = evaluate()
    best = math.inf
    best_step, selected = None, None
    trace = [{"step": 0, "selection_nll": baseline, "selectable": False}]
    counts = Counter(full=0, drop_oct=0, drop_colpo=0, all_neutral=0)
    epoch, step = 0, 0
    while step < policy.steps:
        order = np.random.default_rng(policy.seed + epoch * 1000).permutation(len(source_train))
        epoch += 1
        chunks = np.array_split(order, max(1, math.ceil(len(order) / policy.accumulation_patients)))
        for chunk in chunks:
            if step >= policy.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            model.train()
            if hasattr(model, "keep_full_lm_logits"):
                model.keep_full_lm_logits = True
            for index in chunk:
                patient = source_train[int(index)]
                question = deepcopy(patient_questions()[0])
                target = int(patient.label)
                if rng.getrandbits(1):
                    question["candidates"].reverse()
                    target = 1 - target
                views, branch = policy.apply_views(patient.views, rng, neutral_factory)
                counts[branch] += 1
                if (policy.modality == "O" and branch == "drop_oct") or (
                    policy.modality == "K" and branch == "drop_colpo"
                ):
                    counts["all_neutral"] += 1
                prepared = prepare_training(model, views, question)
                loss = loss_callback(model, prepared, target)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("The patient training loss must be finite and scalar")
                (loss / len(chunk)).backward()
            torch.nn.utils.clip_grad_norm_(list(named_parameters.values()), policy.max_grad_norm)
            optimizer.param_groups[0]["lr"] = policy.scheduled_learning_rate(step)
            optimizer.step()
            step += 1
            if step % policy.selection_interval == 0 or step == policy.steps:
                score = evaluate()
                trace.append({"step": step, "selection_nll": score, "selectable": True})
                if score < best:
                    best, best_step = score, step
                    selected = {name: p.detach().cpu().clone() for name, p in named_parameters.items()}
    with torch.no_grad():
        for name, p in named_parameters.items():
            p.copy_(selected[name].to(device=p.device, dtype=p.dtype))
    model.eval()
    if hasattr(model, "keep_full_lm_logits"):
        model.keep_full_lm_logits = False
    return BackboneTrainingResult(best_step, best, baseline, tuple(trace), dict(counts), selected)
