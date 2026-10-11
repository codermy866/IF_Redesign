"""Fixed-slot Visual-JEV inference and a portable RAW patient readout.

The separately installed upstream ``vdm`` package supplies model execution and
prompt helpers. No upstream source, pretrained weights, or fitted parameters
are distributed here. Every inference prepares and executes a fresh prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import inspect

import numpy as np
from scipy.special import erf, expit


CONTEXT = (
    "Views 1-12 are cervical OCT; views 13-20 are colposcopy. "
    "Uniform gray views contain no visual evidence. Assess image content only."
)


def patient_questions():
    """Return the fixed patient question and twenty independent view questions."""
    questions = [{
        "qtype": "choice",
        "instruction": "Considering the displayed cervical imaging evidence, is this patient likely to have CIN2 or worse?",
        "candidates": ["CIN2+ absent", "CIN2+ present"],
    }]
    questions += [{
        "qtype": "choice",
        "instruction": f"Does OCT view {i} show structural features suggestive of a high-grade lesion?",
        "candidates": ["No", "Yes"],
    } for i in range(1, 13)]
    questions += [{
        "qtype": "choice",
        "instruction": f"Does colposcopy view {i} show visual features suggestive of a high-grade lesion?",
        "candidates": ["No", "Yes"],
    } for i in range(1, 9)]
    return questions


def raw_descriptors(view_probabilities):
    """Return the fixed 32 RAW coordinates, retaining each modality separately."""
    q = np.asarray(view_probabilities, dtype=float)
    if q.ndim != 2 or q.shape[1] != 20 or not np.isfinite(q).all():
        raise ValueError("Expected finite view probabilities with shape (N, 20)")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("View probabilities must be in [0, 1]")
    blocks = []
    for values in (q[:, :12], q[:, 12:]):
        z = np.sort(values, axis=1)
        blocks.append(np.column_stack([
            z.mean(axis=1), z.max(axis=1), z[:, -2:].mean(axis=1),
            z.std(axis=1),
            np.quantile(z, .75, axis=1) - np.quantile(z, .25, axis=1),
            (z > .5).mean(axis=1), z,
        ]))
    return np.concatenate(blocks, axis=1)


@dataclass(frozen=True)
class RawPatientReadout:
    """The source-selected REF_ERM_RAW readout, before clinical anchoring.

``weights`` contains 33*4+8=140 parameters. ``feature_mean`` and
``feature_scale`` are the fitted RAW scaler parameters, including the final
global option-logit difference. GELU uses its exact erf formulation, matching
the original ``torch.nn.functional.gelu(..., approximate='none')``.
"""

    weights: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    rho: float = 1.0
    temperature: float = 1.0

    def __post_init__(self):
        for name, size in (("weights", 140), ("feature_mean", 33), ("feature_scale", 33)):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, tuple(values.tolist()))
        if np.any(np.asarray(self.feature_scale) <= 0):
            raise ValueError("feature_scale must be positive")
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be positive and finite")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")

    @property
    def effective_scale(self):
        """Supply this scale to the clinical-anchoring intervention extractor."""
        return float(self.rho / self.temperature)

    def score_value(self, option_logits, clinical_logit):
        """Compute h(X,c); the caller subtracts h(neutral,c) and adds c once."""
        logits = np.asarray(option_logits, dtype=float)
        single = logits.ndim == 2
        if single:
            logits = logits[None]
        if logits.ndim != 3 or logits.shape[1:] != (21, 2) or not np.isfinite(logits).all():
            raise ValueError("Expected finite option logits with shape (21, 2) or (N, 21, 2)")
        c = np.asarray(clinical_logit, dtype=float)
        if c.ndim == 0:
            c = np.full(len(logits), float(c))
        if c.shape != (len(logits),) or not np.isfinite(c).all():
            raise ValueError("clinical_logit must be a scalar or one finite value per patient")
        differences = logits[:, :, 1] - logits[:, :, 0]
        features = np.column_stack([raw_descriptors(expit(differences[:, 1:])), differences[:, 0]])
        x = (features - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        w = np.asarray(self.weights)
        preactivation = x @ w[:132].reshape(33, 4)
        hidden = .5 * preactivation * (1 + erf(preactivation / np.sqrt(2)))
        score = np.column_stack([hidden, hidden * np.tanh(c[:, None] / 3)]) @ w[132:]
        if not np.isfinite(score).all():
            raise ValueError("Patient readout produced a nonfinite score")
        return float(score[0]) if single else score

    def to_dict(self):
        """Return parameters that can be saved as ordinary JSON."""
        return {
            "representation": "RAW", "parameters": 140,
            "weights": list(self.weights), "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "rho": float(self.rho), "temperature": float(self.temperature),
        }

    @classmethod
    def from_dict(cls, parameters):
        """Load the canonical JSON or ``params/scaler/rho/T`` numeric form."""
        if parameters.get("representation", "RAW") != "RAW":
            raise ValueError("Only the fixed RAW patient readout is supported")
        if "weights" not in parameters:
            return cls(
                parameters["params"], parameters["scaler"]["mean"], parameters["scaler"]["scale"],
                parameters["rho"], parameters["T"],
            )
        return cls(
            parameters["weights"], parameters["feature_mean"], parameters["feature_scale"],
            parameters["rho"], parameters["temperature"],
        )


@dataclass(frozen=True)
class UpstreamBindings:
    """Inject immutable upstream interfaces, also allowing small fake-model tests."""
    prepared_group_factory: Callable
    prefix_text: Callable
    suffix_text: Callable
    split_ids: Callable


def _upstream_bindings():
    try:
        from vdm.models.vdm_model import PreparedGroup
        from vdm.models.prompts import prefix_text, suffix_text, split_ids
    except ImportError as exc:
        raise ImportError(
            "VisualJEVAdapter requires the separately installed Visual-JEV vdm package; "
            "supply a compatible pretrained VDM or CompatVDM model."
        ) from exc
    return UpstreamBindings(PreparedGroup, prefix_text, suffix_text, split_ids)


def prepare_patient(model, views, questions=None, *, bindings=None):
    """Prepare fresh fixed-slot input without changing model mode or gradients.

Pass one patient question during training and leave ``questions`` unspecified
for the twenty-one independent inference questions. The helper uses no model
or language-prefix caches and never freezes trainable adapter parameters.
"""
    import torch
    bindings = bindings or _upstream_bindings()
    images = VisualJEVAdapter._images(views)
    image_processor = getattr(model.processor, "image_processor", None)
    if image_processor is not None:
        image_processor.max_pixels = 224 * 224
        image_processor.min_pixels = 224 * 224
    questions = patient_questions() if questions is None else list(questions)
    if not questions:
        raise ValueError("At least one independent question is required")
    tags = "\n".join(f"View {i}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(1, 21))
    prefix = bindings.prefix_text(CONTEXT).replace(
        "<|vision_start|><|image_pad|><|vision_end|>", tags,
    ) + "<|im_end|>"
    encoded = model.processor(text=[prefix], images=images, return_tensors="pt")
    if len(encoded["image_grid_thw"]) != 20:
        raise ValueError("The image processor must preserve all twenty image slots")
    suffixes, positions = [], []
    rope = model.vl_model.get_rope_index
    for q in questions:
        suffix = "\n<|im_start|>user" + bindings.suffix_text(q["instruction"], q["candidates"], q["qtype"])
        bindings.split_ids(model.tokenizer, prefix, prefix + suffix)
        ids = torch.tensor(model.tokenizer(suffix, add_special_tokens=False)["input_ids"])
        suffixes.append(ids)
        full = torch.cat([encoded["input_ids"][0], ids])[None].to(model.device_str)
        kwargs = {"image_grid_thw": encoded["image_grid_thw"].to(model.device_str)}
        if "mm_token_type_ids" in inspect.signature(rope).parameters:
            kwargs["mm_token_type_ids"] = (full == model.image_token_id).to(torch.int32)
        position, _ = rope(full, **kwargs)
        positions.append(position[:, 0])
    return bindings.prepared_group_factory(
        encoded["input_ids"][0], suffixes, encoded["pixel_values"],
        encoded["image_grid_thw"], positions,
        [q["qtype"] for q in questions], [len(q["candidates"]) for q in questions],
        model.image_token_id,
    )


class VisualJEVAdapter:
    """Execute the fixed patient extension using a supplied upstream model.

The caller loads its selected language adapter into ``model`` before wrapping
it. The model is frozen for inference. Only deterministic resizing is done
here; no image selection, pathology labels, or clinical context enters the VLM
prompt. Clinical information enters only the RAW patient readout.
"""

    def __init__(self, model, *, bindings: UpstreamBindings | None = None):
        self.model = model
        self.bindings = bindings
        self.model.eval()
        self.model.requires_grad_(False)
        if hasattr(self.model, "keep_full_lm_logits"):
            self.model.keep_full_lm_logits = False
        image_processor = getattr(model.processor, "image_processor", None)
        if image_processor is not None:
            image_processor.max_pixels = 224 * 224
            image_processor.min_pixels = 224 * 224

    @classmethod
    def from_upstream(cls, model_path, *, device="cuda:0", model_factory=None, **model_kwargs):
        """Load caller-supplied weights with a separately installed model factory.

Supply ``model_factory`` when the environment needs a Transformers-compatible
VDM subclass. Loading and selecting a language adapter remains the caller's
responsibility; a bare base model does not reproduce the trained method.
"""
        if model_factory is None:
            from vdm.models.vdm_model import VDM
            model_factory = VDM
        model = model_factory(str(model_path), device=device, with_heads=False, **model_kwargs)
        return cls(model)

    @staticmethod
    def neutral_view():
        from PIL import Image
        return Image.new("RGB", (224, 224), (128, 128, 128))

    @staticmethod
    def _images(views: Sequence):
        from PIL import Image
        if len(views) != 20:
            raise ValueError("Exactly twenty ordered image slots are required")
        images = []
        for view in views:
            if view is None:
                image = VisualJEVAdapter.neutral_view()
            elif isinstance(view, (str, Path)):
                with Image.open(view) as opened:
                    image = opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            elif isinstance(view, Image.Image):
                image = view.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            elif isinstance(view, np.ndarray) and view.dtype == np.uint8 and view.ndim == 3 and view.shape[2] == 3:
                image = Image.fromarray(view).resize((224, 224), Image.Resampling.BILINEAR)
            else:
                raise TypeError("Each view must be a PIL image, uint8 RGB array, image path, or None")
            images.append(image)
        return images

    def prepare(self, views):
        """Build a new complete prefix and all independent question suffixes."""
        return prepare_patient(self.model, views, patient_questions(), bindings=self.bindings)

    def logits(self, views):
        """Recompute a fresh language prefix for this exact twenty-view input."""
        import torch
        self.model.eval()
        with torch.inference_mode():
            prepared = self.prepare(views)
            values = self.model.run_prefix_share_batch(prepared)["lm_option_logits"][:, :2]
            result = values.float().cpu().numpy()
        if result.shape != (21, 2) or not np.isfinite(result).all():
            raise ValueError("Upstream execution must return twenty-one finite binary option logits")
        return result

    def scorer(self, readout: RawPatientReadout):
        """Return the unanchored ``scorer(views, clinical_logit)`` callback."""
        def score(views, clinical_logit):
            return readout.score_value(self.logits(views), clinical_logit)
        return score
