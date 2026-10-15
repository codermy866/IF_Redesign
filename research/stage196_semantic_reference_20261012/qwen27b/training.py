"""Portable source-only version of the current task-adaptation training loop."""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from .adapter import images, save, sha
from .readout import center_auc


@dataclass(frozen=True)
class AdaptationPolicy:
    updates: int = 512
    checkpoints: tuple[int, ...] = (128, 256, 512)
    learning_rate: float = 3e-5
    proximal_weight: float = .01
    weight_decay: float = .01
    gradient_clip: float = 1.
    walltime_hours: float = 12.

    def __post_init__(self):
        if (self.updates < 1 or not self.checkpoints or
                tuple(sorted(set(self.checkpoints))) != self.checkpoints or
                self.checkpoints[0] < 1 or self.checkpoints[-1] > self.updates):
            raise ValueError('Invalid checkpoint schedule')
        if (self.learning_rate <= 0 or self.proximal_weight < 0 or
                self.weight_decay < 0 or self.gradient_clip <= 0 or
                self.walltime_hours <= 0):
            raise ValueError('Invalid training hyperparameters')


def validate_source_partitions(targets, train_keys, selection_keys, outer_center, *, require_reread=False):
    if not train_keys or not selection_keys:
        raise ValueError('Both source partitions are required')
    if (len(set(train_keys)) != len(train_keys) or
            len(set(selection_keys)) != len(selection_keys) or
            set(train_keys) & set(selection_keys)):
        raise ValueError('Source training and selection patients must be disjoint and unique')
    for key in list(train_keys) + list(selection_keys):
        row = targets[key]
        if row['center'] == outer_center:
            raise ValueError('Outer-center patient in source learning')
        if row['y'] not in (0, 1):
            raise ValueError('Patient target must be binary pathology')
        if (require_reread and key in train_keys and row.get('reread_valid', False) and
                (len(row.get('reread', [])) != 12 or any(v not in (-1, 0, 1) for v in row['reread']))):
            raise ValueError('Expected twelve weak position labels; -1 is unresolved')


def train_source_adapter(model, manifest, targets, train_keys, selection_keys, *,
                         outer_center, variant, output_dir, seed=17,
                         policy=AdaptationPolicy(), progress=None):
    """Fit and restore a source-selected medical LoRA checkpoint.

    ``manifest[key]`` has only ``oct_sites`` and ``colpo_paths``. Labels live
    separately in ``targets``. PATIENT_ONLY requires no weak labels. DUAL_TASK
    checks both usable weak classes in each training center. Source selection chooses
    trained checkpoints; it is not an independent validation set. Output is
    local checkpoints and metric traces, never patient records.
    """
    if variant not in ('PATIENT_ONLY', 'DUAL_TASK'):
        raise ValueError('Adaptation variant must be PATIENT_ONLY or DUAL_TASK')
    validate_source_partitions(targets, train_keys, selection_keys, outer_center,
                               require_reread=variant == 'DUAL_TASK')
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Use an empty output directory to preserve previous runs')
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'MEDICAL_ADAPTER.safetensors'
    model.reset()
    model.enable_training(True)
    t = model.torch
    t.manual_seed(seed + 195)
    t.cuda.manual_seed_all(seed + 195)
    rng = np.random.default_rng(seed + 195)
    diagnosis, weak = {}, {}
    for key in train_keys:
        target = targets[key]
        diagnosis.setdefault((target['center'], int(target['y'])), []).append(key)
        if variant == 'DUAL_TASK' and target.get('reread_valid', False):
            for site, label in enumerate(target['reread'], 1):
                if label in (0, 1) and any(p for p in (manifest[key].get('oct_sites', {}).get(str(site)) or []) if p):
                    weak.setdefault((target['center'], int(label)), []).append((key, site))
    centers = sorted({g for g, label in diagnosis})
    if (any((c, label) not in diagnosis for c in centers for label in (0, 1)) or
            (variant == 'DUAL_TASK' and any((c, label) not in weak
                                          for c in centers for label in (0, 1)))):
        model.enable_training(False)
        raise ValueError('Empty center/task class prevents balanced source sampling')
    opt = t.optim.AdamW(list(model.parameters.values()),
                       lr=policy.learning_rate, weight_decay=policy.weight_decay)
    best, trace = None, []
    started = time.monotonic()
    try:
        for step in range(1, policy.updates + 1):
            center = centers[int(rng.integers(len(centers)))]
            label = int(rng.integers(2))
            task = 'reread' if variant == 'DUAL_TASK' and step % 2 == 0 else 'diagnosis'
            if task == 'diagnosis':
                entries = diagnosis[(center, label)]
                key = entries[int(rng.integers(len(entries)))]
                site, phase = None, int(rng.integers(10))
            else:
                entries = weak[(center, label)]
                key, site = entries[int(rng.integers(len(entries)))]
                phase = int(rng.integers(2))
            prepared = model.prepared(images(manifest[key], task, site, phase), task)
            opt.zero_grad(set_to_none=True)
            row = model.forward(*prepared, grad=True)
            loss = (t.nn.functional.cross_entropy(
                row['logits'][None], t.tensor([label], device=model.device)) +
                policy.proximal_weight * model.proximal())
            loss.backward()
            gradients = [p.grad for p in model.parameters.values() if p.grad is not None]
            if not gradients or not all(t.isfinite(g).all() for g in gradients):
                raise RuntimeError('Nonfinite/empty medical adaptation gradients')
            t.nn.utils.clip_grad_norm_(list(model.parameters.values()), policy.gradient_clip)
            opt.step()
            if progress is not None and step % 16 == 0:
                progress(dict(step=step, task=task, loss=float(loss.detach()),
                              seconds=time.monotonic() - started))
            if step in policy.checkpoints:
                model.enable_training(False)
                values = []
                for key in selection_keys:
                    inp = model.prepared(images(manifest[key], phase=4), 'diagnosis')
                    logits = model.forward(*inp)['logits'].cpu().numpy()
                    values.append(float(logits[1] - logits[0]))
                y = np.array([targets[k]['y'] for k in selection_keys])
                g = np.array([targets[k]['center'] for k in selection_keys])
                metric = dict(step=step, AUROC=float(roc_auc_score(y, values)),
                              center_AUROC=center_auc(y, np.asarray(values), g),
                              seconds=time.monotonic() - started)
                trace.append(metric)
                save(output_dir / 'ADAPTER_SELECTION_TRACE.json',
                     dict(rows=trace, source_only=True))
                if best is None or (metric['AUROC'], metric['center_AUROC'], -step) > (
                        best['AUROC'], best['center_AUROC'], -best['step']):
                    model.save_adapter(checkpoint)
                    best = metric
                model.enable_training(True)
            if time.monotonic() - started > policy.walltime_hours * 3600:
                raise RuntimeError('Adapter training walltime exceeded')
    finally:
        model.enable_training(False)
    model.load_adapter(checkpoint)
    result = dict(
        state='SOURCE_ADAPTER_FROZEN', variant=variant, selected_checkpoint=best,
        adapter_sha256=sha(checkpoint), updates=policy.updates, seed=seed,
        trainable_tensors=len(model.parameters),
        trainable_parameters=sum(p.numel() for p in model.parameters.values()),
        native_vision_frozen=True, base_weights_frozen=True,
        diagnosis_updates=policy.updates if variant == 'PATIENT_ONLY' else (policy.updates + 1)//2,
        reread_updates=0 if variant == 'PATIENT_ONLY' else policy.updates//2,
        source_only=True, trace=trace,
    )
    save(output_dir / 'ADAPTER_COMPLETE.json', result)
    return result
