import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from if_redesign.qwen27b.adapter import images
from if_redesign.qwen27b import training


def test_raw_frame_bags_and_twenty_slots(tmp_path):
    paths = []
    for index in range(10):
        path = tmp_path / f'{index}.png'
        Image.new('RGB', (300, 200), (index*20, 0, 0)).save(path)
        paths.append(str(path))
    item = dict(oct_sites={str(site): paths for site in range(1, 13)}, colpo_paths=[])
    for phase, indices in ((0, [0, 3, 6, 9]), (1, [1, 2, 5, 8])):
        views = images(item, 'reread', site=3, phase=phase)
        assert len(views) == 20 and all(v.size == (224, 224) for v in views)
        assert [v.getpixel((0, 0))[0] for v in views[:4]] == [i*20 for i in indices]
        assert all(v.getpixel((0, 0)) == (128, 128, 128) for v in views[4:])
    full = images(item, phase=4)
    assert all(v.getpixel((0, 0)) == (80, 0, 0) for v in full[:12])
    assert all(v.getpixel((0, 0)) == (128, 128, 128) for v in images(item, colpo_only=True))


def targets():
    return {f'{part}{center}{y}': dict(center=center, y=y, reread=[1-y]*6+[y]*6,
                                      reread_valid=True)
            for part in ('t', 's') for center in ('a', 'b') for y in (0, 1)}


def test_source_overlap_and_outer_center_are_rejected():
    t = targets()
    with pytest.raises(ValueError, match='disjoint'):
        training.validate_source_partitions(t, ['ta0'], ['ta0'], 'outer')
    with pytest.raises(ValueError, match='Outer-center'):
        training.validate_source_partitions(t, ['ta0'], ['sa0'], 'a')
    with pytest.raises(ValueError, match='schedule'):
        training.AdaptationPolicy(updates=2)


def test_dual_task_label_routing_and_best_checkpoint_restore(tmp_path, monkeypatch):
    t = targets()
    manifest = {key: {'key': key} for key in t}
    calls, supervised, snapshots = [], [], []
    def fake_images(item, task='diagnosis', site=None, phase=0):
        return SimpleNamespace(key=item['key'], task=task, site=site, phase=phase)
    monkeypatch.setattr(training, 'images', fake_images)

    class Model:
        def __init__(self):
            self.torch = torch
            self.device = 'cpu'
            self.parameters = {'w': torch.nn.Parameter(torch.tensor(.2))}
            self.initial = self.parameters['w'].detach().clone()
            self.latest = None
        def reset(self):
            with torch.no_grad(): self.parameters['w'].copy_(self.initial)
        def enable_training(self, value):
            self.training = value
            self.parameters['w'].requires_grad_(value)
        def prepared(self, images, task):
            assert images.task == task
            return (images,)
        def forward(self, images, grad=False):
            self.latest = images
            if grad: calls.append(images)
            value = (2*int(images.key[-1])-1)*self.parameters['w']
            with torch.set_grad_enabled(grad):
                return {'logits': torch.stack((-value, value))}
        def proximal(self):
            return (self.parameters['w']-self.initial).square()
        def save_adapter(self, path):
            snapshots.append(float(self.parameters['w'].detach()))
            path.write_text(json.dumps(snapshots[-1]))
        def load_adapter(self, path):
            with torch.no_grad(): self.parameters['w'].fill_(json.loads(path.read_text()))
    model = Model()
    ce = torch.nn.functional.cross_entropy
    def tracked_ce(logits, label):
        row = model.latest
        expected = t[row.key]['y'] if row.task == 'diagnosis' else t[row.key]['reread'][row.site-1]
        assert int(label[0]) == expected
        supervised.append((row.task, int(label[0])))
        return ce(logits, label)
    monkeypatch.setattr(torch.nn.functional, 'cross_entropy', tracked_ce)
    result = training.train_source_adapter(
        model, manifest, t, [k for k in t if k.startswith('t')],
        [k for k in t if k.startswith('s')], outer_center='outer',
        variant='DUAL_TASK', output_dir=tmp_path/'run',
        policy=training.AdaptationPolicy(updates=4, checkpoints=(2, 4)))
    assert [row.task for row in calls] == ['diagnosis', 'reread', 'diagnosis', 'reread']
    assert all(row.key.startswith('t') for row in calls)
    assert result['selected_checkpoint']['step'] == 2
    assert (result['diagnosis_updates'], result['reread_updates']) == (2, 2)
    assert len(supervised) == 4 and not model.training
    assert float(model.parameters['w']) == pytest.approx(snapshots[0])
