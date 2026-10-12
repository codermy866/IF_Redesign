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


@pytest.mark.parametrize('variant', ['PATIENT_ONLY', 'DUAL_TASK'])
def test_task_label_routing_and_best_checkpoint_restore(tmp_path, monkeypatch, variant):
    t = targets()
    if variant == 'PATIENT_ONLY':
        t = {key: {k: value for k, value in row.items() if k in ('center', 'y')}
             for key, row in t.items()}
    manifest = {key: {'key': key, 'oct_sites': {str(s): ['synthetic'] for s in range(1, 13)}} for key in t}
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
        variant=variant, output_dir=tmp_path/'run',
        policy=training.AdaptationPolicy(updates=4, checkpoints=(2, 4)))
    expected_tasks = ['diagnosis']*4 if variant == 'PATIENT_ONLY' else ['diagnosis', 'reread', 'diagnosis', 'reread']
    assert [row.task for row in calls] == expected_tasks
    assert all(row.key.startswith('t') for row in calls)
    assert result['selected_checkpoint']['step'] == 2
    assert (result['diagnosis_updates'], result['reread_updates']) == ((4, 0) if variant == 'PATIENT_ONLY' else (2, 2))
    assert len(supervised) == 4 and not model.training
    assert float(model.parameters['w']) == pytest.approx(snapshots[0])


def test_missing_positions_stay_neutral_and_partial_bags_do_not_index_past_end(tmp_path):
    path = tmp_path/'one.png'
    Image.new('RGB', (300, 200), (20, 30, 40)).save(path)
    item = {'oct_sites': {'2': [str(path)], '3': None, '4': []}}
    full = images(item, phase=9)
    assert len(full) == 20 and full[1].getpixel((0, 0)) == (20, 30, 40)
    assert all(v.getpixel((0, 0)) == (128, 128, 128) for j, v in enumerate(full) if j != 1)
    missing = images(item, 'reread', site=1)
    assert all(v.getpixel((0, 0)) == (128, 128, 128) for v in missing)
    partial = images(item, 'reread', site=2)
    assert partial[0].getpixel((0, 0)) == (20, 30, 40)
    assert all(v.getpixel((0, 0)) == (128, 128, 128) for v in partial[1:])


def test_patient_targets_are_usable_without_any_reread_metadata():
    t = {key: {k: value for k, value in row.items() if k in ('center', 'y')}
         for key, row in targets().items()}
    training.validate_source_partitions(t, ['ta0', 'ta1'], ['sa0', 'sa1'], 'outer')
    # Missing weak metadata is masked; actual dual-task sampling checks usable strata.
    training.validate_source_partitions(t, ['ta0', 'ta1'], ['sa0', 'sa1'], 'outer', require_reread=True)
