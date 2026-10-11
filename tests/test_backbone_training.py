import random
from types import SimpleNamespace

import numpy as np
import pytest

from if_redesign.cfv_jev.backbone_training import (
    BackboneTrainingPolicy, TrainingPatient, patient_answer_loss, train_source_backbone,
    source_selection_nll,
)


def test_frozen_policy_and_budget_control_masking():
    policy = BackboneTrainingPolicy()
    assert policy.steps == 512 and policy.selection_interval == 16
    assert policy.betas == (.9, .95) and policy.weight_decay == 0
    assert (policy.lora_rank, policy.lora_alpha, policy.lora_dropout) == (16, 32, .05)
    assert set(policy.language_projection_names) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }
    views = list(range(20))
    for modality in ("OK", "O", "K"):
        p = BackboneTrainingPolicy(modality=modality)
        evaluation = p.evaluation_views(views, lambda: None)
        assert evaluation[:12] == ([None] * 12 if modality == "K" else views[:12])
        assert evaluation[12:] == ([None] * 8 if modality == "O" else views[12:])
        for draw, branch in ((.1, "full"), (.6, "drop_oct"), (.95, "drop_colpo")):
            masked, actual = p.apply_views(views, SimpleNamespace(random=lambda: draw), lambda: None)
            assert actual == branch and len(masked) == 20 and views == list(range(20))
            if branch == "drop_oct" or modality == "K":
                assert masked[:12] == [None] * 12
            if branch == "drop_colpo" or modality == "O":
                assert masked[12:] == [None] * 8
    assert policy.scheduled_learning_rate(0) == pytest.approx(1e-4 / 51)
    assert policy.scheduled_learning_rate(51) == pytest.approx(1e-4)


def test_mask_policy_reproducibility():
    p = BackboneTrainingPolicy()
    def sequence():
        rng = random.Random(17)
        return [p.apply_views(list(range(20)), rng, lambda: None) for _ in range(100)]
    assert sequence() == sequence()
    assert {branch for _, branch in sequence()} == {"full", "drop_oct", "drop_colpo"}


def test_source_only_checkpoint_selection_restores_best_trained_state():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_weight = torch.nn.Parameter(torch.tensor(.3))
            self.keep_full_lm_logits = False

    train = [TrainingPatient(f"train{i}", list(range(20)), i % 2) for i in range(5)]
    selection = [TrainingPatient("selection", list(range(20)), 1)]
    model = Model()
    evaluations, prepared_questions = [], []
    snapshots = []

    def score(model, patients):
        assert patients is selection
        assert not model.training and not model.keep_full_lm_logits
        snapshots.append(model.language_weight.detach().clone())
        evaluations.append(len(evaluations))
        return [1., 4., 5.][len(evaluations) - 1]

    def prepare(model, views, question):
        assert model.training and model.keep_full_lm_logits
        assert len(views) == 20
        prepared_questions.append(question)
        return views

    result = train_source_backbone(
        model, train, selection, prepare_training=prepare, selection_nll=score,
        policy=BackboneTrainingPolicy(steps=4, selection_interval=2),
        neutral_factory=lambda: None,
        loss_callback=lambda model, prepared, target: (model.language_weight - target).square(),
    )
    assert result.selected_step == 2 and result.selection_nll == 4
    assert result.baseline_selection_nll == 1  # Step zero remains a reference.
    assert [r["step"] for r in result.trace] == [0, 2, 4]
    torch.testing.assert_close(model.language_weight, snapshots[1])
    assert sum(result.mask_counts[k] for k in ("full", "drop_oct", "drop_colpo")) == 20
    assert len(prepared_questions) == 20
    assert all("likely to have CIN2" in q["instruction"] for q in prepared_questions)
    assert not model.training and not model.keep_full_lm_logits
    with pytest.raises(ValueError, match="disjoint"):
        train_source_backbone(model, train, train, prepare_training=prepare, selection_nll=score)


def test_whole_vocabulary_patient_loss_uses_option_token_not_binary_softmax():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2., .4, -1., 3., -.5, .2]], requires_grad=True)
    model = SimpleNamespace(
        device_str="cpu", dtype=torch.float32, option_token_ids=torch.tensor([2, 5]),
        forward_train=lambda *args: {"lm_logits": logits},
    )
    prepared = SimpleNamespace(
        prefix_ids=torch.tensor([1, 2]), suffix_ids=[torch.tensor([3])],
        pixel_values=torch.zeros((20, 3)), image_grid_thw=torch.ones((20, 3)),
        position_ids=[torch.zeros((3, 3), dtype=torch.int64)],
    )
    loss = patient_answer_loss(model, prepared, 1)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([5]))
    torch.testing.assert_close(loss, expected)
    assert float(loss.detach()) != pytest.approx(float(
        torch.nn.functional.cross_entropy(logits[:, [2, 5]], torch.tensor([1])).detach(),
    ))
    loss.backward()
    assert logits.grad.shape == logits.shape and torch.isfinite(logits.grad).all()


def test_default_trainer_prepares_patient_only_loss_and_fresh_source_selection():
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from scipy.special import expit
    from if_redesign.cfv_jev.visual_jev import UpstreamBindings
    prepared_inputs, selection_groups, training_calls = [], [], []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_weight = torch.nn.Parameter(torch.tensor(.3))
            self.device_str, self.dtype, self.image_token_id = "cpu", torch.float32, 9
            self.option_token_ids = torch.tensor([0, 1])
            self.keep_full_lm_logits = False
            self.processor = self.process
            self.tokenizer = lambda *args, **kwargs: {"input_ids": [2, 3]}
            self.vl_model = SimpleNamespace(get_rope_index=self.rope)

        def process(self, *, text, images, return_tensors):
            prepared_inputs.append([np.asarray(v).mean() for v in images])
            return {"input_ids": torch.tensor([[1, 9]]), "pixel_values": torch.zeros((20, 3)),
                    "image_grid_thw": torch.ones((20, 3), dtype=torch.int64)}

        def rope(self, full, image_grid_thw):
            return torch.zeros((3, 1, full.shape[1]), dtype=torch.int64), None

        def run_prefix_share_batch(self, group):
            assert not self.training and not self.keep_full_lm_logits
            assert len(group.suffix_ids) == 21
            selection_groups.append(group)
            result = torch.zeros((21, 2))
            result[0, 1] = self.language_weight
            return {"lm_option_logits": result}

        def forward_train(self, pixels, geometry, full, positions, mask, last, qtypes, counts):
            assert self.training and self.keep_full_lm_logits
            assert qtypes == ["choice"] and counts == [2]
            training_calls.append(full)
            logits = torch.stack([self.language_weight, -self.language_weight, self.language_weight * 0])[None]
            return {"lm_logits": logits}

    def group_factory(prefix, suffixes, pixels, geometry, positions, qtypes, counts, token):
        return SimpleNamespace(prefix_ids=prefix, suffix_ids=suffixes, pixel_values=pixels,
                               image_grid_thw=geometry, position_ids=positions)

    bindings = UpstreamBindings(
        group_factory,
        lambda context: context + "<|vision_start|><|image_pad|><|vision_end|>",
        lambda instruction, candidates, qtype: instruction,
        lambda *args: None,
    )
    views = [np.zeros((24, 25, 3), dtype=np.uint8) for _ in range(20)]
    train = [TrainingPatient("train", views, 1)]
    selection = [TrainingPatient("selection", views, 1)]
    model = Model()
    policy = BackboneTrainingPolicy(modality="O", steps=2, selection_interval=1)
    nll = source_selection_nll(model, selection, policy, bindings=bindings)
    assert nll == pytest.approx(-np.log(expit(.3)), rel=1e-6)
    assert prepared_inputs[0] == [0.] * 12 + [128.] * 8
    assert model.language_weight.requires_grad
    result = train_source_backbone(model, train, selection, policy=policy, bindings=bindings)
    assert len(training_calls) == 2 and len(selection_groups) == 4
    assert len({id(group) for group in selection_groups}) == 4
    assert result.selected_step in (1, 2) and result.trace[0]["selectable"] is False
    assert model.language_weight.requires_grad and not model.training
