"""No weights, upstream installation, GPU, or patient data are needed."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from if_redesign.cfv_jev.visual_jev import (
    CONTEXT, RawPatientReadout, UpstreamBindings, VisualJEVAdapter,
    patient_questions, raw_descriptors,
    prepare_patient,
)


def test_raw_readout_matches_original_torch_equations():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(187)
    logits = rng.normal(size=(7, 21, 2))
    clinical = rng.normal(size=7)
    weights = rng.normal(0, .2, 140)
    mean = rng.normal(0, .2, 33)
    scale = rng.uniform(.5, 1.5, 33)
    readout = RawPatientReadout(weights, mean, scale, .5, 1.5)
    differences = logits[:, :, 1] - logits[:, :, 0]
    q = expit(differences[:, 1:])
    blocks = []
    for v in (q[:, :12], q[:, 12:]):
        z = np.sort(v, axis=1)
        blocks.append(np.c_[z.mean(1), z.max(1), z[:, -2:].mean(1), z.std(1),
                            np.quantile(z, .75, axis=1) - np.quantile(z, .25, axis=1),
                            (z > .5).mean(1), z])
    features = np.c_[np.concatenate(blocks, axis=1), differences[:, 0]]
    x = torch.tensor((features - mean) / scale, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    c = torch.tensor(clinical, dtype=torch.float64)
    hidden = torch.nn.functional.gelu(x @ w[:132].reshape(33, 4), approximate="none")
    expected = (torch.cat([hidden, hidden * torch.tanh(c[:, None] / 3)], dim=1) @ w[132:]).numpy()
    np.testing.assert_allclose(readout.score_value(logits, clinical), expected, atol=1e-13, rtol=1e-13)
    assert readout.score_value(logits[0], clinical[0]) == pytest.approx(expected[0])
    assert readout.effective_scale == pytest.approx(1 / 3)
    restored = RawPatientReadout.from_dict(json.loads(json.dumps(readout.to_dict())))
    np.testing.assert_array_equal(restored.score_value(logits, clinical), readout.score_value(logits, clinical))
    alternate = RawPatientReadout.from_dict({
        "params": weights.tolist(), "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
        "rho": .5, "T": 1.5,
    })
    np.testing.assert_array_equal(alternate.score_value(logits, clinical), readout.score_value(logits, clinical))


def test_raw_descriptor_is_modality_permutation_invariant():
    rng = np.random.default_rng(15)
    q = rng.uniform(size=(3, 20))
    order = np.r_[rng.permutation(12), 12 + rng.permutation(8)]
    assert raw_descriptors(q).shape == (3, 32)
    np.testing.assert_array_equal(raw_descriptors(q), raw_descriptors(q[:, order]))


def test_adapter_prepares_a_fresh_prefix_for_every_intervention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    calls = []

    class Processor:
        image_processor = SimpleNamespace()

        def __call__(self, *, text, images, return_tensors):
            calls.append((text, images))
            assert return_tensors == "pt"
            assert len(images) == 20
            assert all(v.size == (224, 224) and v.mode == "RGB" for v in images)
            return {"input_ids": torch.tensor([[4, 9, 4]]),
                    "pixel_values": torch.tensor([[np.asarray(v).mean()] for v in images]),
                    "image_grid_thw": torch.ones((20, 3), dtype=torch.int64)}

    class Model:
        def __init__(self):
            self.processor = Processor()
            self.tokenizer = lambda text, **kwargs: {"input_ids": [1, 2]}
            self.vl_model = SimpleNamespace(get_rope_index=self.rope)
            self.device_str = "cpu"
            self.image_token_id = 9
            self.seen = []
            self.keep_full_lm_logits = True

        def eval(self):
            return self

        def requires_grad_(self, flag):
            assert flag is False
            return self

        def rope(self, full, image_grid_thw, mm_token_type_ids):
            assert image_grid_thw.shape == (20, 3)
            assert mm_token_type_ids.shape == full.shape
            return torch.zeros((3, 1, full.shape[1]), dtype=torch.int64), None

        def run_prefix_share_batch(self, prepared):
            self.seen.append(prepared)
            return {"lm_option_logits": torch.zeros((21, 2))}

    splits = []
    bindings = UpstreamBindings(
        lambda *args: args,
        lambda context: context + "<|vision_start|><|image_pad|><|vision_end|>",
        lambda instruction, candidates, qtype: instruction,
        lambda tokenizer, prefix, full: splits.append((prefix, full)),
    )
    model = Model()
    adapter = VisualJEVAdapter(model, bindings=bindings)
    images = [np.zeros((28, 30, 3), dtype=np.uint8) for _ in range(20)]
    adapter.logits(images)
    images[3] = None
    adapter.logits(images)
    assert len(calls) == 2 and len(model.seen) == 2 and len(splits) == 42
    assert model.seen[0] is not model.seen[1]
    for text, _ in calls:
        assert CONTEXT in text[0]
        assert text[0].count("<|image_pad|>") == 20
        assert text[0].endswith("<|im_end|>")
    assert np.asarray(calls[1][1][3]).mean() == 128
    assert len(model.seen[0][1]) == 21
    assert model.processor.image_processor.max_pixels == 224 * 224
    assert model.keep_full_lm_logits is False
    readout = RawPatientReadout(np.zeros(140), np.zeros(33), np.ones(33))
    assert adapter.scorer(readout)(images, 1.3) == 0


def test_fixed_patient_prompts_and_readout_validation():
    prompts = patient_questions()
    assert len(prompts) == 21
    assert prompts[0]["candidates"] == ["CIN2+ absent", "CIN2+ present"]
    assert prompts[12]["instruction"].startswith("Does OCT view 12")
    assert prompts[13]["instruction"].startswith("Does colposcopy view 1")
    with pytest.raises(ValueError, match="positive"):
        RawPatientReadout(np.zeros(140), np.zeros(33), np.zeros(33))
    with pytest.raises(ValueError, match="140"):
        RawPatientReadout(np.zeros(26), np.zeros(33), np.ones(33))
    with pytest.raises(ValueError, match="twenty"):
        VisualJEVAdapter._images([None] * 12)


def test_single_question_preparation_preserves_training_and_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    weight = torch.nn.Parameter(torch.tensor(1.))
    recorded = []

    def processor(**kwargs):
        recorded.append(kwargs)
        return {"input_ids": torch.tensor([[1, 9]]), "pixel_values": torch.zeros((20, 3)),
                "image_grid_thw": torch.ones((20, 3), dtype=torch.int64)}

    model = SimpleNamespace(
        training=True, language_weight=weight, processor=processor,
        tokenizer=lambda *args, **kwargs: {"input_ids": [2, 3]},
        device_str="cpu", image_token_id=9,
        vl_model=SimpleNamespace(get_rope_index=lambda full, image_grid_thw:
                                 (torch.zeros((3, 1, full.shape[1])), None)),
    )
    bindings = UpstreamBindings(
        lambda *args: args,
        lambda context: context + "<|vision_start|><|image_pad|><|vision_end|>",
        lambda instruction, candidates, qtype: instruction,
        lambda *args: None,
    )
    question = patient_questions()[0]
    prepared = prepare_patient(model, [None] * 20, [question], bindings=bindings)
    assert len(prepared[1]) == 1 and len(prepared[4]) == 1
    assert prepared[5] == ["choice"] and prepared[6] == [2]
    assert model.training and weight.requires_grad
    assert len(recorded) == 1 and len(recorded[0]["images"]) == 20
