import json
import numpy as np
import pytest
import torch
from cervix_cogalign.evidence_alignment import (
    clinical_prompt, pairwise_rank_loss, dpo_loss, sequence_logp,
    validate_oof, counterfactual_pair, preference_text, language_lora_targets,
)


def test_prompt_allowlist():
    prompt = clinical_prompt(dict(HPV="16", TCT="ASC-US", age=44, pathology="SECRET", center="HIDDEN"), [1,4])
    assert "SECRET" not in prompt and "HIDDEN" not in prompt
    assert "44" in prompt and "ASC-US" in prompt


def test_oof_excludes_both_fit_and_calibration():
    validate_oof(2, dict(gain_indices=[2], fit_indices=[0], cal_indices=[1]))
    with pytest.raises(ValueError):
        validate_oof(2, dict(gain_indices=[2], fit_indices=[0], cal_indices=[2]))


def test_rank_direction_ties_and_gradient():
    gains = torch.tensor([0.1, 0.9])
    good = torch.tensor([-1.,1.], requires_grad=True)
    bad = torch.tensor([1.,-1.])
    assert pairwise_rank_loss(good,gains) < pairwise_rank_loss(bad,gains)
    pairwise_rank_loss(good,gains).backward()
    assert good.grad[0] > 0 and good.grad[1] < 0
    assert pairwise_rank_loss(good,torch.ones(2)).item() == 0


def test_dpo_exact_reference_and_preference_direction():
    same = dpo_loss(torch.tensor(-4.),torch.tensor(-5.),-4.,-5.)
    assert same.item() == pytest.approx(np.log(2))
    assert dpo_loss(torch.tensor(-3.),torch.tensor(-5.),-4.,-5.) < same


def test_prompt_tokens_not_scored():
    logits = torch.zeros(1,4,3)
    labels = torch.tensor([[-100,-100,1,2]])
    assert sequence_logp(logits,labels).item() == pytest.approx(-2*np.log(3))


def test_cf_donor_is_unselected_negative_and_no_label_flip():
    cf = counterfactual_pair([1,2], [0.4,0.1], [1,1,0,-1], [1,2,3,4])
    assert cf == {"removed":1,"donor":3}
    chosen,rejected = preference_text(1,"replacement")
    assert "patient pathology is not changed" in chosen and "patient pathology is not changed" in rejected
    assert counterfactual_pair([1],[-0.1],[1],[1]) is None


def test_no_visual_lora_targets():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.ModuleDict({"q_proj":torch.nn.Linear(2,2)})
            self.language_model = torch.nn.ModuleDict({"q_proj":torch.nn.Linear(2,2)})
    assert language_lora_targets(Tiny()) == ["language_model.q_proj"]


def test_llava_checkpoint_key_conversion():
    from convert_dea_llava import convert_key
    assert convert_key("model.layers.0.self_attn.q_proj.weight") == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert convert_key("model.mm_projector.2.bias") == "model.multi_modal_projector.linear_2.bias"
    assert convert_key("model.vision_tower.vision_tower.vision_model.embeddings.class_embedding") == "model.vision_tower.vision_model.embeddings.class_embedding"
    with pytest.raises(ValueError):convert_key("unknown.weight")


def test_evidence_metrics_handle_invalid_json_and_absent_sites():
    from summarize_dea import grounded_fraction,metrics,bootstrap
    assert grounded_fraction("unstructured text",[1]) is None
    assert grounded_fraction(json.dumps({"evidence":[{"site":1},{"site":2}]}),[1]) == .5
    assert metrics([1,1],[.7,.8])["AUROC"] is None
    result=bootstrap(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]),np.array(["A","A","B","B"]),20,
                     other=np.array([.1,.9,.2,.8]))
    assert result["AUROC"]["CI95"] == [0.,0.]
