"""Task-routed LoRA construction for Qwen3-VL."""
from peft import LoraConfig, get_peft_model
from torch import nn


TASK_TO_ADAPTER = {"diag": "D", "site": "S", "assim": "A"}


def qwen3vl_adaptation_targets(model):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        language = any(f"model.language_model.layers.{index}." in name for index in range(20, 28))
        language = language and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        vision = any(f"model.visual.blocks.{index}." in name for index in (17, 23))
        vision = vision and name.rsplit(".", 1)[-1] in {"qkv", "proj", "linear_fc1", "linear_fc2"}
        merger = "model.visual.merger." in name and name.rsplit(".", 1)[-1] in {"linear_fc1", "linear_fc2"}
        if language or vision or merger:
            targets.append(name)
    if not targets:
        raise RuntimeError("No Qwen3-VL adaptation targets were found")
    return targets


def _config(rank, targets, dropout):
    return LoraConfig(r=int(rank), lora_alpha=2 * int(rank), lora_dropout=float(dropout), target_modules=targets, task_type="CAUSAL_LM", bias="none")


def build_task_routed_lora(base_model, ranks, dropout=0.0, targets=None):
    """Attach D/S/A adapters whose ranks sum to the fixed parameter budget."""
    if set(ranks) != {"D", "S", "A"} or min(ranks.values()) < 1:
        raise ValueError("ranks must contain positive D, S, and A entries")
    targets = targets or qwen3vl_adaptation_targets(base_model)
    model = get_peft_model(base_model, _config(ranks["D"], targets, dropout), adapter_name="D")
    model.add_adapter("S", _config(ranks["S"], targets, dropout))
    model.add_adapter("A", _config(ranks["A"], targets, dropout))
    return model


def set_task_adapter(model, task):
    model.set_adapter(TASK_TO_ADAPTER[task])
