"""Train task-routed Supervision-Matched Adaptation on JSONL task records."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .modeling import build_task_routed_lora, set_task_adapter
from .objectives import evidence_assimilation_loss, evidence_utility
from .sampling import build_training_schedule


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def conversation(prompt, image=None):
    content = []
    if image:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": "Cervical OCT research model."}]},
        {"role": "user", "content": content},
    ]


def representation(model, processor, prompt, image, device):
    inputs = processor.apply_chat_template(
        conversation(prompt, image), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    output = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = output.hidden_states[-1][:, -1].float()
    return F.layer_norm(hidden, (hidden.shape[-1],))


def validate_rows(rows):
    allowed = {"diag", "site", "assim"}
    for index, row in enumerate(rows):
        task = row.get("task")
        if task not in allowed:
            raise ValueError(f"Row {index} has invalid task {task!r}")
        for field in ("patient_key", "prompt", "label"):
            if field not in row:
                raise ValueError(f"Row {index} is missing {field}")
        if task == "assim" and "baseline_prompt" not in row:
            raise ValueError(f"Assimilation row {index} is missing baseline_prompt")


def train(model, processor, risk_head, site_head, rows, config, device):
    schedule = build_training_schedule(
        rows, int(config["steps"]),
        {
            "diag": float(config["task_exposure"]["D"]),
            "site": float(config["task_exposure"]["S"]),
            "assim": float(config["task_exposure"]["A"]),
        },
        float(config["site_positive_fraction"]),
        seed=int(config["seed"]),
    )
    adapter_parameters = [parameter for name, parameter in model.named_parameters() if "lora_" in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(config["learning_rate"])},
            {"params": list(risk_head.parameters()) + list(site_head.parameters()), "lr": float(config["head_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    model.train(); risk_head.train(); site_head.train()
    history = []
    for step, row in enumerate(schedule, start=1):
        task = row["task"]
        set_task_adapter(model, task)
        optimizer.zero_grad(set_to_none=True)
        label = torch.tensor([float(row["label"])], device=device)
        if task == "diag":
            hidden = representation(model, processor, row["prompt"], row.get("image"), device)
            loss = F.binary_cross_entropy_with_logits(risk_head(hidden).squeeze(-1), label)
        elif task == "site":
            hidden = representation(model, processor, row["prompt"], row.get("image"), device)
            loss = F.binary_cross_entropy_with_logits(site_head(hidden).squeeze(-1), label)
        else:
            before = representation(model, processor, row["baseline_prompt"], None, device)
            after = representation(model, processor, row["prompt"], row.get("image"), device)
            potential = torch.tensor([float(row.get("p_gain", 0.0))], device=device)
            loss, _ = evidence_assimilation_loss(
                risk_head(before).squeeze(-1), risk_head(after).squeeze(-1), label, potential,
                direction_margin=config["direction_margin"], gap_delta=config["gap_delta"],
                gap_weight=config["gap_weight"],
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        active = [parameter for parameter in model.parameters() if parameter.requires_grad]
        active.extend(risk_head.parameters()); active.extend(site_head.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(active, float(config["gradient_clip_norm"]))
        optimizer.step()
        history.append({"step": step, "task": task, "loss": float(loss.detach()), "gradient_norm": float(grad_norm)})
    return history


@torch.no_grad()
def evaluate(model, processor, risk_head, site_head, rows, device):
    model.eval(); risk_head.eval(); site_head.eval()
    predictions = []
    for row in rows:
        task = row["task"]
        set_task_adapter(model, task)
        head = site_head if task == "site" else risk_head
        hidden = representation(model, processor, row["prompt"], row.get("image"), device)
        probability = torch.sigmoid(head(hidden).squeeze(-1))
        result = {**row, "probability": float(probability.item())}
        if task == "assim":
            before_hidden = representation(model, processor, row["baseline_prompt"], None, device)
            before = torch.sigmoid(risk_head(before_hidden).squeeze(-1))
            label = torch.tensor([float(row["label"])], device=device)
            utility = evidence_utility(label, before, probability)
            result.update(baseline_probability=float(before.item()), assimilation=float(utility.item()))
        predictions.append(result)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    train_rows = read_jsonl(args.train_jsonl)
    validation_rows = read_jsonl(args.validation_jsonl)
    validate_rows(train_rows); validate_rows(validation_rows)
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    base.requires_grad_(False)
    base.config.use_cache = False
    model = build_task_routed_lora(base, config["ranks"], config.get("lora_dropout", 0.0))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.max_pixels = int(config["max_pixels"])
    hidden_size = int(model.get_base_model().config.text_config.hidden_size)
    risk_head = nn.Linear(hidden_size, 1).to(args.device)
    site_head = nn.Linear(hidden_size, 1).to(args.device)
    nn.init.zeros_(risk_head.weight); nn.init.zeros_(risk_head.bias)
    nn.init.zeros_(site_head.weight); nn.init.zeros_(site_head.bias)
    history = train(model, processor, risk_head, site_head, train_rows, config, args.device)
    predictions = evaluate(model, processor, risk_head, site_head, validation_rows, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "adapters", selected_adapters=["D", "S", "A"])
    torch.save({"risk_head": risk_head.state_dict(), "site_head": site_head.state_dict()}, args.output / "heads.pt")
    with (args.output / "training_history.jsonl").open("w") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")
    with (args.output / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
