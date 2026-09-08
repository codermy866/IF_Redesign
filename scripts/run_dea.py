"""Single-GPU adapter-only SFT, utility alignment, DPO and frozen evaluation.

Two processes may run independently on two A6000s. No full LLM checkpoint is
written. DPO reference log probabilities come from the exact starting adapter.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration
from cervix_cogalign.evidence_alignment import (
    assert_lora_only, language_lora_targets, pairwise_rank_loss, dpo_loss, sequence_logp,
)

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Backend:
    def __init__(self, cfg, backbone, adapter=None, train=False):
        self.cfg = cfg
        self.family = cfg["backbones"][backbone]["family"]
        path = ROOT / "models/pretrained" / (backbone if self.family == "qwen" else "llava_med_hf")
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Backbone is not ready: {path}")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_cls = Qwen2_5_VLForConditionalGeneration if self.family == "qwen" else LlavaForConditionalGeneration
        base = model_cls.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        base.requires_grad_(False)
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=train)
        elif train:
            self.model = get_peft_model(base, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                lora_dropout=0.0, target_modules=language_lora_targets(base), task_type="CAUSAL_LM"))
        else:
            self.model = base
        self.model.to("cuda")
        self.model.config.use_cache = False
        if train:
            assert_lora_only(self.model)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.high = self.tokenizer.encode("high", add_special_tokens=False)
        self.low = self.tokenizer.encode("low", add_special_tokens=False)
        if len(self.high) != 1 or len(self.low) != 1:
            raise ValueError("Importance scoring requires single-token high/low labels")

    def encode(self, row, response=None, suffix=""):
        prompt = row["prompt"] + suffix
        if self.family == "qwen":
            content = [{"type": "image"} for _ in row["images"]] + [{"type": "text", "text": prompt}]
            text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        else:
            text = "[INST] " + "<image>\n" * len(row["images"]) + prompt + " [/INST]"
        images = []
        for path in row["images"]:
            with Image.open(path) as im:
                image = im.convert("RGB")
                if self.family == "llava":
                    # Native LLaVA-Med uses aspect-ratio padding, not center crop.
                    bg = tuple(int(255*v) for v in self.processor.image_processor.image_mean)
                    canvas = Image.new("RGB", (max(image.size), max(image.size)), bg)
                    canvas.paste(image, ((canvas.width-image.width)//2, (canvas.height-image.height)//2))
                    image = canvas
                images.append(image)
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        prefix_length = inputs["input_ids"].shape[1]
        if response is not None:
            # Append tokenized completion after the processor-expanded image tokens.
            # The assistant prefix, all prompt tokens and all image tokens are masked.
            end = "<|im_end|>" if self.family == "qwen" else self.tokenizer.eos_token
            response_ids = self.tokenizer.encode(response + end, add_special_tokens=False)
            ids = torch.tensor([response_ids], dtype=torch.long)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], 1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][:, :prefix_length] = -100
        if inputs["input_ids"].shape[1] > self.cfg["max_length"]:
            raise ValueError("Sequence exceeds max_length; refusing silent truncation of images/targets")
        return {k: v.to("cuda") for k, v in inputs.items()}

    def lm_loss(self, row):
        return self.model(**self.encode(row, row["target"])).loss

    def score(self, row, site):
        suffix = f"\nAssess diagnostic importance of site {site} given the available evidence. Output only high or low.\nImportance:"
        inputs = self.encode(row, suffix=suffix)
        logits = self.model(**inputs).logits[0, -1].float()
        return logits[self.high[0]] - logits[self.low[0]]

    def logp(self, row, response):
        inputs = self.encode(row, response)
        labels = inputs.pop("labels")
        return sequence_logp(self.model(**inputs).logits, labels)[0]

    @torch.no_grad()
    def diagnosis(self, row):
        query = dict(row, prompt=row["prompt"] + "\nFor this scoring query output only the CIN2+ class: positive or negative.")
        scores = torch.stack([self.logp(query, s) for s in ("negative", "positive")])
        return float(scores.softmax(0)[1])

    @torch.no_grad()
    def generate(self, row):
        inputs = self.encode(row)
        generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.cfg["max_new_tokens"], use_cache=True,
                                        pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def snapshot(model, optimizer, output, epoch, step, cfg, **extra):
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    torch.save(dict(optimizer=optimizer.state_dict(), epoch=epoch, step=step,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), output / "training_state.pt")
    (output / "run_config.json").write_text(json.dumps(dict(cfg=cfg, **extra), indent=2))
    return str(output)


def train(cfg, args, directory, output):
    is_dpo = args.arm == "alignment_dpo"
    if is_dpo and not args.adapter:
        source = output.parent / "alignment" / "complete.json"
        if not source.exists():
            raise FileNotFoundError("DPO requires completed alignment SFT")
        args.adapter = json.loads(source.read_text())["adapter"]
    backend = Backend(cfg, args.backbone, adapter=args.adapter, train=True)
    rows = read_rows(directory / ("dpo.jsonl" if is_dpo else "train.jsonl"))
    anchors = {r["id"]:r for r in read_rows(directory / "train.jsonl")} if is_dpo else {}
    if is_dpo:
        # Primary optimization cannot be solved from an intervention announcement.
        # Membership/citation preferences remain available as a separate audit.
        rows = [r for r in rows if r["intervention"] in ("matched_original","matched_replacement")]
    if not rows:
        raise ValueError("Empty training dataset")
    optimizer = torch.optim.AdamW((p for p in backend.model.parameters() if p.requires_grad),
                                 lr=cfg["dpo_learning_rate"] if is_dpo else cfg["learning_rate"])
    (output / "trainable_audit.json").write_text(json.dumps(dict(names=assert_lora_only(backend.model),
        trainable=sum(p.numel() for p in backend.model.parameters() if p.requires_grad),
        total=sum(p.numel() for p in backend.model.parameters())), indent=2))
    refs = []
    if is_dpo:
        backend.model.eval()
        with torch.no_grad():
            for row in rows:
                refs.append([float(backend.logp(row, row[k])) for k in ("chosen", "rejected")])
        np.savez(output / "reference_logps.npz", logps=np.array(refs))
        (output / "reference_provenance.json").write_text(json.dumps(dict(adapter=args.adapter,
            data_sha256=hashlib.sha256((directory / "dpo.jsonl").read_bytes()).hexdigest()), indent=2))
    step, records = 0, []
    accumulation = cfg["gradient_accumulation"]
    epochs = cfg["dpo_epochs"] if is_dpo else cfg["epochs"]
    last_adapter = None
    for epoch in range(epochs):
        backend.model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(rows))
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), accumulation):
            batch = order[start:start+accumulation]
            losses, ranks = [], []
            for index in batch:
                row = rows[int(index)]
                if args.arm == "sft":
                    if "sft_target" in row:
                        row = dict(row, target=row["sft_target"])
                    else:
                        target = json.loads(row["target"])
                        for evidence in target["evidence"]:
                            evidence["importance"] = "unassessed"
                        row = dict(row, target=json.dumps(target))
                if is_dpo:
                    pc, pr = [backend.logp(row, row[k]) for k in ("chosen", "rejected")]
                    rc, rr = refs[int(index)]
                    loss = dpo_loss(pc, pr, rc, rr, cfg["dpo_beta"])
                else:
                    loss = backend.lm_loss(row)
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
                if is_dpo and cfg.get("dpo_factual_anchor_weight",0) > 0:
                    anchor_loss = backend.lm_loss(anchors[row["anchor_id"]])
                    (cfg["dpo_factual_anchor_weight"]*anchor_loss/len(batch)).backward()
                if args.arm == "alignment" and cfg["rank_lambda"] > 0:
                    gains = np.asarray(row["gains"])
                    candidates = [(i,j) for i in range(len(gains)) for j in range(i) if abs(gains[i]-gains[j]) > cfg["rank_min_gap"]]
                    if candidates:
                        a,b = candidates[(epoch + int(index)) % len(candidates)]
                        scores = torch.stack([backend.score(row, row["sites"][k]) for k in (a,b)])
                        rank = pairwise_rank_loss(scores, torch.tensor(gains[[a,b]], device="cuda"), cfg["rank_min_gap"])
                        (cfg["rank_lambda"] * rank / len(batch)).backward()
                        ranks.append(float(rank.detach()))
            torch.nn.utils.clip_grad_norm_((p for p in backend.model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = dict(epoch=epoch+1, step=step, loss=float(np.mean(losses)), rank_loss=float(np.mean(ranks)) if ranks else 0,
                          peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
            records.append(record)
            with (output / "train_log.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
            if step % cfg["save_steps"] == 0 or (args.max_steps and step >= args.max_steps):
                last_adapter = snapshot(backend.model, optimizer, output / f"checkpoint-step-{step}", epoch, step, cfg,
                                        arm=args.arm, seed=args.seed, backbone=args.backbone)
            if args.max_steps and step >= args.max_steps:
                (output / "smoke_complete.json").write_text(json.dumps(dict(adapter=last_adapter, steps=step)))
                return
        last_adapter = snapshot(backend.model, optimizer, output / f"checkpoint-epoch-{epoch+1}", epoch, step, cfg,
                                arm=args.arm, seed=args.seed, backbone=args.backbone)
    (output / "complete.json").write_text(json.dumps(dict(adapter=last_adapter, steps=step, epochs=epochs,
        status="training_complete", selection="fixed final epoch; no test tuning"), indent=2))


def evaluate(cfg, args, directory, output):
    if args.arm != "zero_shot" and not args.adapter:
        args.adapter = json.loads((output / "complete.json").read_text())["adapter"]
    backend = Backend(cfg, args.backbone, adapter=args.adapter)
    backend.model.eval()
    rows = read_rows(directory / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    file = output / f"predictions_{args.split}{'_smoke' if args.limit else ''}.jsonl"
    completed = {r["id"] for r in read_rows(file)} if file.exists() else set()
    with torch.no_grad(), file.open("a") as handle:
        for row in rows:
            if row["id"] in completed:
                continue
            scores = [float(backend.score(row, site)) for site in row["sites"]]
            probability = backend.diagnosis(row)
            text = backend.generate(row)
            interventions = {}
            # Label-free perturbation targets determined by model rank, not y.
            for name, slot in (("remove_high", int(np.argmax(scores))), ("remove_low", int(np.argmin(scores)))):
                modified = dict(row, images=[x for j,x in enumerate(row["images"]) if j != slot],
                                sites=[s for j,s in enumerate(row["sites"]) if j != slot])
                # Replace the full image-order mapping; avoid stale positional labels.
                old = ", ".join(f"site {s}" for s in row["sites"])
                new = ", ".join(f"site {s}" for s in modified["sites"])
                modified["prompt"] = row["prompt"].replace("Image order: " + old, "Image order: " + new)
                interventions[name] = dict(removed_site=row["sites"][slot], probability=backend.diagnosis(modified),
                                           response=backend.generate(modified))
            if "audit_normal_donor" in row:
                slot = int(np.argmax(scores))
                donor = row["audit_normal_donor"]
                modified = dict(row, images=list(row["images"]), sites=list(row["sites"]))
                modified["images"][slot] = donor["image"]
                modified["sites"][slot] = donor["site"]
                old = ", ".join(f"site {s}" for s in row["sites"])
                new = ", ".join(f"site {s}" for s in modified["sites"])
                modified["prompt"] = row["prompt"].replace("Image order: " + old, "Image order: " + new)
                interventions["replace_high_normal"] = dict(removed_site=row["sites"][slot], donor_site=donor["site"],
                    probability=backend.diagnosis(modified), response=backend.generate(modified),
                    annotation_informed_audit=True, counterfactual_pathology_known=False)
            result = dict(id=row["id"], index=row["index"], center=row["center"], label=row["label"],
                          sites=row["sites"], importance_scores=scores, probability=probability,
                          response=text, counterfactuals=interventions)
            handle.write(json.dumps(result, ensure_ascii=False)+"\n")
            handle.flush()
            print(json.dumps(dict(evaluated=row["index"], split=args.split)), flush=True)
    (output / f"evaluation_{args.split}{'_smoke' if args.limit else ''}.json").write_text(json.dumps(dict(
        status="complete", n=len(rows), adapter=args.adapter, threshold=0.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dea_v1.json")
    p.add_argument("--fold", required=True)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--backbone", required=True, choices=("qwen3b", "qwen7b", "llava_med"))
    p.add_argument("--arm", required=True, choices=("zero_shot", "sft", "alignment", "alignment_dpo", "alignment_no_rank"))
    p.add_argument("--mode", choices=("train", "evaluate"), default="train")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--adapter")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    directory = ROOT / cfg["output"] / "datasets" / args.fold / f"seed_{args.seed}"
    output = ROOT / cfg["output"] / ("smoke_runs" if args.max_steps else "runs") / args.backbone / args.fold / f"seed_{args.seed}" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "train" and args.arm == "zero_shot":
        raise ValueError("Zero-shot is evaluation only")
    if args.mode == "train" and ((output / "complete.json").exists() or (output / "smoke_complete.json").exists()):
        return
    if args.mode == "train" and list(output.glob("checkpoint-*")):
        raise RuntimeError("Partial run exists; retain checkpoints and use a new output config for retry")
    (output / "resolved_config.json").write_text(json.dumps(dict(config=cfg, arguments=vars(args)), indent=2))
    (train if args.mode == "train" else evaluate)(cfg, args, directory, output)


if __name__ == "__main__":
    main()
