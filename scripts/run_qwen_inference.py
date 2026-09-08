#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic Qwen3-VL inference with resumable JSONL output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument(
        "--score-diagnosis",
        action="store_true",
        help="Store positive-vs-negative probability at the generated <diagnosis> token",
    )
    return parser.parse_args()


def to_qwen_messages(row: dict) -> list[dict]:
    system = next((m["content"] for m in row["messages"] if m["role"] == "system"), "")
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    user = user.replace("<image>\n", "", len(row.get("images", [])))
    content = [{"type": "image", "image": path} for path in row.get("images", [])]
    content.append({"type": "text", "text": user})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if output.is_file():
        completed = {str(row["id"]) for row in read_jsonl(output)}

    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    processor.tokenizer.padding_side = "left"
    diagnosis_open = processor.tokenizer.encode("<diagnosis>", add_special_tokens=False)
    positive_token = processor.tokenizer.encode("positive", add_special_tokens=False)
    negative_token = processor.tokenizer.encode("negative", add_special_tokens=False)
    if args.score_diagnosis and (len(positive_token) != 1 or len(negative_token) != 1):
        raise RuntimeError("Diagnosis labels must each map to one token")
    pending = [row for row in rows if str(row["id"]) not in completed]
    with output.open("a", encoding="utf-8") as sink, torch.inference_mode():
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            started = time.time()
            messages = [to_qwen_messages(row) for row in batch]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(args.device)
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
            )
            if args.sample:
                generation_kwargs.update(temperature=0.7, top_p=0.8, top_k=20)
            if args.score_diagnosis:
                generation_kwargs.update(return_dict_in_generate=True, output_scores=True)
            generation = model.generate(**generation_kwargs)
            sequences = generation.sequences if args.score_diagnosis else generation
            trimmed = sequences[:, inputs.input_ids.shape[1] :]
            completions = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            elapsed = time.time() - started
            for batch_index, (row, completion) in enumerate(zip(batch, completions), start=1):
                result = dict(row)
                result["completion"] = completion
                result["latency_seconds"] = round(elapsed / len(batch), 4)
                result["model_path"] = str(Path(args.model).resolve())
                result["adapter_path"] = str(Path(args.adapter).resolve()) if args.adapter else None
                if args.score_diagnosis:
                    token_ids = trimmed[batch_index - 1].tolist()
                    score = None
                    for token_index in range(len(token_ids) - len(diagnosis_open)):
                        if token_ids[token_index : token_index + len(diagnosis_open)] == diagnosis_open:
                            label_index = token_index + len(diagnosis_open)
                            logits = generation.scores[label_index][
                                batch_index - 1,
                                [negative_token[0], positive_token[0]],
                            ].float()
                            score = float(torch.softmax(logits, dim=0)[1].item())
                            break
                    result["diagnosis_probability"] = score
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                index = offset + batch_index
                print(f"[{index}/{len(pending)}] {row['id']} {result['latency_seconds']}s", flush=True)
            sink.flush()


if __name__ == "__main__":
    main()
