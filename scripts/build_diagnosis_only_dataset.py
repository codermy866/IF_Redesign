#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_jsonl, write_jsonl  # noqa: E402


def diagnosis_prompt(content: str) -> str:
    prefix = content.split("按以下标签严格", 1)[0].rstrip()
    return prefix + "\n\n只输出：\n<diagnosis>negative 或 positive</diagnosis>\n<probability>0到1之间的CIN2+概率</probability>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the equal-budget diagnosis-only SFT control")
    parser.add_argument("--baseline-dir", default=str(ROOT / "results/pilot_v0/baseline"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/pilot_v0/diagnosis_only"))
    args = parser.parse_args()
    output = Path(args.output_dir)
    for split in ["train", "val"]:
        prompts, sft = [], []
        for row in read_jsonl(Path(args.baseline_dir) / f"{split}.jsonl"):
            item = dict(row)
            messages = [dict(message) for message in row["messages"]]
            messages[-1]["content"] = diagnosis_prompt(messages[-1]["content"])
            item["messages"] = messages
            prompts.append(item)
            trained = dict(item)
            target = "positive" if int(row["label"]) else "negative"
            probability = "1.0" if int(row["label"]) else "0.0"
            trained["messages"] = messages + [
                {
                    "role": "assistant",
                    "content": f"<diagnosis>{target}</diagnosis>\n<probability>{probability}</probability>",
                }
            ]
            trained["solution"] = target
            sft.append(trained)
        write_jsonl(output / f"{split}_prompt.jsonl", prompts)
        write_jsonl(output / f"{split}_sft.jsonl", sft)
    print(f"diagnosis-only datasets written to {output}")


if __name__ == "__main__":
    main()
