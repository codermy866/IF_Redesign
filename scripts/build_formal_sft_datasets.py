#!/usr/bin/env python3
"""Compose fold-specific SFT and three-block ablation datasets."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json, read_jsonl, stable_score, write_jsonl  # noqa: E402
from cervix_cogalign.parsing import extract_json_object  # noqa: E402
from cervix_cogalign.prompts import clean_value, sanitize_observation  # noqa: E402


VARIANTS = (
    "diagnosis_only",
    "without_integration",
    "full_hierarchy",
    "colposcopy_only",
    "oct_only",
    "without_clinical",
    "shuffled_cognition",
)
REQUIRED = {"clinical_context", "colposcopy_morphology", "oct_microstructure", "integration"}


def target(draft: dict, label: int, variant: str) -> str:
    diagnosis = "positive" if int(label) else "negative"
    probability = "1.0" if int(label) else "0.0"
    values = {
        "clinical_context": clean_value(draft.get("clinical_context")),
        "colposcopy_morphology": sanitize_observation(draft.get("colposcopy_morphology")),
        "oct_microstructure": sanitize_observation(draft.get("oct_microstructure")),
        "integration": sanitize_observation(draft.get("integration")),
    }
    fields = []
    if variant not in {"diagnosis_only", "without_clinical"}:
        fields.append(("clinical_context", values["clinical_context"]))
    if variant not in {"diagnosis_only", "oct_only"}:
        fields.append(("colposcopy_morphology", values["colposcopy_morphology"]))
    if variant not in {"diagnosis_only", "colposcopy_only"}:
        fields.append(("oct_microstructure", values["oct_microstructure"]))
    if variant not in {
        "diagnosis_only",
        "without_integration",
        "colposcopy_only",
        "oct_only",
    }:
        fields.append(("integration", values["integration"]))
    fields.extend([("diagnosis", diagnosis), ("probability", probability)])
    return "\n".join(f"<{name}>{value}</{name}>" for name, value in fields)


def remove_instruction(text: str, tag: str) -> str:
    return re.sub(rf"^<{tag}>.*?</{tag}>\n?", "", text, flags=re.MULTILINE)


def deterministic_clinical_context(item: dict) -> str:
    user = next(message["content"] for message in item["messages"] if message["role"] == "user")
    match = re.search(r"临床信息：\n(.*?)\n\n按以下标签", user, flags=re.DOTALL)
    if not match:
        return "临床资料未提供"
    return "；".join(line.strip() for line in match.group(1).splitlines() if line.strip())


def transform_prompt(row: dict, variant: str) -> dict:
    item = copy.deepcopy(row)
    user = next(message for message in item["messages"] if message["role"] == "user")
    content = user["content"]
    if variant == "diagnosis_only":
        prefix = content.split("按以下标签严格", 1)[0].rstrip()
        content = prefix + "\n\n只输出：\n<diagnosis>negative 或 positive</diagnosis>\n<probability>0到1之间的CIN2+概率</probability>"
    if variant == "without_integration":
        content = remove_instruction(content, "integration")
    if variant == "without_clinical":
        content = re.sub(
            r"临床信息：\n.*?\n\n按以下标签",
            "临床信息：已隐藏（消融设置）\n\n按以下标签",
            content,
            count=1,
            flags=re.DOTALL,
        )
        content = remove_instruction(content, "clinical_context")
    if variant == "colposcopy_only":
        content = content.replace("<image>\n<image>\n", "<image>\n", 1)
        content = content.replace(
            "第一张拼图为阴道镜序列，第二张拼图为该位点OCT切片序列。",
            "唯一拼图为阴道镜序列。",
        )
        content = remove_instruction(content, "oct_microstructure")
        content = remove_instruction(content, "integration")
        item["images"] = [item["images"][0]]
    if variant == "oct_only":
        content = content.replace("<image>\n<image>\n", "<image>\n", 1)
        content = content.replace(
            "第一张拼图为阴道镜序列，第二张拼图为该位点OCT切片序列。",
            "唯一拼图为该位点OCT切片序列。",
        )
        content = remove_instruction(content, "colposcopy_morphology")
        content = remove_instruction(content, "integration")
        item["images"] = [item["images"][1]]
    user["content"] = content
    item["variant"] = variant
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/formal_v1.json"))
    parser.add_argument("--drafts", default=str(ROOT / "results/formal_v1/all/evidence_drafts.jsonl"))
    args = parser.parse_args()
    config = read_json(args.config)
    formal_root = Path(config["output_dir"])
    parsed_drafts = {}
    rejected = []
    for row in read_jsonl(args.drafts):
        draft = extract_json_object(row.get("completion", ""))
        missing = sorted(REQUIRED - set(draft or {}))
        if draft is None or missing:
            rejected.append({"id": row["id"], "missing": missing})
        else:
            parsed_drafts[str(row["id"])] = draft

    audits = []
    for fold in config["centres"].values():
        split_rows = {
            split: read_jsonl(formal_root / "folds" / fold / "baseline" / f"{split}.jsonl")
            for split in ("train", "val", "test")
        }
        for variant in VARIANTS:
            variant_dir = formal_root / "folds" / fold / "variants" / variant
            evaluation_prompts = []
            for split, rows in split_rows.items():
                prompts = [transform_prompt(row, variant) for row in rows]
                write_jsonl(variant_dir / f"{split}_prompt.jsonl", prompts)
                if split in {"val", "test"}:
                    evaluation_prompts.extend(prompts)
                if split == "test":
                    continue
                usable = [row for row in prompts if str(row["id"]) in parsed_drafts]
                donor = {}
                if variant == "shuffled_cognition" and split == "train":
                    ordered = sorted(usable, key=lambda row: stable_score(row["id"], fold, "shuffle", seed=int(config["seed"])))
                    rotated = ordered[1:] + ordered[:1]
                    donor = {str(row["id"]): str(other["id"]) for row, other in zip(ordered, rotated)}
                sft = []
                for item in usable:
                    identifier = str(item["id"])
                    evidence_id = donor.get(identifier, identifier)
                    evidence = dict(parsed_drafts[evidence_id])
                    evidence["clinical_context"] = deterministic_clinical_context(item)
                    trained = copy.deepcopy(item)
                    trained["messages"].append(
                        {
                            "role": "assistant",
                            "content": target(evidence, int(item["label"]), variant),
                        }
                    )
                    trained["solution"] = "positive" if int(item["label"]) else "negative"
                    if evidence_id != identifier:
                        trained["cognition_donor_id"] = evidence_id
                    sft.append(trained)
                write_jsonl(variant_dir / f"{split}_sft.jsonl", sft)
                audits.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "split": split,
                        "available": len(rows),
                        "accepted": len(sft),
                    }
                )
            write_jsonl(variant_dir / "eval_prompt.jsonl", evaluation_prompts)
    with (formal_root / "sft_build_audit.json").open("w", encoding="utf-8") as sink:
        json.dump(
            {"drafts_accepted": len(parsed_drafts), "drafts_rejected": rejected, "datasets": audits},
            sink,
            ensure_ascii=False,
            indent=2,
        )
    print(f"drafts accepted {len(parsed_drafts)}/{len(parsed_drafts) + len(rejected)}")
    print(f"wrote {len(audits)} train/validation variant datasets")


if __name__ == "__main__":
    main()
