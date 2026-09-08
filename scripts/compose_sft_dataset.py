#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_jsonl, write_jsonl  # noqa: E402
from cervix_cogalign.parsing import extract_json_object  # noqa: E402
from cervix_cogalign.prompts import compose_target  # noqa: E402


REQUIRED = {"clinical_context", "colposcopy_morphology", "oct_microstructure", "integration"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose SFT targets from label-blind evidence drafts")
    parser.add_argument("--drafts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted, audit = [], []
    for row in read_jsonl(args.drafts):
        draft = extract_json_object(row.get("completion", ""))
        missing = sorted(REQUIRED - set(draft or {}))
        ok = draft is not None and not missing
        audit.append({"id": row["id"], "accepted": ok, "missing_fields": missing})
        if not ok:
            continue
        messages = list(row["messages"])
        messages[-1] = dict(messages[-1])
        prompt = messages[-1]["content"]
        prompt = prompt.replace("请只记录可观察证据，不得给出疾病名称、分级、良恶性或CIN2+结论。", "请完成分层证据整合并判断是否达到CIN2+。")
        prompt = prompt.split("严格输出", 1)[0].rstrip() + """

严格按以下标签输出：
<clinical_context>临床先验及其局限</clinical_context>
<colposcopy_morphology>阴道镜宏观形态</colposcopy_morphology>
<oct_microstructure>OCT微结构</oct_microstructure>
<integration>跨模态一致与冲突证据</integration>
<diagnosis>negative 或 positive</diagnosis>
<probability>0到1之间的CIN2+概率</probability>"""
        messages[-1]["content"] = prompt
        messages.append({"role": "assistant", "content": compose_target(draft, int(row["label"]))})
        item = {k: v for k, v in row.items() if k not in {"completion", "latency_seconds", "model_path", "adapter_path"}}
        item["messages"] = messages
        item["solution"] = "positive" if int(row["label"]) else "negative"
        accepted.append(item)
    write_jsonl(args.output, accepted)
    with Path(args.audit).open("w", encoding="utf-8") as f:
        json.dump({"accepted": len(accepted), "total": len(audit), "rows": audit}, f, ensure_ascii=False, indent=2)
    print(f"accepted {len(accepted)}/{len(audit)}")


if __name__ == "__main__":
    main()
