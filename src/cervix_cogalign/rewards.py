from __future__ import annotations

import re
from typing import Any


REQUIRED_TAGS = [
    "clinical_context",
    "colposcopy_morphology",
    "oct_microstructure",
    "integration",
    "diagnosis",
    "probability",
]


def format_reward(text: str) -> float:
    positions = []
    for tag in REQUIRED_TAGS:
        match = re.search(fr"<{tag}>.*?</{tag}>", text, flags=re.I | re.S)
        if not match:
            return 0.0
        positions.append(match.start())
    return 1.0 if positions == sorted(positions) else 0.0


def cognition_reward(text: str) -> float:
    sections = {
        "colposcopy_morphology": ["部位", "醋白", "边界", "颜色", "血管", "不可见", "不足"],
        "oct_microstructure": ["上皮", "分层", "厚度", "基底膜", "信号", "衰减", "不可见", "不足"],
        "integration": ["一致", "冲突", "综合", "证据", "不足", "不确定"],
    }
    earned = 0.0
    for tag, words in sections.items():
        match = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.I | re.S)
        if match:
            hits = sum(word in match.group(1) for word in words)
            earned += min(hits / 3.0, 1.0)
    return earned / len(sections)


def diagnosis_reward(text: str, solution: Any) -> float:
    match = re.search(r"<diagnosis>\s*(positive|negative)\s*</diagnosis>", text, flags=re.I)
    if not match:
        return 0.0
    gold = str(solution).strip().lower()
    if gold in {"1", "true", "positive"}:
        gold = "positive"
    elif gold in {"0", "false", "negative"}:
        gold = "negative"
    return float(match.group(1).lower() == gold)


class CervixFormatReward:
    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        return [format_reward(text) for text in completions]


class CervixCognitionReward:
    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        return [cognition_reward(text) for text in completions]


class CervixDiagnosisReward:
    def __call__(self, completions: list[str], solution: list[Any], **kwargs) -> list[float]:
        return [diagnosis_reward(text, gold) for text, gold in zip(completions, solution)]
