from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        value = json.loads(fenced)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", fenced, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def parse_prediction(text: str) -> tuple[int | None, float | None]:
    diagnosis_match = re.search(r"<diagnosis>\s*([^<]+)\s*</diagnosis>", text, flags=re.I)
    diagnostic_text = diagnosis_match.group(1) if diagnosis_match else text[-500:]
    normalized = diagnostic_text.lower()
    if re.search(r"\bpositive\b|\bcin\s*2\+\b|阳性", normalized):
        label = 1
    elif re.search(r"\bnegative\b|阴性", normalized):
        label = 0
    else:
        label = None

    probability_match = re.search(
        r"<probability>\s*([01](?:\.\d+)?)\s*</probability>", text, flags=re.I
    )
    probability = float(probability_match.group(1)) if probability_match else None
    if probability is not None and not 0.0 <= probability <= 1.0:
        probability = None
    return label, probability
