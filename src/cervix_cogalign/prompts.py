from __future__ import annotations

import re
from typing import Any


SYSTEM_PROMPT = (
    "你是用于回顾性科研的宫颈多模态辅助分析模型，不替代病理诊断。"
    "只根据给定临床信息、阴道镜图像和OCT图像作答；证据不足时必须明确说明。"
)


def clean_value(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "未提供" if not text or text.lower() in {"nan", "none", "null"} else text


def normalize_hpv(value: Any) -> str:
    text = clean_value(value)
    if text == "未提供" or text in {"1", "1.0"}:
        return "状态未知"
    if text in {"-", "阴性", "negative", "Negative"}:
        return "阴性"
    if re.search(r"(^|\D)(16|18)(\D|$)", text):
        return f"HPV16/18相关高危型检出（原始记录：{text}）"
    if re.search(r"\d", text) or re.search(r"阳性|positive|高危", text, flags=re.I):
        return f"其他/未分组高危型检出（原始记录：{text}）"
    return "状态未知"


def normalize_tct(value: Any) -> str:
    text = clean_value(value)
    if text == "未提供" or text in {"1", "1.0", ""}:
        return "结果未知"
    upper = text.upper().replace("_", "-")
    for category in ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "AGC"]:
        if category in upper:
            return category
    if re.search(r"癌|恶性|SCC|CARCINOMA", upper):
        return "癌可疑"
    return "结果未知"


def clinical_block(row: dict[str, Any]) -> str:
    return (
        f"年龄：{clean_value(row.get('age'))}\n"
        f"HPV：{normalize_hpv(row.get('hpv'))}\n"
        f"TCT：{normalize_tct(row.get('tct'))}"
    )


def diagnostic_prompt(row: dict[str, Any]) -> str:
    return f"""请评估该检查位点是否达到CIN2+。第一张拼图为阴道镜序列，第二张拼图为该位点OCT切片序列。

临床信息：
{clinical_block(row)}

按以下标签严格、简洁输出；每个证据字段最多40个汉字，不要复述要求，不要添加标签以外的段落：
<clinical_context>临床先验及其局限</clinical_context>
<colposcopy_morphology>部位、醋白、边界、颜色、血管等宏观形态；不可见则说明</colposcopy_morphology>
<oct_microstructure>上皮分层、厚度、基底膜、信号均匀性与衰减等微结构；不可见则说明</oct_microstructure>
<integration>跨模态一致与冲突证据</integration>
<diagnosis>negative 或 positive</diagnosis>
<probability>0到1之间的CIN2+概率</probability>"""


def evidence_prompt(row: dict[str, Any]) -> str:
    return f"""第一张拼图为阴道镜序列，第二张拼图为该检查位点OCT切片序列。请只记录可观察证据，不得给出疾病名称、分级、良恶性或CIN2+结论。

临床信息：
{clinical_block(row)}

严格输出单行JSON；每个字段最多40个汉字：
{{"clinical_context":"临床先验及局限","colposcopy_morphology":"部位、醋白、边界、颜色和血管的观察","oct_microstructure":"上皮分层、厚度、基底膜、信号均匀性和衰减的观察","integration":"两种图像证据是否一致及不可判定项"}}"""


_DIAGNOSTIC_TERMS = re.compile(
    r"(?i)(cin\s*[123]?\+?|hsil|lsil|癌|恶性|高级别|低级别|阳性|阴性|positive|negative)"
)


def sanitize_observation(text: Any) -> str:
    value = clean_value(text)
    return _DIAGNOSTIC_TERMS.sub("[诊断性表述已移除]", value)


def compose_target(draft: dict[str, Any], label: int) -> str:
    diagnosis = "positive" if int(label) == 1 else "negative"
    probability = "1.0" if int(label) == 1 else "0.0"
    return "\n".join(
        [
            f"<clinical_context>{clean_value(draft.get('clinical_context'))}</clinical_context>",
            f"<colposcopy_morphology>{sanitize_observation(draft.get('colposcopy_morphology'))}</colposcopy_morphology>",
            f"<oct_microstructure>{sanitize_observation(draft.get('oct_microstructure'))}</oct_microstructure>",
            f"<integration>{sanitize_observation(draft.get('integration'))}</integration>",
            f"<diagnosis>{diagnosis}</diagnosis>",
            f"<probability>{probability}</probability>",
        ]
    )
