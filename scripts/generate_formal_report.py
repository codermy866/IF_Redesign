#!/usr/bin/env python3
"""Generate the Chinese Formal v1 report directly from locked result tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/formal_v1"
ANALYSIS = FORMAL / "analysis"
LABELS = {
    "clinical_only_lr": "Clinical-only LR",
    "frozen_multimodal_linear": "Frozen multimodal linear",
    "qwen3vl_zero_shot": "Qwen3-VL zero-shot",
    "diagnosis_only_3seed_ensemble": "Diagnosis-only SFT (3-seed)",
    "full_hierarchy_3seed_ensemble": "Hierarchical SFT (3-seed)",
    "diagnosis_only": "Diagnosis-only",
    "without_integration": "Without integration",
    "colposcopy_only": "Colposcopy-only",
    "oct_only": "OCT-only",
    "without_clinical": "Without clinical",
    "shuffled_cognition": "Shuffled cognition",
    "full_hierarchy": "Full hierarchy",
}
MAIN = list(LABELS)[:5]


def ci_cell(metrics: pd.DataFrame, method: str, metric: str) -> str:
    row = metrics[
        (metrics.method == method)
        & (metrics.scale == "macro_centre")
        & (metrics.metric == metric)
    ].iloc[0]
    return f"{row.value:.3f} [{row.ci95_low:.3f}, {row.ci95_high:.3f}]"


def comparison_row(comparisons: pd.DataFrame, group: str, reference: str) -> pd.Series:
    return comparisons[
        (comparisons.comparison_group == group)
        & (comparisons.reference == reference)
        & (comparisons.scale == "macro_centre")
        & (comparisons.metric == "auprc")
    ].iloc[0]


def direction(row: pd.Series) -> str:
    if row.ci95_low > 0:
        return "支持完整方法优于参照"
    if row.ci95_high < 0:
        return "完整方法低于参照"
    return "区间跨 0，证据不确定"


def main() -> None:
    metrics = pd.read_csv(ANALYSIS / "main_metrics_with_ci.csv")
    comparisons = pd.read_csv(ANALYSIS / "paired_comparisons.csv")
    folds = pd.read_csv(ANALYSIS / "fold_metrics.csv")
    cohort = pd.read_csv(FORMAL / "fold_summary.csv")
    test_cohort = cohort[cohort.split == "test"]
    stats = json.loads((ANALYSIS / "statistical_analysis.json").read_text(encoding="utf-8"))

    lines = [
        "# Formal v1a 五中心留一验证结果",
        "",
        "> 本报告由患者级预测自动生成；没有手工录入性能数字。",
        "",
        "## 队列与分析状态",
        "",
        f"共纳入 {int(test_cohort.n.sum())} 个独立患者/检查位点，外层为 5 折逐中心留一验证。",
        f"置信区间使用 {stats['bootstrap_replicates']:,} 次中心内、结局分层的患者 bootstrap。",
        "测试中心保持自然患病率；阈值只在每折源中心验证集确定。",
        "",
        "## 五个主要方法",
        "",
        "| 方法 | 宏平均 AUPRC (95% CI) | 宏平均 AUROC (95% CI) | 宏平均 balanced accuracy (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for method in MAIN:
        lines.append(
            f"| {LABELS[method]} | {ci_cell(metrics, method, 'auprc')} | "
            f"{ci_cell(metrics, method, 'auroc')} | {ci_cell(metrics, method, 'balanced_accuracy')} |"
        )

    lines.extend(
        [
            "",
            "## 逐中心主方法结果",
            "",
            "| 留出中心 | AUPRC | AUROC | Balanced accuracy | 分数覆盖率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    hierarchy_folds = folds[folds.method == "full_hierarchy_3seed_ensemble"].copy()
    for _, row in hierarchy_folds.iterrows():
        lines.append(
            f"| {row.fold} | {row.auprc:.3f} | {row.auroc:.3f} | "
            f"{row.balanced_accuracy:.3f} | {row.score_coverage:.1%} |"
        )

    lines.extend(
        [
            "",
            "## 三个消融板块",
            "",
            "下表为同一种子完整分层方法减去消融变体的宏平均 AUPRC；正值表示完整方法更好。",
            "",
            "| 板块 | 被移除/破坏的部分 | ΔAUPRC (95% CI) | Holm 校正 p | 判断 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for block, references in {
        "认知结构": ("diagnosis_only", "without_integration"),
        "模态贡献": ("colposcopy_only", "oct_only"),
        "监督完整性": ("without_clinical", "shuffled_cognition"),
    }.items():
        group = {
            "认知结构": "cognition_structure",
            "模态贡献": "modality_contribution",
            "监督完整性": "supervision_integrity",
        }[block]
        for reference in references:
            row = comparison_row(comparisons, group, reference)
            adjusted = row.get("holm_adjusted_p_within_family", float("nan"))
            lines.append(
                f"| {block} | {LABELS[reference]} | {row.delta:+.3f} "
                f"[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] | {adjusted:.4f} | {direction(row)} |"
            )

    diagnosis = comparison_row(comparisons, "main", "diagnosis_only_3seed_ensemble")
    frozen = comparison_row(comparisons, "main", "frozen_multimodal_linear")
    centre_spread = hierarchy_folds.auprc.max() - hierarchy_folds.auprc.min()
    conclusion = direction(diagnosis)
    if diagnosis.ci95_low > 0 and frozen.ci95_low > 0:
        attribution = "方法贡献得到支持：分层方法同时优于等预算诊断式 SFT 和冻结多模态表征。"
    elif diagnosis.ci95_high < 0:
        attribution = "当前方法没有带来收益；问题不能归因于仅仅缺少训练，分层监督本身可能有害或目标质量不足。"
    else:
        attribution = "方法增益尚不确定；现有结果更符合数据域偏移与方法效果共同作用，不能把失败单独归因于模型或数据。"
    lines.extend(
        [
            "",
            "## 效果由数据还是方法决定",
            "",
            f"- 主方法相对等预算 Diagnosis-only 的配对判断：{conclusion}。",
            f"- 主方法的逐中心 AUPRC 极差为 {centre_spread:.3f}；若该值较大，说明中心域和患病率对表观效果影响很强。",
            f"- 综合判断：{attribution}",
            "- 数据端已知风险包括五中心阳性率 7.79%–91.01%、临床字段缺失，以及当前标签清单与历史索引存在不一致；旧实验数字因此不能直接复用。",
            "- 方法端已知风险是分层认知目标来自标签盲模型草稿而非医师逐例撰写。Shuffled-cognition 消融若不下降，就不能声称认知链本身有效。",
            "",
            "## 声明边界",
            "",
            "当前结果只检验 Formal v1a 的分层 SFT。由于没有经临床复核的病灶 ROI，反事实 GRPO 仍被门控；不得把本结果写成完整复现论文的病灶擦除强化学习，也不得作临床部署结论。",
            "",
        ]
    )
    target = ROOT / "docs/FORMAL_V1_RESULTS.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
