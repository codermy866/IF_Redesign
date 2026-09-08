#!/usr/bin/env python3
"""Aggregate the pilot evidence into one machine-readable result file."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "pilot_v0"


def load(relative: str) -> dict:
    with (RESULTS / relative).open(encoding="utf-8") as source:
        return json.load(source)


def diagnostic(relative: str) -> dict:
    report = load(relative)
    return {
        "n": report["n"],
        "accuracy": report["metrics"]["accuracy"],
        "auroc": report["metrics"]["auroc"],
        "auprc": report["metrics"]["auprc"],
        "brier": report["metrics"]["brier"],
        "format_pass_rate": report["format_pass_rate"],
        "mean_cognition_reward": report["mean_cognition_reward"],
    }


def main() -> None:
    controls = load("controls/metrics.json")
    models = {
        "qwen3vl_zero_shot": diagnostic("evaluation/qwen3vl_zero_shot_val/metrics.json"),
        "diagnosis_only_sft": diagnostic("evaluation/diagnosis_only_sft_val/metrics.json"),
        "hierarchical_sft": diagnostic("evaluation/hierarchical_sft_val/metrics.json"),
    }
    ablations = {
        "within_center_opposite_clinical": diagnostic("evaluation/hierarchical_sft_val_within_center_opposite_clinical/metrics.json"),
        "within_center_opposite_oct": diagnostic("evaluation/hierarchical_sft_val_within_center_opposite_oct/metrics.json"),
        "within_center_opposite_colposcopy": diagnostic("evaluation/hierarchical_sft_val_within_center_opposite_colposcopy/metrics.json"),
        "within_center_opposite_both_images": diagnostic("evaluation/hierarchical_sft_val_within_center_opposite_images/metrics.json"),
    }
    paired = {
        "hierarchical_vs_zero_shot": load("evaluation/compare_zero_shot_vs_hierarchical_sft.json"),
        "hierarchical_vs_diagnosis_only": load("evaluation/compare_diagnosis_only_vs_hierarchical_sft.json"),
        "within_center_opposite_clinical_vs_original": load("evaluation/compare_original_vs_within_center_opposite_clinical.json"),
        "within_center_opposite_oct_vs_original": load("evaluation/compare_original_vs_within_center_opposite_oct.json"),
        "within_center_opposite_colposcopy_vs_original": load("evaluation/compare_original_vs_within_center_opposite_colposcopy.json"),
        "within_center_opposite_both_images_vs_original": load("evaluation/compare_original_vs_within_center_opposite_images.json"),
    }
    base = models["hierarchical_sft"]
    report = {
        "scope": {
            "split": "validation only; pilot test remains unopened",
            "n_train": 64,
            "n_validation": 20,
            "validation_balance": {"negative": 10, "positive": 10},
            "inference_level": "case/site; one case per patient in pilot",
        },
        "model_results": models,
        "data_controls": controls,
        "paired_comparisons": paired,
        "ablation_results": ablations,
        "ablation_accuracy_drop_from_original": {
            name: base["accuracy"] - result["accuracy"] for name, result in ablations.items()
        },
        "evidence_grade": {
            "data_signal_present": True,
            "method_signal_present_in_pilot": True,
            "clinical_shortcut_primary_in_this_pilot": False,
            "colposcopy_dependency_stronger_than_oct_dependency": True,
            "full_cogalign_grpo_evaluated": False,
            "formal_clinical_claim_supported": False,
        },
        "interpretation": [
            "Frozen image embeddings are separable, so the selected data contain diagnostic signal.",
            "Hierarchical SFT outperforms zero-shot and equal-budget diagnosis-only SFT on the 20-case validation pilot.",
            "Opposite-label image replacement reverses much of the performance while opposite-label clinical replacement has only a small effect; the pilot model mainly follows images rather than the clinical fields.",
            "Colposcopy contributes more than OCT in this pilot, although replacing OCT alone also reduces performance.",
            "The pilot is too small and balanced, has no expert cognition-chain labels, and does not evaluate reviewed-ROI GRPO; these results are feasibility evidence only.",
        ],
    }
    target = RESULTS / "evaluation" / "data_vs_method_summary.json"
    with target.open("w", encoding="utf-8") as sink:
        json.dump(report, sink, ensure_ascii=False, indent=2)
    print(target)


if __name__ == "__main__":
    main()
