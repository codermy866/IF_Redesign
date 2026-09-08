#!/usr/bin/env python3
"""Aggregate LOCO controls, zero-shot, SFT comparisons, and three ablation blocks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cervix_cogalign.io import read_json, read_jsonl  # noqa: E402
from cervix_cogalign.metrics import binary_metrics, select_balanced_accuracy_threshold  # noqa: E402
from cervix_cogalign.parsing import parse_prediction  # noqa: E402
from cervix_cogalign.rewards import cognition_reward, format_reward  # noqa: E402
from run_formal_training_queue import ABLATIONS, FOLDS, PRIMARY, SEEDS  # noqa: E402


def extract_rows(path: Path) -> pd.DataFrame:
    rows = []
    for row in read_jsonl(path):
        parsed_label, parsed_probability = parse_prediction(row.get("completion", ""))
        score = row.get("diagnosis_probability")
        if score is None:
            score = parsed_probability
        rows.append(
            {
                "id": str(row["id"]),
                "patient_id": str(row["patient_id"]),
                "center": row["center"],
                "split": row["split"],
                "label": int(row["label"]),
                "generated_label": parsed_label,
                "probability": score,
                "literal_probability": parsed_probability,
                "latency_seconds": row.get("latency_seconds"),
                "format_reward": format_reward(row.get("completion", "")),
                "cognition_reward": cognition_reward(row.get("completion", "")),
            }
        )
    return pd.DataFrame(rows)


def evaluate_partition(frame: pd.DataFrame, threshold: float) -> dict:
    valid = frame.dropna(subset=["probability"])
    result = binary_metrics(valid.label, valid.probability, threshold)
    result.update(
        {
            "score_coverage": float(frame.probability.notna().mean()),
            "parse_coverage": float(frame.generated_label.notna().mean()),
            "format_pass_rate": float(frame.format_reward.mean()),
            "mean_cognition_reward": float(frame.cognition_reward.mean()),
        }
    )
    return result


def evaluate_vlm_job(frame: pd.DataFrame, variant: str, fold: str, seed: str) -> tuple[dict, pd.DataFrame]:
    validation = frame[frame.split == "val"]
    test = frame[frame.split == "test"].copy()
    if validation.empty or test.empty:
        raise ValueError(f"Missing validation/test rows for {(variant, fold, seed)}")
    valid_validation = validation.dropna(subset=["probability"])
    score_coverage = float(len(valid_validation) / len(validation))
    if score_coverage < 0.95:
        raise ValueError(
            f"Validation diagnosis-score coverage is only {score_coverage:.1%} "
            f"for {(variant, fold, seed)}"
        )
    if valid_validation.label.nunique() < 2:
        raise ValueError(f"Validation scores lack both classes for {(variant, fold, seed)}")
    threshold = select_balanced_accuracy_threshold(
        valid_validation.label, valid_validation.probability
    )
    metrics = evaluate_partition(test, threshold)
    metrics.update({"method": variant, "fold": fold, "seed": seed, "validation_threshold": threshold})
    test["method"] = variant
    test["fold"] = fold
    test["seed"] = seed
    test["threshold"] = threshold
    test["predicted_label"] = np.where(
        test.probability.notna(), (test.probability >= threshold).astype(int), np.nan
    )
    return metrics, test


def zero_shot_by_fold(formal_root: Path, assignment: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    raw = extract_rows(formal_root / "all/zero_shot_predictions.jsonl").set_index("id", drop=False)
    summaries, predictions = [], []
    for fold in FOLDS:
        fold_assignment = assignment[assignment.fold == fold].set_index("case_id")
        ids = fold_assignment.index.astype(str)
        frame = raw.loc[ids].copy()
        frame["split"] = frame.id.map(fold_assignment.split)
        metrics, test = evaluate_vlm_job(frame, "qwen3vl_zero_shot", fold, "fixed")
        summaries.append(metrics)
        predictions.append(test)
    return summaries, pd.concat(predictions, ignore_index=True)


def ensemble_primary(predictions: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    summaries, output = [], []
    for variant in PRIMARY:
        part = predictions[predictions.method == variant]
        for fold in FOLDS:
            fold_part = part[part.fold == fold]
            averaged = fold_part.groupby(
                ["id", "patient_id", "center", "split", "label"], as_index=False
            ).agg(
                probability=("probability", "mean"),
                format_reward=("format_reward", "mean"),
                cognition_reward=("cognition_reward", "mean"),
                generated_label=("generated_label", lambda x: int(np.mean(x.dropna()) >= 0.5) if x.notna().any() else np.nan),
            )
            metrics, test = evaluate_vlm_job(averaged, variant + "_3seed_ensemble", fold, "ensemble")
            summaries.append(metrics)
            output.append(test)
    return summaries, pd.concat(output, ignore_index=True)


def macro_summary(metrics: pd.DataFrame) -> dict:
    columns = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "auroc", "auprc", "brier", "ece", "format_pass_rate", "mean_cognition_reward"]
    output = {}
    for method, part in metrics.groupby("method"):
        output[method] = {
            f"macro_{column}_mean": float(part[column].mean())
            for column in columns if column in part
        }
        output[method].update(
            {
                f"macro_{column}_std": float(part[column].std())
                for column in columns if column in part
            }
        )
        output[method]["fold_seed_rows"] = int(len(part))
    return output


def training_summary(formal_root: Path) -> pd.DataFrame:
    rows = []
    for variant in PRIMARY + ABLATIONS:
        seeds = SEEDS if variant in PRIMARY else (SEEDS[0],)
        for fold in FOLDS:
            for seed in seeds:
                log = formal_root / "runs" / variant / fold / f"seed_{seed}" / "checkpoints/logging.jsonl"
                events = read_jsonl(log)
                completed = [event for event in events if "model_parameter_info" in event]
                runtimes = [event for event in events if "train_runtime" in event]
                if not completed or not runtimes:
                    raise ValueError(f"Incomplete training log: {(variant, fold, seed)}")
                final = completed[-1]
                runtime = runtimes[-1]
                rows.append(
                    {
                        "variant": variant,
                        "fold": fold,
                        "seed": seed,
                        "global_step": final["global_step"],
                        "best_metric_eval_loss": final.get("best_metric"),
                        "best_model_checkpoint": final.get("best_model_checkpoint"),
                        "train_runtime_seconds": runtime["train_runtime"],
                        "peak_memory_gib": final.get("memory"),
                        "trainable_parameter_description": final["model_parameter_info"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/formal_v1.json"))
    args = parser.parse_args()
    config = read_json(args.config)
    formal_root = Path(config["output_dir"])
    assignment = pd.read_csv(formal_root / "fold_assignments.csv", dtype={"case_id": str})
    all_metrics, all_predictions, all_eval_frames = [], [], []
    variants = PRIMARY + ABLATIONS
    for variant in variants:
        seeds = SEEDS if variant in PRIMARY else (SEEDS[0],)
        for fold in FOLDS:
            for seed in seeds:
                path = formal_root / "runs" / variant / fold / f"seed_{seed}" / "eval_predictions.jsonl"
                frame = extract_rows(path)
                frame["method"] = variant
                frame["fold"] = fold
                frame["seed"] = str(seed)
                all_eval_frames.append(frame)
                metrics, test = evaluate_vlm_job(frame, variant, fold, str(seed))
                all_metrics.append(metrics)
                all_predictions.append(test)
    zero_metrics, zero_predictions = zero_shot_by_fold(formal_root, assignment)
    all_metrics.extend(zero_metrics)
    all_predictions.append(zero_predictions)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    ensemble_metrics, ensemble_predictions = ensemble_primary(pd.concat(all_eval_frames, ignore_index=True))
    all_metrics.extend(ensemble_metrics)
    all_predictions.append(ensemble_predictions)
    metric_frame = pd.DataFrame(all_metrics)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)

    control_metrics = pd.read_csv(formal_root / "controls/fold_metrics.csv")
    control_metrics["seed"] = "fixed"
    metric_frame = pd.concat([metric_frame, control_metrics], ignore_index=True, sort=False)
    control_predictions = pd.read_csv(
        formal_root / "controls/predictions.csv", dtype={"id": str, "patient_id": str}
    )
    control_predictions["split"] = "test"
    control_predictions["seed"] = "fixed"
    prediction_frame = pd.concat(
        [prediction_frame, control_predictions], ignore_index=True, sort=False
    )
    output = formal_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(output / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(output / "vlm_test_predictions.csv", index=False)
    prediction_frame.to_csv(output / "all_test_predictions.csv", index=False)
    training_summary(formal_root).to_csv(output / "training_summary.csv", index=False)
    latency = prediction_frame.dropna(subset=["latency_seconds"])
    if not latency.empty:
        latency.groupby(["method", "fold", "seed"], as_index=False).agg(
            n=("id", "size"),
            mean_latency_seconds=("latency_seconds", "mean"),
            median_latency_seconds=("latency_seconds", "median"),
        ).to_csv(output / "inference_latency.csv", index=False)
    report = {
        "primary_metric": "macro centre AUPRC",
        "primary_comparators": config["primary_comparators"],
        "ablation_blocks": config["ablation_blocks"],
        "macro_summary": macro_summary(metric_frame),
        "limitations": [
            "The five centres have strongly heterogeneous prevalence.",
            "Cognition targets are label-blind model drafts, not clinician-authored ground truth.",
            "Counterfactual GRPO remains gated because no reviewed lesion ROI exists.",
        ],
    }
    with (output / "formal_summary.json").open("w", encoding="utf-8") as sink:
        json.dump(report, sink, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
