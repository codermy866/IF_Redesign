#!/usr/bin/env python3
"""Summarize source-only M1 reference-frame sensitivity without test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")
SEEDS = (20260905, 20260906, 20260907)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--frame1-config", default=str(ROOT / "configs/ices_v1_m1_frame01_supplement.json"))
    parser.add_argument("--frame10-config", default=str(ROOT / "configs/ices_v1_m1_frame10_supplement.json"))
    return parser.parse_args()


def config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(output: Path, arm: str, fold: str, seed: int, policy: str = "attention_fixed_cost") -> float:
    path = output / "runs" / arm / fold / f"seed_{seed}" / "source_validation_metrics.json"
    return float(json.loads(path.read_text(encoding="utf-8"))[policy]["macro_auprc"])


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()), "n": len(array)}


def main() -> None:
    args = parse_args()
    base, frame1, frame10 = map(config, (args.base_config, args.frame1_config, args.frame10_config))
    definitions = {
        "frame_1_same_CS": (Path(frame1["output_dir"]), "raw_m1_ablation"),
        "frame_5_same_CS": (Path(base["output_dir"]), "raw_m1_ablation"),
        "frame_10_same_CS": (Path(frame10["output_dir"]), "raw_m1_ablation"),
        "ten_frame_CS_cluster": (Path(base["output_dir"]), "cluster_m1_m2"),
    }
    reports = {}
    for name, (output, arm) in definitions.items():
        values = [metric(output, arm, fold, seed) for fold in FOLDS for seed in SEEDS]
        reports[name] = summary(values)
    cluster_values = [metric(Path(base["output_dir"]), "cluster_m1_m2", fold, seed) for fold in FOLDS for seed in SEEDS]
    contrasts = {}
    for name, (output, arm) in definitions.items():
        if name == "ten_frame_CS_cluster":
            continue
        values = [metric(Path(base["output_dir"]), "cluster_m1_m2", fold, seed) - metric(output, arm, fold, seed) for fold in FOLDS for seed in SEEDS]
        contrasts[f"cluster_minus_{name}"] = summary(values)
    result_root = Path(base["output_dir"]) / "supplement_m1_frame_sensitivity"
    result_root.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis": "post-screen exploratory source-only M1 reference-frame sensitivity", "test_labels_opened": False,
        "outer_test_predictions_written": False, "single_frame_controls": reports, "cluster_minus_single_frame": contrasts,
        "interpretation": "The analysis tests whether the M1 comparison depends on the selected same-position frame. It does not establish temporal or anatomical OCT semantics.",
    }
    (result_root / "m1_frame_sensitivity_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ICES M1 source-only reference-frame sensitivity", "", "Exploratory source-only analysis; no outer-test labels were evaluated.", "", "| Representation | median macro-AUPRC |", "|---|---:|"]
    for name, values in reports.items():
        lines.append(f"| {name} | {values['median']:.4f} |")
    lines.append("\nTen-frame pooling is only supported if it is consistently favorable across the prespecified frame-1, frame-5 and frame-10 controls.")
    (result_root / "M1_FRAME_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(result_root / "m1_frame_sensitivity_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
