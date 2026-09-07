#!/usr/bin/env python3
"""Summarize the locked source-only M2 performance-cost frontier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    return parser.parse_args()


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "median": float(np.median(array)), "min": float(np.min(array)), "max": float(np.max(array)), "n": len(array)}


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["output_dir"]) / "supplement_m2_cost_frontier"
    payloads = []
    for fold in FOLDS:
        for seed in config["seeds"]:
            path = root / fold / f"seed_{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing M2 frontier run: {path}")
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    names = [f"attention_fixed_{k}" for k in range(1, int(config["policy"]["adaptive_max_visual_units"]) + 1)] + ["adaptive_locked_margin"]
    frontier = []
    for name in names:
        frontier.append({
            "policy": name,
            "macro_auprc": summary([float(item["reports"][name]["macro_auprc"]) for item in payloads]),
            "visual_cost": summary([float(item["reports"][name]["mean_visual_cost"]) for item in payloads]),
        })
    nondominated = 0
    per_run = []
    for item in payloads:
        adaptive = item["reports"]["adaptive_locked_margin"]
        candidates = [item["reports"][f"attention_fixed_{k}"] for k in range(1, int(config["policy"]["adaptive_max_visual_units"]) + 1)]
        dominated = any(float(candidate["mean_visual_cost"]) <= float(adaptive["mean_visual_cost"]) and float(candidate["macro_auprc"]) >= float(adaptive["macro_auprc"]) for candidate in candidates)
        nondominated += int(not dominated)
        per_run.append({"fold": item["fold"], "seed": item["seed"], "adaptive_dominated_by_fixed_lower_or_equal_cost": dominated})
    report = {
        "experiment_id": config["experiment_id"], "analysis": "post-screen exploratory source-only locked-margin M2 frontier", "test_labels_opened": False,
        "outer_test_predictions_written": False, "frontier": frontier,
        "adaptive_non_dominated_runs": {"count": nondominated, "total": len(payloads), "per_run": per_run},
        "interpretation": "This frontier compares the locked adaptive rule with fixed global budgets. It does not establish prospective acquisition efficiency or tune the adaptive threshold after outcomes.",
    }
    (root / "m2_cost_frontier_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ICES M2 locked-margin source-only cost frontier", "", "Exploratory source-only analysis; no outer-test label was evaluated.", "", "| Policy | median macro-AUPRC | median visual units |", "|---|---:|---:|"]
    for item in frontier:
        lines.append(f"| {item['policy']} | {item['macro_auprc']['median']:.4f} | {item['visual_cost']['median']:.2f} |")
    lines.append(f"\nAdaptive rule was non-dominated by a fixed global budget in {nondominated}/{len(payloads)} source-validation runs.")
    (root / "M2_COST_FRONTIER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(root / "m2_cost_frontier_summary.json"), "adaptive_non_dominated_runs": nondominated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
