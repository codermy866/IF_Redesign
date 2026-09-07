#!/usr/bin/env python3
"""Evaluate the locked M2 policy on a source-only fixed-budget frontier.

This does not tune the confidence threshold: the completed v1 checkpoint and
its locked adaptive margin are reused.  The result is an exploratory
performance-cost description, never an outer-test analysis.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def trainer_module():
    spec = importlib.util.spec_from_file_location("ices_train", ROOT / "scripts" / "train_ices_backbone.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load ICES training utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact(report: dict) -> dict:
    return {key: value for key, value in report.items() if not hasattr(value, "shape")}


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train = trainer_module()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    root = Path(config["output_dir"]) / "supplement_m2_cost_frontier"
    root.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        data = train.load_fold_data(config, fold)
        indices = data["source_indices"]["val"]
        for seed in config["seeds"]:
            destination = root / fold / f"seed_{seed}.json"
            if destination.is_file():
                print(f"already complete: {destination}")
                continue
            checkpoint = Path(config["output_dir"]) / "runs" / "cluster_m1_m2" / fold / f"seed_{seed}" / "checkpoint.pt"
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if state.get("test_labels_opened") or state.get("held_out_prediction_written"):
                raise AssertionError("M2 frontier refuses a checkpoint marked as having used held-out outcomes")
            model = train.create_model(config, data, device)
            model.load_state_dict(state["model_state"])
            reports = {}
            for budget in range(1, int(config["policy"]["adaptive_max_visual_units"]) + 1):
                local = copy.deepcopy(config)
                local["policy"]["fixed_visual_budget"] = budget
                reports[f"attention_fixed_{budget}"] = compact(train.evaluate_policy(model, data, indices, device, local, "attention_fixed_cost", int(seed), "cluster_m1_m2"))
                reports[f"random_fixed_{budget}"] = compact(train.evaluate_policy(model, data, indices, device, local, "random_fixed_cost", int(seed), "cluster_m1_m2"))
            reports["adaptive_locked_margin"] = compact(train.evaluate_policy(model, data, indices, device, config, "adaptive_retention", int(seed), "cluster_m1_m2"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "fold": fold, "seed": int(seed), "checkpoint": str(checkpoint), "reports": reports,
                "test_labels_opened": False, "outer_test_predictions_written": False,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"M2 frontier complete fold={fold} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
