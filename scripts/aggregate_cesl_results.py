#!/usr/bin/env python3
"""Aggregate completed CESL seed/fold predictions without changing any policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.io import read_json, read_jsonl  # noqa: E402
from cervix_cogalign.metrics import select_balanced_accuracy_threshold  # noqa: E402


FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")
LINEAR_METHODS = (
    "clinical_only_lr",
    "raw_colposcopy_meanpool_linear",
    "raw_oct_meanpool_linear",
    "raw_full_meanpool_linear",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str, "patient_id": str, "seed": str, "fold": str, "policy": str})


def neural_paths(output: Path, seeds: list[int]) -> list[Path]:
    paths = []
    for fold in FOLDS:
        for seed in seeds:
            root = output / "evaluations" / fold / f"seed_{seed}"
            marker = root / "evaluation_complete.json"
            path = root / "test_predictions.csv"
            val = root / "validation_predictions.csv"
            if not marker.is_file() or not path.is_file() or not val.is_file():
                raise FileNotFoundError(f"Missing completed CESL evaluation artifact: {root}")
            paths.extend([path, val])
    return paths


def ensemble_predictions(frame: pd.DataFrame, *, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fold", "policy", "id", "patient_id", "center", "label"]
    aggregate = (
        frame.groupby(keys, as_index=False)
        .agg(
            probability=("probability", "mean"),
            selected_visual_atoms=("selected_visual_atoms", "mean"),
            relative_evidence_cost=("relative_evidence_cost", "mean"),
            stopping_criterion_met=("stopping_criterion_met", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    if (aggregate.n_seeds != 3).any():
        raise AssertionError("Every neural CESL ensemble case must contain exactly three seed predictions")
    thresholds = []
    if split == "test":
        raise ValueError("Test thresholds must be supplied from source-validation ensembles")
    for (fold, policy), part in aggregate.groupby(["fold", "policy"], sort=True):
        threshold = select_balanced_accuracy_threshold(part.label.astype(int), part.probability.astype(float))
        thresholds.append({"fold": fold, "policy": policy, "threshold_source_validation_ensemble": threshold})
    return aggregate, pd.DataFrame(thresholds)


def apply_thresholds(frame: pd.DataFrame, thresholds: pd.DataFrame, method_family: str) -> pd.DataFrame:
    merged = frame.merge(thresholds, on=["fold", "policy"], how="left", validate="many_to_one")
    if merged.threshold_source_validation_ensemble.isna().any():
        raise AssertionError("A test prediction lacks its source-validation threshold")
    merged["predicted_label"] = (merged.probability >= merged.threshold_source_validation_ensemble).astype(int)
    merged["method"] = merged.policy
    merged["method_family"] = method_family
    merged["seed"] = "three_seed_ensemble"
    return merged


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(config["output_dir"])
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    paths = neural_paths(output, [int(seed) for seed in config["seeds"]])
    neural_test = pd.concat([read_csv(path) for path in paths if path.name == "test_predictions.csv"], ignore_index=True)
    neural_val = pd.concat([read_csv(path) for path in paths if path.name == "validation_predictions.csv"], ignore_index=True)
    val_ensemble, thresholds = ensemble_predictions(neural_val, split="val")
    test_keys = ["fold", "policy", "id", "patient_id", "center", "label"]
    test_ensemble = (
        neural_test.groupby(test_keys, as_index=False)
        .agg(
            probability=("probability", "mean"),
            selected_visual_atoms=("selected_visual_atoms", "mean"),
            relative_evidence_cost=("relative_evidence_cost", "mean"),
            stopping_criterion_met=("stopping_criterion_met", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    if (test_ensemble.n_seeds != 3).any():
        raise AssertionError("Every neural CESL test ensemble case must contain exactly three seed predictions")
    neural_val_final = apply_thresholds(val_ensemble, thresholds, "cesl_neural")
    neural_test_final = apply_thresholds(test_ensemble, thresholds, "cesl_neural")

    linear_test = []
    linear_val = []
    for fold in FOLDS:
        root = output / "linear_controls" / fold
        if not (root / "complete.json").is_file():
            raise FileNotFoundError(f"Missing raw-atom linear controls for fold={fold}")
        linear_test.append(read_csv(root / "test_predictions.csv"))
        linear_val.append(read_csv(root / "validation_predictions.csv"))
    linear_test_frame = pd.concat(linear_test, ignore_index=True)
    linear_val_frame = pd.concat(linear_val, ignore_index=True)
    if set(linear_test_frame.policy) != set(LINEAR_METHODS):
        raise AssertionError("Linear controls do not match the locked comparator set")
    linear_thresholds = []
    for (fold, policy), part in linear_val_frame.groupby(["fold", "policy"], sort=True):
        linear_thresholds.append(
            {
                "fold": fold,
                "policy": policy,
                "threshold_source_validation_ensemble": select_balanced_accuracy_threshold(part.label.astype(int), part.probability.astype(float)),
            }
        )
    linear_threshold_frame = pd.DataFrame(linear_thresholds)
    linear_val_final = apply_thresholds(linear_val_frame, linear_threshold_frame, "frozen_raw_linear")
    linear_test_final = apply_thresholds(linear_test_frame, linear_threshold_frame, "frozen_raw_linear")
    final_test = pd.concat([neural_test_final, linear_test_final], ignore_index=True)
    final_val = pd.concat([neural_val_final, linear_val_final], ignore_index=True)
    expected = len(read_jsonl(config["atom_manifest"]))
    coverage = final_test.groupby("method").id.nunique()
    if not (coverage == expected).all():
        raise AssertionError(f"A method does not cover the complete held-out CESL cohort: expected={expected}, got={coverage.to_dict()}")
    final_test.to_csv(analysis / "all_test_predictions.csv", index=False)
    final_val.to_csv(analysis / "all_validation_predictions.csv", index=False)
    thresholds.to_csv(analysis / "ensemble_thresholds.csv", index=False)
    linear_threshold_frame.to_csv(analysis / "linear_thresholds.csv", index=False)

    ancillary = {
        "mechanism_audit.csv": "mechanism_audit.csv",
        "rank_profiles.csv": "rank_profiles.csv",
        "q_sensitivity.csv": "q_sensitivity.csv",
    }
    for target_name, source_name in ancillary.items():
        files = [output / "evaluations" / fold / f"seed_{seed}" / source_name for fold in FOLDS for seed in config["seeds"]]
        if not all(path.is_file() for path in files):
            raise FileNotFoundError(f"Missing ancillary CESL artifact for {target_name}")
        pd.concat([pd.read_csv(path) for path in files], ignore_index=True).to_csv(analysis / target_name, index=False)
    donor_index = [
        {"fold": fold, "seed": int(seed), "path": str((output / "evaluations" / fold / f"seed_{seed}" / "donor_edges.csv.gz").resolve())}
        for fold in FOLDS
        for seed in config["seeds"]
    ]
    (analysis / "donor_audit_index.json").write_text(json.dumps(donor_index, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "experiment_id": config["experiment_id"],
        "n_test_cases_per_method": {method: int(part.id.nunique()) for method, part in final_test.groupby("method")},
        "methods": sorted(final_test.method.unique().tolist()),
        "neural_seed_ensemble": 3,
        "held_out_policy_labels_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    (analysis / "aggregation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
