#!/usr/bin/env python3
"""Evaluate fixed CESL policies under predeclared evidence-availability stress.

The policies are not reoptimized under a stress scenario.  This is therefore a
post-selection input-availability stress test, not a prospective missing-data
prevalence or adaptive-reacquisition study.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cervix_cogalign.cesl import SLOT_IDS, sigmoid  # noqa: E402
from cervix_cogalign.io import read_json  # noqa: E402
from evaluate_cesl_fold import evaluation_root, infer, load_model  # noqa: E402
from train_cesl_backbone import load_fold_data  # noqa: E402


POLICY_BACKBONE = {
    "C0_full_evidence": "c0_full",
    "C3_cev_adaptive": "selection",
    "C4_cesl_css": "css",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold")
    parser.add_argument("seed", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def scenario_masks(base: np.ndarray, scenario: str) -> np.ndarray:
    result = base.copy()
    if scenario == "missing_clinical":
        result[:, 0] = False
    elif scenario == "missing_colposcopy":
        result[:, 1:5] = False
    elif scenario == "missing_oct":
        result[:, 5:] = False
    elif scenario == "reduced_colposcopy_atoms":
        result[:, [2, 4]] = False
    elif scenario == "reduced_oct_atoms":
        result[:, [6, 8, 10]] = False
    else:
        raise ValueError(f"Unknown stress scenario: {scenario}")
    if (~result).all(axis=1).any():
        raise AssertionError(f"Stress scenario {scenario} removed every selected atom from at least one case")
    return result


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = evaluation_root(config, args.fold, args.seed)
    marker = output / "availability_stress_complete.json"
    if marker.is_file() and (output / "availability_stress_predictions.csv").is_file():
        print(f"already complete: {output}")
        return
    test_path = output / "test_predictions.csv"
    calibration_path = output / "calibration.json"
    if not test_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("Availability stress requires completed frozen CESL evaluation")
    data = load_fold_data(config, args.fold)
    test_indices = data["parts"]["test"]["index"].astype(int).tolist()
    index_by_id = {str(identifier): index for index, identifier in enumerate(data["ids"])}
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(test_path, dtype={"id": str, "patient_id": str})
    labels = data["payload"]["labels"][test_indices].numpy().astype(int)
    rows: list[dict] = []
    for policy, backbone in POLICY_BACKBONE.items():
        part = frame[frame.policy == policy].copy().set_index("id").loc[[str(data["ids"][index]) for index in test_indices]].reset_index()
        if len(part) != len(test_indices):
            raise AssertionError(f"Stress test input is incomplete for policy={policy}")
        base = np.zeros((len(part), len(SLOT_IDS)), dtype=bool)
        for local, selected_json in enumerate(part.selected_slots):
            for name in json.loads(selected_json):
                base[local, SLOT_IDS.index(name)] = True
            if policy == "C0_full_evidence":
                base[local] = data["payload"]["available"][test_indices[local]].numpy().astype(bool)
            else:
                base[local, 0] = True
        model = load_model(config, data, args.fold, backbone, args.seed, device)
        values = calibration[policy]
        temperature = float(values["temperature_source_validation"])
        threshold = float(values["threshold_source_validation"])
        for scenario in config["robustness_scenarios"]:
            mask = scenario_masks(base, scenario)
            logits, _ = infer(model, data, test_indices, mask, device, int(config["model"]["batch_size"]))
            probability = np.asarray(sigmoid(logits / temperature), dtype=float)
            for local, global_index in enumerate(test_indices):
                rows.append(
                    {
                        "id": str(data["ids"][global_index]),
                        "patient_id": str(data["payload"]["patient_ids"][global_index]),
                        "center": str(data["payload"]["centers"][global_index]),
                        "fold": args.fold,
                        "seed": args.seed,
                        "policy": policy,
                        "scenario": scenario,
                        "label": int(labels[local]),
                        "probability": float(probability[local]),
                        "threshold": threshold,
                        "predicted_label": int(probability[local] >= threshold),
                        "remaining_visual_atoms": int(mask[local, 1:].sum()),
                        "fixed_policy_stress": True,
                    }
                )
    pd.DataFrame(rows).to_csv(output / "availability_stress_predictions.csv", index=False)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "fold": args.fold,
                "seed": args.seed,
                "policies": list(POLICY_BACKBONE),
                "scenarios": config["robustness_scenarios"],
                "interpretation": "post-selection input-availability stress only; policies were not reoptimized",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"CESL AVAILABILITY STRESS COMPLETE fold={args.fold} seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
