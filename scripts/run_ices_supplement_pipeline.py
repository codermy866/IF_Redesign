#!/usr/bin/env python3
"""Execute the finite ICES post-screen source-only supplementary protocol.

Stages are intentionally sequential to avoid GPU contention and to make the
state durable. A failed stage stops the pipeline; it never falls through to an
outer-test evaluation or parameter search.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def write_status(path: Path, *, stage: str, status: str, detail: str = "") -> None:
    path.write_text(json.dumps({"stage": stage, "status": status, "detail": detail, "updated_at_unix": time.time(), "outer_test_labels_opened": False, "outer_test_predictions_written": False}, ensure_ascii=False, indent=2), encoding="utf-8")


def complete(path: Path) -> bool:
    return path.is_file()


def run_stage(name: str, command: list[str], log: Path, status: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    write_status(status, stage=name, status="running", detail=" ".join(command))
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        write_status(status, stage=name, status="failed", detail=f"returncode={process.returncode}; see {log}")
        raise SystemExit(f"ICES supplementary stage failed: {name}")
    write_status(status, stage=name, status="complete", detail=str(log))


def run_m1_extraction(gpus: list[str], status: Path, logs: Path, target: Path) -> None:
    if target.is_file() and target.with_suffix(".audit.json").is_file():
        return
    name = "m1_frame_sensitivity_feature_extraction"
    write_status(status, stage=name, status="running")
    processes = []
    handles = []
    for index, gpu in enumerate(gpus):
        log = logs / f"{name}_shard{index}.log"
        handle = log.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/extract_ices_m1_frame_sensitivity.py"), "--device", "cuda", "--shard-index", str(index), "--num-shards", str(len(gpus)), "--output", str(target)]
        handles.append(handle)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    if any(codes):
        write_status(status, stage=name, status="failed", detail=f"shard return codes={codes}")
        raise SystemExit("ICES M1 frame-sensitivity feature extraction failed")
    run_stage(name + "_merge", [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/extract_ices_m1_frame_sensitivity.py"), "--merge", "--num-shards", str(len(gpus)), "--output", str(target)], logs / f"{name}_merge.log", status)


def main() -> None:
    args = parse_args()
    if len(args.gpus) < 1:
        raise ValueError("At least one GPU must be specified")
    output = ROOT / "results/ices_v1"
    logs = output / "supplement_pipeline_logs"
    status = output / "supplement_pipeline_status.json"
    py = str(ROOT / ".venv/bin/python")
    base = ROOT / "configs/ices_v1_exploratory.json"
    clean = ROOT / "configs/ices_v1_clean_m3_supplement.json"
    frame1 = ROOT / "configs/ices_v1_m1_frame01_supplement.json"
    frame10 = ROOT / "configs/ices_v1_m1_frame10_supplement.json"

    frontier_summary = output / "supplement_m2_cost_frontier/m2_cost_frontier_summary.json"
    if not complete(frontier_summary):
        run_stage("m2_locked_cost_frontier", [py, str(ROOT / "scripts/run_ices_m2_cost_frontier.py"), "--config", str(base), "--device", "cuda"], logs / "m2_frontier.log", status)
        run_stage("m2_locked_cost_frontier_aggregate", [py, str(ROOT / "scripts/aggregate_ices_m2_cost_frontier.py"), "--config", str(base)], logs / "m2_frontier_aggregate.log", status)

    clean_summary = Path(json.loads(clean.read_text(encoding="utf-8"))["output_dir"]) / "clean_m3_supplement_summary.json"
    if not complete(clean_summary):
        run_stage("clean_m3_training", [py, str(ROOT / "scripts/run_ices_training_queue.py"), "--config", str(clean), "--gpus", *args.gpus, "--backbones", "cluster_selected_control", "cluster_sufficiency_control", "cluster_pure_m3"], logs / "clean_m3_training.log", status)
        run_stage("clean_m3_aggregate", [py, str(ROOT / "scripts/aggregate_ices_clean_m3_supplement.py"), "--config", str(clean)], logs / "clean_m3_aggregate.log", status)

    target = output / "supplement_m1_frame_sensitivity/features/m1_frame_sensitivity_resnet50.pt"
    run_m1_extraction(args.gpus, status, logs, target)
    m1_summary = output / "supplement_m1_frame_sensitivity/m1_frame_sensitivity_summary.json"
    if not complete(m1_summary):
        for name, config in (("m1_frame01_training", frame1), ("m1_frame10_training", frame10)):
            run_stage(name, [py, str(ROOT / "scripts/run_ices_training_queue.py"), "--config", str(config), "--gpus", *args.gpus, "--backbones", "raw_m1_ablation"], logs / f"{name}.log", status)
        run_stage("m1_frame_sensitivity_aggregate", [py, str(ROOT / "scripts/aggregate_ices_m1_frame_sensitivity.py"), "--base-config", str(base), "--frame1-config", str(frame1), "--frame10-config", str(frame10)], logs / "m1_frame_sensitivity_aggregate.log", status)
    write_status(status, stage="all_supplementary_source_only_experiments", status="complete", detail="No additional tuning and no outer-test evaluation were run.")
    print("ICES supplementary source-only protocol complete")


if __name__ == "__main__":
    main()
