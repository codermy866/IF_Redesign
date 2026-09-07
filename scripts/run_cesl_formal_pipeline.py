#!/usr/bin/env python3
"""Resumable CESL-v2 formal retrospective experiment supervisor."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def write_status(path: Path, *, status: str, step: str, completed: list[str], detail: str = "") -> None:
    path.write_text(
        json.dumps(
            {"status": status, "current_step": step, "completed_steps": completed, "detail": detail, "updated_unix": time.time()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(command: list[str], *, log) -> None:
    log.write("$ " + " ".join(command) + "\n")
    log.flush()
    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Pipeline command failed with returncode={completed.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    feature_path = output / config["image_encoder"]["feature_file"]
    if not feature_path.is_file():
        raise FileNotFoundError(f"Cannot start CESL formal pipeline before feature extraction: {feature_path}")
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = output / "formal_pipeline_status.json"
    completed: list[str] = []
    with (log_root / "formal_pipeline.log").open("a", encoding="utf-8") as log:
        try:
            write_status(status_path, status="running", step="linear_controls_and_training", completed=completed)
            linear = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_cesl_linear_queue.py"), "--config", str(args.config)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_cesl_training_queue.py"), "--config", str(args.config), "--gpus", *args.gpus],
                log=log,
            )
            if linear.wait() != 0:
                raise RuntimeError("CESL linear-control queue failed")
            completed.extend(["linear_controls", "backbone_training"])
            write_status(status_path, status="running", step="frozen_held_out_evaluation", completed=completed)
            run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_cesl_evaluation_queue.py"), "--config", str(args.config), "--gpus", *args.gpus],
                log=log,
            )
            completed.append("frozen_held_out_evaluation")
            write_status(status_path, status="running", step="availability_stress", completed=completed)
            run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_cesl_stress_queue.py"), "--config", str(args.config), "--gpus", *args.gpus],
                log=log,
            )
            completed.append("availability_stress")
            write_status(status_path, status="running", step="aggregation_and_statistics", completed=completed)
            run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/aggregate_cesl_results.py"), "--config", str(args.config)], log=log)
            run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_cesl_statistics.py"), "--config", str(args.config)], log=log)
            run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/aggregate_cesl_availability_stress.py"), "--config", str(args.config)], log=log)
            run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/audit_cesl_evidence_gates.py"), "--config", str(args.config)], log=log)
            completed.extend(["aggregation", "statistics", "evidence_gate_audit"])
            write_status(status_path, status="complete", step="complete", completed=completed)
        except Exception as error:
            write_status(status_path, status="failed", step="failed", completed=completed, detail=str(error))
            raise
    print("CESL FORMAL PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
