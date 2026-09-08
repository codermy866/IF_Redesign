#!/usr/bin/env python3
"""Retrain only unreadable Formal v1 seed-20260902 adapters into an isolated tree."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Xiangyang seed-20260902 retained a later readable adapter after its recovery.
# The four earlier held-out centres retain only owner-private adapter files.
JOBS = tuple(
    (variant, fold, 20260902)
    for fold in ("shiyan", "enshi", "wuhan", "jingzhou")
    for variant in ("full_hierarchy", "diagnosis_only")
)
RECOVERED = ROOT / "results/formal_v1/recovered_runs"
STATUS = ROOT / "results/formal_v1/recovered_adapter_campaign_status.json"


def checkpoint_root(job: tuple[str, str, int]) -> Path:
    variant, fold, seed = job
    return RECOVERED / variant / fold / f"seed_{seed}" / "checkpoints"


def complete(job: tuple[str, str, int]) -> bool:
    root = checkpoint_root(job)
    adapters = list(root.glob("checkpoint-*/adapter_model.safetensors"))
    if not adapters:
        return False
    log = root.parent / "train_console.log"
    if log.is_file() and "End time of running main:" in log.read_text(encoding="utf-8", errors="replace")[-20000:]:
        return True
    for state_path in root.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) >= int(state.get("max_steps", 10**12)):
            return True
    return False


def write_status(completed: int, pending: list[tuple[str, str, int]], active: dict[str, tuple[subprocess.Popen, tuple[str, str, int], object]], failures: list[dict]) -> None:
    STATUS.write_text(json.dumps({
        "planned": len(JOBS), "completed": completed, "pending": len(pending),
        "active": [{"gpu": gpu, "job": job} for gpu, (_, job, _) in active.items()],
        "failures": failures, "output_root": str(RECOVERED),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    args = parser.parse_args()
    pending = [job for job in JOBS if not complete(job)]
    completed = len(JOBS) - len(pending)
    active: dict[str, tuple[subprocess.Popen, tuple[str, str, int], object]] = {}
    failures: list[dict] = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            variant, fold, seed = job
            log_path = checkpoint_root(job).parent / "train_console.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "FORMAL_RUNS_ROOT": str(RECOVERED), "FORMAL_SKIP_COMPLETION_MARKER": "true"})
            process = subprocess.Popen(["bash", str(ROOT / "scripts/train_formal_sft.sh"), fold, variant, str(seed)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, job, log)
        time.sleep(5)
        for gpu in list(active):
            process, job, log = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code == 0 and complete(job):
                completed += 1
            else:
                failures.append({"job": job, "returncode": code})
            del active[gpu]
        write_status(completed, pending, active, failures)
    if failures:
        raise SystemExit(f"Adapter recovery finished with {len(failures)} failures")


if __name__ == "__main__":
    main()
