#!/usr/bin/env python3
"""Resumable two-GPU queue for the frozen source-only VEC screen."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/vec_v1_source_screen"))
    parser.add_argument("--job-script", default=str(ROOT / "scripts/run_vec_source_screen.py"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--draws", type=int, default=3)
    return parser.parse_args()


def complete(output: Path, fold: str, seed: int) -> bool:
    root = output / "jobs" / fold / f"seed_{seed}"
    return (root / "complete.json").is_file() and (root / "source_validation_arrays.pt").is_file()


def write_status(path: Path, jobs: list[tuple[str, int]], active: dict[str, tuple], output: Path, failures: list[dict]) -> None:
    active_jobs = {job for _, job, _, _ in active.values()}
    path.write_text(
        json.dumps(
            {
                "planned": len(jobs),
                "completed": sum(complete(output, *job) for job in jobs),
                "pending": sum(not complete(output, *job) and job not in active_jobs for job in jobs),
                "active": [{"gpu": gpu, "fold": job[0], "seed": job[1], "pid": process.pid} for gpu, (process, job, _, _) in active.items()],
                "failures": failures,
                "held_out_test_labels_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    base_config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    assignments = pd.read_csv(base_config["fold_assignments"], dtype={"fold": str})
    folds = sorted(assignments.fold.unique().tolist())
    jobs = [(fold, int(seed)) for fold in folds for seed in base_config["seeds"]]
    output = Path(args.output_dir)
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not complete(output, *job)]
    active: dict[str, tuple[subprocess.Popen, tuple[str, int], object, float]] = {}
    failures: list[dict] = []
    status_path = output / "source_screen_status.json"
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            fold, seed = pending.pop(0)
            log = (log_root / f"source_screen_{fold}_{seed}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(args.job_script),
                fold,
                str(seed),
                "--base-config",
                str(args.base_config),
                "--output-dir",
                str(output),
                "--budget",
                str(args.budget),
                "--draws",
                str(args.draws),
                "--device",
                "cuda",
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, (fold, seed), log, time.time())
            print(f"START gpu={gpu} fold={fold} seed={seed}", flush=True)
        write_status(status_path, jobs, active, output, failures)
        time.sleep(5)
        for gpu in list(active):
            process, job, log, started = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code == 0 and complete(output, *job):
                print(f"DONE gpu={gpu} job={job} minutes={(time.time() - started) / 60.0:.1f}", flush=True)
            else:
                failures.append({"gpu": gpu, "fold": job[0], "seed": job[1], "returncode": code})
                print(f"FAILED gpu={gpu} job={job} code={code}", flush=True)
            del active[gpu]
        write_status(status_path, jobs, active, output, failures)
    if failures:
        raise SystemExit(f"VEC source screen finished with {len(failures)} failures")
    print(f"VEC SOURCE SCREEN COMPLETE {len(jobs)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
