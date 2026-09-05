#!/usr/bin/env python3
"""Run the prespecified CESL availability-stress matrix on two GPUs."""
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
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def complete(config: dict, job: tuple[str, int]) -> bool:
    fold, seed = job
    root = Path(config["output_dir"]) / "evaluations" / fold / f"seed_{seed}"
    return (root / "availability_stress_complete.json").is_file() and (root / "availability_stress_predictions.csv").is_file()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folds = tuple(str(fold) for fold in config["folds"])
    planned = [(fold, int(seed)) for fold in folds for seed in config["seeds"]]
    not_evaluated = [job for job in planned if not (Path(config["output_dir"]) / "evaluations" / job[0] / f"seed_{job[1]}" / "evaluation_complete.json").is_file()]
    if not_evaluated:
        raise RuntimeError(f"Stress evaluation needs frozen policy predictions; first incomplete={not_evaluated[0]}")
    output = Path(config["output_dir"])
    status_path = output / "availability_stress_status.json"
    pending = [job for job in planned if not complete(config, job)]
    active: dict[str, tuple[subprocess.Popen, tuple[str, int], object, float]] = {}
    failures: list[dict] = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            fold, seed = pending.pop(0)
            root = output / "evaluations" / fold / f"seed_{seed}"
            log = (root / "availability_stress_console.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [sys.executable, str(ROOT / "scripts/run_cesl_availability_stress.py"), fold, str(seed), "--config", str(args.config), "--device", "cuda"]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, (fold, seed), log, time.time())
            print(f"START gpu={gpu} stress={fold}/{seed}", flush=True)
        active_jobs = {job for _, job, _, _ in active.values()}
        status_path.write_text(
            json.dumps(
                {
                    "planned": len(planned),
                    "completed": sum(complete(config, job) for job in planned),
                    "pending": sum(not complete(config, job) and job not in active_jobs for job in planned),
                    "active": [{"gpu": gpu, "fold": job[0], "seed": job[1], "pid": process.pid} for gpu, (process, job, _, _) in active.items()],
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        time.sleep(5)
        for gpu in list(active):
            process, job, log, started = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code == 0 and complete(config, job):
                print(f"DONE gpu={gpu} stress={job} minutes={(time.time() - started) / 60.0:.1f}", flush=True)
            else:
                failures.append({"gpu": gpu, "fold": job[0], "seed": job[1], "returncode": code})
                print(f"FAILED gpu={gpu} stress={job} code={code}", flush=True)
            del active[gpu]
    if failures:
        raise SystemExit(f"CESL stress queue completed with {len(failures)} failures")
    print(f"CESL STRESS COMPLETE {len(planned)}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
