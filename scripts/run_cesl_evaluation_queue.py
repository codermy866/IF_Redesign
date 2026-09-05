#!/usr/bin/env python3
"""Run frozen held-out CESL policy inference after all backbone training finishes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("c0_full", "selection", "css")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def training_complete(config: dict, fold: str, seed: int) -> bool:
    root = Path(config["output_dir"]) / "runs"
    return all((root / backbone / fold / f"seed_{seed}" / "training_complete.json").is_file() for backbone in BACKBONES)


def complete(config: dict, job: tuple[str, int]) -> bool:
    fold, seed = job
    root = Path(config["output_dir"]) / "evaluations" / fold / f"seed_{seed}"
    return (root / "evaluation_complete.json").is_file() and (root / "test_predictions.csv").is_file()


def write_status(path: Path, config: dict, planned: list[tuple[str, int]], active: dict[str, tuple], failures: list[dict]) -> None:
    active_jobs = {job for _, job, _, _ in active.values()}
    path.write_text(
        json.dumps(
            {
                "planned": len(planned),
                "completed": sum(complete(config, job) for job in planned),
                "pending": sum(not complete(config, job) and job not in active_jobs for job in planned),
                "active": [
                    {"gpu": gpu, "fold": job[0], "seed": job[1], "pid": process.pid}
                    for gpu, (process, job, _, _) in active.items()
                ],
                "failures": failures,
                "held_out_labels_used_only_after_policy_construction": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folds = tuple(str(fold) for fold in config["folds"])
    planned = [(fold, int(seed)) for fold in folds for seed in config["seeds"]]
    incomplete = [job for job in planned if not training_complete(config, *job)]
    if incomplete:
        raise RuntimeError(f"CESL evaluation requires all backbones complete; first incomplete fold/seed={incomplete[0]}")
    output_root = Path(config["output_dir"])
    status_path = output_root / "evaluation_campaign_status.json"
    pending = [job for job in planned if not complete(config, job)]
    active: dict[str, tuple[subprocess.Popen, tuple[str, int], object, float]] = {}
    failures: list[dict] = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            fold, seed = pending.pop(0)
            root = output_root / "evaluations" / fold / f"seed_{seed}"
            root.mkdir(parents=True, exist_ok=True)
            log = (root / "evaluation_console.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(ROOT / "scripts/evaluate_cesl_fold.py"),
                fold,
                str(seed),
                "--config",
                str(args.config),
                "--device",
                "cuda",
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, (fold, seed), log, time.time())
            print(f"START gpu={gpu} fold={fold} seed={seed}", flush=True)
        write_status(status_path, config, planned, active, failures)
        time.sleep(5)
        for gpu in list(active):
            process, job, log, started = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code == 0 and complete(config, job):
                print(f"DONE gpu={gpu} job={job} minutes={(time.time() - started) / 60.0:.1f}", flush=True)
            else:
                failures.append({"gpu": gpu, "fold": job[0], "seed": job[1], "returncode": code})
                print(f"FAILED gpu={gpu} job={job} code={code}", flush=True)
            del active[gpu]
        write_status(status_path, config, planned, active, failures)
    if failures:
        raise SystemExit(f"CESL evaluation completed with {len(failures)} failures")
    print(f"CESL EVALUATION COMPLETE {len(planned)}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
