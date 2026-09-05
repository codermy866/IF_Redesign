#!/usr/bin/env python3
"""Run the three CESL backbone families across locked LOCO folds and seeds."""
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


def run_root(config: dict, backbone: str, fold: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "runs" / backbone / fold / f"seed_{seed}"


def complete(config: dict, job: tuple[str, str, int]) -> bool:
    backbone, fold, seed = job
    root = run_root(config, backbone, fold, seed)
    return (root / "training_complete.json").is_file() and (root / "checkpoint.pt").is_file()


def write_status(path: Path, config: dict, planned: list[tuple[str, str, int]], active: dict[str, tuple], failures: list[dict]) -> None:
    active_jobs = {job for _, job, _, _ in active.values()}
    completed = sum(complete(config, job) for job in planned)
    pending = sum(not complete(config, job) and job not in active_jobs for job in planned)
    path.write_text(
        json.dumps(
            {
                "planned": len(planned),
                "completed": completed,
                "pending": pending,
                "active": [
                    {"gpu": gpu, "backbone": job[0], "fold": job[1], "seed": job[2], "pid": process.pid}
                    for gpu, (process, job, _, _) in active.items()
                ],
                "failures": failures,
                "held_out_labels_opened_by_training": False,
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
    feature_path = Path(config["output_dir"]) / config["image_encoder"]["feature_file"]
    if not feature_path.is_file():
        raise FileNotFoundError(f"CESL training cannot start before the frozen feature cache exists: {feature_path}")
    planned = [(backbone, fold, int(seed)) for backbone in BACKBONES for fold in folds for seed in config["seeds"]]
    output_root = Path(config["output_dir"])
    status_path = output_root / "training_campaign_status.json"
    pending = [job for job in planned if not complete(config, job)]
    active: dict[str, tuple[subprocess.Popen, tuple[str, str, int], object, float]] = {}
    failures: list[dict] = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            backbone, fold, seed = pending.pop(0)
            root = run_root(config, backbone, fold, seed)
            root.mkdir(parents=True, exist_ok=True)
            log = (root / "train_console.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(ROOT / "scripts/train_cesl_backbone.py"),
                fold,
                backbone,
                str(seed),
                "--config",
                str(args.config),
                "--device",
                "cuda",
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, (backbone, fold, seed), log, time.time())
            print(f"START gpu={gpu} backbone={backbone} fold={fold} seed={seed}", flush=True)
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
                failures.append({"gpu": gpu, "backbone": job[0], "fold": job[1], "seed": job[2], "returncode": code})
                print(f"FAILED gpu={gpu} job={job} code={code}", flush=True)
            del active[gpu]
        write_status(status_path, config, planned, active, failures)
    if failures:
        raise SystemExit(f"CESL training completed with {len(failures)} failures")
    print(f"CESL TRAINING COMPLETE {len(planned)}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
