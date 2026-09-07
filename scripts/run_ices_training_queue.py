#!/usr/bin/env python3
"""Run the locked ICES M1--M3 source-only ablation grid on available GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = (
    "raw_m1_ablation",
    "cluster_m1_m2",
    "cluster_m1_m2_m3",
    "cluster_selected_control",
    "cluster_sufficiency_control",
    "cluster_pure_m3",
)
FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    return parser.parse_args()


def run_root(config: dict, backbone: str, fold: str, seed: int) -> Path:
    return Path(config["output_dir"]) / "runs" / backbone / fold / f"seed_{seed}"


def complete(config: dict, job: tuple[str, str, int]) -> bool:
    backbone, fold, seed = job
    root = run_root(config, backbone, fold, seed)
    return (root / "training_complete.json").is_file() and (root / "checkpoint.pt").is_file() and (root / "source_validation_metrics.json").is_file()


def write_status(path: Path, config: dict, planned: list[tuple[str, str, int]], active: dict[str, tuple], failures: list[dict]) -> None:
    active_jobs = {job for _, job, _, _ in active.values()}
    completed = sum(complete(config, job) for job in planned)
    path.write_text(json.dumps({
        "stage": "source_only_training",
        "planned": len(planned), "completed": completed,
        "pending": sum(not complete(config, job) and job not in active_jobs for job in planned),
        "active": [{"gpu": gpu, "backbone": job[0], "fold": job[1], "seed": job[2], "pid": process.pid} for gpu, (process, job, _, _) in active.items()],
        "failures": failures,
        "outer_test_labels_opened_by_campaign": False,
        "outer_test_predictions_written_by_campaign": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cache = Path(config.get("feature_cache", Path(config["output_dir"]) / config["image_encoder"]["feature_file"]))
    if not cache.is_file():
        raise FileNotFoundError(f"ICES training starts only after the path-free feature cache is complete: {cache}")
    planned = [(backbone, fold, int(seed)) for backbone in args.backbones for fold in FOLDS for seed in config["seeds"]]
    pending = [job for job in planned if not complete(config, job)]
    output = Path(config["output_dir"])
    status_path = output / "training_campaign_status.json"
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
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/train_ices_backbone.py"), fold, backbone, str(seed), "--config", str(args.config), "--device", "cuda"]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
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
        raise SystemExit(f"ICES source-only campaign completed with {len(failures)} failures")
    print(f"ICES SOURCE-ONLY TRAINING COMPLETE {len(planned)}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
