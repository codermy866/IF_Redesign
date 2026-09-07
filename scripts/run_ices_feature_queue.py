#!/usr/bin/env python3
"""Run deterministic ICES feature shards, then merge only if all succeed."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    feature = output / config["image_encoder"]["feature_file"]
    status_path = output / "feature_campaign_status.json"
    if feature.is_file() and feature.with_suffix(".audit.json").is_file():
        print(f"ICES feature cache already complete: {feature}")
        return
    jobs: dict[str, tuple[subprocess.Popen, object]] = {}
    for shard, gpu in enumerate(args.gpus):
        log_path = output / "features" / f"extract_shard{shard:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/extract_ices_cluster_features.py"),
            "--config", str(args.config), "--device", "cuda",
            "--shard-index", str(shard), "--num-shards", str(len(args.gpus)),
        ]
        jobs[str(gpu)] = (subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log)
    failures: list[dict] = []
    while jobs:
        status_path.write_text(json.dumps({
            "stage": "extracting", "planned_shards": len(args.gpus),
            "active": [{"gpu": gpu, "pid": process.pid} for gpu, (process, _) in jobs.items()],
            "failures": failures, "outcome_used_for_feature_extraction": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(5)
        for gpu in list(jobs):
            process, log = jobs[gpu]
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code:
                failures.append({"gpu": gpu, "returncode": code})
            del jobs[gpu]
    if failures:
        status_path.write_text(json.dumps({"stage": "failed", "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"ICES feature extraction failed: {failures}")
    merge = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/extract_ices_cluster_features.py"), "--config", str(args.config), "--merge", "--num-shards", str(len(args.gpus))]
    subprocess.run(merge, cwd=ROOT, check=True)
    status_path.write_text(json.dumps({
        "stage": "complete", "feature_cache": str(feature), "raw_paths_stored": False,
        "outcome_used_for_feature_extraction": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ICES FEATURE EXTRACTION COMPLETE: {feature}")


if __name__ == "__main__":
    main()
