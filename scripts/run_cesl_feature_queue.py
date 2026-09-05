#!/usr/bin/env python3
"""Run resumable CESL raw-atom feature extraction on available GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    return parser.parse_args()


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return bool(payload.get("records"))
    except Exception:
        return False


def write_status(path: Path, *, planned: int, active: dict[str, tuple], failures: list[dict], shards: list[Path]) -> None:
    path.write_text(
        json.dumps(
            {
                "planned": planned,
                "completed": sum(complete(shard) for shard in shards),
                "pending": sum(not complete(shard) for shard in shards) - len(active),
                "active": [
                    {"gpu": gpu, "shard_index": job[1], "pid": job[0].pid}
                    for gpu, job in active.items()
                ],
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target = Path(config["output_dir"]) / config["image_encoder"]["feature_file"]
    num_shards = len(args.gpus)
    shards = [target.with_name(f"{target.stem}.shard{index:02d}of{num_shards:02d}{target.suffix}") for index in range(num_shards)]
    output_root = Path(config["output_dir"])
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "feature_extraction_status.json"
    pending = [index for index, path in enumerate(shards) if not complete(path)]
    active: dict[str, tuple[subprocess.Popen, int, object, float]] = {}
    failures: list[dict] = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            shard_index = pending.pop(0)
            log = (log_root / f"feature_shard_{shard_index:02d}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(ROOT / "scripts/extract_cesl_atom_features.py"),
                "--config",
                str(args.config),
                "--device",
                "cuda",
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(num_shards),
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, shard_index, log, time.time())
            print(f"START gpu={gpu} feature_shard={shard_index}/{num_shards}", flush=True)
        write_status(status_path, planned=num_shards, active=active, failures=failures, shards=shards)
        time.sleep(5)
        for gpu in list(active):
            process, shard_index, log, started = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log.close()
            okay = code == 0 and complete(shards[shard_index])
            if okay:
                print(f"DONE gpu={gpu} feature_shard={shard_index} minutes={(time.time() - started) / 60:.1f}", flush=True)
            else:
                failures.append({"gpu": gpu, "shard_index": shard_index, "returncode": code})
                print(f"FAILED gpu={gpu} feature_shard={shard_index} code={code}", flush=True)
            del active[gpu]
        write_status(status_path, planned=num_shards, active=active, failures=failures, shards=shards)
    if failures:
        raise SystemExit(f"CESL feature extraction finished with {len(failures)} failed shards")
    merge_command = [
        sys.executable,
        str(ROOT / "scripts/extract_cesl_atom_features.py"),
        "--config",
        str(args.config),
        "--num-shards",
        str(num_shards),
        "--merge",
    ]
    subprocess.run(merge_command, cwd=ROOT, check=True)
    write_status(status_path, planned=num_shards, active={}, failures=[], shards=shards)
    print(f"CESL FEATURE EXTRACTION COMPLETE {num_shards}/{num_shards}", flush=True)


if __name__ == "__main__":
    main()
