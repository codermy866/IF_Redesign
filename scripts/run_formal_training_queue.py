#!/usr/bin/env python3
"""Run the frozen Formal v1 SFT matrix on two GPUs with resumable jobs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("shiyan", "enshi", "wuhan", "jingzhou", "xiangyang")
PRIMARY = ("full_hierarchy", "diagnosis_only")
ABLATIONS = (
    "without_integration",
    "colposcopy_only",
    "oct_only",
    "without_clinical",
    "shuffled_cognition",
)
SEEDS = (20260902, 20260903, 20260904)


def run_dir(variant: str, fold: str, seed: int) -> Path:
    return ROOT / "results/formal_v1/runs" / variant / fold / f"seed_{seed}" / "checkpoints"


def completion_marker(variant: str, fold: str, seed: int) -> Path:
    return run_dir(variant, fold, seed).parent / "training_complete.json"


def is_complete(variant: str, fold: str, seed: int) -> bool:
    marker = completion_marker(variant, fold, seed)
    if marker.is_file() and any(run_dir(variant, fold, seed).glob("checkpoint-*/adapter_model.safetensors")):
        return True
    states = sorted(run_dir(variant, fold, seed).glob("checkpoint-*/trainer_state.json"))
    for path in states:
        state = json.loads(path.read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) >= int(state.get("max_steps", 10**12)):
            return True
    # With save_total_limit=1, SWIFT can delete the final checkpoint and retain
    # only an earlier best-validation checkpoint. Its console completion plus a
    # loadable retained adapter is therefore the legacy completion evidence.
    log = run_dir(variant, fold, seed).parent / "train_console.log"
    if log.is_file() and any(run_dir(variant, fold, seed).glob("checkpoint-*/adapter_model.safetensors")):
        tail = log.read_text(encoding="utf-8", errors="replace")[-20000:]
        if "End time of running main:" in tail and "'train_runtime':" in tail:
            return True
    return False


def mark_complete(variant: str, fold: str, seed: int) -> None:
    retained = sorted(path.parent for path in run_dir(variant, fold, seed).glob("checkpoint-*/adapter_model.safetensors"))
    if not retained:
        raise FileNotFoundError(f"No retained adapter for {(variant, fold, seed)}")
    completion_marker(variant, fold, seed).write_text(
        json.dumps(
            {
                "variant": variant,
                "fold": fold,
                "seed": seed,
                "status": "complete",
                "retained_checkpoints": [str(path) for path in retained],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def jobs() -> list[tuple[str, str, int]]:
    planned = []
    for seed in SEEDS:
        for fold in FOLDS:
            for variant in PRIMARY:
                planned.append((variant, fold, seed))
    for fold in FOLDS:
        for variant in ABLATIONS:
            planned.append((variant, fold, SEEDS[0]))
    return planned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    args = parser.parse_args()
    planned = jobs()
    for job in planned:
        if is_complete(*job) and not completion_marker(*job).is_file():
            mark_complete(*job)
    pending = [job for job in planned if not is_complete(*job)]
    status_path = ROOT / "results/formal_v1/training_campaign_status.json"
    active: dict[str, tuple[subprocess.Popen, tuple[str, str, int], object, float]] = {}
    failures = []
    completed = len(planned) - len(pending)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            variant, fold, seed = pending.pop(0)
            job_root = run_dir(variant, fold, seed).parent
            job_root.mkdir(parents=True, exist_ok=True)
            log_handle = (job_root / "train_console.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                ["bash", str(ROOT / "scripts/train_formal_sft.sh"), fold, variant, str(seed)],
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, (variant, fold, seed), log_handle, time.time())
            print(f"START gpu={gpu} variant={variant} fold={fold} seed={seed}", flush=True)
        time.sleep(5)
        for gpu in list(active):
            process, job, log_handle, started = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            variant, fold, seed = job
            if code == 0:
                mark_complete(variant, fold, seed)
            okay = code == 0 and is_complete(variant, fold, seed)
            if okay:
                completed += 1
                print(
                    f"DONE gpu={gpu} variant={variant} fold={fold} seed={seed} minutes={(time.time()-started)/60:.1f}",
                    flush=True,
                )
            else:
                failures.append({"variant": variant, "fold": fold, "seed": seed, "returncode": code})
                print(f"FAILED gpu={gpu} variant={variant} fold={fold} seed={seed} code={code}", flush=True)
            del active[gpu]
        status = {
            "planned": len(planned),
            "completed": completed,
            "pending": len(pending),
            "active": [
                {"gpu": gpu, "variant": job[0], "fold": job[1], "seed": job[2]}
                for gpu, (_, job, _, _) in active.items()
            ],
            "failures": failures,
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"Formal training finished with {len(failures)} failed jobs")
    print(f"FORMAL TRAINING COMPLETE {completed}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
