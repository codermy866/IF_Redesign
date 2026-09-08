#!/usr/bin/env python3
"""Run validation + held-centre inference for every completed Formal v1 adapter."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_formal_training_queue import FOLDS, PRIMARY, ABLATIONS, SEEDS, is_complete, run_dir


ROOT = Path(__file__).resolve().parents[1]
RECOVERED_RUNS = ROOT / "results/formal_v1/recovered_runs"


def planned_jobs() -> list[tuple[str, str, int]]:
    result = []
    for seed in SEEDS:
        for fold in FOLDS:
            for variant in PRIMARY:
                result.append((variant, fold, seed))
    for fold in FOLDS:
        for variant in ABLATIONS:
            result.append((variant, fold, SEEDS[0]))
    return result


def best_checkpoint(variant: str, fold: str, seed: int) -> Path:
    roots = [RECOVERED_RUNS / variant / fold / f"seed_{seed}" / "checkpoints", run_dir(variant, fold, seed)]
    for root in roots:
        candidates = sorted(
            (path for path in root.glob("checkpoint-*") if (path / "adapter_model.safetensors").is_file() and os.access(path / "adapter_model.safetensors", os.R_OK)),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if not candidates:
            continue
        for state_path in reversed([path / "trainer_state.json" for path in candidates]):
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                selected = state.get("best_model_checkpoint")
                if selected and (Path(selected) / "adapter_model.safetensors").is_file() and os.access(Path(selected) / "adapter_model.safetensors", os.R_OK):
                    return Path(selected)
        return candidates[-1]
    raise FileNotFoundError(f"No readable adapter checkpoint for {(variant, fold, seed)}")


def paths(variant: str, fold: str, seed: int) -> tuple[Path, Path]:
    prompt = ROOT / "results/formal_v1/folds" / fold / "variants" / variant / "eval_prompt.jsonl"
    output = ROOT / "results/formal_v1/runs" / variant / fold / f"seed_{seed}" / "eval_predictions.jsonl"
    return prompt, output


def jsonl_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as source:
        return {str(json.loads(line)["id"]) for line in source if line.strip()}


def inference_complete(job: tuple[str, str, int]) -> bool:
    prompt, output = paths(*job)
    return prompt.is_file() and jsonl_ids(prompt) == jsonl_ids(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    args = parser.parse_args()
    planned = planned_jobs()
    not_trained = [job for job in planned if not is_complete(*job)]
    if not_trained:
        raise RuntimeError(f"Training incomplete for {len(not_trained)} jobs; first={not_trained[0]}")
    pending = [job for job in planned if not inference_complete(job)]
    completed = len(planned) - len(pending)
    active = {}
    failures = []
    status_path = ROOT / "results/formal_v1/inference_campaign_status.json"
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            variant, fold, seed = job
            prompt, output = paths(*job)
            output.parent.mkdir(parents=True, exist_ok=True)
            log_handle = (output.parent / "inference_console.log").open("a", encoding="utf-8")
            max_tokens = "64" if variant == "diagnosis_only" else "320"
            batch_size = "16" if variant == "diagnosis_only" else "8"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONPATH"] = str(ROOT / "src")
            command = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/run_qwen_inference.py"),
                "--model", str(ROOT / "models/pretrained/Qwen3-VL-2B-Instruct"),
                "--adapter", str(best_checkpoint(*job)),
                "--input", str(prompt),
                "--output", str(output),
                "--device", "cuda:0",
                "--max-new-tokens", max_tokens,
                "--batch-size", batch_size,
                "--score-diagnosis",
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
            active[gpu] = (process, job, log_handle, time.time())
            print(f"START gpu={gpu} variant={variant} fold={fold} seed={seed}", flush=True)
        time.sleep(5)
        for gpu in list(active):
            process, job, log_handle, started = active[gpu]
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            okay = code == 0 and inference_complete(job)
            if okay:
                completed += 1
                print(f"DONE gpu={gpu} job={job} minutes={(time.time()-started)/60:.1f}", flush=True)
            else:
                failures.append({"job": job, "returncode": code})
                print(f"FAILED gpu={gpu} job={job} code={code}", flush=True)
            del active[gpu]
        status_path.write_text(
            json.dumps(
                {
                    "planned": len(planned),
                    "completed": completed,
                    "pending": len(pending),
                    "active": [{"gpu": gpu, "job": value[1]} for gpu, value in active.items()],
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if failures:
        raise SystemExit(f"Inference finished with {len(failures)} failures")
    print(f"FORMAL INFERENCE COMPLETE {completed}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
