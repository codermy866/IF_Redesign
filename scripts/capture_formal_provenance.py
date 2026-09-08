#!/usr/bin/env python3
"""Capture immutable inputs and software provenance for Formal v1."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    config_path = ROOT / "configs/formal_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_root = Path(config["model"])
    code_files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    locked_inputs = [
        config_path,
        Path(config["manifest"]),
        Path(config["feature_cache"]),
        ROOT / "results/formal_v1/fold_assignments.csv",
        ROOT / "results/formal_v1/cohort_audit.csv",
        model_root / "config.json",
        model_root / "generation_config.json",
        model_root / "preprocessor_config.json",
        model_root / "tokenizer_config.json",
        model_root / "model.safetensors.index.json",
    ]
    provenance = {
        "experiment_id": config["experiment_id"],
        "git_commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ).splitlines(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "ms-swift", "peft", "trl", "numpy", "pandas", "scikit-learn")
        },
        "locked_inputs": [digest(path) for path in locked_inputs],
        "implementation_digest": digest(ROOT / "requirements-lock.txt"),
        "code_files": [digest(path) for path in code_files],
        "model_weight_files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size}
            for path in sorted(model_root.glob("*.safetensors"))
        ],
        "note": "Weight shard sizes and the official model-index digest are recorded; full multi-GB shard hashing is omitted.",
    }
    target = ROOT / "results/formal_v1/formal_provenance.json"
    target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
