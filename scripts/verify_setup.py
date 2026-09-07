#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


EXPECTED_MODEL_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(root / "models/pretrained/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    model = Path(args.model)
    missing = [name for name in REQUIRED_FILES if not (model / name).is_file()]
    packages = {
        name: version(name)
        for name in ["torch", "transformers", "ms-swift", "peft", "trl", "datasets", "accelerate"]
    }
    observed_hash = None if args.skip_hash or missing else sha256(model / "model.safetensors")
    report = {
        "model": str(model.resolve()),
        "missing_files": missing,
        "model_sha256": observed_hash,
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
        "model_hash_ok": observed_hash == EXPECTED_MODEL_SHA256 if observed_hash else None,
        "packages": packages,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing or (observed_hash is not None and observed_hash != EXPECTED_MODEL_SHA256):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
