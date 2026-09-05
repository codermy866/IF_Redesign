#!/usr/bin/env python3
"""Run deterministic raw-atom linear comparator controls after feature extraction."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_example.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folds = tuple(str(fold) for fold in config["folds"])
    output = Path(config["output_dir"])
    feature_path = output / config["image_encoder"]["feature_file"]
    if not feature_path.is_file():
        raise FileNotFoundError(f"Linear controls require frozen CESL feature cache: {feature_path}")
    completed = 0
    for fold in folds:
        marker = output / "linear_controls" / fold / "complete.json"
        if marker.is_file():
            completed += 1
            continue
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_cesl_linear_controls.py"), fold, "--config", str(args.config)],
            cwd=ROOT,
            check=True,
        )
        completed += 1
    print(f"CESL LINEAR CONTROLS COMPLETE {completed}/{len(folds)}", flush=True)


if __name__ == "__main__":
    main()
