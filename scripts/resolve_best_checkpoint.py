#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    candidates = sorted(
        args.run_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints below {args.run_dir}")
    best = None
    state_files = list(args.run_dir.glob("checkpoint-*/trainer_state.json"))
    for state_file in state_files:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        value = state.get("best_model_checkpoint")
        if value and Path(value).is_dir():
            best = Path(value)
    print((best or candidates[-1]).resolve())


if __name__ == "__main__":
    main()
