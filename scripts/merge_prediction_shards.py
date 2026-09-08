#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge unique JSONL prediction shards")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = {}
    for path in args.inputs:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row["id"])
                if key in rows:
                    raise ValueError(f"Duplicate id {key}")
                rows[key] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for key in sorted(rows):
            sink.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} unique rows into {args.output}")


if __name__ == "__main__":
    main()
