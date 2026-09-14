"""Command-line interface for source-derived SMA allocation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .allocation import derive_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-energy", type=Path, required=True)
    parser.add_argument("--gradient-variance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-rank", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=120)
    args = parser.parse_args()
    result = derive_allocation(pd.read_csv(args.spectral_energy), pd.read_csv(args.gradient_variance), total_rank=args.total_rank, total_steps=args.total_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
