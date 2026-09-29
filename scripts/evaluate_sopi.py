#!/usr/bin/env python3
"""Compute SOPI and posterior-movement diagnostics from held-out predictions."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from if_redesign.sopi import estimate_sopi, posterior_movement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--y", default="y")
    parser.add_argument("--q0", default="q0")
    parser.add_argument("--q1", default="q1")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.y].to_numpy()
    q0 = df[args.q0].to_numpy()
    q1 = df[args.q1].to_numpy()

    result = {
        "n": int(len(df)),
        "J_brier": estimate_sopi(y, q0, q1, score="brier"),
        "V_log": estimate_sopi(y, q0, q1, score="log"),
        "mean_abs_movement": float(posterior_movement(q0, q1, metric="absolute").mean()),
        "mean_squared_movement": float(posterior_movement(q0, q1, metric="squared").mean()),
        "mean_kl_movement": float(posterior_movement(q0, q1, metric="kl").mean()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
