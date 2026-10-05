#!/usr/bin/env python
"""Evaluate patient-specific Evidence Advantage from a held-out prediction table.

Required CSV columns:
    y, clinical_logit, updated_logit

Optional column:
    advantage_score

No patient identifiers are required.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from if_redesign.evidence_advantage import (
    evidence_advantage,
    evaluate_selective_update,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--loss", choices=["log", "brier"], default="log")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"y", "clinical_logit", "updated_logit"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    advantage = evidence_advantage(
        df["y"].to_numpy(),
        df["clinical_logit"].to_numpy(),
        df["updated_logit"].to_numpy(),
        loss=args.loss,
    )

    out = {
        "n": int(len(df)),
        "mean_advantage": float(advantage.mean()),
        "beneficial_fraction": float((advantage > 0).mean()),
        "harmful_fraction": float((advantage < 0).mean()),
    }

    if "advantage_score" in df.columns:
        ev = evaluate_selective_update(
            df["y"].to_numpy(),
            df["clinical_logit"].to_numpy(),
            df["updated_logit"].to_numpy(),
            df["advantage_score"].to_numpy(),
            threshold=args.threshold,
            loss=args.loss,
        )
        out["selective_policy"] = ev.__dict__

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
