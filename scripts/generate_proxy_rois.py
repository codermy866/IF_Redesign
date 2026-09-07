#!/usr/bin/env python3
"""Generate deterministic, explicitly unreviewed ROI boxes for engineering screening."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.counterfactual import proxy_gradient_energy_bbox  # noqa: E402
from cervix_cogalign.io import read_jsonl, write_jsonl  # noqa: E402


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "results/formal_v1/all/evidence_prompts.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "results/formal_v1/proxy_roi/auto_proxy_rois.jsonl"))
    parser.add_argument("--audit", default=str(ROOT / "results/formal_v1/proxy_roi/auto_proxy_roi_audit.json"))
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--window-fraction", type=float, default=0.35)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    rois = []
    for row in rows:
        images = row.get("images", [])
        if not 0 <= args.image_index < len(images):
            raise IndexError(f"missing image index {args.image_index} for {row['id']}")
        image_path = Path(images[args.image_index])
        with Image.open(image_path) as image:
            bbox, diagnostics = proxy_gradient_energy_bbox(
                image,
                window_fraction=args.window_fraction,
                grid_size=args.grid_size,
            )
        rois.append(
            {
                "id": str(row["id"]),
                "image_index": args.image_index,
                "bbox": [round(value, 6) for value in bbox],
                "source": "automatic_proxy_gradient_energy_v1",
                "reviewed": False,
                "claim_status": "engineering_proxy_only_not_clinically_reviewed",
                "image_sha256": file_sha256(image_path),
                "selection": diagnostics,
            }
        )
    if len({row["id"] for row in rois}) != len(rois):
        raise ValueError("proxy ROI output must contain one box per case")
    write_jsonl(args.output, rois)
    audit = {
        "n": len(rois),
        "dataset": str(Path(args.dataset).resolve()),
        "image_index": args.image_index,
        "method": "gradient energy over a deterministic grid",
        "reviewed": False,
        "permitted_use": "engineering screening and pipeline tests only",
        "prohibited_use": "clinical review claim, lesion ground truth, causal claim, or negative-target GRPO",
    }
    target = Path(args.audit)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
