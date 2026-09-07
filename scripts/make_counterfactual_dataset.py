#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.counterfactual import save_blur_pair  # noqa: E402
from cervix_cogalign.io import read_jsonl, stable_score, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lesion-blur and matched random-blur counterfactual pairs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rois", required=True, help="JSONL: id,image_index,bbox,source,reviewed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--assign-negative", action="store_true", help="Create paper-style GRPO targets; requires reviewed ROI")
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = {str(row["id"]): row for row in read_jsonl(args.dataset)}
    rois = read_jsonl(args.rois)
    output = Path(args.output_dir)
    lesion_rows, random_rows, audit = [], [], []
    for roi in rois:
        case_id = str(roi["id"])
        if case_id not in rows:
            raise KeyError(f"ROI references unknown id {case_id}")
        reviewed = bool(roi.get("reviewed", False))
        if not reviewed and not args.allow_unreviewed:
            raise RuntimeError(f"ROI {case_id} is unreviewed; pass --allow-unreviewed for proxy-only screening")
        if args.assign_negative and not reviewed:
            raise RuntimeError("Paper-style negative targets are forbidden for unreviewed proxy ROIs")
        source = rows[case_id]
        image_index = int(roi["image_index"])
        image_path = source["images"][image_index]
        lesion_path = output / "assets" / f"{case_id}_image{image_index}_lesion_blur.jpg"
        random_path = output / "assets" / f"{case_id}_image{image_index}_random_blur.jpg"
        save_blur_pair(
            image_path,
            roi["bbox"],
            lesion_path,
            random_path,
            seed=stable_score(case_id, seed=args.seed),
        )
        lesion = json.loads(json.dumps(source))
        random_control = json.loads(json.dumps(source))
        lesion["images"][image_index] = str(lesion_path.resolve())
        random_control["images"][image_index] = str(random_path.resolve())
        for item, kind in [(lesion, "lesion_blur"), (random_control, "random_blur")]:
            item["id"] = f"{case_id}::{kind}"
            item["counterfactual_kind"] = kind
            item["original_id"] = case_id
            item["roi_source"] = roi.get("source", "unspecified")
            item["roi_reviewed"] = reviewed
        if args.assign_negative:
            lesion["label"] = 0
            lesion["solution"] = "negative"
            lesion["counterfactual_target_assumption"] = "lesion removal is sufficient to normalize diagnosis"
        lesion_rows.append(lesion)
        random_rows.append(random_control)
        audit.append(
            {
                "id": case_id,
                "image_index": image_index,
                "bbox": roi["bbox"],
                "source": roi.get("source", "unspecified"),
                "reviewed": reviewed,
                "assigned_negative": args.assign_negative,
            }
        )
    write_jsonl(output / "lesion_blur.jsonl", lesion_rows)
    write_jsonl(output / "random_blur.jsonl", random_rows)
    with (output / "counterfactual_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(audit)} counterfactual pairs to {output}")


if __name__ == "__main__":
    main()
