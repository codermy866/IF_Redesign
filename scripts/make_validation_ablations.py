#!/usr/bin/env python3
"""Create deterministic, paired validation inputs for shortcut diagnostics.

The donor for every case has the opposite label.  This intentionally breaks the
association between the selected input component and the target while leaving
all other fields unchanged.  These files are evaluation-only and must never be
used for training or model selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


CLINICAL = re.compile(r"(临床信息：\n)(.*?)(\n\n按以下标签)", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def user_text(row: dict) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def set_user_text(row: dict, text: str) -> None:
    next(message for message in row["messages"] if message["role"] == "user")["content"] = text


def clinical_block(text: str) -> str:
    match = CLINICAL.search(text)
    if not match:
        raise ValueError("Clinical block not found")
    return match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--match-center",
        action="store_true",
        help="Choose every opposite-label donor from the same acquisition centre",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.match_center:
        pools = {
            (center, label): [
                row for row in rows if row["center"] == center and int(row["label"]) == label
            ]
            for center in {row["center"] for row in rows}
            for label in (0, 1)
        }
        empty = [key for key, pool in pools.items() if not pool]
        if empty:
            raise ValueError(f"Cannot centre-match these centre/label pools: {empty}")
    else:
        pools = {
            label: [row for row in rows if int(row["label"]) == label]
            for label in (0, 1)
        }
    counters = {key: 0 for key in pools}
    image_rows: list[dict] = []
    colposcopy_rows: list[dict] = []
    oct_rows: list[dict] = []
    clinical_rows: list[dict] = []
    audit: list[dict] = []
    for original in rows:
        opposite = 1 - int(original["label"])
        key = (original["center"], opposite) if args.match_center else opposite
        donor_pool = pools[key]
        donor = donor_pool[counters[key] % len(donor_pool)]
        counters[key] += 1

        image_row = copy.deepcopy(original)
        image_row["images"] = copy.deepcopy(donor["images"])
        image_row["ablation"] = "opposite_label_images"
        image_row["donor_id"] = donor["id"]
        image_rows.append(image_row)

        if len(original["images"]) != 2 or len(donor["images"]) != 2:
            raise ValueError("Expected [colposcopy, OCT] image pairs")
        colposcopy_row = copy.deepcopy(original)
        colposcopy_row["images"][0] = donor["images"][0]
        colposcopy_row["ablation"] = "opposite_label_colposcopy"
        colposcopy_row["donor_id"] = donor["id"]
        colposcopy_rows.append(colposcopy_row)

        oct_row = copy.deepcopy(original)
        oct_row["images"][1] = donor["images"][1]
        oct_row["ablation"] = "opposite_label_oct"
        oct_row["donor_id"] = donor["id"]
        oct_rows.append(oct_row)

        clinical_row = copy.deepcopy(original)
        original_text = user_text(clinical_row)
        donor_clinical = clinical_block(user_text(donor))
        replaced, count = CLINICAL.subn(
            lambda match: match.group(1) + donor_clinical + match.group(3),
            original_text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"Could not replace clinical block for {original['id']}")
        set_user_text(clinical_row, replaced)
        clinical_row["ablation"] = "opposite_label_clinical"
        clinical_row["donor_id"] = donor["id"]
        clinical_rows.append(clinical_row)

        audit.append({
            "id": original["id"],
            "label": int(original["label"]),
            "donor_id": donor["id"],
            "donor_label": int(donor["label"]),
            "center": original["center"],
            "donor_center": donor["center"],
        })

    prefix = "within_center_" if args.match_center else ""
    write_jsonl(args.output_dir / f"{prefix}opposite_label_images.jsonl", image_rows)
    write_jsonl(args.output_dir / f"{prefix}opposite_label_colposcopy.jsonl", colposcopy_rows)
    write_jsonl(args.output_dir / f"{prefix}opposite_label_oct.jsonl", oct_rows)
    write_jsonl(args.output_dir / f"{prefix}opposite_label_clinical.jsonl", clinical_rows)
    with (args.output_dir / f"{prefix}audit.json").open("w", encoding="utf-8") as sink:
        json.dump({"n": len(rows), "pairs": audit}, sink, ensure_ascii=False, indent=2)
    match_note = "centre-matched " if args.match_center else ""
    print(f"Wrote {len(rows)} rows for each of four {match_note}ablations to {args.output_dir}")


if __name__ == "__main__":
    main()
