#!/usr/bin/env python3
"""Extract frozen, individual raw-atom image features for CESL-v2.

This script contains no outcome fitting and never writes raw image paths into
the feature cache.  It may be run in deterministic GPU shards and merged only
after every shard has completed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cervix_cogalign.cesl import SLOT_IDS, SLOT_TO_INDEX, nuisance_stratum  # noqa: E402
from cervix_cogalign.io import read_json, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cesl_v2_formal_retrospective.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="Merge completed deterministic shards instead of extracting one shard.")
    return parser.parse_args()


def feature_path(config: dict) -> Path:
    return Path(config["output_dir"]) / config["image_encoder"]["feature_file"]


def shard_path(target: Path, shard_index: int, num_shards: int) -> Path:
    return target.with_name(f"{target.stem}.shard{shard_index:02d}of{num_shards:02d}{target.suffix}")


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        if image.mode in {"I", "I;16", "F"}:
            values = np.asarray(image, dtype=np.float32)
            low, high = np.percentile(values, [1.0, 99.0])
            if high <= low:
                high = low + 1.0
            values = np.clip((values - low) / (high - low), 0.0, 1.0)
            return Image.fromarray((values * 255).astype(np.uint8), mode="L").convert("RGB")
        return ImageOps.exif_transpose(image).convert("RGB")


def merge(config: dict, num_shards: int) -> None:
    target = feature_path(config)
    pieces = []
    for index in range(num_shards):
        path = shard_path(target, index, num_shards)
        if not path.is_file():
            raise FileNotFoundError(f"Missing CESL feature shard {index}/{num_shards}")
        pieces.append(torch.load(path, map_location="cpu", weights_only=False))
    records = []
    for piece in pieces:
        records.extend(piece["records"])
    records.sort(key=lambda row: int(row["global_index"]))
    expected = list(range(len(records)))
    actual = [int(row["global_index"]) for row in records]
    if actual != expected:
        raise AssertionError("CESL feature shards do not reconstruct one contiguous atom cohort")
    features = torch.stack([row["features"] for row in records], dim=0).contiguous()
    available = torch.stack([row["available"] for row in records], dim=0).bool().contiguous()
    payload = {
        "experiment_id": config["experiment_id"],
        "encoder": config["image_encoder"],
        "slot_ids": list(SLOT_IDS),
        "features": features,
        "available": available,
        "ids": [row["id"] for row in records],
        "patient_ids": [row["patient_id"] for row in records],
        "centers": [row["center"] for row in records],
        "labels": torch.tensor([row["label"] for row in records], dtype=torch.long),
        "ages": [row["age"] for row in records],
        "hpvs": [row["hpv"] for row in records],
        "tcts": [row["tct"] for row in records],
        "nuisance_strata": [row["nuisance_stratum"] for row in records],
        "feature_audit": {
            "n_cases": len(records),
            "feature_shape": list(features.shape),
            "dtype": str(features.dtype),
            "unreadable_images": int(sum(int(row["unreadable_images"]) for row in records)),
            "raw_paths_stored": False,
            "outcome_used_for_feature_extraction": False,
        }
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    audit = target.with_suffix(".audit.json")
    audit.write_text(json.dumps(payload["feature_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"merged": str(target), **payload["feature_audit"]}, ensure_ascii=False))


def extract(config: dict, args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    target = feature_path(config)
    output = shard_path(target, args.shard_index, args.num_shards)
    atom_rows = read_jsonl(config["atom_manifest"])
    manifest = pd.read_csv(config["manifest"], dtype={"case_id": str}).set_index("case_id", drop=False)
    selected = [(index, row) for index, row in enumerate(atom_rows) if index % args.num_shards == args.shard_index]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    transform = weights.transforms(antialias=True)
    batch_size = int(config["image_encoder"]["batch_size"])
    records = []
    tasks: list[tuple[int, int, str]] = []
    for local_index, (global_index, row) in enumerate(selected):
        case_id = str(row["id"])
        if case_id not in manifest.index:
            raise KeyError(f"Raw atom case is absent from frozen manifest: {case_id}")
        clinical = manifest.loc[case_id]
        records.append(
            {
                "global_index": global_index,
                "id": case_id,
                "patient_id": str(row["patient_id"]),
                "center": str(row["center"]),
                "label": int(row["label"]),
                "age": clinical.get("age"),
                "hpv": clinical.get("hpv"),
                "tct": clinical.get("tct"),
                "nuisance_stratum": nuisance_stratum(clinical.get("age"), clinical.get("hpv"), clinical.get("tct")),
                "features": torch.zeros((len(SLOT_IDS), int(config["image_encoder"]["feature_dim"])), dtype=torch.float16),
                "available": torch.zeros(len(SLOT_IDS), dtype=torch.bool),
                "unreadable_images": 0,
            }
        )
        records[-1]["available"][0] = True
        for atom in row["atoms"]:
            atom_id = str(atom["atom_id"])
            if atom_id == "clinical":
                continue
            if atom_id not in SLOT_TO_INDEX:
                raise ValueError(f"Unexpected atom id: {atom_id}")
            tasks.append((local_index, SLOT_TO_INDEX[atom_id], str(atom["path"])))
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        valid: list[tuple[int, int]] = []
        for local_index, slot_index, path in chunk:
            try:
                tensors.append(transform(load_rgb(path)))
                valid.append((local_index, slot_index))
            except Exception:
                records[local_index]["unreadable_images"] += 1
        if not tensors:
            continue
        inputs = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(inputs).detach().float().cpu().to(torch.float16)
        for (local_index, slot_index), vector in zip(valid, outputs):
            records[local_index]["features"][slot_index] = vector
            records[local_index]["available"][slot_index] = True
        if start == 0 or (start + len(chunk)) % max(batch_size * 50, 1) == 0:
            print(f"shard={args.shard_index}/{args.num_shards} images={min(start + len(chunk), len(tasks))}/{len(tasks)}", flush=True)
    for record in records:
        if not record["available"][1:].any():
            raise AssertionError("A complete-modality raw-atom record lost all visual atoms during feature extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "shard_index": args.shard_index, "num_shards": args.num_shards}, output)
    print(json.dumps({"shard": str(output), "n_cases": len(records), "n_images": len(tasks)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.merge:
        merge(config, args.num_shards)
    else:
        extract(config, args)


if __name__ == "__main__":
    main()
