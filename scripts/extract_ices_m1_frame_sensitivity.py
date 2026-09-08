#!/usr/bin/env python3
"""Extract first- and tenth-frame OCT controls for the ICES M1 sensitivity audit.

Colposcopy tokens are copied from the validated path-free cluster cache.  Only
the OCT token is replaced with frame 1 or frame 10 from the *same parsed C/S
position*.  No outcome is used and the result stores no image path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cervix_cogalign.ices import N_SLOTS, OCT_INDICES, SLOT_IDS  # noqa: E402
from cervix_cogalign.io import read_jsonl  # noqa: E402
from cervix_cogalign.sequence_evidence import parse_oct_position_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--output", default=str(ROOT / "results/ices_v1/supplement_m1_frame_sensitivity/features/m1_frame_sensitivity_resnet50.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


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


def shard_path(target: Path, index: int, total: int) -> Path:
    return target.with_name(f"{target.stem}.shard{index:02d}of{total:02d}{target.suffix}")


def read_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_cache(config: dict) -> Path:
    return Path(config["output_dir"]) / config["image_encoder"]["feature_file"]


def extract(config: dict, args: argparse.Namespace) -> None:
    target = Path(args.output)
    base = torch.load(base_cache(config), map_location="cpu", weights_only=False)
    rows = read_jsonl(Path(config["output_dir"]) / "cluster_manifest.jsonl")
    if tuple(base["slot_ids"]) != SLOT_IDS or [str(row["id"]) for row in rows] != [str(value) for value in base["ids"]]:
        raise AssertionError("M1 sensitivity manifest/cache alignment failed")
    selected = list(range(args.shard_index, len(rows), args.num_shards))
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms(antialias=True)
    records = []
    tasks: list[tuple[int, int, int, str]] = []
    for local, global_index in enumerate(selected):
        row = rows[global_index]
        record = {
            "global_index": global_index,
            "frame1_features": base["cluster_features"][global_index].clone(),
            "frame10_features": base["cluster_features"][global_index].clone(),
        }
        records.append(record)
        units = [unit for unit in row["evidence_units"] if unit["kind"] == "oct_position_cluster"]
        if len(units) != len(OCT_INDICES):
            raise AssertionError("M1 sensitivity requires all twelve OCT C/S units")
        for position, unit in enumerate(units):
            paths = list(unit["frame_paths"])
            if len(paths) != 10 or parse_oct_position_frame(paths[0])[2] != 1 or parse_oct_position_frame(paths[-1])[2] != 10:
                raise AssertionError("M1 sensitivity requires ordered complete C/S frame groups")
            tasks.append((local, OCT_INDICES[position], 1, paths[0]))
            tasks.append((local, OCT_INDICES[position], 10, paths[-1]))
    batch_size = int(config["image_encoder"]["batch_size"])
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        tensors, valid = [], []
        for local, slot, frame, path in chunk:
            tensors.append(transform(load_rgb(path)))
            valid.append((local, slot, frame))
        inputs = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            vectors = model(inputs).detach().float().cpu().to(torch.float16)
        for (local, slot, frame), vector in zip(valid, vectors):
            records[local]["frame1_features" if frame == 1 else "frame10_features"][slot] = vector
        if start == 0 or (start + len(chunk)) % max(50 * batch_size, 1) == 0:
            print(f"shard={args.shard_index}/{args.num_shards} images={min(start + len(chunk), len(tasks))}/{len(tasks)}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "shard_index": args.shard_index, "num_shards": args.num_shards}, shard_path(target, args.shard_index, args.num_shards))
    print(json.dumps({"shard": args.shard_index, "n_cases": len(records), "n_images": len(tasks)}, ensure_ascii=False))


def merge(config: dict, args: argparse.Namespace) -> None:
    target = Path(args.output)
    base = torch.load(base_cache(config), map_location="cpu", weights_only=False)
    pieces = [torch.load(shard_path(target, index, args.num_shards), map_location="cpu", weights_only=False) for index in range(args.num_shards)]
    records = [item for piece in pieces for item in piece["records"]]
    records.sort(key=lambda item: item["global_index"])
    if [item["global_index"] for item in records] != list(range(len(base["ids"]))):
        raise AssertionError("M1 sensitivity shards are incomplete")
    frame1 = torch.stack([item["frame1_features"] for item in records]).contiguous()
    frame10 = torch.stack([item["frame10_features"] for item in records]).contiguous()
    if tuple(frame1.shape[1:]) != tuple(base["cluster_features"].shape[1:]) or tuple(frame10.shape) != tuple(frame1.shape):
        raise AssertionError("M1 frame-control feature schema mismatch")
    payload = {key: value for key, value in base.items() if key not in {"cluster_features", "raw_features"}}
    payload.update({
        "experiment_id": "ices_v1_m1_frame_sensitivity",
        "slot_ids": list(SLOT_IDS),
        "frame1_features": frame1,
        "frame10_features": frame10,
        "feature_audit": {
            "n_cases": len(records), "feature_shape": list(frame1.shape), "raw_paths_stored": False,
            "outcome_used_for_feature_extraction": False,
            "comparison": "frame 1 and frame 10 from the same parsed C/S position; colposcopy tokens copied from the validated cache",
        },
    })
    torch.save(payload, target)
    target.with_suffix(".audit.json").write_text(json.dumps(payload["feature_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"merged": str(target), **payload["feature_audit"]}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    config = read_config(args.config)
    if args.merge:
        merge(config, args)
    else:
        extract(config, args)


if __name__ == "__main__":
    main()
