#!/usr/bin/env python3
"""Extract frozen path-free ICES image features in deterministic GPU shards.

The output intentionally retains neither raw image paths nor frame filenames.
For each parsed OCT C/S group it writes (a) the mean of all ten frame features
for M1 and (b) the matched fifth-frame feature for the M1 ablation.  Labels are
copied only after image processing and are never consulted to construct a unit
or feature.
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

from cervix_cogalign.cesl import clinical_matrix, nuisance_stratum  # noqa: E402
from cervix_cogalign.ices import MAX_COLPOSCOPY_VIEWS, N_SLOTS, OCT_INDICES, SLOT_IDS  # noqa: E402
from cervix_cogalign.io import read_jsonl  # noqa: E402
from cervix_cogalign.sequence_evidence import parse_oct_position_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/ices_v1_exploratory.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def read_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _merge(config: dict, num_shards: int) -> None:
    target = feature_path(config)
    pieces = []
    for index in range(num_shards):
        path = shard_path(target, index, num_shards)
        if not path.is_file():
            raise FileNotFoundError(f"Missing ICES feature shard {index}/{num_shards}: {path}")
        pieces.append(torch.load(path, map_location="cpu", weights_only=False))
    records = [row for piece in pieces for row in piece["records"]]
    records.sort(key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in records] != list(range(len(records))):
        raise AssertionError("ICES feature shards do not reconstruct one contiguous frozen cohort")
    cluster_features = torch.stack([row["cluster_features"] for row in records]).contiguous()
    raw_features = torch.stack([row["raw_features"] for row in records]).contiguous()
    available = torch.stack([row["available"] for row in records]).bool().contiguous()
    if tuple(cluster_features.shape[1:]) != (N_SLOTS, int(config["image_encoder"]["feature_dim"])):
        raise AssertionError("ICES cluster-cache tensor schema is inconsistent with the locked protocol")
    if not torch.equal(available[:, 0], torch.ones(len(records), dtype=torch.bool)):
        raise AssertionError("ICES clinical token must be present for every case")
    if not torch.equal(available[:, list(OCT_INDICES)].sum(dim=1), torch.full((len(records),), len(OCT_INDICES), dtype=torch.long)):
        raise AssertionError("ICES cluster cache must retain all twelve validated OCT positions")
    payload = {
        "experiment_id": config["experiment_id"],
        "encoder": config["image_encoder"],
        "slot_ids": list(SLOT_IDS),
        "cluster_features": cluster_features,
        "raw_features": raw_features,
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
            "feature_shape": list(cluster_features.shape),
            "raw_frame_control_shape": list(raw_features.shape),
            "dtype": str(cluster_features.dtype),
            "max_colposcopy_slots": MAX_COLPOSCOPY_VIEWS,
            "oct_position_clusters": len(OCT_INDICES),
            "unreadable_images": int(sum(int(row["unreadable_images"]) for row in records)),
            "raw_paths_stored": False,
            "outcome_used_for_unit_construction": False,
            "outcome_used_for_feature_extraction": False,
            "m1_comparison": "all-ten-frame mean per C/S position versus fifth frame from the same C/S position",
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    target.with_suffix(".audit.json").write_text(json.dumps(payload["feature_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"merged": str(target), **payload["feature_audit"]}, ensure_ascii=False))


def _extract(config: dict, args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    target = feature_path(config)
    output = shard_path(target, args.shard_index, args.num_shards)
    rows = read_jsonl(Path(config["output_dir"]) / "cluster_manifest.jsonl")
    manifest = pd.read_csv(config["manifest"], dtype={"case_id": str}).set_index("case_id", drop=False)
    selected = [(index, row) for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms(antialias=True)
    batch_size = int(config["image_encoder"]["batch_size"])
    feature_dim = int(config["image_encoder"]["feature_dim"])
    records: list[dict] = []
    tasks: list[tuple[int, int, str, str]] = []
    for local_index, (global_index, row) in enumerate(selected):
        case_id = str(row["id"])
        if case_id not in manifest.index:
            raise KeyError(f"ICES case is absent from the frozen manifest: {case_id}")
        clinical = manifest.loc[case_id]
        units = row["evidence_units"]
        colposcopy = [unit for unit in units if unit["kind"] == "colposcopy_view"]
        oct_units = [unit for unit in units if unit["kind"] == "oct_position_cluster"]
        if len(colposcopy) > MAX_COLPOSCOPY_VIEWS or len(oct_units) != len(OCT_INDICES):
            raise AssertionError("ICES manifest no longer matches its fixed evidence schema")
        record = {
            "global_index": global_index,
            "id": case_id,
            "patient_id": str(row["patient_id"]),
            "center": str(row["center"]),
            "label": int(row["label"]),
            "age": clinical.get("age"),
            "hpv": clinical.get("hpv"),
            "tct": clinical.get("tct"),
            "nuisance_stratum": nuisance_stratum(clinical.get("age"), clinical.get("hpv"), clinical.get("tct")),
            "cluster_features": torch.zeros((N_SLOTS, feature_dim), dtype=torch.float16),
            "raw_features": torch.zeros((N_SLOTS, feature_dim), dtype=torch.float16),
            "available": torch.zeros(N_SLOTS, dtype=torch.bool),
            "cluster_sums": torch.zeros((N_SLOTS, feature_dim), dtype=torch.float32),
            "cluster_counts": torch.zeros(N_SLOTS, dtype=torch.long),
            "raw_seen": torch.zeros(N_SLOTS, dtype=torch.bool),
            "unreadable_images": 0,
        }
        record["available"][0] = True
        records.append(record)
        for view_index, unit in enumerate(colposcopy):
            tasks.append((local_index, 1 + view_index, str(unit["path"]), "colposcopy"))
        for oct_index, unit in enumerate(oct_units):
            slot = OCT_INDICES[oct_index]
            frame_paths = list(unit["frame_paths"])
            if len(frame_paths) != 10:
                raise AssertionError("ICES OCT cluster must contain exactly ten parsed frames")
            for path in frame_paths:
                tasks.append((local_index, slot, str(path), "oct"))
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        valid: list[tuple[int, int, str, str]] = []
        for local_index, slot, path, kind in chunk:
            try:
                tensors.append(transform(load_rgb(path)))
                valid.append((local_index, slot, path, kind))
            except Exception:
                records[local_index]["unreadable_images"] += 1
        if tensors:
            inputs = torch.stack(tensors).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(inputs).detach().float().cpu()
            for (local_index, slot, path, kind), vector in zip(valid, outputs):
                record = records[local_index]
                if kind == "colposcopy":
                    record["cluster_features"][slot] = vector.to(torch.float16)
                    record["raw_features"][slot] = vector.to(torch.float16)
                    record["available"][slot] = True
                else:
                    record["cluster_sums"][slot] += vector
                    record["cluster_counts"][slot] += 1
                    if parse_oct_position_frame(path)[2] == 5:
                        record["raw_features"][slot] = vector.to(torch.float16)
                        record["raw_seen"][slot] = True
        if start == 0 or (start + len(chunk)) % max(batch_size * 50, 1) == 0:
            print(f"shard={args.shard_index}/{args.num_shards} images={min(start + len(chunk), len(tasks))}/{len(tasks)}", flush=True)
    for record in records:
        counts = record["cluster_counts"][list(OCT_INDICES)]
        if record["unreadable_images"] or not torch.equal(counts, torch.full_like(counts, 10)) or not record["raw_seen"][list(OCT_INDICES)].all():
            raise RuntimeError("ICES fails closed: an OCT C/S cluster lost a required frame during feature extraction")
        for slot in OCT_INDICES:
            record["cluster_features"][slot] = (record["cluster_sums"][slot] / record["cluster_counts"][slot]).to(torch.float16)
            record["available"][slot] = True
        del record["cluster_sums"], record["cluster_counts"], record["raw_seen"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "shard_index": args.shard_index, "num_shards": args.num_shards}, output)
    print(json.dumps({"shard": str(output), "n_cases": len(records), "n_images": len(tasks)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    if args.merge:
        _merge(config, args.num_shards)
    else:
        _extract(config, args)


if __name__ == "__main__":
    main()
