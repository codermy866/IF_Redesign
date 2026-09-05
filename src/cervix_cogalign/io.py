from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_path_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def existing_paths(value: Any) -> list[str]:
    return [p for p in parse_path_list(value) if Path(p).is_file()]


def stable_score(*parts: object, seed: int = 0) -> int:
    payload = "\x1f".join(map(str, (*parts, seed))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def evenly_spaced(items: Sequence[str], limit: int) -> list[str]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=limit).round().astype(int)
    return [items[int(i)] for i in indices]


def select_colposcopy(paths: Sequence[str], limit: int) -> list[str]:
    """Cover acquisition phases when filenames expose them, then fill evenly."""
    ordered = sorted(paths)
    selected: list[str] = []
    phase_tokens = (("ori", "original"), ("pre", "green"), ("post", "acid", "iod"))
    for tokens in phase_tokens:
        candidates = [p for p in ordered if any(t in Path(p).name.lower() for t in tokens)]
        if candidates:
            selected.append(candidates[len(candidates) // 2])
    for path in evenly_spaced(ordered, limit):
        if path not in selected:
            selected.append(path)
    return selected[:limit]


def _load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        if im.mode in {"I", "I;16", "F"}:
            arr = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(arr, [1.0, 99.0])
            if hi <= lo:
                hi = lo + 1.0
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
        return ImageOps.exif_transpose(im).convert("RGB")


def build_contact_sheet(
    paths: Sequence[str],
    output_path: str | Path,
    *,
    tile_size: int = 280,
    columns: int = 3,
) -> Path:
    if not paths:
        raise ValueError("Cannot build a contact sheet without images")
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(paths):
        image = ImageEnhance.Contrast(_load_rgb(path)).enhance(1.03)
        image.thumbnail((tile_size - 8, tile_size - 8), Image.Resampling.LANCZOS)
        x0 = (idx % columns) * tile_size + (tile_size - image.width) // 2
        y0 = (idx // columns) * tile_size + (tile_size - image.height) // 2
        canvas.paste(image, (x0, y0))
        draw.rectangle(
            (idx % columns * tile_size, idx // columns * tile_size,
             (idx % columns + 1) * tile_size - 1, (idx // columns + 1) * tile_size - 1),
            outline=(96, 96, 96),
            width=2,
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="JPEG", quality=92, subsampling=0)
    return target
