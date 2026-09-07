from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np

from PIL import Image, ImageFilter


BBox = tuple[int, int, int, int]


def normalized_to_pixels(box: Sequence[float], size: tuple[int, int]) -> BBox:
    if len(box) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")
    width, height = size
    x1, y1, x2, y2 = [float(v) for v in box]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("normalized bbox coordinates must satisfy 0 <= x1 < x2 <= 1")
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def gaussian_blur_roi(image: Image.Image, bbox: BBox, radius: float = 18.0) -> Image.Image:
    result = image.convert("RGB").copy()
    crop = result.crop(bbox).filter(ImageFilter.GaussianBlur(radius=radius))
    result.paste(crop, bbox)
    return result


def matched_random_bbox(bbox: BBox, size: tuple[int, int], seed: int) -> BBox:
    width, height = size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    rng = random.Random(seed)
    candidates: list[BBox] = []
    for _ in range(128):
        nx1 = rng.randint(0, max(0, width - bw))
        ny1 = rng.randint(0, max(0, height - bh))
        candidate = (nx1, ny1, nx1 + bw, ny1 + bh)
        ix = max(0, min(x2, candidate[2]) - max(x1, candidate[0]))
        iy = max(0, min(y2, candidate[3]) - max(y1, candidate[1]))
        if ix * iy <= 0.1 * bw * bh:
            return candidate
        candidates.append(candidate)
    return min(candidates, key=lambda b: max(0, min(x2, b[2]) - max(x1, b[0])))


def proxy_gradient_energy_bbox(
    image: Image.Image,
    *,
    window_fraction: float = 0.35,
    grid_size: int = 13,
) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    """Return a deterministic, unreviewed proxy ROI from local image gradients.

    This is an engineering locator, not a lesion detector.  It must retain
    explicit unreviewed provenance and cannot be used to create clinical or
    causal claims by itself.
    """
    if not 0 < window_fraction <= 1:
        raise ValueError("window_fraction must be in (0, 1]")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    if min(width, height) < 4:
        raise ValueError("image is too small for proxy ROI selection")
    window_width = max(2, min(width, int(round(width * window_fraction))))
    window_height = max(2, min(height, int(round(height * window_fraction))))
    gradient = np.zeros_like(gray)
    gradient[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    max_x, max_y = width - window_width, height - window_height
    xs = np.unique(np.linspace(0, max_x, num=grid_size).round().astype(int))
    ys = np.unique(np.linspace(0, max_y, num=grid_size).round().astype(int))
    candidates: list[tuple[float, int, int]] = []
    for y in ys:
        for x in xs:
            local_gray = gray[y : y + window_height, x : x + window_width]
            content_fraction = float((local_gray > 8).mean())
            local_energy = float(gradient[y : y + window_height, x : x + window_width].mean())
            candidates.append((local_energy * content_fraction, int(x), int(y)))
    scores = np.asarray([value[0] for value in candidates], dtype=np.float64)
    best_index = int(np.argmax(scores))
    best_score, x1, y1 = candidates[best_index]
    median_score = float(np.median(scores))
    confidence = float((best_score - median_score) / (abs(median_score) + 1e-8))
    return (
        (x1 / width, y1 / height, (x1 + window_width) / width, (y1 + window_height) / height),
        {
            "proxy_score": float(best_score),
            "median_candidate_score": median_score,
            "relative_separation": confidence,
            "content_fraction": float((gray[y1 : y1 + window_height, x1 : x1 + window_width] > 8).mean()),
        },
    )


def save_blur_pair(
    image_path: str | Path,
    normalized_bbox: Sequence[float],
    lesion_output: str | Path,
    random_output: str | Path,
    *,
    seed: int,
    radius: float = 18.0,
) -> tuple[Path, Path]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    lesion_box = normalized_to_pixels(normalized_bbox, image.size)
    random_box = matched_random_bbox(lesion_box, image.size, seed)
    lesion_target, random_target = Path(lesion_output), Path(random_output)
    lesion_target.parent.mkdir(parents=True, exist_ok=True)
    random_target.parent.mkdir(parents=True, exist_ok=True)
    gaussian_blur_roi(image, lesion_box, radius).save(lesion_target, quality=94)
    gaussian_blur_roi(image, random_box, radius).save(random_target, quality=94)
    return lesion_target, random_target
