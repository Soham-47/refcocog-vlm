"""Small, dependency-free helpers for 2D grounding boxes."""

from __future__ import annotations

import re


NUMBER = r"-?\d+(?:\.\d+)?"
BOX_PATTERNS = (
    re.compile(rf"\[\s*\[?\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*\]?\s*\]"),
    re.compile(rf"bbox_2d\s*[\"']?\s*:\s*\[\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*\]", re.I),
    re.compile(rf"<\|box_start\|>\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)\s*,\s*\(\s*({NUMBER})\s*,\s*({NUMBER})\s*\)"),
)


def parse_box(text: str) -> list[float] | None:
    """Parse one valid normalized [x1, y1, x2, y2] box (0–1000)."""
    for pattern in BOX_PATTERNS:
        match = pattern.search(text)
        if match:
            box = [float(value) for value in match.groups()]
            if all(0 <= value <= 1000 for value in box) and box[0] < box[2] and box[1] < box[3]:
                return box
    return None


def iou(a: list[float], b: list[float]) -> float:
    """Calculate intersection-over-union for [x1, y1, x2, y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalized_to_pixels(box: list[float], width: int, height: int) -> list[float]:
    """Convert a 0–1000 normalized box to image pixel coordinates."""
    return [box[0] * width / 1000, box[1] * height / 1000,
            box[2] * width / 1000, box[3] * height / 1000]


def source_bbox_to_pixels(box: list[float], source_width: int, source_height: int,
                          image_width: int, image_height: int) -> list[float]:
    """Map a COCO annotation box into the dimensions of its cached image."""
    return [box[0] * image_width / source_width, box[1] * image_height / source_height,
            box[2] * image_width / source_width, box[3] * image_height / source_height]


def iou_reward(completion: str, target: list[float], width: int, height: int) -> float:
    """Return pixel-space IoU, or a small penalty for an invalid box."""
    prediction = parse_box(completion)
    if prediction is None:
        return -0.1
    return iou(normalized_to_pixels(prediction, width, height), target)
