"""Segmenter interface and shared helpers (tiling, non-max suppression)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol

from PIL import Image

from ..schema import Graph, iou


@dataclass
class Detection:
    bbox: list[float]  # x0, y0, x1, y1 in sheet pixels
    score: float = 1.0
    hint: Optional[str] = None  # class hint from the prompt that found it (not trusted)


class Segmenter(Protocol):
    def detect(self, image: Image.Image) -> list[Detection]: ...


def tile_boxes(width: int, height: int, size: int = 1024, overlap: int = 128) -> list[tuple[int, int, int, int]]:
    if width <= size and height <= size:
        return [(0, 0, width, height)]
    step = max(1, size - overlap)
    xs = list(range(0, max(1, width - overlap), step))
    ys = list(range(0, max(1, height - overlap), step))
    out = []
    for y in ys:
        for x in xs:
            out.append((x, y, min(width, x + size), min(height, y + size)))
    return out


def nms(dets: list[Detection], iou_thr: float = 0.5, contain_thr: float = 0.85) -> list[Detection]:
    """Greedy NMS by score, plus suppression of boxes mostly contained in a kept box."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        drop = False
        a = d.bbox
        area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        for k in kept:
            b = k.bbox
            if iou(a, b) >= iou_thr:
                drop = True
                break
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if ix * iy / area_a >= contain_thr:
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


def filter_boxes(dets: list[Detection], image_size, min_side: float = 14, max_frac: float = 0.3) -> list[Detection]:
    w, h = image_size
    out = []
    for d in dets:
        bw, bh = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
        if bw < min_side or bh < min_side:
            continue
        if bw * bh > max_frac * w * h:
            continue
        out.append(d)
    return out


class OracleSegmenter:
    """Returns ground-truth boxes (optionally degraded).  Tests and UI development ONLY."""

    def __init__(self, gt: Graph, jitter: int = 0, drop: float = 0.0, false_positives: int = 0, seed: int = 0):
        self.gt, self.jitter, self.drop, self.fp, self.seed = gt, jitter, drop, false_positives, seed

    def detect(self, image: Image.Image) -> list[Detection]:
        rng = random.Random(self.seed)
        out = []
        for n in self.gt.nodes:
            if rng.random() < self.drop:
                continue
            j = self.jitter
            dx, dy = (rng.randint(-j, j), rng.randint(-j, j)) if j else (0, 0)
            x0, y0, x1, y1 = n.bbox
            out.append(Detection([x0 + dx, y0 + dy, x1 + dx, y1 + dy], 0.99, n.cls))
        w, h = image.size
        for _ in range(self.fp):
            x, y = rng.randint(0, w - 60), rng.randint(0, h - 60)
            out.append(Detection([x, y, x + 40, y + 24], 0.5, None))
        return out
