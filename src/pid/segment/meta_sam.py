"""SAM 3.1 localizer, served by the Meta Model API.

SAM takes a short noun phrase ("pump") and an image and returns every matching object as a box plus a mask,
encoded as text tokens in the response:

    <0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600;c=0.96|><|mask;...|>

The sheet is cut into overlapping tiles; each tile is sent once per class phrase.  Only the boxes are used
(the line tracer does its own pixel work), and boxes from all tiles and phrases are mapped back to sheet
coordinates and merged.  Call `python -m pid.smoke localize` to measure its recall before relying on it.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image

from ..models.llm import ChatClient, image_data_url
from .base import Detection, filter_boxes, nms, tile_boxes

DEFAULT_PROMPTS: dict[str, str] = {
    "tank": "storage tank",
    "vessel": "pressure vessel",
    "pump": "pump",
    "valve": "valve",
    "check_valve": "check valve",
    "control_valve": "control valve",
    "instrument": "instrument bubble",
    "relief_valve": "relief valve",
}

SAM_PRICE_PER_IMAGE_USD = 2.50 / 1000  # listed price: $2.50 per 1,000 images

_BOX = re.compile(r"<\|box;([^|]*)\|>")


def response_text(data: dict) -> str:
    """Pull the output text out of a (non-streaming) Responses-API body."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output") or []:
        for c in item.get("content") or []:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts)


def parse_boxes(text: str) -> list[dict]:
    """Parse every box token into {x1,y1,x2,y2,w,h,c}.  Malformed tokens are skipped."""
    out = []
    for m in _BOX.finditer(text or ""):
        kv = {}
        for pair in m.group(1).split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    kv[k.strip()] = float(v)
                except ValueError:
                    pass
        if all(k in kv for k in ("x1", "y1", "x2", "y2")):
            out.append(kv)
    return out


class MetaSamLocalizer:
    def __init__(
        self,
        client: ChatClient,
        prompts: Optional[dict[str, str]] = None,
        tile: int = 1280,
        overlap: int = 128,
        min_score: float = 0.3,
        workers: int = 4,
        model: str = "sam-3.1",
    ):
        self.client, self.prompts, self.tile, self.overlap = client, prompts or DEFAULT_PROMPTS, tile, overlap
        self.min_score, self.workers, self.model = min_score, workers, model

    def _one(self, crop: Image.Image, cls: str, phrase: str, ox: int, oy: int) -> list[Detection]:
        w, h = crop.size
        payload = {
            "model": self.model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": phrase},
                        {"type": "input_image", "image_url": image_data_url(crop)},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }
        data = self.client.post_raw("responses", payload, purpose="localize", fixed_cost=SAM_PRICE_PER_IMAGE_USD)
        dets = []
        for b in parse_boxes(response_text(data)):
            sx = w / b["w"] if b.get("w") else 1.0
            sy = h / b["h"] if b.get("h") else 1.0
            x0, y0 = b["x1"] * sx, b["y1"] * sy
            x1, y1 = (b["x2"] + 1) * sx, (b["y2"] + 1) * sy  # x2 / y2 are inclusive pixel coordinates
            score = b.get("c", 0.6)
            if x1 <= x0 or y1 <= y0 or score < self.min_score:
                continue
            dets.append(Detection([x0 + ox, y0 + oy, x1 + ox, y1 + oy], float(score), cls))
        return dets

    def detect(self, image: Image.Image) -> list[Detection]:
        image = image.convert("RGB")
        W, H = image.size
        jobs = []
        for x0, y0, x1, y1 in tile_boxes(W, H, self.tile, self.overlap):
            crop = image.crop((x0, y0, x1, y1))
            for cls, phrase in self.prompts.items():
                jobs.append((crop, cls, phrase, x0, y0))
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(lambda j: self._one(*j), jobs))
        return nms(filter_boxes([d for r in results for d in r], image.size))
