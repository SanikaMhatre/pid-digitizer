"""Fallback localizer: ask the vision-language model for symbol boxes tile by tile.

EXPERIMENTAL. It exists as the documented fallback if SAM 3.1 does not work well on engineering
drawings.  Whether the model returns accurate pixel boxes is an empirical question; measure it
with `python -m pid.eval.run_eval --segmenter vlm` before relying on it.
"""
from __future__ import annotations

from PIL import Image

from ..models.llm import ChatClient, image_part
from ..schema import SYMBOL_CLASSES
from .base import Detection, filter_boxes, nms, tile_boxes

PROMPT = (
    "This is a tile of a P&ID drawing, {w} pixels wide and {h} pixels high. List every equipment or "
    "instrument symbol (not pipes, not text) fully visible in the tile. Return JSON: "
    '{{"symbols": [{{"class": one of {classes}, "box": [x0, y0, x1, y1]}}]}} with boxes in the pixel '
    "coordinates of THIS image (origin top-left). Boxes must be tight around the symbol."
)


class VLMBoxSegmenter:
    def __init__(self, client: ChatClient, tile: int = 768, overlap: int = 96, reasoning_effort: str = "medium"):
        self.client, self.tile, self.overlap, self.effort = client, tile, overlap, reasoning_effort

    def detect(self, image: Image.Image) -> list[Detection]:
        image = image.convert("RGB")
        W, H = image.size
        dets: list[Detection] = []
        for (x0, y0, x1, y1) in tile_boxes(W, H, self.tile, self.overlap):
            crop = image.crop((x0, y0, x1, y1))
            w, h = crop.size
            res = self.client.chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT.format(w=w, h=h, classes=SYMBOL_CLASSES)},
                            image_part(crop, max_side=max(w, h)),
                        ],
                    }
                ],
                purpose="localize",
                response_format={"type": "json_object"},
                reasoning_effort=self.effort,
                max_tokens=2000,
            )
            try:
                items = res.json().get("symbols", [])
            except Exception:
                continue
            for it in items:
                try:
                    bx0, by0, bx1, by1 = [float(v) for v in it["box"]]
                except Exception:
                    continue
                if max(bx0, by0, bx1, by1) <= 1.0:  # normalized coordinates
                    bx0, bx1, by0, by1 = bx0 * w, bx1 * w, by0 * h, by1 * h
                if bx1 <= bx0 or by1 <= by0:
                    continue
                dets.append(Detection([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0], 0.6, it.get("class")))
        return nms(filter_boxes(dets, image.size))
