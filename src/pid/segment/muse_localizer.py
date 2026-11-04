"""Muse Spark localizer: find symbol boxes by asking the vision model, tile by tile.

The sheet is cut into overlapping tiles.  Each tile is sent to Muse Spark Open with the legend of
symbol classes and a request for tight pixel boxes in strict JSON.  Boxes from all tiles are mapped
back to sheet coordinates, de-duplicated, and passed on to the classifier and the line tracer.

This is the alternative to the SAM 3.1 localizer (`meta_sam.py`): the same model that reads the symbols also finds them.  How *accurate* its boxes are is an empirical question: measure it with
`python -m pid.smoke localize` (box recall against generated ground truth) before relying on it.
"""
from __future__ import annotations

from PIL import Image

from ..classify import DEFAULT_LEGEND
from ..models.llm import ChatClient, image_part
from ..schema import SYMBOL_CLASSES
from .base import Detection, filter_boxes, nms, tile_boxes

PROMPT = (
    "This is a tile of a P&ID drawing, {w} pixels wide and {h} pixels high (origin at the top-left).\n"
    "Symbol classes on this drawing set:\n{legend}\n\n"
    "List every equipment or instrument symbol that is FULLY visible in the tile. Do not list pipes, "
    "signal lines or text labels. For each symbol give a TIGHT bounding box in the pixel coordinates of THIS image.\n"
    'Return JSON only: {{"symbols": [{{"class": one of {classes}, "box": [x0, y0, x1, y1]}}]}}'
)


class MuseLocalizer:
    def __init__(
        self,
        client: ChatClient,
        tile: int = 768,
        overlap: int = 96,
        reasoning_effort: str = "medium",
        legend: dict[str, str] | None = None,
    ):
        self.client, self.tile, self.overlap, self.effort = client, tile, overlap, reasoning_effort
        self.legend = legend or DEFAULT_LEGEND

    def _legend_text(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.legend.items())

    def detect(self, image: Image.Image) -> list[Detection]:
        image = image.convert("RGB")
        W, H = image.size
        dets: list[Detection] = []
        for x0, y0, x1, y1 in tile_boxes(W, H, self.tile, self.overlap):
            crop = image.crop((x0, y0, x1, y1))
            w, h = crop.size
            res = self.client.chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PROMPT.format(w=w, h=h, legend=self._legend_text(), classes=SYMBOL_CLASSES),
                            },
                            image_part(crop, max_side=max(w, h)),
                        ],
                    }
                ],
                purpose="localize",
                response_format={"type": "json_object"},
                reasoning_effort=self.effort,
                max_tokens=2500,
            )
            try:
                items = res.json().get("symbols", [])
            except (ValueError, AttributeError):
                continue  # one unreadable tile must not sink the sheet; the pipeline warns if too few symbols come back
            for it in items:
                try:
                    bx0, by0, bx1, by1 = (float(v) for v in it["box"])
                except (KeyError, TypeError, ValueError):
                    continue
                if max(bx0, by0, bx1, by1) <= 1.0:  # model answered in normalized coordinates
                    bx0, bx1, by0, by1 = bx0 * w, bx1 * w, by0 * h, by1 * h
                if bx1 <= bx0 or by1 <= by0:
                    continue
                dets.append(Detection([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0], 0.6, it.get("class")))
        return nms(filter_boxes(dets, image.size))
