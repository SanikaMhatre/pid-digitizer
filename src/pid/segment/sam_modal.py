"""Client for the SAM 3.1 function deployed on Modal (see infra/modal_sam.py).

The sheet is cut into overlapping tiles, each tile is sent to the GPU function with a list of
short noun-phrase prompts, and the returned boxes are mapped back to sheet coordinates and merged.
"""
from __future__ import annotations

import io
from typing import Optional

from PIL import Image

from .base import Detection, filter_boxes, nms, tile_boxes

DEFAULT_PROMPTS = [
    "valve",
    "pump",
    "tank",
    "pressure vessel",
    "instrument bubble",
    "check valve",
    "control valve",
    "relief valve",
]


class ModalSamSegmenter:
    def __init__(
        self,
        app_name: str = "pid-sam",
        cls_name: str = "SamServer",
        prompts: Optional[list[str]] = None,
        tile: int = 1024,
        overlap: int = 128,
        score_thr: float = 0.35,
        refine_with_exemplars: bool = True,
        server=None,
    ):
        self.prompts = prompts or DEFAULT_PROMPTS
        self.tile, self.overlap, self.score_thr = tile, overlap, score_thr
        self.refine = refine_with_exemplars
        if server is not None:  # injected (tests)
            self.server = server
        else:
            import modal  # imported lazily: only needed for live runs

            self.server = modal.Cls.from_name(app_name, cls_name)()

    def detect(self, image: Image.Image) -> list[Detection]:
        image = image.convert("RGB")
        w, h = image.size
        dets: list[Detection] = []
        for (x0, y0, x1, y1) in tile_boxes(w, h, self.tile, self.overlap):
            buf = io.BytesIO()
            image.crop((x0, y0, x1, y1)).save(buf, format="PNG")
            out = self.server.segment.remote(buf.getvalue(), self.prompts, self.score_thr, self.refine)
            for r in out:
                bx0, by0, bx1, by1 = r["box"]
                dets.append(Detection([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0], float(r["score"]), r.get("prompt")))
        dets = filter_boxes(dets, image.size)
        return nms(dets)
