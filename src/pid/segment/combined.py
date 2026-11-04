"""Union of several localizers: every box any of them proposes, merged by non-max suppression."""
from __future__ import annotations

from PIL import Image

from .base import Detection, nms


class CombinedLocalizer:
    def __init__(self, *localizers):
        self.localizers = localizers

    def detect(self, image: Image.Image) -> list[Detection]:
        return nms([d for loc in self.localizers for d in loc.detect(image)])
