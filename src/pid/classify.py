"""Symbol classification and tag reading with Muse Spark Open (vision + structured output)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol

from PIL import Image

from .models.llm import ChatClient, LLMError, image_part
from .schema import SYMBOL_CLASSES, Graph, iou

# The "legend": what each class looks like.  For a real drawing set this comes from the
# sheet's legend (parse_legend); this default matches the synthetic sheets.
DEFAULT_LEGEND = {
    "tank": "rectangle with a horizontal roof line near the top (atmospheric storage tank)",
    "vessel": "tall capsule / rounded rectangle (pressure vessel)",
    "pump": "circle containing a triangle pointing in the flow direction (centrifugal pump)",
    "valve": "bow-tie: two triangles touching at their points (isolation / gate valve)",
    "check_valve": "single triangle pointing in the flow direction with a solid bar at its tip",
    "control_valve": "bow-tie with a stem going up to a small circle (actuator) on top",
    "instrument": "circle with a horizontal line and an instrument code inside (instrument bubble)",
    "relief_valve": "downward-pointing triangle under a horizontal bar, with a stem on top",
}

CLASS_ENUM = SYMBOL_CLASSES + ["not_a_symbol"]

SCHEMA = {
    "name": "pid_symbol",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": CLASS_ENUM},
            "class_confidence": {"type": "number"},
            "tag": {"type": ["string", "null"]},
            "tag_confidence": {"type": "number"},
            "evidence": {"type": "string"},
        },
        "required": ["class", "class_confidence", "tag", "tag_confidence", "evidence"],
        "additionalProperties": False,
    },
}

SYSTEM = (
    "You are an expert process engineer reading a piping and instrumentation diagram (P&ID). "
    "You are shown crops from one sheet. Identify the single symbol the first image is centred on "
    "and read its tag. Answer ONLY with the requested JSON object. Never invent a tag: if no tag is "
    "visible next to the symbol, return null and a low tag_confidence."
)


@dataclass
class ClassResult:
    cls: str
    class_conf: float
    tag: Optional[str]
    tag_conf: float
    evidence: str = ""


class SymbolClassifier(Protocol):
    def classify(self, image: Image.Image, bbox: list[float], reasoning_effort: str = "low") -> ClassResult: ...


def _pad_box(bbox, pad_l, pad_t, pad_r, pad_b, size):
    x0, y0, x1, y1 = bbox
    w, h = size
    return (max(0, int(x0 - pad_l)), max(0, int(y0 - pad_t)), min(w, int(x1 + pad_r)), min(h, int(y1 + pad_b)))


def _upscale(img: Image.Image, min_side: int = 192) -> Image.Image:
    w, h = img.size
    s = max(1.0, min_side / max(1, min(w, h)))
    if s > 1.0:
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def make_crops(image: Image.Image, bbox: list[float]) -> tuple[Image.Image, Image.Image]:
    """Tight symbol crop plus a wider context crop where its tag label usually sits."""
    tight = image.crop(_pad_box(bbox, 10, 10, 10, 10, image.size))
    wide = image.crop(_pad_box(bbox, 24, 16, 96, 48, image.size))
    return _upscale(tight), _upscale(wide, 256)


class VLMClassifier:
    def __init__(self, client: ChatClient, legend: Optional[dict[str, str]] = None, strict_schema: bool = True):
        self.client = client
        self.legend = legend or DEFAULT_LEGEND
        self.strict_schema = strict_schema

    def _prompt(self) -> str:
        legend = "\n".join(f"- {k}: {v}" for k, v in self.legend.items())
        return (
            "Legend of symbol classes on this drawing set:\n"
            f"{legend}\n- not_a_symbol: text, a pipe fragment or noise rather than a symbol\n\n"
            "Image 1 is a tight crop of ONE candidate symbol. Image 2 is the surrounding area where its "
            "tag label may appear (usually directly below, or to the right). The tag that belongs to this "
            "symbol is the one closest to it; ignore tags of neighbouring symbols.\n"
            "Return: class (from the legend), class_confidence 0-1, tag (exact text or null), "
            "tag_confidence 0-1, evidence (one short sentence about visible features)."
        )

    def classify(self, image: Image.Image, bbox: list[float], reasoning_effort: str = "low") -> ClassResult:
        tight, wide = make_crops(image, bbox)
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [{"type": "text", "text": self._prompt()}, image_part(tight), image_part(wide)],
            },
        ]
        rf = {"type": "json_schema", "json_schema": SCHEMA} if self.strict_schema else {"type": "json_object"}
        try:
            res = self.client.chat(
                messages,
                purpose="classify",
                response_format=rf,
                reasoning_effort=reasoning_effort,
                max_tokens=600,
            )
            d = res.json()
            cls = d.get("class", "not_a_symbol")
            if cls not in CLASS_ENUM:
                cls = "not_a_symbol"
            tag = d.get("tag")
            tag = tag.strip() if isinstance(tag, str) and tag.strip() and tag.strip().lower() != "null" else None
            return ClassResult(
                cls=cls,
                class_conf=float(max(0.0, min(1.0, d.get("class_confidence", 0.0)))),
                tag=tag,
                tag_conf=float(max(0.0, min(1.0, d.get("tag_confidence", 0.0)))),
                evidence=str(d.get("evidence", ""))[:300],
            )
        except LLMError:
            raise  # auth / network / quota problems must fail loudly, never become an empty graph
        except (ValueError, KeyError, TypeError, AttributeError) as e:  # malformed model output: degrade one symbol, not the sheet
            return ClassResult("not_a_symbol", 0.0, None, 0.0, f"classification failed: {type(e).__name__}: {e}"[:300])


def parse_legend(client: ChatClient, legend_image: Image.Image) -> dict[str, str]:
    """Stage 1: ask the model to describe the legend sheet. Falls back to the default legend."""
    prompt = (
        "This image is the legend of a P&ID drawing set. Return a JSON object mapping each of these "
        f"class names to a one-sentence visual description of its symbol: {SYMBOL_CLASSES}. "
        "Only include classes you can see."
    )
    try:
        res = client.chat(
            [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(legend_image)]}],
            purpose="legend",
            response_format={"type": "json_object"},
            max_tokens=900,
        )
        d = res.json()
        out = {k: str(v) for k, v in d.items() if k in SYMBOL_CLASSES}
        return {**DEFAULT_LEGEND, **out}
    except Exception:
        return dict(DEFAULT_LEGEND)


class SimulatedClassifier:
    """Offline stand-in for the VLM, used in tests and UI development ONLY.

    It looks the answer up in a ground-truth graph.  With `error_rate > 0` it makes mistakes
    at low reported confidence at the default effort, and answers correctly when asked again
    at a higher reasoning effort, so the verification-agent loop can be exercised without any
    model calls.  Results from this class are never reported as model accuracy.
    """

    def __init__(self, gt: Graph, error_rate: float = 0.0, seed: int = 0):
        self.gt = gt
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self._bad: dict[str, str] = {}

    def _match(self, bbox):
        best, best_iou = None, 0.0
        for n in self.gt.nodes:
            v = iou(bbox, n.bbox)
            if v > best_iou:
                best, best_iou = n, v
        return best if best_iou >= 0.3 else None

    def classify(self, image, bbox, reasoning_effort="low") -> ClassResult:
        n = self._match(bbox)
        if n is None:
            return ClassResult("not_a_symbol", 0.9, None, 0.0, "simulated: no ground-truth symbol here")
        if reasoning_effort in ("high", "max", "medium"):
            return ClassResult(n.cls, 0.95, n.tag, 0.95, "simulated: re-checked at higher effort")
        if n.id not in self._bad:
            self._bad[n.id] = "err" if self.rng.random() < self.error_rate else "ok"
        if self._bad[n.id] == "err":
            wrong = self.rng.choice([c for c in SYMBOL_CLASSES if c != n.cls])
            return ClassResult(wrong, 0.45, n.tag, 0.4, "simulated: low-confidence mistake")
        return ClassResult(n.cls, 0.97, n.tag, 0.97, "simulated")
