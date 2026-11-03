"""Overlay rendering: symbols, nets and violations drawn on the original sheet."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .rules.engine import Violation
from .schema import Graph
from .trace import TraceResult

CLASS_COLORS = {
    "tank": (31, 119, 180),
    "vessel": (44, 160, 44),
    "pump": (214, 39, 40),
    "valve": (148, 103, 189),
    "check_valve": (140, 86, 75),
    "control_valve": (227, 119, 194),
    "instrument": (23, 190, 207),
    "relief_valve": (255, 127, 14),
}
NET_COLORS = {"process": (0, 114, 178), "signal": (230, 159, 0)}
SEVERITY_COLORS = {"error": (213, 0, 0), "warning": (230, 100, 0), "info": (90, 90, 90)}


def draw_overlay(
    image: Image.Image,
    graph: Graph,
    violations: list[Violation] | None = None,
    trace: TraceResult | None = None,
    show_tags: bool = True,
) -> Image.Image:
    base = image.convert("RGB").copy()
    violations = violations or []

    if trace is not None:
        arr = np.asarray(base).copy()
        for net in graph.nets:
            lab = trace.net_label.get(net.id)
            if lab is None:
                continue
            mask = trace.labels == lab
            color = np.array(NET_COLORS.get(net.type, (0, 114, 178)))
            arr[mask] = (0.35 * arr[mask] + 0.65 * color).astype(np.uint8)
        base = Image.fromarray(arr)

    d = ImageDraw.Draw(base)
    font = ImageFont.load_default(size=13)
    for n in graph.nodes:
        color = CLASS_COLORS.get(n.cls, (80, 80, 80))
        x0, y0, x1, y1 = n.bbox
        width = 2 if not n.needs_review else 3
        d.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=color, width=width)
        if show_tags:
            label = f"{n.cls}" + (f" {n.tag}" if n.tag else "")
            d.text((x0 - 3, y0 - 18), label, fill=color, font=font)
        if n.needs_review:
            d.text((x1 - 6, y0 - 18), "?", fill=(213, 0, 0), font=font)

    for v in violations:
        color = SEVERITY_COLORS.get(v.severity, (213, 0, 0))
        boxes = []
        nm = graph.node_map()
        for nid in v.node_ids:
            if nid in nm:
                boxes.append(nm[nid].bbox)
        if v.bbox and not v.node_ids:
            boxes.append(v.bbox)
        if not boxes and v.bbox:
            boxes.append(v.bbox)
        for b in boxes:
            d.rectangle([b[0] - 9, b[1] - 9, b[2] + 9, b[3] + 9], outline=color, width=3)
        if boxes:
            b = boxes[0]
            d.text((b[0] - 9, b[3] + 12), v.rule_id.split("-")[0], fill=color, font=font)
        if v.net_ids and v.bbox:
            b = v.bbox
            d.ellipse([b[2] - 8, (b[1] + b[3]) / 2 - 8, b[2] + 8, (b[1] + b[3]) / 2 + 8], outline=color, width=3)
    return base
