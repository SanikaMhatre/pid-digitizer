"""Synthetic P&ID generator.

Why: public P&ID datasets have unclear licenses and no planted errors.  This generator draws
our own sheets (original symbols, nothing copied), knows the exact ground-truth graph, and can
plant design errors whose expected rule violations are known by construction.

Layout: horizontal process lines (rows).  Items are placed on a column grid, joined by pipes.
Optional extras: a relief valve above the end vessel, an instrument bubble above each control
valve (dashed signal line), and a vertical tee that cross-connects adjacent rows.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..schema import SYMBOL_CLASSES, Attachment, Graph, Net, Node

CELL_W = 190
ROW_H = 270
LEFT = 70
TOP = 215
LINE_W = 2

ERROR_KINDS = [
    "missing_check_valve",
    "missing_isolation_valve",
    "missing_tag",
    "duplicate_tag",
    "dangling_pipe",
    "missing_signal",
    "missing_relief",
]

TAG_PREFIX = {
    "tank": "T",
    "vessel": "V",
    "pump": "P",
    "valve": "HV",
    "check_valve": "NRV",
    "control_valve": "FCV",
    "instrument": "FIC",
    "relief_valve": "PSV",
}

# (width, height) of each symbol's bounding box
SIZES = {
    "tank": (90, 80),
    "vessel": (70, 110),
    "pump": (56, 56),
    "valve": (50, 36),
    "check_valve": (50, 36),
    "control_valve": (50, 76),
    "instrument": (44, 44),
    "relief_valve": (34, 40),
}

PUMP_ROWS = [
    ["tank", "valve", "pump", "check_valve", "valve", "vessel"],
    ["tank", "valve", "pump", "check_valve", "valve", "valve", "vessel"],
    ["tank", "valve", "pump", "check_valve", "valve", "vessel"],
]
CONTROL_ROWS = [
    ["tank", "valve", "control_valve", "valve", "vessel"],
    ["tank", "valve", "control_valve", "valve", "valve", "vessel"],
]


@dataclass
class Sheet:
    image: Image.Image
    graph: Graph
    planted: list[dict] = field(default_factory=list)
    seed: int = 0


def _font(size: int = 15):
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------- drawing
def _draw_symbol(d: ImageDraw.ImageDraw, cls: str, bbox, body_cy: float):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = LINE_W
    if cls == "tank":
        d.rectangle([x0, y0, x1, y1], outline=0, width=w)
        d.line([x0, y0 + 16, x1, y0 + 16], fill=0, width=w)
    elif cls == "vessel":
        d.rounded_rectangle([x0, y0, x1, y1], radius=30, outline=0, width=w)
    elif cls == "pump":
        d.ellipse([x0, y0, x1, y1], outline=0, width=w)
        d.polygon([(cx - 12, cy - 14), (cx - 12, cy + 14), (cx + 16, cy)], outline=0, width=w)
    elif cls == "valve":
        d.polygon([(x0, y0), (cx, cy), (x0, y1)], outline=0, width=w)
        d.polygon([(x1, y0), (cx, cy), (x1, y1)], outline=0, width=w)
    elif cls == "check_valve":
        d.polygon([(x0, y0), (x1, cy), (x0, y1)], outline=0, width=w)
        d.line([x1, y0, x1, y1], fill=0, width=w + 1)
    elif cls == "control_valve":
        by0, by1 = body_cy - 18, body_cy + 18
        d.polygon([(x0, by0), (cx, body_cy), (x0, by1)], outline=0, width=w)
        d.polygon([(x1, by0), (cx, body_cy), (x1, by1)], outline=0, width=w)
        d.line([cx, by0, cx, by0 - 16], fill=0, width=w)
        d.ellipse([cx - 12, y0, cx + 12, y0 + 24], outline=0, width=w)
    elif cls == "instrument":
        d.ellipse([x0, y0, x1, y1], outline=0, width=w)
        d.line([x0 + 2, cy, x1 - 2, cy], fill=0, width=1)
        f = _font(12)
        d.text((cx, cy - 8), "FIC", fill=0, font=f, anchor="mm")
    elif cls == "relief_valve":
        d.polygon([(x0, y0 + 10), (x1, y0 + 10), (cx, y1)], outline=0, width=w)
        d.line([x0, y0 + 10, x1, y0 + 10], fill=0, width=w + 1)
        d.line([cx, y0, cx, y0 + 10], fill=0, width=w)
    else:  # pragma: no cover
        raise ValueError(cls)


def _draw_seg(d: ImageDraw.ImageDraw, seg, dashed: bool):
    x0, y0, x1, y1 = seg
    if not dashed:
        d.line([x0, y0, x1, y1], fill=0, width=LINE_W)
        return
    length = max(abs(x1 - x0), abs(y1 - y0))
    dash, gap = 8, 5
    pos = 0
    ux = (x1 - x0) / length if length else 0
    uy = (y1 - y0) / length if length else 0
    while pos < length:
        end = min(pos + dash, length)
        d.line([x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end], fill=0, width=LINE_W)
        pos += dash + gap


def _apply_scan_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("L").filter(ImageFilter.GaussianBlur(0.7))
    arr = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr += nrng.normal(0, 10, arr.shape)
    speck = nrng.random(arr.shape) < 0.0008
    arr[speck] = 0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")


# --------------------------------------------------------------------------- layout
def _row_template(rng: random.Random, kind: str) -> list[str]:
    return list(rng.choice(PUMP_ROWS if kind == "pump" else CONTROL_ROWS))


def make_sheet(
    seed: int = 0,
    n_rows: int = 2,
    errors: tuple[str, ...] | list[str] = (),
    scan_noise: bool = False,
    sheet_id: str | None = None,
) -> Sheet:
    for e in errors:
        if e not in ERROR_KINDS:
            raise ValueError(f"unknown error kind '{e}', expected one of {ERROR_KINDS}")
    if "missing_signal" in errors and n_rows < 2:
        raise ValueError("missing_signal needs a control row (n_rows >= 2)")
    rng = random.Random(seed)

    kinds = [("pump" if r % 2 == 0 else "control") for r in range(n_rows)]
    rows_cls = [_row_template(rng, k) for k in kinds]

    # --- apply item-level errors to the class lists before layout
    if "missing_check_valve" in errors:
        rows_cls[0].remove("check_valve")
    if "missing_isolation_valve" in errors:
        pi = rows_cls[0].index("pump")
        del rows_cls[0][pi - 1]  # valve right upstream of the pump

    max_cols = max(len(r) for r in rows_cls)
    width = LEFT + max_cols * CELL_W + 40
    height = TOP + (n_rows - 1) * ROW_H + 130

    # --- tags
    counters: dict[str, int] = {}

    def next_tag(cls: str) -> str:
        counters[cls] = counters.get(cls, 0) + 1
        base = 100 + counters[cls] + (seed % 7) * 10
        suffix = "A" if cls == "pump" else ""
        return f"{TAG_PREFIX[cls]}-{base}{suffix}"

    items: list[dict] = []  # all drawable items
    rows: list[list[dict]] = []

    def add_item(cls, cx, cy_row, bbox, row, col, tag=True):
        it = {
            "id": f"n{len(items)}",
            "cls": cls,
            "bbox": bbox,
            "cx": cx,
            "cy": cy_row,  # body / port centre line
            "row": row,
            "col": col,
            "tag": next_tag(cls),
            "show_tag": tag,
        }
        items.append(it)
        return it

    for r, classes in enumerate(rows_cls):
        cy = TOP + r * ROW_H
        row_items = []
        for c, cls in enumerate(classes):
            cx = LEFT + CELL_W / 2 + c * CELL_W
            w, h = SIZES[cls]
            if cls == "control_valve":
                bbox = [cx - w / 2, cy - 58, cx + w / 2, cy + 18]
            else:
                bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            row_items.append(add_item(cls, cx, cy, bbox, r, c))
        rows.append(row_items)

    # --- extras: relief valve over the last vessel, instrument over each control valve
    extra_nets: list[dict] = []
    for r, row_items in enumerate(rows):
        last = row_items[-1]
        if last["cls"] == "vessel" and not ("missing_relief" in errors and r == 0):
            w, h = SIZES["relief_valve"]
            vtop = last["bbox"][1]
            bbox = [last["cx"] - w / 2, vtop - 36 - h, last["cx"] + w / 2, vtop - 36]
            rv = add_item("relief_valve", last["cx"], (bbox[1] + bbox[3]) / 2, bbox, r, last["col"])
            extra_nets.append(
                {
                    "type": "process",
                    "att": [(last, "N"), (rv, "S")],
                    "segs": [(last["cx"], vtop, last["cx"], vtop - 36)],
                }
            )
        for it in list(row_items):
            if it["cls"] == "control_valve":
                w, h = SIZES["instrument"]
                top = it["bbox"][1]
                bbox = [it["cx"] - w / 2, top - 60 - h, it["cx"] + w / 2, top - 60]
                ins = add_item("instrument", it["cx"], (bbox[1] + bbox[3]) / 2, bbox, r, it["col"])
                if "missing_signal" in errors and r == 1:
                    continue
                extra_nets.append(
                    {
                        "type": "signal",
                        "att": [(ins, "S"), (it, "N")],
                        "segs": [(it["cx"], bbox[3], it["cx"], top)],
                    }
                )

    # --- pipes between consecutive items in a row
    nets: list[dict] = []
    gap_net: dict[tuple[int, int], dict] = {}
    for r, row_items in enumerate(rows):
        for c in range(len(row_items) - 1):
            a, b = row_items[c], row_items[c + 1]
            y = a["cy"]
            x_a, x_b = a["bbox"][2], b["bbox"][0]
            net = {"type": "process", "att": [(a, "E"), (b, "W")], "segs": [(x_a, y, x_b, y)]}
            if "dangling_pipe" in errors and r == 0 and c == len(row_items) - 2:
                # keep only a stub attached to the upstream symbol
                net = {
                    "type": "process",
                    "att": [(a, "E")],
                    "segs": [(x_a, y, x_a + 0.4 * (x_b - x_a), y)],
                }
            nets.append(net)
            gap_net[(r, c)] = net

    # --- vertical tee between adjacent rows at gap 0
    for r in range(n_rows - 1):
        n0, n1 = gap_net[(r, 0)], gap_net[(r + 1, 0)]
        xm = (n0["segs"][0][0] + n0["segs"][0][2]) / 2
        y0, y1 = rows[r][0]["cy"], rows[r + 1][0]["cy"]
        n0["att"] = n0["att"] + n1["att"]
        n0["segs"] = n0["segs"] + n1["segs"] + [(xm, y0, xm, y1)]
        nets.remove(n1)
        gap_net[(r + 1, 0)] = n0
    nets += extra_nets

    # --- tag-level errors
    planted: list[dict] = []
    pump = next((i for i in rows[0] if i["cls"] == "pump"), None)
    valves0 = [i for i in rows[0] if i["cls"] == "valve"]
    if "missing_tag" in errors:
        target = valves0[-1]
        target["tag"] = None
        target["show_tag"] = False
    if "duplicate_tag" in errors:
        valves0[1 if len(valves0) > 1 else 0]["tag"] = valves0[0]["tag"] if len(valves0) > 1 else rows[1][1]["tag"]
        if len(valves0) == 1:
            valves0[0]["tag"] = rows[1][1]["tag"]

    # --- render
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    for net in nets:
        for seg in net["segs"]:
            _draw_seg(d, seg, dashed=net["type"] == "signal")
    for it in items:
        _draw_symbol(d, it["cls"], it["bbox"], it["cy"])
    f = _font(15)
    for it in items:
        if not it["show_tag"] or not it["tag"]:
            continue
        x0, y0, x1, y1 = it["bbox"]
        if it["cls"] in ("instrument", "relief_valve"):
            d.text((x1 + 8, (y0 + y1) / 2), it["tag"], fill=0, font=f, anchor="lm")
        else:
            d.text(((x0 + x1) / 2, y1 + 6), it["tag"], fill=0, font=f, anchor="mt")
    if scan_noise:
        img = _apply_scan_noise(img, rng)

    # --- ground-truth graph
    nodes = [
        Node(id=i["id"], cls=i["cls"], tag=i["tag"], bbox=[float(v) for v in i["bbox"]], source="gt")
        for i in items
    ]
    gnets = []
    for k, net in enumerate(nets):
        xs = [v for s in net["segs"] for v in (s[0], s[2])]
        ys = [v for s in net["segs"] for v in (s[1], s[3])]
        length = sum(abs(s[2] - s[0]) + abs(s[3] - s[1]) for s in net["segs"])
        gnets.append(
            Net(
                id=f"e{k}",
                type=net["type"],
                attachments=[Attachment(node=it["id"], side=side) for it, side in net["att"]],
                bbox=[min(xs), min(ys), max(xs), max(ys)],
                length_px=float(length),
            )
        )
    graph = Graph(sheet_id=sheet_id or f"synthetic_{seed}", width=width, height=height, nodes=nodes, nets=gnets)

    # --- planted-error bookkeeping (what a correct checker should flag)
    expect = {
        "missing_check_valve": ("R4-PUMP-CHECK-VALVE", pump and pump["id"]),
        "missing_isolation_valve": ("R5A-PUMP-SUCTION-ISOLATION", pump and pump["id"]),
        "missing_tag": ("R2-TAG-PRESENT", valves0[-1]["id"] if valves0 else None),
        "duplicate_tag": ("R1-UNIQUE-TAG", valves0[0]["id"] if valves0 else None),
        "dangling_pipe": ("R3-DANGLING-PIPE", rows[0][-2]["id"]),
        "missing_signal": ("R6-CONTROL-VALVE-LOOP", next((i["id"] for i in rows[1] if i["cls"] == "control_valve"), None)) if n_rows > 1 else ("", None),
        "missing_relief": ("R7-VESSEL-RELIEF", rows[0][-1]["id"]),
    }
    for e in errors:
        rule_id, node_id = expect[e]
        planted.append({"kind": e, "rule_id": rule_id, "node_id": node_id})

    return Sheet(image=img, graph=graph, planted=planted, seed=seed)


def render_legend() -> Image.Image:
    """A legend sheet: one example of each symbol class with its name."""
    width, height = 1100, 330
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    f = _font(14)
    d.text((20, 14), "LEGEND", fill=0, font=_font(18))
    for k, cls in enumerate(SYMBOL_CLASSES):
        col, row = k % 4, k // 4
        cx = 150 + col * 260
        cy = 130 + row * 150
        w, h = SIZES[cls]
        if cls == "control_valve":
            bbox = [cx - w / 2, cy - 58, cx + w / 2, cy + 18]
        else:
            bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        _draw_symbol(d, cls, bbox, cy)
        d.text((cx, bbox[3] + 8), cls.replace("_", " "), fill=0, font=f, anchor="mt")
    return img
