"""Classical line tracing: turn pipe pixels into nets attached to symbols.

Steps
  1. binarize the sheet, blank out every symbol box (padded) so only pipes and text remain
  2. bridge small gaps (dashed signal lines) with directional closing
  3. label connected components; each one is a net
  4. a net attaches to every symbol whose padded box it touches; the side (W/E/N/S) comes from
     where the pixels meet the box
  5. a component with a low ink-to-bridged-area ratio is a dashed line -> signal net

Limitations (documented, tested honestly): pixel connectivity cannot tell a line *crossing*
from a *junction*; both would merge.  Drawings with crossings need the verification agent or
explicit jump symbols.  Text that touches a pipe is treated as part of the pipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from .schema import Attachment, Net, Node


@dataclass
class TraceCfg:
    ink_threshold: int = 170
    mask_pad: int = 6  # symbol boxes are blanked out with this padding
    touch_reach: int = 6  # how far outside the padded box a pipe pixel still counts as touching
    dash_gap: int = 8  # largest gap bridged when joining dashes
    dash_ratio: float = 0.8  # below this ink ratio a net is a dashed (signal) line
    min_len: int = 10  # minimum skeleton length (px) for a net to be kept
    text_max_height: int = 30  # text-like strips are at most this tall (px)...
    text_max_width: int = 160  # ...and this wide
    text_branchiness: float = 1.2  # skeleton length / (w+h) above this means text, not a pipe
    text_min_ratio: float = 0.5  # ink ratio below this (dashes are ~0.6) means text, not a line


@dataclass
class TraceResult:
    nets: list[Net]
    labels: np.ndarray  # label image (0 = background)
    net_label: dict[str, int] = field(default_factory=dict)


def _side(node: Node, px: np.ndarray, py: np.ndarray) -> str:
    x0, y0, x1, y1 = node.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max((x1 - x0) / 2, 1), max((y1 - y0) / 2, 1)
    dx = (px.mean() - cx) / hw
    dy = (py.mean() - cy) / hh
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def trace_nets(image: Image.Image, nodes: list[Node], cfg: TraceCfg | None = None) -> TraceResult:
    cfg = cfg or TraceCfg()
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    ink = gray < cfg.ink_threshold

    sym = np.zeros_like(ink)
    for n in nodes:
        x0, y0, x1, y1 = n.bbox
        sym[max(0, int(y0) - cfg.mask_pad) : min(h, int(y1) + cfg.mask_pad + 1),
            max(0, int(x0) - cfg.mask_pad) : min(w, int(x1) + cfg.mask_pad + 1)] = True
    line = ink & ~sym

    g = cfg.dash_gap
    closed = ndimage.binary_closing(line, structure=np.ones((1, g + 1), bool)) | ndimage.binary_closing(
        line, structure=np.ones((g + 1, 1), bool)
    )
    closed &= ~sym
    labels, n_labels = ndimage.label(closed, structure=np.ones((3, 3), bool))
    slices = ndimage.find_objects(labels)

    # which components touch which symbols, and on which side
    touches: dict[int, list[Attachment]] = {}
    reach = cfg.mask_pad + cfg.touch_reach
    for n in nodes:
        x0, y0, x1, y1 = n.bbox
        ya, yb = max(0, int(y0) - reach), min(h, int(y1) + reach + 1)
        xa, xb = max(0, int(x0) - reach), min(w, int(x1) + reach + 1)
        region = labels[ya:yb, xa:xb]
        for lab in np.unique(region):
            if lab == 0:
                continue
            ys, xs = np.nonzero(region == lab)
            side = _side(n, xs + xa, ys + ya)
            att = Attachment(node=n.id, side=side)
            lst = touches.setdefault(int(lab), [])
            if att not in lst:
                lst.append(att)

    nets: list[Net] = []
    net_label: dict[str, int] = {}
    keep = []
    for lab, atts in touches.items():
        sl = slices[lab - 1]
        comp = labels[sl] == lab
        skel_len = float(skeletonize(comp).sum())
        if skel_len < cfg.min_len:
            continue
        ink_px = float((line[sl] & comp).sum())
        ratio = ink_px / max(float(comp.sum()), 1.0)
        # Tag text: a short, low strip.  Either its skeleton is much longer than its extent
        # (many strokes; a pipe's skeleton is about w+h), or its bridged area is mostly empty (ink ratio far below the ~0.6
        # of a dashed line).  A straight pipe stub has skeleton length ~= extent, ratio ~= 1.
        comp_h, comp_w = comp.shape
        if min(comp_h, comp_w) <= cfg.text_max_height and max(comp_h, comp_w) <= cfg.text_max_width:
            if skel_len > cfg.text_branchiness * (comp_h + comp_w) or ratio < cfg.text_min_ratio:
                continue
        ys, xs = sl
        keep.append((ys.start, xs.start, lab, atts, skel_len, ratio, [xs.start, ys.start, xs.stop, ys.stop]))

    for k, (_, _, lab, atts, skel_len, ratio, bbox) in enumerate(sorted(keep)):
        net_id = f"e{k}"
        nets.append(
            Net(
                id=net_id,
                type="signal" if ratio < cfg.dash_ratio else "process",
                attachments=atts,
                bbox=[float(v) for v in bbox],
                length_px=skel_len,
                confidence=1.0,
            )
        )
        net_label[net_id] = lab
    return TraceResult(nets=nets, labels=labels, net_label=net_label)
