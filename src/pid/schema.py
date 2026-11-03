"""Graph schema for a digitized P&ID sheet.

A sheet is a graph with two kinds of elements:
  * nodes: symbols (tank, pump, valve, ...), each with a class, optional tag and bounding box
  * nets:  connected pieces of pipe / signal line.  A net touches one or more symbols;
           each touch is an attachment (node id + side of the symbol the line meets).

Representing a net explicitly (instead of node-to-node edges) keeps junctions honest:
a tee that feeds two valves is one net with three attachments, not a clique of edges.
Flow direction comes from the side of attachment: by convention the inlet of a symbol
is its west side and its outlet is its east side.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SYMBOL_CLASSES = [
    "tank",
    "vessel",
    "pump",
    "valve",
    "check_valve",
    "control_valve",
    "instrument",
    "relief_valve",
]

Side = Literal["W", "E", "N", "S"]
NetType = Literal["process", "signal"]

INLET_SIDES = {"W"}
OUTLET_SIDES = {"E"}


class Node(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    cls: str = Field(alias="class")
    tag: Optional[str] = None
    bbox: list[float]  # x0, y0, x1, y1 in pixels
    confidence: float = 1.0
    tag_confidence: float = 1.0
    source: str = "gt"
    needs_review: bool = False
    notes: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2

    def label(self) -> str:
        return self.tag or f"{self.cls}:{self.id}"


class Attachment(BaseModel):
    node: str
    side: Side


class Net(BaseModel):
    id: str
    type: NetType = "process"
    attachments: list[Attachment]
    bbox: Optional[list[float]] = None
    length_px: float = 0.0
    confidence: float = 1.0


class Graph(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sheet_id: str = "sheet"
    width: int = 0
    height: int = 0
    nodes: list[Node] = Field(default_factory=list)
    nets: list[Net] = Field(default_factory=list)
    review_queue: list[dict] = Field(default_factory=list)

    # ---- lookups -------------------------------------------------------
    def node_map(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def nets_of(self, node_id: str) -> list[tuple[Net, str]]:
        out = []
        for net in self.nets:
            for a in net.attachments:
                if a.node == node_id:
                    out.append((net, a.side))
        return out

    # ---- derived views -------------------------------------------------
    def edges(self) -> list[dict]:
        """Node-to-node view: every pair of symbols that share a net."""
        edges = []
        for net in self.nets:
            atts = net.attachments
            for i in range(len(atts)):
                for j in range(i + 1, len(atts)):
                    if atts[i].node != atts[j].node:
                        edges.append(
                            {
                                "a": atts[i].node,
                                "b": atts[j].node,
                                "net": net.id,
                                "type": net.type,
                            }
                        )
        return edges

    def connected_pairs(self) -> set[frozenset]:
        return {frozenset((e["a"], e["b"])) for e in self.edges()}

    def bom(self) -> list[dict]:
        rows = []
        for n in sorted(self.nodes, key=lambda n: (n.cls, n.tag or "")):
            rows.append(
                {
                    "id": n.id,
                    "class": n.cls,
                    "tag": n.tag or "",
                    "x": round(n.center[0]),
                    "y": round(n.center[1]),
                    "confidence": round(n.confidence, 3),
                }
            )
        return rows

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        d = self.model_dump(by_alias=True)
        d["edges"] = self.edges()
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> Graph:
        d = {k: v for k, v in d.items() if k != "edges"}
        return cls.model_validate(d)


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0
