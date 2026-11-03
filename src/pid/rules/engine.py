"""Deterministic design-rule engine.

Rules are data (YAML).  The language model never decides whether a rule is violated:
it can only help *write* a rule (see nl_compiler.py).  Evaluation is plain code over the
graph, so every finding is reproducible and can be unit tested.

Rule format (YAML list):

  - id: R4-PUMP-CHECK-VALVE
    severity: warning                 # error | warning | info
    description: Pump discharge needs a check valve within 2 hops downstream
    applies_to: {class: pump}         # node matcher (or {kind: net})
    require:                          # exactly one requirement
      path:
        direction: downstream         # downstream | upstream | any
        edge_type: process            # process | signal | any
        max_hops: 2                   # number of symbols to walk through
        reaches: {class: check_valve}

Other requirements:  unique: tag | has_attr: tag | connected: true | min_symbols: N (nets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schema import INLET_SIDES, OUTLET_SIDES, SYMBOL_CLASSES, Graph, Net, Node


# --------------------------------------------------------------------------- models
class Matcher(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    cls: Optional[str] = Field(default=None, alias="class")
    class_in: Optional[list[str]] = None
    has: Optional[Literal["tag"]] = None
    missing: Optional[Literal["tag"]] = None
    kind: Literal["node", "net"] = "node"

    @field_validator("cls")
    @classmethod
    def _known_class(cls, v):
        if v is not None and v not in SYMBOL_CLASSES:
            raise ValueError(f"unknown class '{v}', expected one of {SYMBOL_CLASSES}")
        return v

    @field_validator("class_in")
    @classmethod
    def _known_classes(cls, v):
        if v is not None:
            bad = [c for c in v if c not in SYMBOL_CLASSES]
            if bad:
                raise ValueError(f"unknown classes {bad}, expected one of {SYMBOL_CLASSES}")
        return v


class PathReq(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["downstream", "upstream", "any"] = "any"
    edge_type: Literal["process", "signal", "any"] = "any"
    max_hops: int = Field(default=1, ge=1, le=6)
    reaches: Matcher


class Require(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unique: Optional[Literal["tag"]] = None
    has_attr: Optional[Literal["tag"]] = None
    path: Optional[PathReq] = None
    connected: Optional[bool] = None
    min_symbols: Optional[int] = Field(default=None, ge=1, le=10)

    @model_validator(mode="after")
    def _exactly_one(self):
        set_ = [k for k in ("unique", "has_attr", "path", "connected", "min_symbols") if getattr(self, k) is not None]
        if len(set_) != 1:
            raise ValueError(f"require must contain exactly one condition, got {set_}")
        return self


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Literal["error", "warning", "info"] = "warning"
    description: str
    applies_to: Matcher = Field(default_factory=Matcher)
    require: Require
    enabled: bool = True

    @model_validator(mode="after")
    def _kind_matches_requirement(self):
        is_net = self.applies_to.kind == "net"
        if is_net and self.require.min_symbols is None:
            raise ValueError("rules on nets only support the 'min_symbols' requirement")
        if not is_net and self.require.min_symbols is not None:
            raise ValueError("'min_symbols' only applies to rules with applies_to: {kind: net}")
        return self


@dataclass
class Violation:
    rule_id: str
    severity: str
    message: str
    node_ids: list[str] = field(default_factory=list)
    net_ids: list[str] = field(default_factory=list)
    bbox: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "node_ids": self.node_ids,
            "net_ids": self.net_ids,
            "bbox": self.bbox,
        }


# --------------------------------------------------------------------------- loading
def load_rules(path: str | Path) -> list[Rule]:
    data = yaml.safe_load(Path(path).read_text())
    return parse_rules(data)


def parse_rules(data) -> list[Rule]:
    if isinstance(data, dict):
        data = data.get("rules", [data])
    return [Rule.model_validate(r) for r in data]


def default_rules() -> list[Rule]:
    return load_rules(Path(__file__).with_name("starter_pack.yaml"))


# --------------------------------------------------------------------------- evaluation
def _match(node: Node, m: Matcher) -> bool:
    if m.cls is not None and node.cls != m.cls:
        return False
    if m.class_in is not None and node.cls not in m.class_in:
        return False
    if m.has == "tag" and not node.tag:
        return False
    if m.missing == "tag" and node.tag:
        return False
    return True


class _Index:
    """Adjacency index with direction awareness."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.nodes = graph.node_map()
        self.by_node: dict[str, list[tuple[Net, str]]] = {n.id: [] for n in graph.nodes}
        for net in graph.nets:
            for a in net.attachments:
                if a.node in self.by_node:
                    self.by_node[a.node].append((net, a.side))

    def neighbors(self, node_id: str, direction: str, edge_type: str) -> list[str]:
        out = []
        for net, my_side in self.by_node.get(node_id, []):
            if edge_type != "any" and net.type != edge_type:
                continue
            directional = direction != "any" and net.type == "process"
            if directional:
                if direction == "downstream" and my_side not in OUTLET_SIDES:
                    continue
                if direction == "upstream" and my_side not in INLET_SIDES:
                    continue
            for a in net.attachments:
                if a.node == node_id or a.node not in self.nodes:
                    continue
                if directional:
                    if direction == "downstream" and a.side not in INLET_SIDES:
                        continue
                    if direction == "upstream" and a.side not in OUTLET_SIDES:
                        continue
                out.append(a.node)
        return out

    def reach(self, start: str, req: PathReq) -> list[str]:
        seen = {start}
        frontier = [start]
        hits: list[str] = []
        for _ in range(req.max_hops):
            nxt = []
            for n in frontier:
                for m in self.neighbors(n, req.direction, req.edge_type):
                    if m in seen:
                        continue
                    seen.add(m)
                    nxt.append(m)
                    if _match(self.nodes[m], req.reaches):
                        hits.append(m)
            frontier = nxt
            if not frontier:
                break
        return hits


def _describe_path(req: PathReq) -> str:
    target = req.reaches.cls or (req.reaches.class_in and "/".join(req.reaches.class_in)) or "symbol"
    return f"no {target} within {req.max_hops} hop(s) ({req.direction}, {req.edge_type})"


def evaluate(graph: Graph, rules: list[Rule]) -> list[Violation]:
    idx = _Index(graph)
    violations: list[Violation] = []

    for rule in rules:
        if not rule.enabled:
            continue
        req = rule.require

        if rule.applies_to.kind == "net":
            for net in graph.nets:
                n_symbols = len({a.node for a in net.attachments})
                if n_symbols < (req.min_symbols or 1):
                    labels = [idx.nodes[a.node].label() for a in net.attachments if a.node in idx.nodes]
                    where = f" at {', '.join(labels)}" if labels else ""
                    violations.append(
                        Violation(
                            rule.id,
                            rule.severity,
                            f"{rule.description}{where}",
                            node_ids=sorted({a.node for a in net.attachments}),
                            net_ids=[net.id],
                            bbox=net.bbox,
                        )
                    )
            continue

        subjects = [n for n in graph.nodes if _match(n, rule.applies_to)]

        if req.unique is not None:
            groups: dict[str, list[Node]] = {}
            for n in subjects:
                if n.tag:
                    groups.setdefault(n.tag, []).append(n)
            for tag, members in groups.items():
                if len(members) > 1:
                    for n in members:
                        violations.append(
                            Violation(
                                rule.id,
                                rule.severity,
                                f"{rule.description}: '{tag}' is used by {len(members)} symbols",
                                node_ids=[n.id],
                                bbox=n.bbox,
                            )
                        )
            continue

        for n in subjects:
            ok = True
            detail = ""
            if req.has_attr is not None:
                ok = bool(n.tag)
                detail = "missing tag"
            elif req.connected is not None:
                ok = bool(idx.by_node.get(n.id)) == req.connected
                detail = "not connected to any pipe or signal line"
            elif req.path is not None:
                ok = bool(idx.reach(n.id, req.path))
                detail = _describe_path(req.path)
            if not ok:
                violations.append(
                    Violation(
                        rule.id,
                        rule.severity,
                        f"{rule.description}: {n.label()} - {detail}",
                        node_ids=[n.id],
                        bbox=n.bbox,
                    )
                )
    return violations
