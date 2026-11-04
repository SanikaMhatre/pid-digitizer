"""End-to-end pipeline: segment -> classify -> trace -> graph -> verify -> rules -> overlay."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .agent import AgentConfig, VerificationAgent, flag_only
from .classify import SymbolClassifier
from .models.llm import ChatClient
from .rules.engine import Rule, Violation, default_rules, evaluate
from .schema import Graph, Node
from .segment.base import Segmenter
from .trace import TraceCfg, TraceResult, trace_nets
from .viz import draw_overlay

log = logging.getLogger("pid.pipeline")


@dataclass
class PipelineConfig:
    classify_effort: str = "low"
    run_agent: bool = True
    max_workers: int = 8
    trace: TraceCfg = field(default_factory=TraceCfg)
    agent: AgentConfig = field(default_factory=AgentConfig)
    min_class_conf: float = 0.0  # drop symbols below this confidence (0 keeps everything)
    max_candidates: int = 300  # hard cap on model classification calls per sheet (credit protection)


@dataclass
class PipelineResult:
    graph: Graph
    violations: list[Violation]
    overlay: Image.Image
    timings: dict
    cost: dict
    audit: list[dict]
    trace: TraceResult
    warnings: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        segmenter: Segmenter,
        classifier: SymbolClassifier,
        client: Optional[ChatClient] = None,
        rules: Optional[list[Rule]] = None,
        cfg: Optional[PipelineConfig] = None,
    ):
        self.segmenter, self.classifier, self.client = segmenter, classifier, client
        self.rules = rules if rules is not None else default_rules()
        self.cfg = cfg or PipelineConfig()

    def run(self, image: Image.Image, sheet_id: str = "sheet") -> PipelineResult:
        image = image.convert("RGB")
        t = {}
        mark = self.client.ledger.snapshot() if self.client else 0

        t0 = time.time()
        dets = self.segmenter.detect(image)
        t["segment_s"] = time.time() - t0
        capped = 0
        if len(dets) > self.cfg.max_candidates:
            capped = len(dets) - self.cfg.max_candidates
            dets = sorted(dets, key=lambda d: d.score, reverse=True)[: self.cfg.max_candidates]

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as ex:
            results = list(ex.map(lambda d: self.classifier.classify(image, d.bbox, self.cfg.classify_effort), dets))
        t["classify_s"] = time.time() - t0

        keep = [(d, r) for d, r in zip(dets, results, strict=True) if r.cls != "not_a_symbol" and r.class_conf >= self.cfg.min_class_conf]
        keep.sort(key=lambda dr: (round(dr[0].bbox[1] / 40), dr[0].bbox[0]))  # reading order
        nodes = [
            Node(
                id=f"n{i}",
                cls=r.cls,
                tag=r.tag,
                bbox=[float(v) for v in d.bbox],
                confidence=r.class_conf,
                tag_confidence=r.tag_conf,
                source="muse",
                notes=r.evidence or None,
            )
            for i, (d, r) in enumerate(keep)
        ]

        t0 = time.time()
        tr = trace_nets(image, nodes, self.cfg.trace)
        t["trace_s"] = time.time() - t0

        graph = Graph(sheet_id=sheet_id, width=image.size[0], height=image.size[1], nodes=nodes, nets=tr.nets)

        t0 = time.time()
        audit: list[dict] = []
        if self.cfg.run_agent and self.client is not None:
            res = VerificationAgent(self.client, self.classifier, image, self.cfg.agent).run(graph)
            audit = res.audit
        else:
            flag_only(graph, self.cfg.agent.low_conf)
        t["verify_s"] = time.time() - t0

        t0 = time.time()
        violations = evaluate(graph, self.rules)
        t["rules_s"] = time.time() - t0

        warnings = _sanity_warnings(len(dets), len(nodes), graph)
        if capped:
            warnings.insert(0, f"The localizer returned {capped + len(dets)} candidates; only the {len(dets)} highest-scoring were classified (max_candidates).")
        overlay = draw_overlay(image, graph, violations, tr)
        t["total_s"] = sum(t.values())
        cost = self.client.ledger.since(mark) if self.client else {"cost_usd": 0.0, "calls": 0}
        log.info(
            "sheet=%s symbols=%d nets=%d violations=%d review=%d seconds=%.2f cost_usd=%.4f",
            sheet_id, len(graph.nodes), len(graph.nets), len(violations), len(graph.review_queue), t["total_s"], cost.get("cost_usd", 0.0),
        )
        for w in warnings:
            log.warning("sheet=%s %s", sheet_id, w)
        return PipelineResult(graph, violations, overlay, {k: round(v, 3) for k, v in t.items()}, cost, audit, tr, warnings)


def _sanity_warnings(n_candidates: int, n_symbols: int, graph: Graph) -> list[str]:
    """Cheap checks that catch a result that is syntactically fine but probably wrong."""
    out = []
    if n_candidates == 0:
        out.append("The localizer found no candidate symbols; the sheet may be blank or too large, or the model endpoint misbehaving.")
    elif n_symbols == 0:
        out.append(f"All {n_candidates} candidates were rejected as 'not a symbol'; check the model endpoint and the legend.")
    elif n_symbols < 0.4 * n_candidates:
        out.append(f"{n_candidates - n_symbols} of {n_candidates} candidates were rejected; the localizer may be over-triggering.")
    if n_symbols and not graph.nets:
        out.append("Symbols were found but no connecting pipes; line tracing may need tuning for this drawing style.")
    if graph.review_queue:
        out.append(f"{len(graph.review_queue)} item(s) need human review before this result is relied on.")
    return out
