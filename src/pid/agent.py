"""Verification agent: re-examines uncertain items with tools instead of guessing.

The agent never edits the graph directly.  It can only call the tools below; every call is
validated and appended to an audit log.  'flag_for_human' is a first-class outcome: a system that
knows when it is unsure is more useful than one that is always confident.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .classify import SymbolClassifier
from .models.llm import ChatClient, image_part
from .schema import SYMBOL_CLASSES, Graph

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "zoom",
            "description": "Get a magnified crop around a symbol so you can inspect it. The image arrives in the next message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "margin": {"type": "integer", "description": "extra pixels around the symbol (default 40)"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reclassify",
            "description": "Ask the vision model again about a symbol at a higher reasoning effort. Returns its answer; does NOT change the graph.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}, "effort": {"type": "string", "enum": ["medium", "high"]}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_change",
            "description": "Change a symbol's class or tag in the graph. Only commit when you have evidence. For a symbol that truly has no visible tag, commit tag = null.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "field": {"type": "string", "enum": ["class", "tag"]},
                    "value": {"type": ["string", "null"]},
                    "evidence": {"type": "string"},
                },
                "required": ["node_id", "field", "value", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_human",
            "description": "Put a symbol in the human review queue when you cannot resolve it with confidence.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["node_id", "reason"],
            },
        },
    },
]

SYSTEM = (
    "You are a verification agent for a P&ID digitizer. You are given a list of symbols whose class or "
    "tag the first-pass model was unsure about. For each one: zoom, optionally reclassify at higher effort, "
    "then either commit a correction backed by evidence, confirm it is fine by committing nothing, or flag "
    "it for a human. Never invent a tag; if none is visible commit tag=null. Be economical: you have a "
    "fixed budget of tool calls. Stop (reply without tool calls) when every listed symbol is resolved."
)


@dataclass
class AgentConfig:
    max_tool_calls: int = 14
    low_conf: float = 0.75
    effort: str = "medium"


@dataclass
class AgentResult:
    graph: Graph
    audit: list[dict] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    tool_calls: int = 0


def flag_uncertain(graph: Graph, low_conf: float = 0.75) -> list[dict]:
    items = []
    for n in graph.nodes:
        reasons = []
        if n.confidence < low_conf:
            reasons.append(f"class confidence {n.confidence:.2f}")
        if not n.tag:
            reasons.append("no tag read")
        elif n.tag_confidence < low_conf:
            reasons.append(f"tag confidence {n.tag_confidence:.2f}")
        if reasons:
            items.append({"node_id": n.id, "class": n.cls, "tag": n.tag, "reasons": reasons})
    return items


class VerificationAgent:
    def __init__(self, client: ChatClient, classifier: SymbolClassifier, image: Image.Image, cfg: Optional[AgentConfig] = None):
        self.client, self.classifier, self.image = client, classifier, image
        self.cfg = cfg or AgentConfig()

    # ------------------------------------------------------------------ tools
    def _zoom(self, graph: Graph, node_id: str, margin: int = 40) -> tuple[dict, Optional[Image.Image]]:
        n = graph.node_map().get(node_id)
        if n is None:
            return {"error": f"unknown node {node_id}"}, None
        x0, y0, x1, y1 = n.bbox
        w, h = self.image.size
        box = (max(0, int(x0 - margin)), max(0, int(y0 - margin)), min(w, int(x1 + margin + 60)), min(h, int(y1 + margin + 30)))
        crop = self.image.crop(box)
        s = max(1.0, 320 / max(1, min(crop.size)))
        if s > 1:
            crop = crop.resize((int(crop.size[0] * s), int(crop.size[1] * s)), Image.LANCZOS)
        return {"ok": True, "note": f"magnified crop of {node_id} attached in the next message"}, crop

    def _reclassify(self, graph: Graph, node_id: str, effort: str = "high") -> dict:
        n = graph.node_map().get(node_id)
        if n is None:
            return {"error": f"unknown node {node_id}"}
        r = self.classifier.classify(self.image, n.bbox, reasoning_effort=effort if effort in ("medium", "high") else "high")
        return {"class": r.cls, "class_confidence": r.class_conf, "tag": r.tag, "tag_confidence": r.tag_conf, "evidence": r.evidence}

    def _commit(self, graph: Graph, node_id: str, field_: str, value, evidence: str) -> dict:
        n = graph.node_map().get(node_id)
        if n is None:
            return {"error": f"unknown node {node_id}"}
        if field_ == "class":
            if value not in SYMBOL_CLASSES:
                return {"error": f"class must be one of {SYMBOL_CLASSES}"}
            n.cls = value
            n.confidence = max(n.confidence, 0.9)
        elif field_ == "tag":
            n.tag = value.strip() if isinstance(value, str) and value.strip() else None
            n.tag_confidence = max(n.tag_confidence, 0.9)
        else:
            return {"error": "field must be 'class' or 'tag'"}
        n.source = n.source.split("+")[0] + "+agent"
        n.notes = (evidence or "")[:200]
        return {"ok": True, "node": node_id, field_: value}

    def _flag(self, graph: Graph, node_id: str, reason: str) -> dict:
        n = graph.node_map().get(node_id)
        if n is None:
            return {"error": f"unknown node {node_id}"}
        n.needs_review = True
        graph.review_queue.append({"item": node_id, "reason": reason[:200]})
        return {"ok": True, "flagged": node_id}

    # ------------------------------------------------------------------ loop
    def run(self, graph: Graph) -> AgentResult:
        flagged = flag_uncertain(graph, self.cfg.low_conf)
        result = AgentResult(graph=graph)
        if not flagged:
            return result

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Symbols to verify:\n" + json.dumps(flagged, indent=1)},
        ]
        used = 0
        touched: set[str] = set()
        while used < self.cfg.max_tool_calls:
            res = self.client.chat(
                messages,
                purpose="agent",
                tools=TOOLS,
                tool_choice="auto",
                reasoning_effort=self.cfg.effort,
                max_tokens=800,
            )
            if not res.tool_calls:
                break
            messages.append(res.message)
            pending_images: list[Image.Image] = []
            for call in res.tool_calls:
                if used >= self.cfg.max_tool_calls:
                    out = {"error": "tool-call budget exhausted"}
                else:
                    used += 1
                    out = self._dispatch(graph, call["name"], call["arguments"], pending_images)
                    touched.add(str(call["arguments"].get("node_id", "")))
                result.audit.append({"step": used, "tool": call["name"], "args": call["arguments"], "result": out})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(out)})
            if pending_images:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Requested crop(s):"}] + [image_part(im) for im in pending_images],
                    }
                )

        # anything the agent never resolved or flagged goes to a human: no silent guesses
        queued = {q["item"] for q in graph.review_queue}
        for item in flagged:
            nid = item["node_id"]
            n = graph.node_map()[nid]
            still_bad = n.confidence < self.cfg.low_conf or (not n.tag and nid not in touched)
            if still_bad and nid not in queued:
                self._flag(graph, nid, "unresolved after agent budget: " + "; ".join(item["reasons"]))
        result.tool_calls = used
        result.flagged = [q["item"] for q in graph.review_queue]
        return result

    def _dispatch(self, graph: Graph, name: str, args: dict, images: list) -> dict:
        try:
            if name == "zoom":
                out, img = self._zoom(graph, args["node_id"], int(args.get("margin", 40)))
                if img is not None:
                    images.append(img)
                return out
            if name == "reclassify":
                return self._reclassify(graph, args["node_id"], args.get("effort", "high"))
            if name == "commit_change":
                return self._commit(graph, args["node_id"], args["field"], args.get("value"), args.get("evidence", ""))
            if name == "flag_for_human":
                return self._flag(graph, args["node_id"], args.get("reason", ""))
            return {"error": f"unknown tool {name}"}
        except KeyError as e:
            return {"error": f"missing argument {e}"}


def flag_only(graph: Graph, low_conf: float = 0.75) -> Graph:
    """No-LLM fallback: queue uncertain items for a human without trying to resolve them."""
    queued = {q["item"] for q in graph.review_queue}
    for item in flag_uncertain(graph, low_conf):
        if item["node_id"] not in queued:
            graph.node_map()[item["node_id"]].needs_review = True
            graph.review_queue.append({"item": item["node_id"], "reason": "; ".join(item["reasons"])})
    return graph
