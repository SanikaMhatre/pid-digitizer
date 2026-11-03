"""Natural-language -> rule DSL compiler.

The model only *authors* a rule.  What it returns is validated against the Rule schema (unknown
classes and malformed conditions are rejected), previewed on reference sheets, and shown to the
person for approval.  The deterministic engine then decides violations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from ..models.llm import ChatClient
from ..schema import SYMBOL_CLASSES, Graph
from .engine import Rule, evaluate

SYSTEM = f"""You convert a plain-English P&ID design rule into ONE strict JSON object (no prose).

Schema:
{{
  "id": "R-<SHORT-UPPERCASE-NAME>",
  "severity": "error" | "warning" | "info",
  "description": "<short sentence>",
  "applies_to": {{ "class": <one class> }} | {{ "class_in": [<classes>] }} | {{ "has": "tag" }} | {{ "kind": "net" }} | {{}},
  "require": EXACTLY ONE of:
      {{"path": {{"direction": "downstream"|"upstream"|"any", "edge_type": "process"|"signal"|"any",
                  "max_hops": 1-6, "reaches": {{"class": <class>}} or {{"class_in": [<classes>]}} }} }}
    | {{"unique": "tag"}} | {{"has_attr": "tag"}} | {{"connected": true}} | {{"min_symbols": <int>}}   (min_symbols only with kind=net)
}}
Allowed classes: {SYMBOL_CLASSES}.
Flow runs west to east: 'downstream of X' means following X's outlet. 'hops' counts symbols walked through.

Example 1 input: "Every pump must have a check valve within 2 symbols after it"
Example 1 output: {{"id":"R-PUMP-NRV","severity":"warning","description":"Pump needs a check valve within 2 hops downstream","applies_to":{{"class":"pump"}},"require":{{"path":{{"direction":"downstream","edge_type":"process","max_hops":2,"reaches":{{"class":"check_valve"}}}}}}}}
Example 2 input: "No two symbols may share a tag"
Example 2 output: {{"id":"R-UNIQUE","severity":"error","description":"Tags must be unique","applies_to":{{"has":"tag"}},"require":{{"unique":"tag"}}}}
If the request cannot be expressed in this schema, return {{"error": "<why>"}}."""


@dataclass
class CompileResult:
    rule: Optional[Rule]
    error: Optional[str]
    raw: str
    preview: dict


def compile_rule(
    client: ChatClient,
    text: str,
    reference_graphs: Optional[dict[str, Graph]] = None,
    reasoning_effort: str = "medium",
    max_attempts: int = 2,
) -> CompileResult:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    raw, err = "", None
    for _ in range(max_attempts):
        res = client.chat(
            messages,
            purpose="nl_rule",
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
            max_tokens=700,
        )
        raw = res.text
        try:
            data = res.json()
        except Exception as e:
            err = f"model did not return JSON: {e}"
        else:
            if "error" in data and len(data) == 1:
                return CompileResult(None, str(data["error"]), raw, {})
            try:
                rule = Rule.model_validate(data)
            except ValidationError as e:
                err = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            else:
                return CompileResult(rule, None, raw, _preview(rule, reference_graphs))
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That was rejected: {err}. Return a corrected JSON object only."},
        ]
    return CompileResult(None, err, raw, {})


def _preview(rule: Rule, refs: Optional[dict[str, Graph]]) -> dict:
    out = {}
    for name, g in (refs or {}).items():
        v = evaluate(g, [rule])
        out[name] = {"violations": len(v), "examples": [x.message for x in v[:3]]}
    return out


def rule_to_yaml(rule: Rule) -> str:
    import yaml

    return yaml.safe_dump([json.loads(rule.model_dump_json(by_alias=True, exclude_none=True))], sort_keys=False)
