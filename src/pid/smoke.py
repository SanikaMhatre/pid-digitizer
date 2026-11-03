"""Smoke tests for the two live model dependencies.  Run these FIRST once credits arrive.

  python -m pid.smoke llm        # Muse Spark Open: basic call, vision, strict JSON, tool calling, reasoning param
  python -m pid.smoke sam        # SAM 3.1 on Modal: does it find the symbols on a synthetic sheet?
  python -m pid.smoke all

Each check prints PASS / FAIL with the reason, and the LLM checks print the tokens and dollars used,
so you also learn what a sheet will cost before running an evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys

from .classify import VLMClassifier
from .models.llm import ChatClient, LLMError, image_part
from .schema import iou
from .synth.generator import make_sheet


def check_llm(client: ChatClient) -> list[dict]:
    results = []

    def run(name, fn):
        try:
            detail = fn()
            results.append({"check": name, "ok": True, "detail": detail})
        except Exception as e:
            results.append({"check": name, "ok": False, "detail": f"{type(e).__name__}: {str(e)[:300]}"})

    sheet = make_sheet(seed=0)
    target = next(n for n in sheet.graph.nodes if n.cls == "pump")

    def basic():
        r = client.chat([{"role": "user", "content": "Reply with exactly: pong"}], purpose="smoke", max_tokens=20, use_cache=False)
        assert "pong" in r.text.lower(), f"unexpected reply {r.text!r}"
        return r.text.strip()

    def reasoning():
        r = client.chat([{"role": "user", "content": "What is 17*23? Answer with the number only."}], purpose="smoke",
                        reasoning_effort="low", max_tokens=300, use_cache=False)
        assert "391" in r.text, f"unexpected reply {r.text!r}"
        return "reasoning_effort accepted (or silently dropped); answer ok"

    def vision():
        res = VLMClassifier(client).classify(sheet.image, target.bbox, "low")
        assert res.cls != "not_a_symbol", f"model said not_a_symbol ({res.evidence})"
        return f"truth pump / {target.tag}; model said {res.cls} / {res.tag} (class conf {res.class_conf:.2f})"

    def strict_json():
        r = client.chat(
            [{"role": "user", "content": "Give a colour name and a number 1-3."}],
            purpose="smoke",
            response_format={"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {
                "type": "object", "properties": {"colour": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["colour", "n"], "additionalProperties": False}}},
            max_tokens=100, use_cache=False)
        d = r.json()
        assert set(d) == {"colour", "n"}, f"schema not honoured: {d}"
        return d

    def tools():
        tool = [{"type": "function", "function": {"name": "zoom", "description": "magnify a symbol",
                 "parameters": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}}}]
        r = client.chat([{"role": "user", "content": "Zoom into node n7."}], purpose="smoke", tools=tool,
                        tool_choice="auto", max_tokens=100, use_cache=False)
        assert r.tool_calls and r.tool_calls[0]["name"] == "zoom", f"no tool call returned: {r.text!r}"
        return r.tool_calls[0]["arguments"]

    def image_in_tool_followup():
        r = client.chat([{"role": "user", "content": [{"type": "text", "text": "How many separate symbols are in this image? Number only."},
                                                      image_part(sheet.image.crop((60, 140, 700, 330)))]}],
                        purpose="smoke", max_tokens=200, use_cache=False)
        return f"model answered {r.text.strip()[:40]!r}"

    for name, fn in [("basic call", basic), ("reasoning_effort", reasoning), ("vision + structured classification", vision),
                     ("strict JSON schema", strict_json), ("tool calling", tools), ("image in follow-up message", image_in_tool_followup)]:
        run(name, fn)
    return results


def check_sam(segmenter=None, n_sheets: int = 2) -> list[dict]:
    if segmenter is None:
        from .segment.sam_modal import ModalSamSegmenter

        segmenter = ModalSamSegmenter()
    out = []
    for seed in range(n_sheets):
        sheet = make_sheet(seed=seed)
        dets = segmenter.detect(sheet.image)
        hit = sum(1 for n in sheet.graph.nodes if any(iou(n.bbox, d.bbox) >= 0.5 for d in dets))
        out.append({"seed": seed, "gt_symbols": len(sheet.graph.nodes), "detections": len(dets),
                    "recall_at_iou_0.5": round(hit / len(sheet.graph.nodes), 3)})
    return out


def _print_llm(results, client):
    ok = True
    for r in results:
        ok &= r["ok"]
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['check']}: {r['detail']}")
    print("usage:", json.dumps(client.ledger.totals()))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["llm", "sam", "all"])
    args = ap.parse_args(argv)
    ok = True
    if args.target in ("llm", "all"):
        try:
            with ChatClient.from_env(cache_dir=False) as client:
                print(f"endpoint {client.base_url}  model {client.model}")
                ok &= _print_llm(check_llm(client), client)
        except LLMError as e:
            print("FAIL", e)
            ok = False
    if args.target in ("sam", "all"):
        try:
            rows = check_sam()
            for r in rows:
                print(r)
            good = all(r["recall_at_iou_0.5"] >= 0.5 for r in rows)
            print(f"[{'PASS' if good else 'WEAK'}] SAM localization recall >= 0.5 on synthetic sheets")
            ok &= good
        except Exception as e:
            print(f"[FAIL] SAM: {type(e).__name__}: {str(e)[:300]}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
