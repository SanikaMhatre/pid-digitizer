import json

import httpx
import pytest
from conftest import openai_body
from PIL import Image

from pid.classify import VLMClassifier
from pid.models.llm import LLMError, extract_json
from pid.synth.generator import make_sheet


def test_request_shape_auth_and_parse(make_client):
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=openai_body("hello"))

    c = make_client(handler)
    r = c.chat([{"role": "user", "content": "hi"}], reasoning_effort="low", max_tokens=50)
    assert r.text == "hello"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "m" and seen["body"]["reasoning_effort"] == "low"


def test_cost_ledger_and_cache(make_client):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, json=openai_body("ok", prompt_tokens=1_000_000, completion_tokens=1_000_000))

    c = make_client(handler)
    c.chat([{"role": "user", "content": "x"}], purpose="a")
    c.chat([{"role": "user", "content": "x"}], purpose="a")  # identical -> served from cache
    assert calls["n"] == 1
    t = c.ledger.totals()
    assert t["live_calls"] == 1 and t["cached_calls"] == 1
    assert t["cost_usd"] == pytest.approx(1.25 + 4.25)


def test_retry_on_429_then_success(make_client):
    seq = [httpx.Response(429, text="slow down"), httpx.Response(200, json=openai_body("fine"))]

    def handler(req):
        return seq.pop(0)

    assert make_client(handler, cache=False).chat([{"role": "user", "content": "x"}]).text == "fine"


def test_drops_reasoning_effort_if_server_rejects_it(make_client):
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, text="unknown parameter: reasoning_effort")
        return httpx.Response(200, json=openai_body("ok"))

    c = make_client(handler, cache=False)
    assert c.chat([{"role": "user", "content": "x"}], reasoning_effort="high").text == "ok"
    assert "reasoning_effort" not in bodies[-1]


def test_hard_error_raises(make_client):
    c = make_client(lambda req: httpx.Response(401, text="bad key"), cache=False)
    with pytest.raises(LLMError, match="401"):
        c.chat([{"role": "user", "content": "x"}])


def test_tool_calls_parsed(make_client):
    tc = [{"id": "a", "type": "function", "function": {"name": "zoom", "arguments": '{"node_id": "n1"}'}}]
    c = make_client(lambda req: httpx.Response(200, json=openai_body(None, tool_calls=tc)), cache=False)
    r = c.chat([{"role": "user", "content": "x"}])
    assert r.tool_calls == [{"id": "a", "name": "zoom", "arguments": {"node_id": "n1"}}]


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} hope that helps'],
)
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def _img():
    return Image.new("RGB", (400, 300), "white")


def _clf(make_client, content):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=openai_body(content))

    return VLMClassifier(make_client(handler, cache=False)), bodies


def test_classifier_parses_and_sends_two_images_with_schema(make_client):
    good = json.dumps({"class": "pump", "class_confidence": 0.9, "tag": " P-101A ", "tag_confidence": 0.8, "evidence": "circle+triangle"})
    clf, bodies = _clf(make_client, good)
    r = clf.classify(_img(), [100, 100, 156, 156])
    assert (r.cls, r.tag, r.class_conf) == ("pump", "P-101A", 0.9)
    parts = bodies[0]["messages"][1]["content"]
    assert sum(1 for p in parts if p["type"] == "image_url") == 2
    assert bodies[0]["response_format"]["type"] == "json_schema"


def test_classifier_normalizes_null_and_unknown_class(make_client):
    clf, _ = _clf(make_client, json.dumps({"class": "flux_capacitor", "class_confidence": 2, "tag": "null", "tag_confidence": -1, "evidence": ""}))
    r = clf.classify(_img(), [10, 10, 60, 60])
    assert r.cls == "not_a_symbol" and r.tag is None and r.class_conf == 1.0 and r.tag_conf == 0.0


def test_classifier_survives_garbage(make_client):
    clf, _ = _clf(make_client, "I cannot help with that")
    r = clf.classify(_img(), [10, 10, 60, 60])
    assert r.cls == "not_a_symbol" and r.class_conf == 0.0 and "failed" in r.evidence


def test_classifier_propagates_endpoint_failures(make_client):
    """A dead endpoint or bad key must not silently turn into an empty graph."""
    clf = VLMClassifier(make_client(lambda req: httpx.Response(401, text="bad key"), cache=False))
    with pytest.raises(LLMError):
        clf.classify(_img(), [10, 10, 60, 60])


def test_smoke_check_llm_reports_pass_and_fail(make_client):
    """The smoke checker must report per-check results and never raise."""
    from pid.smoke import check_llm

    def handler(req):
        body = json.loads(req.content)
        if "tools" in body:
            return httpx.Response(200, json=openai_body("no tool for you"))
        if "response_format" in body and body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["name"] == "x":
            return httpx.Response(200, json=openai_body('{"colour": "red", "n": 2}'))
        if "response_format" in body:
            return httpx.Response(200, json=openai_body(json.dumps({"class": "pump", "class_confidence": 0.9, "tag": "P-101A", "tag_confidence": 0.9, "evidence": "e"})))
        text = json.dumps(body["messages"])
        return httpx.Response(200, json=openai_body("pong" if "pong" in text else "391" if "17*23" in text else "3"))

    res = {r["check"]: r["ok"] for r in check_llm(make_client(handler, cache=False))}
    assert res["basic call"] and res["strict JSON schema"] and res["vision + structured classification"]
    assert res["tool calling"] is False  # this fake endpoint never returns tool calls -> reported as FAIL, not an exception


def test_smoke_check_sam_with_injected_segmenter():
    from pid.segment.base import OracleSegmenter
    from pid.smoke import check_sam

    class PerfectSeg:
        def detect(self, image):
            return OracleSegmenter(make_sheet(seed=0).graph).detect(image)

    rows = check_sam(PerfectSeg(), n_sheets=1)
    assert rows[0]["recall_at_iou_0.5"] == 1.0
