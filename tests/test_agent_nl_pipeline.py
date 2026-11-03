import base64
import io
import json

import pytest
import yaml
from conftest import ScriptedClient, text_reply, tool_reply
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from pid.agent import AgentConfig, VerificationAgent, flag_uncertain
from pid.app.api import create_app
from pid.classify import SimulatedClassifier
from pid.config import build_simulated_pipeline
from pid.eval.metrics import match_nodes, planted_recall, score_graph, score_violations
from pid.models.llm import ChatResult, Ledger
from pid.pipeline import Pipeline, PipelineConfig
from pid.rules.engine import default_rules, parse_rules
from pid.rules.nl_compiler import compile_rule
from pid.segment.base import OracleSegmenter
from pid.synth.generator import make_sheet


# --------------------------------------------------------------------------- agent
class PolicyClient:
    """A rule-based fake of a competent verifier (NOT a model): reclassify every flagged symbol,
    then commit the class it got back.  Exercises the full tool loop deterministically."""

    def __init__(self):
        self.ledger = Ledger()
        self.n = 0

    def chat(self, messages, **kw):
        self.n += 1
        last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        tool_msgs = [m for m in messages[last_assistant + 1 :] if m["role"] == "tool"]
        if last_assistant == -1:
            flagged = json.loads(messages[1]["content"].split("\n", 1)[1])
            return tool_reply(*[("reclassify", {"node_id": f["node_id"], "effort": "high"}) for f in flagged])
        # we asked for reclassify; now commit what came back
        prev = messages[last_assistant]["tool_calls"]
        calls = []
        for tc, tm in zip(prev, tool_msgs, strict=True):
            if tc["function"]["name"] != "reclassify":
                continue
            args = json.loads(tc["function"]["arguments"])
            out = json.loads(tm["content"])
            calls.append(("commit_change", {"node_id": args["node_id"], "field": "class", "value": out["class"], "evidence": "reclassified at high effort"}))
        return tool_reply(*calls) if calls else ChatResult("", [], {"role": "assistant", "content": ""})


def _noisy_graph(error_rate=0.4):
    sheet = make_sheet(seed=5, n_rows=2)
    clf = SimulatedClassifier(sheet.graph, error_rate=error_rate, seed=1)
    g = sheet.graph.model_copy(deep=True)
    for n in g.nodes:  # first-pass answers at low effort
        r = clf.classify(sheet.image, n.bbox, "low")
        n.cls, n.confidence, n.tag_confidence = r.cls, r.class_conf, r.tag_conf
    return sheet, clf, g


def test_agent_repairs_low_confidence_classes():
    sheet, clf, g = _noisy_graph()
    wrong_before = sum(a.cls != b.cls for a, b in zip(g.nodes, sheet.graph.nodes, strict=True))
    assert wrong_before > 0
    res = VerificationAgent(PolicyClient(), clf, sheet.image, AgentConfig()).run(g)
    assert sum(a.cls != b.cls for a, b in zip(g.nodes, sheet.graph.nodes, strict=True)) == 0
    assert {a["tool"] for a in res.audit} == {"reclassify", "commit_change"}
    assert g.review_queue == []


def test_agent_budget_exhaustion_sends_leftovers_to_human():
    sheet, clf, g = _noisy_graph()
    n_bad = len(flag_uncertain(g))
    res = VerificationAgent(PolicyClient(), clf, sheet.image, AgentConfig(max_tool_calls=1)).run(g)
    assert res.tool_calls == 1
    assert len(g.review_queue) >= n_bad - 1  # nothing silently guessed


def test_agent_rejects_invalid_commit_and_unknown_node():
    sheet, clf, g = _noisy_graph(error_rate=1.0)
    nid = g.nodes[0].id
    client = ScriptedClient(
        [
            tool_reply(
                ("commit_change", {"node_id": nid, "field": "class", "value": "flux_capacitor", "evidence": "x"}),
                ("zoom", {"node_id": "nope"}),
                ("flag_for_human", {"node_id": nid, "reason": "cannot tell"}),
            ),
            text_reply("done"),
        ]
    )
    res = VerificationAgent(client, clf, sheet.image, AgentConfig()).run(g)
    results = [a["result"] for a in res.audit]
    assert "error" in results[0] and "error" in results[1] and results[2].get("ok")
    assert g.node_map()[nid].cls != "flux_capacitor"
    assert any(q["item"] == nid for q in g.review_queue)


def test_agent_zoom_returns_image_message():
    sheet, clf, g = _noisy_graph(error_rate=1.0)
    nid = g.nodes[0].id
    client = ScriptedClient([tool_reply(("zoom", {"node_id": nid})), text_reply("done")])
    VerificationAgent(client, clf, sheet.image, AgentConfig()).run(g)
    second_call_msgs = client.calls[1]["messages"]
    assert second_call_msgs[-1]["role"] == "user"
    assert any(p.get("type") == "image_url" for p in second_call_msgs[-1]["content"])


def test_agent_does_nothing_when_all_confident():
    sheet = make_sheet(seed=1)
    client = ScriptedClient([])
    res = VerificationAgent(client, SimulatedClassifier(sheet.graph), sheet.image).run(sheet.graph.model_copy(deep=True))
    assert client.calls == [] and res.audit == []


# --------------------------------------------------------------------------- NL rule compiler
GOOD = {
    "id": "R-RELIEF-NEAR-TANK",
    "severity": "warning",
    "description": "Tank needs a valve within 1 hop",
    "applies_to": {"class": "tank"},
    "require": {"path": {"direction": "downstream", "edge_type": "process", "max_hops": 1, "reaches": {"class": "valve"}}},
}


def _refs():
    return {"clean": make_sheet(seed=0).graph}


def test_nl_compiler_valid_rule_with_preview():
    c = ScriptedClient([text_reply(json.dumps(GOOD))])
    out = compile_rule(c, "every tank needs a valve right after it", _refs())
    assert out.error is None and out.rule.id == "R-RELIEF-NEAR-TANK"
    assert out.preview["clean"]["violations"] == 0


def test_nl_compiler_repairs_invalid_rule():
    bad = json.loads(json.dumps(GOOD))
    bad["applies_to"] = {"class": "reactor"}
    c = ScriptedClient([text_reply(json.dumps(bad)), text_reply(json.dumps(GOOD))])
    out = compile_rule(c, "tanks need valves", _refs())
    assert out.rule is not None
    assert "unknown class" in c.calls[1]["messages"][-1]["content"]


def test_nl_compiler_gives_up_cleanly():
    bad = {"id": "x", "description": "d", "require": {"unique": "tag", "connected": True}}
    c = ScriptedClient([text_reply(json.dumps(bad)), text_reply(json.dumps(bad))])
    out = compile_rule(c, "nonsense", _refs())
    assert out.rule is None and "exactly one" in out.error


def test_nl_compiler_model_declines():
    c = ScriptedClient([text_reply('{"error": "cannot express pressure ratings"}')])
    out = compile_rule(c, "design pressure must exceed 10 bar", _refs())
    assert out.rule is None and "pressure" in out.error


def test_rule_dsl_rejects_net_rule_with_wrong_requirement():
    with pytest.raises(ValidationError):
        parse_rules([{"id": "x", "description": "d", "applies_to": {"kind": "net"}, "require": {"connected": True}}])


# --------------------------------------------------------------------------- pipeline + metrics
def test_simulated_pipeline_flags_planted_errors_end_to_end():
    sheet = make_sheet(seed=7, errors=["missing_check_valve", "missing_relief"])
    res = build_simulated_pipeline(sheet.graph).run(sheet.image)
    mapping = match_nodes(sheet.graph, res.graph)
    pr = planted_recall(sheet.planted, res.violations, mapping)
    assert pr["found"] == pr["planted"] == 2
    assert score_graph(sheet.graph, res.graph)["connectivity"]["f1"] == 1.0
    assert res.overlay.size == sheet.image.size


def test_pipeline_drops_false_positive_boxes_and_tolerates_jitter():
    sheet = make_sheet(seed=3)
    pipe = Pipeline(
        OracleSegmenter(sheet.graph, jitter=2, false_positives=4, seed=1),
        SimulatedClassifier(sheet.graph),
        cfg=PipelineConfig(run_agent=False),
    )
    res = pipe.run(sheet.image)
    assert len(res.graph.nodes) == len(sheet.graph.nodes)  # false positives were classified not_a_symbol
    assert score_graph(sheet.graph, res.graph)["connectivity"]["f1"] == 1.0


def test_metrics_penalize_errors():
    sheet = make_sheet(seed=2)
    pred = sheet.graph.model_copy(deep=True)
    pred.nodes[0].cls = "pump" if pred.nodes[0].cls != "pump" else "valve"  # wrong class
    del pred.nodes[-1]  # missed symbol
    sc = score_graph(sheet.graph, pred)
    assert sc["symbol_localization"]["recall"] < 1.0
    assert sc["symbol_class"]["f1"] < sc["symbol_localization"]["f1"] <= 1.0
    assert score_graph(sheet.graph, sheet.graph)["symbol_class"]["f1"] == 1.0
    assert score_violations(sheet.graph, sheet.graph, default_rules())["f1"] == 1.0


# --------------------------------------------------------------------------- API
def _png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_analyze_roundtrip():
    sheet = make_sheet(seed=4, errors=["missing_tag"])
    app = create_app(pipeline_factory=lambda rules: build_simulated_pipeline(sheet.graph, rules=rules))
    c = TestClient(app)
    r = c.post("/analyze", files={"file": ("s.png", _png(sheet.image), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert any(v["rule_id"] == "R2-TAG-PRESENT" for v in body["violations"])
    assert body["bom_csv"].startswith("id,class,tag")
    Image.open(io.BytesIO(base64.b64decode(body["overlay_png_b64"])))  # valid image


def test_api_rejects_bad_input():
    app = create_app(pipeline_factory=lambda rules: None)
    c = TestClient(app)
    assert c.post("/analyze", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400
    sheet = make_sheet(seed=0)
    r = c.post("/analyze", files={"file": ("s.png", _png(sheet.image), "image/png")}, data={"rules_yaml": "- id: x"})
    assert r.status_code == 422
    assert c.get("/synthetic", params={"errors": "bogus"}).status_code == 422


def test_api_custom_rules_and_compile_endpoint():
    sheet = make_sheet(seed=0)
    app = create_app(
        pipeline_factory=lambda rules: build_simulated_pipeline(sheet.graph, rules=rules),
        llm_factory=lambda: ScriptedClient([text_reply(json.dumps(GOOD))]),
    )
    c = TestClient(app)
    custom = yaml.safe_dump([GOOD])
    r = c.post("/analyze", files={"file": ("s.png", _png(sheet.image), "image/png")}, data={"rules_yaml": custom})
    assert r.status_code == 200 and r.json()["violations"] == []
    r2 = c.post("/rules/compile", json={"text": "tanks need a valve"})
    assert r2.status_code == 200 and r2.json()["rule"]["id"] == "R-RELIEF-NEAR-TANK" and "clean sheet" in r2.json()["preview"]


def test_pipeline_warns_when_everything_is_rejected():
    sheet = make_sheet(seed=3)

    class RejectAll:
        def classify(self, image, bbox, reasoning_effort="low"):
            from pid.classify import ClassResult

            return ClassResult("not_a_symbol", 0.9, None, 0.0)

    res = Pipeline(OracleSegmenter(sheet.graph), RejectAll(), cfg=PipelineConfig(run_agent=False)).run(sheet.image)
    assert res.graph.nodes == [] and any("rejected" in w for w in res.warnings)


def test_pipeline_warns_on_empty_segmentation():
    sheet = make_sheet(seed=3)
    res = Pipeline(OracleSegmenter(sheet.graph, drop=1.0), SimulatedClassifier(sheet.graph), cfg=PipelineConfig(run_agent=False)).run(sheet.image)
    assert any("no candidate symbols" in w for w in res.warnings)


def test_api_surfaces_upstream_failure_as_502_and_enforces_auth():
    from pid.models.llm import LLMError

    def boom(rules):
        raise LLMError("HTTP 401: bad key sk-secret-123")

    sheet = make_sheet(seed=0)
    c = TestClient(create_app(pipeline_factory=boom, api_key="s3cret"))
    files = {"file": ("s.png", _png(sheet.image), "image/png")}
    assert c.post("/analyze", files=files).status_code == 401
    assert c.get("/health").status_code == 200
    r = c.post("/analyze", files=files, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 502 and "sk-secret" not in r.text  # upstream details are not leaked


def test_api_rejects_oversized_image_dimensions(monkeypatch):
    import pid.app.api as api

    monkeypatch.setattr(api, "MAX_PIXELS", 1000)
    sheet = make_sheet(seed=0)
    r = TestClient(create_app(pipeline_factory=lambda rules: None)).post("/analyze", files={"file": ("s.png", _png(sheet.image), "image/png")})
    assert r.status_code == 413


def test_pipeline_caps_candidates_to_protect_credits():
    sheet = make_sheet(seed=3)
    calls = {"n": 0}

    class Counting(SimulatedClassifier):
        def classify(self, image, bbox, reasoning_effort="low"):
            calls["n"] += 1
            return super().classify(image, bbox, reasoning_effort)

    pipe = Pipeline(
        OracleSegmenter(sheet.graph, false_positives=50, seed=2),
        Counting(sheet.graph),
        cfg=PipelineConfig(run_agent=False, max_candidates=20),
    )
    res = pipe.run(sheet.image)
    assert calls["n"] == 20 and any("max_candidates" in w for w in res.warnings)
