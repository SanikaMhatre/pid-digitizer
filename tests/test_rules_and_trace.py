import random

import pytest

from pid.rules.engine import default_rules, evaluate
from pid.schema import Node
from pid.synth.generator import ERROR_KINDS, make_sheet
from pid.trace import trace_nets


def _pairs(graph):
    return graph.connected_pairs()


def _typed_pairs(graph):
    return {(frozenset((e["a"], e["b"])), e["type"]) for e in graph.edges()}


@pytest.mark.parametrize("seed", range(6))
def test_clean_sheets_have_no_violations(seed):
    sheet = make_sheet(seed=seed, n_rows=2 + seed % 2)
    assert evaluate(sheet.graph, default_rules()) == []


@pytest.mark.parametrize("kind", ERROR_KINDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_planted_error_is_flagged_on_ground_truth(kind, seed):
    sheet = make_sheet(seed=seed, n_rows=2, errors=[kind])
    violations = evaluate(sheet.graph, default_rules())
    plant = sheet.planted[0]
    hits = [v for v in violations if v.rule_id == plant["rule_id"] and plant["node_id"] in v.node_ids]
    assert hits, f"{kind}: expected {plant}, got {[(v.rule_id, v.node_ids) for v in violations]}"


@pytest.mark.parametrize("seed", range(5))
def test_tracing_recovers_ground_truth_connectivity(seed):
    sheet = make_sheet(seed=seed, n_rows=2 + seed % 2)
    res = trace_nets(sheet.image, sheet.graph.nodes)
    pred = sheet.graph.model_copy(update={"nets": res.nets})
    assert _typed_pairs(pred) == _typed_pairs(sheet.graph)


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_tracing_with_planted_errors(kind):
    sheet = make_sheet(seed=3, n_rows=2, errors=[kind])
    res = trace_nets(sheet.image, sheet.graph.nodes)
    pred = sheet.graph.model_copy(update={"nets": res.nets})
    assert _typed_pairs(pred) == _typed_pairs(sheet.graph)
    gt_v = {(v.rule_id, tuple(v.node_ids)) for v in evaluate(sheet.graph, default_rules())}
    pr_v = {(v.rule_id, tuple(v.node_ids)) for v in evaluate(pred, default_rules())}
    assert gt_v == pr_v


def test_tracing_tolerates_box_jitter():
    sheet = make_sheet(seed=1, n_rows=2)
    rng = random.Random(0)
    jittered = []
    for n in sheet.graph.nodes:
        dx, dy = rng.randint(-3, 3), rng.randint(-3, 3)
        x0, y0, x1, y1 = n.bbox
        jittered.append(Node(id=n.id, cls=n.cls, tag=n.tag, bbox=[x0 + dx, y0 + dy, x1 + dx, y1 + dy]))
    res = trace_nets(sheet.image, jittered)
    pred = sheet.graph.model_copy(update={"nets": res.nets})
    assert _typed_pairs(pred) == _typed_pairs(sheet.graph)


def test_tracing_on_scan_noise():
    sheet = make_sheet(seed=2, n_rows=2, scan_noise=True)
    res = trace_nets(sheet.image, sheet.graph.nodes)
    pred = sheet.graph.model_copy(update={"nets": res.nets})
    assert _typed_pairs(pred) == _typed_pairs(sheet.graph)
