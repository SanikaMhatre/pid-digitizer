"""Scoring: predicted graph vs ground-truth graph.

Nodes are matched by box overlap (IoU >= 0.5, greedy by IoU).  Connectivity is scored on the set of
(symbol pair, net type) that share a net, after mapping predicted symbols onto ground-truth ones.
Rule-violation scoring compares what the engine flags on the predicted graph with what it flags on
the ground-truth graph, so a vision error is reported as such rather than hidden or double counted.
"""
from __future__ import annotations

from ..rules.engine import Rule, Violation, evaluate
from ..schema import Graph, iou


def prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    r = tp / n_gt if n_gt else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "pred": n_pred, "gt": n_gt}


def match_nodes(gt: Graph, pred: Graph, thr: float = 0.5) -> dict[str, str]:
    cands = []
    for p in pred.nodes:
        for g in gt.nodes:
            v = iou(p.bbox, g.bbox)
            if v >= thr:
                cands.append((v, p.id, g.id))
    cands.sort(reverse=True)
    used_p, used_g, mapping = set(), set(), {}
    for _, pid, gid in cands:
        if pid in used_p or gid in used_g:
            continue
        mapping[pid] = gid
        used_p.add(pid)
        used_g.add(gid)
    return mapping


def _typed_pairs(g: Graph) -> set:
    return {(frozenset((e["a"], e["b"])), e["type"]) for e in g.edges()}


def score_graph(gt: Graph, pred: Graph) -> dict:
    mapping = match_nodes(gt, pred)
    gmap, pmap = gt.node_map(), pred.node_map()

    loc = prf(len(mapping), len(pred.nodes), len(gt.nodes))
    cls_tp = sum(1 for pid, gid in mapping.items() if pmap[pid].cls == gmap[gid].cls)
    cls = prf(cls_tp, len(pred.nodes), len(gt.nodes))

    both = [(pmap[p], gmap[g]) for p, g in mapping.items()]
    tag_correct = sum(1 for p, g in both if (p.tag or None) == (g.tag or None))
    tag_acc = tag_correct / len(both) if both else 0.0
    tagged = [(p, g) for p, g in both if g.tag]
    tag_read = sum(1 for p, g in tagged if p.tag == g.tag) / len(tagged) if tagged else 1.0

    gt_pairs = _typed_pairs(gt)
    pred_pairs = set()
    for pair, typ in _typed_pairs(pred):
        a, b = tuple(pair)
        if a in mapping and b in mapping:
            pred_pairs.add((frozenset((mapping[a], mapping[b])), typ))
        else:
            pred_pairs.add((pair, typ + "_unmapped"))
    conn = prf(len(gt_pairs & pred_pairs), len(_typed_pairs(pred)), len(gt_pairs))

    return {
        "symbol_localization": loc,
        "symbol_class": cls,
        "tag_accuracy": round(tag_acc, 4),
        "tag_read_accuracy": round(tag_read, 4),
        "connectivity": conn,
    }


def _vkey(v: Violation, mapping: dict[str, str] | None = None):
    ids = v.node_ids
    if mapping is not None:
        ids = [mapping.get(i, f"?{i}") for i in ids]
    return (v.rule_id, tuple(sorted(ids)))


def score_violations(gt: Graph, pred: Graph, rules: list[Rule], pred_violations: list[Violation] | None = None) -> dict:
    mapping = match_nodes(gt, pred)
    gt_v = {_vkey(v) for v in evaluate(gt, rules)}
    pv = pred_violations if pred_violations is not None else evaluate(pred, rules)
    pred_v = {_vkey(v, mapping) for v in pv}
    return prf(len(gt_v & pred_v), len(pred_v), len(gt_v))


def planted_recall(plants: list[dict], pred_violations: list[Violation], mapping: dict[str, str]) -> dict:
    """Did the pipeline flag each deliberately planted error?"""
    inv = {g: p for p, g in mapping.items()}
    found = 0
    detail = []
    for p in plants:
        pid = inv.get(p["node_id"])
        hit = any(v.rule_id == p["rule_id"] and pid in v.node_ids for v in pred_violations) if pid else False
        found += hit
        detail.append({"kind": p["kind"], "found": bool(hit)})
    return {"found": found, "planted": len(plants), "detail": detail}
