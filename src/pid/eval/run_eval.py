"""Evaluation CLI.

  python -m pid.eval.run_eval --mode simulated --n 14          # classical stages only, no model calls
  python -m pid.eval.run_eval --mode live --n 6                # real models (spends credits)
  python -m pid.eval.run_eval --mode live --localizer muse     # ablation: Muse model finds the boxes instead of SAM
  python -m pid.eval.run_eval --mode live --no-agent ...       # ablation: without the verification agent

Sheets are synthetic, with planted errors cycling through every error kind plus clean sheets.
Seeds below TEST_SEED_START are for development; seeds from TEST_SEED_START up are held-out test.
In `simulated` mode the model stages are ground-truth lookups, so the numbers describe tracing and
rules only and must NOT be reported as model accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import build_live_pipeline, build_simulated_pipeline
from ..pipeline import PipelineConfig
from ..rules.engine import default_rules
from ..synth.generator import ERROR_KINDS, make_sheet
from .metrics import match_nodes, planted_recall, score_graph, score_violations

TEST_SEED_START = 1000


def plan_sheets(n: int, split: str):
    start = TEST_SEED_START if split == "test" else 0
    sheets = []
    for i in range(n):
        # every 4th sheet is clean; the rest cycle through each planted error kind
        errors = [] if i % 4 == 3 else [ERROR_KINDS[(i - i // 4) % len(ERROR_KINDS)]]
        sheets.append((start + i, errors))
    return sheets


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulated", "live"], default="simulated")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--scan-noise", action="store_true")
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--localizer", choices=["sam", "muse"], default=None, help="default: PID_LOCALIZER env or sam")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args(argv)

    rules = default_rules()
    cfg = PipelineConfig(run_agent=not args.no_agent)
    rows = []
    live_pipe = build_live_pipeline(rules=rules, cfg=cfg, localizer=args.localizer) if args.mode == "live" else None

    for seed, errors in plan_sheets(args.n, args.split):
        sheet = make_sheet(seed=seed, n_rows=args.rows, errors=errors, scan_noise=args.scan_noise)
        pipe = live_pipe or build_simulated_pipeline(sheet.graph, rules=rules, cfg=cfg)
        t0 = time.time()
        res = pipe.run(sheet.image, sheet_id=sheet.graph.sheet_id)
        sc = score_graph(sheet.graph, res.graph)
        mapping = match_nodes(sheet.graph, res.graph)
        row = {
            "seed": seed,
            "errors": errors,
            "scores": sc,
            "violations": score_violations(sheet.graph, res.graph, rules, res.violations),
            "planted": planted_recall(sheet.planted, res.violations, mapping),
            "seconds": round(time.time() - t0, 2),
            "cost_usd": res.cost.get("cost_usd", 0.0),
            "review_items": len(res.graph.review_queue),
            "agent_tool_calls": len(res.audit),
        }
        rows.append(row)
        print(f"seed {seed:>5} {','.join(errors) or 'clean':<26} sym-F1 {sc['symbol_class']['f1']:.3f}  "
              f"conn-F1 {sc['connectivity']['f1']:.3f}  viol-F1 {row['violations']['f1']:.3f}  "
              f"planted {row['planted']['found']}/{row['planted']['planted']}  ${row['cost_usd']:.4f}")

    planted = sum(r["planted"]["planted"] for r in rows)
    found = sum(r["planted"]["found"] for r in rows)
    summary = {
        "mode": args.mode,
        "split": args.split,
        "sheets": len(rows),
        "symbol_class_f1": mean(r["scores"]["symbol_class"]["f1"] for r in rows),
        "tag_read_accuracy": mean(r["scores"]["tag_read_accuracy"] for r in rows),
        "connectivity_f1": mean(r["scores"]["connectivity"]["f1"] for r in rows),
        "violation_f1_vs_ground_truth": mean(r["violations"]["f1"] for r in rows),
        "planted_errors_found": f"{found}/{planted}",
        "mean_seconds_per_sheet": mean(r["seconds"] for r in rows),
        "mean_cost_usd_per_sheet": mean(r["cost_usd"] for r in rows),
        "mean_review_items": mean(r["review_items"] for r in rows),
        "note": "simulated mode = classical stages only (model stages are ground-truth lookups)" if args.mode == "simulated" else "live models",
    }
    print("\n" + json.dumps(summary, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{args.split}{'_' + args.localizer if args.localizer else ''}{'_noagent' if args.no_agent else ''}{'_noise' if args.scan_noise else ''}"
    (out / f"{tag}.json").write_text(json.dumps({"summary": summary, "sheets": rows}, indent=2))
    print(f"wrote {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
