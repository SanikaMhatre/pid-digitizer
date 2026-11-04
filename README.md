# P&ID Digitizer

**Scanned engineering drawing in → verified graph, bill of materials and design-rule violations out.**

Built for the Meta Global AI Developer Hackathon (Push the Frontier: manufacturing and energy).
**Built with Meta models:** SAM 3.1 finds the symbols, a Muse model reads them and verifies uncertain ones. Both are called through the Meta Model API.

Process plants hold decades of drawings as images. Their structure (which pump feeds which vessel, which valves isolate what)
is locked in pixels, which blocks plant modifications, safety reviews and digital twins. This project reads a P&ID and produces:

1. a machine-readable **graph** (symbols, tags, and the pipes / signal lines connecting them)
2. a **bill of materials** (CSV)
3. an **annotated overlay** of the original drawing
4. **design-rule violations**, each pointing at the exact place on the drawing; new rules can be written in plain English

> **Design principle: models propose, deterministic code decides.** SAM 3.1 and the Muse model produce candidate symbols, tags
> and links with confidences. A verification agent re-examines the uncertain ones with tools. The rule engine that raises violations
> is ordinary code over the graph, so every finding is reproducible, testable and auditable. A language model can *author* a rule,
> but it can never decide whether a rule is violated.

---

## Quickstart (no API keys needed)

```bash
pip install -e ".[dev]"
pytest -q                                           # 87 tests
python -m pid.eval.run_eval --mode simulated --n 14 # classical stages only
streamlit run src/pid/app/ui.py                     # choose "Offline simulated" in the sidebar
python demo/make_demo_assets.py                     # clean_sheet.png, planted_errors.png, noisy_scan.png, legend.png
```

## Going live (credits needed)

```bash
cp .env.example .env     # fill LLM_API_KEY; export the variables or use your shell's dotenv loader
```

1. **Muse model smoke test**: `python -m pid.smoke llm` → PASS/FAIL for basic call, `reasoning_effort`, vision classification,
   strict JSON schema, tool calling, image in follow-up; prints tokens and dollars used.
2. **Localizer check**: `python -m pid.smoke localize` → box recall (IoU ≥ 0.5) against generated ground truth, plus cost per sheet
   (`--localizer muse` to test the Muse model as the localizer instead). The recall is the ceiling for everything downstream.
3. **Measure**: `python -m pid.eval.run_eval --mode live --n 6`
   - ablations: `--no-agent` (without verification), `--localizer muse` (Muse model finds the boxes instead of SAM), `--scan-noise`
   - held-out numbers: `--split test` (seeds ≥ 1000; do not tune on them)
4. **Use**: `streamlit run src/pid/app/ui.py` (Live models) or `uvicorn pid.app.api:app`.

## Architecture

```
sheet image
  │
  ├─ localize ── SAM 3.1 (Meta Model API): tiled, one text prompt per symbol class, boxes NMS-merged
  │
  ├─ classify ── Muse model vision + strict JSON schema: class, tag, confidences (parallel, cached)
  │
  ├─ trace ───── classical CV: mask symbols → bridge dashes → connected components → nets with W/E/N/S attachments
  │
  ├─ graph ───── nodes (symbols) + nets (pipes / signal lines)
  │
  ├─ verify ──── agent with tools: zoom · reclassify (higher reasoning) · commit_change · flag_for_human
  │              fixed tool-call budget, schema-validated arguments, full audit log, unresolved items → human queue
  │
  ├─ rules ───── deterministic YAML rule engine (direction-aware path queries over the graph)
  │
  └─ outputs ─── graph JSON · BOM CSV · annotated overlay · violations · warnings · cost ledger
```

**Why nets, not edges.** A tee feeding two valves is one net with three attachments, not a clique of edges. Flow direction comes from the
side a pipe meets a symbol (west = inlet, east = outlet), which is what lets rules say "downstream of the pump".

### Meta models and their jobs

Both are served by the Meta Model API. The localizer is pluggable (`PID_LOCALIZER=sam|muse`), so the two approaches can be compared on the same sheets.

| Model | Role |
|---|---|
| SAM 3.1 | localization: finds symbol bounding boxes from text prompts, tile by tile |
| Muse model (vision input) | symbol classification and tag reading from crops; legend parsing |
| Muse model (reasoning levels) | low for routine crops; medium/high for disputed items in the verification loop |
| Muse model (tool calling) | the verification agent |
| Muse model (structured output) | schema-constrained classification; natural-language → rule DSL |
| Muse model (vision, tiled), optional | alternative localizer for comparison (`--localizer muse`) |

## Rules

Rules are YAML. Example:

```yaml
- id: R4-PUMP-CHECK-VALVE
  severity: warning
  description: Pump discharge needs a check valve within 2 hops downstream
  applies_to: {class: pump}
  require:
    path: {direction: downstream, edge_type: process, max_hops: 2, reaches: {class: check_valve}}
```

Requirements: `path`, `unique: tag`, `has_attr: tag`, `connected: true`, and `min_symbols: N` (for `kind: net`). The starter pack
(`src/pid/rules/starter_pack.yaml`) is **illustrative**: real plants apply their own engineering standards, and passing these checks does
not establish safety compliance.

Natural-language rules ("every pump needs an isolation valve on both sides") are compiled by the Muse model into this DSL, **validated
against the schema** (unknown classes and malformed conditions are rejected and sent back for repair), previewed on reference sheets, and
applied only after you approve them.

## Evaluation protocol

- Sheets are generated by `pid.synth` (original symbols, so there are no data-rights issues) with planted errors; the expected violations
  are known by construction. Seeds < 1000 are development, ≥ 1000 are held-out test.
- Metrics: symbol localization/class F1 (IoU 0.5), tag read accuracy, connectivity F1 on (symbol pair, net type), violation F1 against
  the violations the engine raises on the ground-truth graph, planted-error recall, seconds and dollars per sheet, items sent to review.
- `--mode simulated` replaces the model stages with ground-truth lookups, so it scores tracing, rules and metrics only; it is not a measure of model accuracy.
- Public P&ID datasets (e.g. the PID2Graph benchmark) can be added for real-drawing evaluation; **check their licenses first** and adapt
  a loader to `pid.schema.Graph`. None are bundled.

### Results

| Run | Symbol F1 | Tag acc. | Connectivity F1 | Planted found | s/sheet | $/sheet |
|---|---|---|---|---|---|---|
| live, SAM + Muse + agent | | | | | | |
| live, no agent | | | | | | |
| live, Muse localizer + agent | | | | | | |
| live, with scan noise | | | | | | |

## Security and privacy

- **Prompt injection:** drawings can contain arbitrary text. Model outputs are schema-constrained, agent tools are a closed whitelist with
  validated arguments, and nothing a model returns is executed or used as a rule verdict. The worst a hostile tag can do is produce a wrong
  tag value, which lands in the review queue when confidence is low.
- **Confidential drawings:** use the Standard model (not the `-contributor` one, which lets Meta train on inputs). Muse Spark Open's weights are
  downloadable, so the model stages can also run on infrastructure you control. Responses are cached on disk (`PID_CACHE_DIR`), keyed by a hash of the request; delete `.cache/` to purge. Request bodies containing images
  are not written to logs.
- **API:** upload size and pixel limits, optional bearer-token auth (`PID_API_KEY`, constant-time compare), request ids, no upstream error
  details returned to clients, non-root container.
- **Secrets:** read from the environment only; `.env` is git-ignored.

## Limitations (known, deliberate)

- Line **crossings** vs **junctions** cannot be distinguished by pixel connectivity; both merge. Drawings with crossings need jump symbols or
  a dedicated check (the agent can flag them; not implemented).
- Tag text that touches a pipe is treated as part of the pipe.
- Flow direction assumes west-to-east conventions; drawings with other layouts need the attachment-side convention generalized.
- One sheet at a time: off-page connectors and multi-sheet consistency are not implemented.
- Pipe sizes, line numbers and specifications are not read.
- The generator's drawing style is simple; real scans will be harder.

## Project layout

```
src/pid/
  schema.py            graph schema (nodes, nets, derived edges, BOM)
  rules/               engine.py · starter_pack.yaml · nl_compiler.py
  trace.py             classical net tracing
  classify.py          Muse model classifier (+ simulated stand-in for tests)
  segment/             meta_sam.py (SAM 3.1) · muse_localizer.py · NMS/tiling · oracle (tests)
  agent.py             verification agent
  pipeline.py          orchestration, warnings, cost accounting
  models/llm.py        OpenAI-compatible client: retries, cache, ledger
  synth/generator.py   synthetic sheets with planted errors
  eval/                metrics.py · run_eval.py
  app/                 api.py (FastAPI) · ui.py (Streamlit)
  smoke.py             live smoke tests
tests/                 87 tests
docs/                  RUNBOOK.md · DEMO_SCRIPT.md
```

## Development note

This project was developed with AI coding assistance (Claude). Record your own contribution and decisions in the commit history and the
hackathon submission as the rules require.

*Built with Muse Spark Open.*
