# P&ID Digitizer

**Scanned engineering drawing in → verified graph, bill of materials and design-rule violations out.**

Built for the Meta Global AI Developer Hackathon (Push the Frontier: manufacturing and energy).
**Built with Muse Spark Open.** Uses SAM 3.1 for localization.

Process plants hold decades of drawings as images. Their structure (which pump feeds which vessel, which valves isolate what)
is locked in pixels, which blocks plant modifications, safety reviews and digital twins. This project reads a P&ID and produces:

1. a machine-readable **graph** (symbols, tags, and the pipes / signal lines connecting them)
2. a **bill of materials** (CSV)
3. an **annotated overlay** of the original drawing
4. **design-rule violations**, each pointing at the exact place on the drawing; new rules can be written in plain English

> **Design principle: models propose, deterministic code decides.** SAM 3.1 and Muse Spark Open produce candidate symbols, tags
> and links with confidences. A verification agent re-examines the uncertain ones with tools. The rule engine that raises violations
> is ordinary code over the graph, so every finding is reproducible, testable and auditable. A language model can *author* a rule,
> but it can never decide whether a rule is violated.

---

## Status: what is verified and what is not

| Area | Status |
|---|---|
| Graph schema, rule engine (9 starter rules, YAML DSL), natural-language rule compiler (validation + repair loop) | **Tested** (77 automated tests) |
| Classical line tracing (symbols masked → nets, W/E/N/S attachment, dashed signal lines, text rejection) | **Tested** on generated sheets incl. box jitter and scan noise; **not tested on real scanned drawings** |
| Synthetic sheet generator with 7 plantable design errors, scoring harness, ablation flags | **Tested** |
| LLM client (retries, on-disk cache, cost ledger, strict-JSON and tool-call plumbing), classifier, verification agent loop, API, UI | **Tested against fake endpoints / scripted models**; UI checked in a real browser in offline mode |
| **Live Muse Spark Open calls** (vision accuracy, strict JSON, tool calling, `reasoning_effort`) | **Not yet run.** `python -m pid.smoke llm` checks each capability in one command |
| **SAM 3.1 on Modal** | **Written, not yet run.** The adapter for SAM's output format must be confirmed with `modal run infra/modal_sam.py::inspect` |
| **Accuracy on real P&IDs** | **Unknown.** No real-drawing results exist yet; do not quote numbers until you run them |

The simulated evaluation (`--mode simulated`) replaces the model stages with ground-truth lookups, so it scores only the classical
stages (tracing, rules, metrics). **Its perfect scores are not model accuracy** and must not be presented as such.

## Quickstart (no keys, no GPU)

```bash
pip install -e ".[dev]"
pytest -q                                           # 77 tests
python -m pid.eval.run_eval --mode simulated --n 14 # classical stages only
streamlit run src/pid/app/ui.py                     # choose "Offline simulated" in the sidebar
python demo/make_demo_assets.py                     # clean_sheet.png, planted_errors.png, noisy_scan.png, legend.png
```

## Going live (credits needed)

```bash
cp .env.example .env     # fill LLM_API_KEY; export the variables or use your shell's dotenv loader
```

1. **Muse Spark Open smoke test**: `python -m pid.smoke llm` → PASS/FAIL for basic call, `reasoning_effort`, vision classification,
   strict JSON schema, tool calling, image in follow-up; prints tokens and dollars used.
2. **Deploy SAM 3.1**: request access to the gated weights on Hugging Face, then
   ```bash
   pip install modal && modal setup
   modal secret create huggingface-secret HF_TOKEN=hf_xxx
   modal deploy infra/modal_sam.py
   modal run infra/modal_sam.py::inspect --image demo/clean_sheet.png     # confirm SAM's output keys / shapes
   python -m pid.smoke sam                                                # localization recall on generated sheets
   ```
   Set a spend limit in the Modal dashboard first. The container scales to zero when idle.
3. **Measure**: `python -m pid.eval.run_eval --mode live --n 6 --segmenter modal`
   - ablations: `--no-agent` (without verification), `--segmenter vlm` (VLM-box fallback), `--scan-noise`
   - held-out numbers: `--split test` (seeds ≥ 1000; do not tune on them)
4. **Use**: `streamlit run src/pid/app/ui.py` (Live models) or `uvicorn pid.app.api:app`.

If SAM underperforms on thin lines or unusual symbols, the localizer is pluggable: `--segmenter vlm` uses the vision model to
propose boxes; the classical tracer works with any box source.

## Architecture

```
sheet image
  │
  ├─ localize ── SAM 3.1 (Modal GPU), tiled, text prompts + self-exemplar refinement ─┐   (or VLM-box fallback)
  │                                                                                    ▼
  ├─ classify ── Muse Spark Open vision + strict JSON schema: class, tag, confidences (parallel, cached)
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

| Model | Role |
|---|---|
| Muse Spark Open (vision input) | symbol classification and tag reading from crops; legend parsing |
| Muse Spark Open (reasoning levels) | low for routine crops; medium/high for disputed items in the verification loop |
| Muse Spark Open (tool calling) | the verification agent |
| Muse Spark Open (structured output) | schema-constrained classification; natural-language → rule DSL |
| SAM 3.1 | promptable concept segmentation to find symbol candidates on large sheets |

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

Natural-language rules ("every pump needs an isolation valve on both sides") are compiled by Muse Spark Open into this DSL, **validated
against the schema** (unknown classes and malformed conditions are rejected and sent back for repair), previewed on reference sheets, and
applied only after you approve them.

## Evaluation protocol

- Sheets are generated by `pid.synth` (original symbols, so there are no data-rights issues) with planted errors; the expected violations
  are known by construction. Seeds < 1000 are development, ≥ 1000 are held-out test.
- Metrics: symbol localization/class F1 (IoU 0.5), tag read accuracy, connectivity F1 on (symbol pair, net type), violation F1 against
  the violations the engine raises on the ground-truth graph, planted-error recall, seconds and dollars per sheet, items sent to review.
- Public P&ID datasets (e.g. the PID2Graph benchmark) can be added for real-drawing evaluation; **check their licenses first** and adapt
  a loader to `pid.schema.Graph`. None are bundled.

### Results to fill in after live runs

| Run | Symbol F1 | Tag acc. | Connectivity F1 | Planted found | s/sheet | $/sheet |
|---|---|---|---|---|---|---|
| live, SAM + Muse Spark Open + agent | | | | | | |
| live, no agent | | | | | | |
| live, VLM-box localizer | | | | | | |

## Security and privacy

- **Prompt injection:** drawings can contain arbitrary text. Model outputs are schema-constrained, agent tools are a closed whitelist with
  validated arguments, and nothing a model returns is executed or used as a rule verdict. The worst a hostile tag can do is produce a wrong
  tag value, which lands in the review queue when confidence is low.
- **Confidential drawings:** with Muse Spark Open you can serve the model on infrastructure you control, so drawings need not go to a closed API.
  Responses are cached on disk (`PID_CACHE_DIR`), keyed by a hash of the request; delete `.cache/` to purge. Request bodies containing images
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
  classify.py          Muse Spark Open classifier (+ simulated stand-in for tests)
  segment/             SAM (Modal client) · VLM-box fallback · NMS/tiling · oracle (tests)
  agent.py             verification agent
  pipeline.py          orchestration, warnings, cost accounting
  models/llm.py        OpenAI-compatible client: retries, cache, ledger
  synth/generator.py   synthetic sheets with planted errors
  eval/                metrics.py · run_eval.py
  app/                 api.py (FastAPI) · ui.py (Streamlit)
  smoke.py             live smoke tests
infra/modal_sam.py     SAM 3.1 GPU function
tests/                 77 tests
docs/                  RUNBOOK.md · DEMO_SCRIPT.md
```

## Development note

This project was developed with AI coding assistance (Claude). Record your own contribution and decisions in the commit history and the
hackathon submission as the rules require.

*Built with Muse Spark Open.*
