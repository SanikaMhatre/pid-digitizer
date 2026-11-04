# Runbook: from this repo to a submission

Deadline: **Wed 4 Nov, 9:00 PM ET** (Thu 5 Nov, about 7:30 AM IST). Late submissions are not considered.
Aim to submit well before the deadline; keep the last hours for the video and the submission form.

## Phase A: first hour
1. `pip install -e ".[dev]"` then `pytest -q` (expect 89 passed). Create the public GitHub repo, push, and keep commits meaningful.
2. Confirm credits arrived (Meta billing page; Baseten/Fireworks emails). Add a spend-limit alert on every provider.
3. `cp .env.example .env`, set `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` (Baseten: `meta-models/muse-spark-open-1026`), and set your provider's real prices in `LLM_PRICE_IN/OUT`.
4. **Gate A**: `python -m pid.smoke llm`. Every check should PASS.
   - strict JSON FAIL → set `strict_schema=False` in `VLMClassifier` (falls back to `json_object`).
   - tool calling FAIL → the agent cannot run on this endpoint; try another provider's endpoint or disable the agent (`--no-agent`).
   - vision FAIL → check the model id; the text-only model will not accept images.

## Phase B: localizer gate
5. **Gate B**: `python -m pid.smoke localize` (SAM 3.1), then `python -m pid.smoke localize --localizer muse`. Each prints box recall (IoU ≥ 0.5) and cost.
   Use the better one (set `PID_LOCALIZER`). SAM low → edit `DEFAULT_PROMPTS` in `segment/meta_sam.py` (concept phrases matter) or lower `min_score`.

## Phase C: measure (this is what the Technical Depth score rewards)
6. `python -m pid.eval.run_eval --mode live --n 6 --split dev` then the ablations (`--no-agent`, `--localizer muse`, `--scan-noise`).
7. Tune thresholds on **dev** only (`PipelineConfig.agent.low_conf`, `TraceCfg`). Then run once on `--split test` and report that.
8. Fill the results table in the README with the real numbers, including the bad ones. Cost per sheet comes from the ledger.

## Phase D: product and video
9. Run the UI on `demo/planted_errors.png` (upload) and on a generated sheet. Record fallback takes.
10. Follow `docs/DEMO_SCRIPT.md`. Keep the video to 3:00.
11. Final checks: repo public, README accurate, video link opens logged-out, contributors listed, "Built with Muse Spark Open" visible in UI and README.

## If something breaks
| Symptom | Likely cause | Fix |
|---|---|---|
| 401 from LLM | wrong key / base URL | re-check `.env`; `python -m pid.smoke llm` |
| All symbols rejected ("not a symbol") | model id lacks vision, or legend mismatch | smoke test; warnings panel in UI says so |
| Many false "dangling pipe" | text blobs traced as pipes on real scans | tune `TraceCfg.text_*`; check overlay |
| Nets merged across a symbol | boxes too small | increase `TraceCfg.mask_pad` |
| Credits draining | repeated runs without cache | cache is on by default (`.cache/llm`); lower `max_candidates`; use fewer sheets |
