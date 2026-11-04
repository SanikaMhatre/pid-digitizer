"""Streamlit interface.   streamlit run src/pid/app/ui.py

Two modes
  * Live models      - SAM 3.1 finds symbols (or the Muse model does); the Muse model reads and verifies (LLM_* env)
  * Offline simulated - model stages are replaced by ground-truth lookups on a GENERATED sheet.
                        For developing the interface and rehearsing the demo without credits.
                        Clearly labelled; its numbers measure the classical stages only.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

try:  # allow `streamlit run` from a source checkout without installing the package
    import pid  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from PIL import Image

from pid.config import build_live_pipeline, build_simulated_pipeline
from pid.eval.metrics import match_nodes, planted_recall, score_graph
from pid.models.llm import ChatClient, LLMError
from pid.pipeline import PipelineConfig
from pid.rules.engine import Rule, default_rules
from pid.rules.nl_compiler import compile_rule, rule_to_yaml
from pid.synth.generator import ERROR_KINDS, make_sheet

st.set_page_config(page_title="P&ID Digitizer", layout="wide")

SEV_ICON = {"error": "🔴", "warning": "🟠", "info": "⚪"}


def _rules() -> list[Rule]:
    if "rules" not in st.session_state:
        st.session_state.rules = default_rules()
    return st.session_state.rules


def _stretch(fn, *args, **kw):
    """Full-width rendering across Streamlit versions (the keyword changed in recent releases)."""
    try:
        return fn(*args, width="stretch", **kw)
    except TypeError:
        return fn(*args, use_container_width=True, **kw)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- sidebar
st.title("P&ID Digitizer")
st.caption("Drawing in, verified graph and design-rule violations out. Models propose; a deterministic rule engine decides.")

with st.sidebar:
    st.header("Models")
    mode = st.radio(
        "Mode",
        ["Live models", "Offline simulated (UI testing only)"],
        help="Offline mode replaces the model stages with ground-truth lookups on a generated sheet.",
    )
    live = mode == "Live models"
    if live:
        use_agent = st.checkbox("Verification agent", value=True)
        localizer = st.selectbox("Localizer", ["sam", "muse", "both"], help="sam = SAM 3.1 on the Meta Model API. muse = the Muse model finds the boxes itself. both = union of the two.")
        key_ok = bool(os.getenv("LLM_API_KEY"))
       
    else:
        st.warning("Simulated models: results here are NOT model accuracy.")
        use_agent, localizer = False, None

    st.header("Rules")
    st.caption(f"{len(_rules())} active")
    if st.button("Reset to starter pack"):
        st.session_state.rules = default_rules()
        st.rerun()
    st.markdown("---\nBuilt with Muse Spark Open")

# ----------------------------------------------------------------------------- input
left, right = st.columns([1, 2])
with left:
    st.subheader("1. Choose a sheet")
    source = st.radio("Source", ["Generate synthetic sheet", "Upload image"], horizontal=True, label_visibility="collapsed")
    gt_sheet = None
    image = None
    if source == "Generate synthetic sheet":
        seed = st.number_input("Seed", 0, 100000, 7)
        n_rows = st.slider("Process lines", 1, 3, 2)
        errs = st.multiselect("Plant design errors", ERROR_KINDS, default=["missing_check_valve"])
        try:
            gt_sheet = make_sheet(seed=int(seed), n_rows=n_rows, errors=errs)
            image = gt_sheet.image
        except ValueError as e:
            st.error(str(e))
    else:
        up = st.file_uploader("P&ID image (PNG or JPEG)", type=["png", "jpg", "jpeg"])
        if up is not None:
            try:
                image = Image.open(up).convert("RGB")
                if image.size[0] * image.size[1] > 40_000_000:
                    st.error("Image is too large (limit 40 megapixels).")
                    image = None
            except Exception:
                st.error("Could not read that file as an image.")
        if not live and image is not None:
            st.error("Offline mode needs a generated sheet (it looks the answers up).")
            image = None

    run = st.button("Run analysis", type="primary", disabled=image is None)

    st.subheader("Natural-language rule")
    rule_text = st.text_area("Describe a rule", placeholder="Every pump must have an isolation valve on both sides", height=80, disabled=not live)
    if st.button("Compile rule", disabled=not live or not rule_text.strip()):
        try:
            with st.spinner("Compiling..."):
                refs = {"clean sheet": make_sheet(seed=0).graph, "sheet with planted errors": make_sheet(seed=1, errors=["missing_check_valve", "missing_relief"]).graph}
                out = compile_rule(ChatClient.from_env(), rule_text, refs)
            st.session_state.compiled = out
        except LLMError as e:
            st.error(f"Model service error: {str(e)[:200]}")
    out = st.session_state.get("compiled")
    if out is not None:
        if out.error:
            st.error(f"Could not compile: {out.error}")
        else:
            st.code(rule_to_yaml(out.rule), language="yaml")
            st.json(out.preview)
            if st.button("Approve and add to active rules"):
                st.session_state.rules = [r for r in _rules() if r.id != out.rule.id] + [out.rule]
                st.session_state.pop("compiled")
                st.rerun()

# ----------------------------------------------------------------------------- run
if run and image is not None:
    try:
        cfg = PipelineConfig(run_agent=use_agent)
        with st.spinner("Analyzing sheet..."):
            if live:
                pipe = build_live_pipeline(rules=_rules(), cfg=cfg, localizer=localizer)
            else:
                pipe = build_simulated_pipeline(gt_sheet.graph, rules=_rules(), cfg=cfg)
            st.session_state.result = pipe.run(image)
            st.session_state.gt = gt_sheet
    except LLMError as e:
        st.error(f"The model service failed: {str(e)[:300]}")
    except Exception as e:  # keep the UI alive; details go to the server log
        st.error(f"Analysis failed: {type(e).__name__}")
        raise

res = st.session_state.get("result")
with right:
    st.subheader("2. Result")
    if res is None:
        st.info("Run an analysis to see the overlay, graph and violations.")
    else:
        for w in res.warnings:
            st.warning(w)
        g = res.graph
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbols", len(g.nodes))
        c2.metric("Nets", len(g.nets))
        c3.metric("Violations", len(res.violations))
        c4.metric("Cost", f"${res.cost.get('cost_usd', 0):.4f}")

        tabs = st.tabs(["Overlay", "Violations", "Graph / BOM", "Agent log", "Review queue", "Performance"])
        with tabs[0]:
            _stretch(st.image, res.overlay)
            st.download_button("Download overlay (PNG)", _png(res.overlay), "overlay.png", "image/png")
        with tabs[1]:
            if not res.violations:
                st.success("No rule violations.")
            for v in res.violations:
                st.markdown(f"{SEV_ICON.get(v.severity, '')} **{v.rule_id}** — {v.message}")
            gt = st.session_state.get("gt")
            if gt is not None and gt.planted:
                pr = planted_recall(gt.planted, res.violations, match_nodes(gt.graph, res.graph))
                st.info(f"Planted errors found: {pr['found']}/{pr['planted']}")
        with tabs[2]:
            bom = pd.DataFrame(g.bom())
            _stretch(st.dataframe, bom, hide_index=True)
            st.download_button("Download BOM (CSV)", bom.to_csv(index=False), "bom.csv", "text/csv")
            st.download_button("Download graph (JSON)", g.to_json(indent=1), "graph.json", "application/json")
            gt = st.session_state.get("gt")
            if gt is not None:
                st.caption("Score against the generator's ground truth")
                st.json(score_graph(gt.graph, g))
        with tabs[3]:
            if not res.audit:
                st.write("The verification agent made no tool calls (nothing was uncertain, or it is disabled).")
            for a in res.audit:
                st.code(f"{a['step']:>2}. {a['tool']}({json.dumps(a['args'])}) -> {json.dumps(a['result'])[:300]}")
        with tabs[4]:
            if not g.review_queue:
                st.success("Nothing needs human review.")
            for q in g.review_queue:
                n = g.node_map().get(q["item"])
                st.markdown(f"- **{n.label() if n else q['item']}**: {q['reason']}")
        with tabs[5]:
            st.json({"timings_s": res.timings, "cost": res.cost})
