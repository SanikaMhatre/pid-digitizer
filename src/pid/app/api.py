"""FastAPI backend.

  GET  /health           liveness
  POST /analyze          multipart: file=<image>, rules_yaml=<optional text>
  POST /rules/compile    json: {"text": "..."}  -> compiled rule + preview (the rule is not applied until you send it back)
  GET  /synthetic        ?seed=1&errors=missing_check_valve  -> a generated sheet (png base64) and its planted errors

Run:  uvicorn pid.app.api:app --port 8000

Hardening
  * upload size and pixel-count limits (decompression-bomb safe)
  * optional shared-secret auth: set PID_API_KEY and send  Authorization: Bearer <key>  (health stays open)
  * blocking model work runs in a worker thread pool (endpoints are plain `def`, not `async def`)
  * request id on every response, structured log line per request, no secrets or image data in logs
  * errors from upstream model services map to 502, bad input to 4xx; internals are not leaked
"""
from __future__ import annotations

import base64
import csv
import hmac
import io
import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from ..config import build_live_pipeline
from ..models.llm import ChatClient, LLMError
from ..pipeline import Pipeline
from ..rules.engine import Rule, default_rules, parse_rules
from ..rules.nl_compiler import compile_rule, rule_to_yaml
from ..synth.generator import ERROR_KINDS, make_sheet

log = logging.getLogger("pid.api")

MAX_UPLOAD_BYTES = int(os.getenv("PID_MAX_UPLOAD_MB", "15")) * 1024 * 1024
MAX_PIXELS = int(os.getenv("PID_MAX_PIXELS", str(40_000_000)))
MAX_RULES_BYTES = 64 * 1024


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _bom_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


def load_image_safely(data: bytes) -> Image.Image:
    """Decode an upload with size limits.  Raises HTTPException with a clear message."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (limit {MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size  # header only; no pixel decode yet
            fmt = probe.format
        if fmt not in ("PNG", "JPEG", "TIFF", "BMP", "WEBP"):
            raise HTTPException(415, f"unsupported image format {fmt}; use PNG or JPEG")
        if w * h > MAX_PIXELS:
            raise HTTPException(413, f"image too large ({w}x{h}); limit is {MAX_PIXELS:,} pixels")
        return Image.open(io.BytesIO(data)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "could not read the upload as an image (PNG or JPEG expected)")


class CompileRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def create_app(
    pipeline_factory: Optional[Callable[[list[Rule]], Pipeline]] = None,
    llm_factory: Optional[Callable[[], ChatClient]] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="P&ID Digitizer", version="0.1.0", description="Built with Muse Spark Open")
    pipeline_factory = pipeline_factory or (lambda rules: build_live_pipeline(rules=rules))
    llm_factory = llm_factory or ChatClient.from_env
    secret = api_key if api_key is not None else os.getenv("PID_API_KEY", "")

    def require_auth(authorization: Optional[str] = Header(default=None)):
        if not secret:
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), secret.encode()):
            raise HTTPException(401, "missing or invalid API key")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.time()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500, headers={"x-request-id": rid})
        response.headers["x-request-id"] = rid
        log.info("rid=%s %s %s -> %s in %.2fs", rid, request.method, request.url.path, response.status_code, time.time() - t0)
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "version": app.version}

    @app.post("/analyze", dependencies=[Depends(require_auth)])
    def analyze(file: UploadFile = File(...), rules_yaml: Optional[str] = Form(None)):
        image = load_image_safely(file.file.read(MAX_UPLOAD_BYTES + 1))
        if rules_yaml and len(rules_yaml.encode()) > MAX_RULES_BYTES:
            raise HTTPException(413, "rules too large")
        try:
            rules = parse_rules(yaml.safe_load(rules_yaml)) if rules_yaml and rules_yaml.strip() else default_rules()
        except Exception as e:
            raise HTTPException(422, f"invalid rules: {str(e)[:400]}")
        try:
            result = pipeline_factory(rules).run(image, sheet_id=(file.filename or "sheet")[:80])
        except HTTPException:
            raise
        except LLMError as e:
            log.warning("upstream model error: %s", str(e)[:300])
            raise HTTPException(502, "the model service failed or is unavailable; try again shortly")
        except Exception:
            log.exception("analysis failed")
            raise HTTPException(500, "analysis failed")
        g = result.graph
        return {
            "graph": g.to_dict(),
            "violations": [v.to_dict() for v in result.violations],
            "bom": g.bom(),
            "bom_csv": _bom_csv(g.bom()),
            "overlay_png_b64": _png_b64(result.overlay),
            "timings": result.timings,
            "cost": result.cost,
            "agent_audit": result.audit,
            "review_queue": g.review_queue,
            "warnings": result.warnings,
        }

    @app.post("/rules/compile", dependencies=[Depends(require_auth)])
    def compile_(req: CompileRequest):
        refs = {
            "clean sheet": make_sheet(seed=0).graph,
            "sheet with planted errors": make_sheet(seed=1, errors=["missing_check_valve", "missing_relief"]).graph,
        }
        try:
            out = compile_rule(llm_factory(), req.text, refs)
        except LLMError as e:
            log.warning("upstream model error: %s", str(e)[:300])
            raise HTTPException(502, "the model service failed or is unavailable; try again shortly")
        return {
            "rule": out.rule.model_dump(by_alias=True, exclude_none=True) if out.rule else None,
            "yaml": rule_to_yaml(out.rule) if out.rule else None,
            "error": out.error,
            "preview": out.preview,
        }

    @app.get("/synthetic", dependencies=[Depends(require_auth)])
    def synthetic(seed: int = 0, errors: str = "", rows: int = 2):
        kinds = [e for e in errors.split(",") if e]
        bad = [e for e in kinds if e not in ERROR_KINDS]
        if bad:
            raise HTTPException(422, f"unknown error kinds {bad}; choose from {ERROR_KINDS}")
        if not 0 <= seed <= 1_000_000:
            raise HTTPException(422, "seed out of range")
        try:
            sheet = make_sheet(seed=seed, n_rows=max(1, min(rows, 3)), errors=kinds)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"png_b64": _png_b64(sheet.image), "planted": sheet.planted, "size": sheet.image.size}

    return app


app = create_app()
