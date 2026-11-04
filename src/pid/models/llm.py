"""OpenAI-compatible chat client with on-disk caching and a cost ledger.

Works with any OpenAI-style /chat/completions endpoint: Baseten, Fireworks, or the Meta
Model API (set LLM_BASE_URL accordingly).  Two features exist to protect a small credit budget:

  * every request is cached on disk by a hash of the payload, so re-running a pipeline during
    development costs nothing
  * every call is written to a ledger (tokens, latency, estimated cost) so cost per sheet is a
    measured number, not a guess

Environment variables
  LLM_BASE_URL   default https://inference.baseten.co/v1
  LLM_API_KEY    required for live calls
  LLM_MODEL      default meta-models/muse-spark-open-1026   (the hackathon model id on Baseten)
  LLM_PRICE_IN   USD per 1M input tokens  (placeholder 1.25: set your provider's real price)
  LLM_PRICE_OUT  USD per 1M output tokens (default 4.25)
  PID_CACHE_DIR  default .cache/llm
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
from PIL import Image

DEFAULT_BASE_URL = "https://inference.baseten.co/v1"
DEFAULT_MODEL = "meta-models/muse-spark-open-1026"


class LLMError(RuntimeError):
    pass


@dataclass
class CallRecord:
    purpose: str
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool


class Ledger:
    """Thread-safe record of model calls (classification runs in a thread pool)."""

    def __init__(self):
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()

    def add(self, rec: CallRecord):
        with self._lock:
            self.calls.append(rec)

    def totals(self) -> dict:
        live = [c for c in self.calls if not c.cached]
        return {
            "calls": len(self.calls),
            "live_calls": len(live),
            "cached_calls": len(self.calls) - len(live),
            "prompt_tokens": sum(c.prompt_tokens for c in live),
            "completion_tokens": sum(c.completion_tokens for c in live),
            "cost_usd": round(sum(c.cost_usd for c in live), 5),
            "latency_s": round(sum(c.latency_s for c in live), 2),
        }

    def by_purpose(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for c in self.calls:
            d = out.setdefault(c.purpose or "other", {"calls": 0, "cost_usd": 0.0})
            d["calls"] += 1
            d["cost_usd"] = round(d["cost_usd"] + (0 if c.cached else c.cost_usd), 5)
        return out

    def snapshot(self) -> int:
        with self._lock:
            return len(self.calls)

    def since(self, mark: int) -> dict:
        sub = Ledger()
        with self._lock:
            sub.calls = self.calls[mark:]
        return sub.totals()


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict]
    message: dict
    cached: bool = False
    raw: dict = field(default_factory=dict)

    def json(self) -> Any:
        return extract_json(self.text)


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and surrounding prose."""
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start_candidates = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not start_candidates:
        raise ValueError(f"no JSON found in: {text[:200]!r}")
    start = min(start_candidates)
    opener = t[start]
    closer = "}" if opener == "{" else "]"
    end = t.rfind(closer)
    if end <= start:
        raise ValueError(f"unterminated JSON in: {text[:200]!r}")
    return json.loads(t[start : end + 1])


def image_part(img: Image.Image, max_side: int = 1280, detail: str = "high") -> dict:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}}


class ChatClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[str | Path | bool] = None,
        price_in: float = 1.25,
        price_out: float = 4.25,
        timeout: float = 180.0,
        max_retries: int = 4,
        transport: Optional[httpx.BaseTransport] = None,
        ledger: Optional[Ledger] = None,
        sleep=time.sleep,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.price_in = float(os.getenv("LLM_PRICE_IN", price_in))
        self.price_out = float(os.getenv("LLM_PRICE_OUT", price_out))
        # cache_dir=None -> default location (env PID_CACHE_DIR or .cache/llm); False or "" -> caching off
        if cache_dir is None:
            cd = os.getenv("PID_CACHE_DIR", ".cache/llm")
        else:
            cd = cache_dir
        self.cache_dir = Path(cd) if cd else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.ledger = ledger or Ledger()
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout, transport=transport)

    @classmethod
    def from_env(cls, **kw) -> ChatClient:
        return cls(**kw)

    # ------------------------------------------------------------------ core call
    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str = "",
        response_format: Optional[dict] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str | dict] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if use_cache and cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text())
            res = self._parse(data, cached=True)
            self.ledger.add(CallRecord(purpose, self.model, 0.0, 0, 0, 0.0, True))
            return res

        t0 = time.time()
        data = self._post(payload)
        latency = time.time() - t0
        usage = data.get("usage") or {}
        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        cost = pt / 1e6 * self.price_in + ct / 1e6 * self.price_out
        self.ledger.add(CallRecord(purpose, self.model, latency, pt, ct, cost, False))
        if use_cache and cache_file:
            cache_file.write_text(json.dumps(data))
        return self._parse(data, cached=False)

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        dropped_reasoning = False
        for attempt in range(self.max_retries + 1):
            try:
                r = self._http.post(url, headers=headers, json=payload)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"network error: {e}") from e
                self._sleep(min(2**attempt, 20))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400 and "reasoning" in r.text.lower() and "reasoning_effort" in payload and not dropped_reasoning:
                payload = {k: v for k, v in payload.items() if k != "reasoning_effort"}
                dropped_reasoning = True
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                self._sleep(min(2**attempt, 20))
                continue
            raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
        raise LLMError("exhausted retries")  # pragma: no cover

    @staticmethod
    def _parse(data: dict, cached: bool) -> ChatResult:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {"_unparsed": raw_args}
            calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})
        return ChatResult(text=content or "", tool_calls=calls, message=msg, cached=cached, raw=data)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
