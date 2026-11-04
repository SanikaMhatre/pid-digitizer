"""Build pipelines from environment configuration.

Live mode (real models):
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL     the Muse model that classifies, reads tags and verifies
  PID_LOCALIZER                            "sam" (default) or "muse"
  SAM_API_KEY, SAM_BASE_URL                SAM 3.1 on the Meta Model API (defaults: LLM_API_KEY, https://api.meta.ai/v1)
"""
from __future__ import annotations

import os
from typing import Optional

from .classify import SimulatedClassifier, VLMClassifier
from .models.llm import ChatClient
from .pipeline import Pipeline, PipelineConfig
from .rules.engine import Rule
from .schema import Graph
from .segment.base import OracleSegmenter
from .segment.meta_sam import MetaSamLocalizer
from .segment.muse_localizer import MuseLocalizer


def build_localizer(client: ChatClient, kind: Optional[str] = None):
    """"sam": SAM 3.1 on the Meta Model API.  "muse": the Muse model finds the boxes itself."""
    kind = (kind or os.getenv("PID_LOCALIZER") or "sam").lower()
    if kind == "muse":
        return MuseLocalizer(client)
    if kind != "sam":
        raise ValueError(f"unknown localizer {kind!r}: use 'sam' or 'muse'")
    sam_client = ChatClient(
        base_url=os.getenv("SAM_BASE_URL") or "https://api.meta.ai/v1",
        api_key=os.getenv("SAM_API_KEY") or client.api_key,
        model="sam-3.1",
        ledger=client.ledger,
    )
    return MetaSamLocalizer(sam_client)


def build_live_pipeline(
    client: Optional[ChatClient] = None,
    rules: Optional[list[Rule]] = None,
    cfg: Optional[PipelineConfig] = None,
    localizer: Optional[str] = None,
) -> Pipeline:
    """SAM 3.1 (or the Muse model) finds symbols; the Muse model classifies, reads tags and verifies."""
    client = client or ChatClient.from_env()
    return Pipeline(build_localizer(client, localizer), VLMClassifier(client), client=client, rules=rules, cfg=cfg)


def build_simulated_pipeline(gt: Graph, rules: Optional[list[Rule]] = None, cfg: Optional[PipelineConfig] = None) -> Pipeline:
    """Pipeline whose model stages are replaced by ground-truth lookups.

    Exercises tracing, graph building, rules and overlay without any model call. Used by tests and
    the offline UI mode. Numbers produced this way measure the classical stages only.
    """
    cfg = cfg or PipelineConfig(run_agent=False)
    return Pipeline(OracleSegmenter(gt), SimulatedClassifier(gt), client=None, rules=rules, cfg=cfg)
