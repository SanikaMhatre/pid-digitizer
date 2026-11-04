"""Build pipelines from environment configuration.

Live mode (real models) needs only the Muse Spark Open endpoint:
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
"""
from __future__ import annotations

from typing import Optional

from .classify import SimulatedClassifier, VLMClassifier
from .models.llm import ChatClient
from .pipeline import Pipeline, PipelineConfig
from .rules.engine import Rule
from .schema import Graph
from .segment.base import OracleSegmenter
from .segment.muse_localizer import MuseLocalizer


def build_live_pipeline(
    client: Optional[ChatClient] = None,
    rules: Optional[list[Rule]] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Pipeline:
    """Muse Spark Open does localization, classification, tag reading and verification."""
    client = client or ChatClient.from_env()
    return Pipeline(MuseLocalizer(client), VLMClassifier(client), client=client, rules=rules, cfg=cfg)


def build_simulated_pipeline(gt: Graph, rules: Optional[list[Rule]] = None, cfg: Optional[PipelineConfig] = None) -> Pipeline:
    """Pipeline whose model stages are replaced by ground-truth lookups.

    Exercises tracing, graph building, rules and overlay without any model call. Used by tests and
    the offline UI mode. Numbers produced this way measure the classical stages only.
    """
    cfg = cfg or PipelineConfig(run_agent=False)
    return Pipeline(OracleSegmenter(gt), SimulatedClassifier(gt), client=None, rules=rules, cfg=cfg)
