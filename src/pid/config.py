"""Build pipelines from environment configuration.

Live mode (real models):
  LLM_API_KEY, LLM_BASE_URL, LLM_MODEL         -> Muse Spark Open endpoint
  PID_SEGMENTER = modal | vlm                    -> SAM 3.1 on Modal, or the VLM-box fallback
  PID_MODAL_APP / PID_MODAL_CLS                  -> default pid-sam / SamServer
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


def build_live_pipeline(
    segmenter: Optional[str] = None,
    client: Optional[ChatClient] = None,
    rules: Optional[list[Rule]] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Pipeline:
    client = client or ChatClient.from_env()
    segmenter = (segmenter or os.getenv("PID_SEGMENTER", "modal")).lower()
    if segmenter == "modal":
        from .segment.sam_modal import ModalSamSegmenter

        seg = ModalSamSegmenter(os.getenv("PID_MODAL_APP", "pid-sam"), os.getenv("PID_MODAL_CLS", "SamServer"))
    elif segmenter == "vlm":
        from .segment.vlm_boxes import VLMBoxSegmenter

        seg = VLMBoxSegmenter(client)
    else:
        raise ValueError(f"unknown segmenter '{segmenter}' (expected modal or vlm)")
    return Pipeline(seg, VLMClassifier(client), client=client, rules=rules, cfg=cfg)


def build_simulated_pipeline(gt: Graph, rules: Optional[list[Rule]] = None, cfg: Optional[PipelineConfig] = None) -> Pipeline:
    """Pipeline whose model stages are replaced by ground-truth lookups.

    Exercises tracing, graph building, rules and overlay without any model call. Used by tests and
    the offline UI mode. Numbers produced this way measure the classical stages only.
    """
    cfg = cfg or PipelineConfig(run_agent=False)
    return Pipeline(OracleSegmenter(gt), SimulatedClassifier(gt), client=None, rules=rules, cfg=cfg)
