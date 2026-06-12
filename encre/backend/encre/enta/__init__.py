#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTA — Encre Train Agent.

An autonomous training agent driven by Agens-2.0-Flash (or configured model).
EnTA autonomously decides what to teach, which teacher to call, when to use
round-table discussions, and how to update the student model — all through
LLM-driven tool calls.

Architecture:
    EnTA itself is an Agent. It uses:
    - backends/    → call teacher models for training data
    - curriculum/  → track student weak areas, suggest next tasks
    - roundtable/  → multi-teacher consensus for creative tasks
    - sandbox/     → execute tasks, get objective reward signals
    - bridge/      → feed data into PiscesL1 training pipeline

    The agent's decisions are made by {config.agent_model} via tool calls.
    The training loop is the agent's own thinking loop, not hardcoded Python.
"""

from encre.enta.trainer import launch_enta
from encre.enta.config import (
    get_model_backend_config,
    create_backend_for_model,
    list_configured_models,
)
from encre.enta.agent import (
    EnTAConfig,
    EnTAAgent,
    EnTAToolHandler,
    TrainingState,
    TrainingStage,
    ENTA_SYSTEM_PROMPT,
)
from encre.enta.curriculum import EnTACurriculum, TaskTemplate, SkillArea
from encre.enta.roundtable import EnTARoundTable, RoundTableResult
from encre.enta.sandbox import EnTASandbox
from encre.enta.bridge import EnTABridge, BridgeConfig, TrainingSample

__all__ = [
    "launch_enta",
    "get_model_backend_config", "create_backend_for_model", "list_configured_models",
    "EnTAConfig", "EnTAAgent", "EnTAToolHandler", "TrainingState", "TrainingStage",
    "ENTA_SYSTEM_PROMPT",
    "EnTACurriculum", "TaskTemplate", "SkillArea",
    "EnTARoundTable", "RoundTableResult",
    "EnTASandbox",
    "EnTABridge", "BridgeConfig", "TrainingSample",
]
