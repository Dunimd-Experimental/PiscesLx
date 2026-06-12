#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTA Trainer — Launch entry point.

Initializes all training components and launches the EnTA agent.
All components are optional — the system degrades gracefully when
model or subconscious trainer are not available.

Usage:
    from encre.enta import launch_enta
    summary = await launch_enta(teacher="deepseek-r1")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from encre.enta.agent import (
    EnTAConfig,
    EnTAAgent,
    EnTAToolHandler,
    TrainingState,
    TrainingStage,
)
from encre.enta.curriculum import EnTACurriculum
from encre.enta.roundtable import EnTARoundTable
from encre.enta.sandbox import EnTASandbox
from encre.enta.bridge import EnTABridge, BridgeConfig

logger = logging.getLogger("encre.enta")


async def launch_enta(
    teacher: str = "deepseek-r1",
    agent_model: str = "agens-2.0-flash",
    aux_teachers: Optional[List[str]] = None,
    max_tasks: int = 0,
    tasks_per_stage: int = 5000,
    subconscious_enabled: bool = True,
    starting_stage: str = "foundation",
    record_interval: int = 100,
    dry_run: bool = False,
    model_path: str = "",
) -> Dict[str, Any]:
    """Initialize components and launch the EnTA agent.

    Args:
        teacher: Primary teacher model (config key from env).
        agent_model: Model driving EnTA's decisions.
        aux_teachers: Additional teachers for round-table.
        max_tasks: Total tasks (0 = unlimited).
        tasks_per_stage: Tasks per curriculum stage.
        subconscious_enabled: Train 0.5B subconscious head.
        starting_stage: Initial training stage.
        record_interval: Log frequency.
        dry_run: Validate without executing.
        model_path: Path to PiscesL1 7B model checkpoint (optional).

    Returns:
        Training summary dict.
    """
    if aux_teachers is None:
        aux_teachers = [
            "deepseek-v3.2", "qwen3.6",
            "agens-2.0-flash", "agens-2.0-video",
        ]

    cfg = EnTAConfig(
        agent_model=agent_model,
        teacher_model=teacher,
        auxiliary_teachers=aux_teachers,
        max_tasks=max_tasks,
        tasks_per_stage=tasks_per_stage,
        subconscious_enabled=subconscious_enabled,
        record_interval=record_interval,
    )

    # Stage scheduling
    stages_order = [
        TrainingStage.FOUNDATION,
        TrainingStage.INTEGRATION,
        TrainingStage.ADVANCED,
        TrainingStage.SPECIALIZATION,
        TrainingStage.SELF_PLAY,
    ]
    stage_map = {s.value: s for s in stages_order}
    if starting_stage in stage_map:
        start_idx = [s.value for s in stages_order].index(starting_stage)
        cfg.curriculum_schedule = stages_order[start_idx:]

    # Shared training state
    state = TrainingState(stage=starting_stage)

    # Curriculum
    curriculum = EnTACurriculum()

    # Teacher backends (lazy init on first tool call)
    teacher_backend = None
    aux_backends: Dict[str, Any] = {}

    # Sandbox
    sandbox = EnTASandbox(tools={})

    # Round-table (needs >= 2 aux teachers)
    round_table = None
    if len(aux_teachers) >= 2:
        round_table = EnTARoundTable(teacher_backends={})

    # ── Load the 7B student model (optional) ──────────────────────────
    student_model = None
    if model_path:
        try:
            import json
            from model.core.model import YvModelForCausalLM
            from model.config import YvConfig

            logger.info(f"Loading student model from {model_path}...")
            config_path = os.path.join(model_path, "config.json")
            config = YvConfig.from_json(config_path) if os.path.exists(config_path) else YvConfig()
            student_model = YvModelForCausalLM(config)
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                import torch
                student_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            student_model.eval()
            logger.info(f"Student model loaded: {config.hidden_size} hidden, {config.n_layer} layers")
        except Exception as e:
            logger.warning(f"Failed to load student model: {e}")

    # ── Subconscious trainer (optional) ───────────────────────────────
    sc_trainer = None
    sc_system = None
    if subconscious_enabled:
        try:
            from opss.subconscious_trainer import SubconsciousTrainer, SubconsciousTrainingConfig, TrainingPhase

            # Get the subconscious system from the model if loaded
            if student_model is not None and hasattr(student_model, 'subconscious'):
                sc_system = student_model.subconscious

            sc_cfg = SubconsciousTrainingConfig(phase=TrainingPhase.HEAD_RL)
            sc_trainer = SubconsciousTrainer(
                subconscious_system=sc_system,
                config=sc_cfg,
            )
            logger.info(f"Subconscious trainer initialized (system={'loaded' if sc_system else 'None'})")
        except Exception as e:
            logger.warning(f"Subconscious trainer init failed (will continue without): {e}")

    # ── Bridge ─────────────────────────────────────────────────────────
    # Register SFT training op if available
    training_ops = {}
    try:
        from opss.train import POPSSSFTTrainingOperator
        training_ops["sft"] = POPSSSFTTrainingOperator()
        logger.info("SFT operator registered")
    except Exception:
        logger.debug("No SFT operator available (will buffer samples)")

    bridge = EnTABridge(
        config=BridgeConfig(
            subconscious_enabled=subconscious_enabled and sc_trainer is not None,
            sft_enabled=bool(training_ops.get("sft")),
        ),
        student_model=student_model,
        subconscious_trainer=sc_trainer,
        training_ops=training_ops,
    )

    # ── Tool handler ──────────────────────────────────────────────────
    tool_handler = EnTAToolHandler(
        curriculum=curriculum,
        sandbox=sandbox,
        bridge=bridge,
        teacher_backend=teacher_backend,
        aux_backends=aux_backends,
        round_table=round_table,
        state=state,
    )

    # ── Dry run ───────────────────────────────────────────────────────
    if dry_run:
        sample_task = curriculum.next_task(starting_stage)
        logger.info(f"=== EnTA Dry Run ===")
        logger.info(f"  agent_model:      {agent_model}")
        logger.info(f"  teacher:          {teacher}")
        logger.info(f"  aux_teachers:     {aux_teachers}")
        logger.info(f"  subconscious:     {subconscious_enabled} (trainer={'yes' if sc_trainer else 'no'})")
        logger.info(f"  model:            {'loaded' if student_model else 'None'}")
        logger.info(f"  starting_stage:   {starting_stage}")
        logger.info(f"  max_tasks:        {max_tasks or 'unlimited'}")
        logger.info(f"  sample_task:      {sample_task.description if sample_task else 'none'}")
        logger.info(f"  bridge_stats:     {bridge.get_stats()}")
        logger.info("EnTA dry run: pipeline validated")
        return {"dry_run": True, "status": "validated"}

    # ── Launch agent ──────────────────────────────────────────────────
    agent = EnTAAgent(
        config=cfg,
        tool_handler=tool_handler,
        state=state,
    )
    summary = await agent.run(max_tasks=max_tasks)
    return summary
