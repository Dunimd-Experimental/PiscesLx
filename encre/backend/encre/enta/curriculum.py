#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTACurriculum — Adaptive training curriculum for the 7B student model.

Generates tasks across 7 skill areas with automatic difficulty adjustment.
Tracks per-skill performance to focus training on weak areas.

Skill areas (aligned with the 7B model's target capabilities):
    TOOL_OPERATION    → file system, code exec, shell
    REASONING         → math, logic, chain-of-thought
    KNOWLEDGE_RETRIEVAL → search, multi-hop, fact lookup
    CREATIVE          → writing, brainstorming, open-ended
    CODE              → generation, debug, refactor
    TOOL_CHAIN        → multi-step tool composition
    ERROR_RECOVERY    → failure handling, edge cases
"""

from __future__ import annotations

import enum
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("encre.enta")


class SkillArea(enum.Enum):
    TOOL_OPERATION = "tool_operation"
    REASONING = "reasoning"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CREATIVE = "creative"
    CODE = "code"
    TOOL_CHAIN = "tool_chain"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class TaskTemplate:
    skill_area: SkillArea
    description: str
    difficulty: float = 0.3
    requires_creativity: bool = False
    tools_required: List[str] = field(default_factory=list)
    teacher_hint: str = ""
    metadata: Dict = field(default_factory=dict)


class EnTACurriculum:
    """Adaptive curriculum with per-skill performance tracking.

    Samples tasks from a pool weighted by each skill area's recent performance.
    Under-performing areas get proportionally more tasks.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.stage_name: str = "foundation"
        self.task_history: List[TaskTemplate] = []

        # Per-skill: weight (higher = more likely to be sampled)
        self.skill_weights: Dict[SkillArea, float] = {s: 1.0 for s in SkillArea}

        # Per-skill: reward history (used to adjust weights)
        self.skill_rewards: Dict[SkillArea, List[float]] = {s: [] for s in SkillArea}

        self._pools = self._build_pools()

        logger.info(f"Curriculum: {len(SkillArea)} areas, {sum(len(v) for v in self._pools.values())} tasks")

    def next_task(self, stage: Any) -> Optional[TaskTemplate]:
        """Sample next task weighted by skill performance.

        Args:
            stage: TrainingStage (or string name).

        Returns:
            TaskTemplate or None if no tasks for this stage.
        """
        name = stage.value if hasattr(stage, "value") else str(stage)
        pool = self._pools.get(name)
        if not pool:
            # Fall back to foundation
            pool = self._pools.get("foundation", [])
        if not pool:
            return None

        # Sample weighted by skills
        weights = [self.skill_weights[t.skill_area] for t in pool]
        weights = [max(0.1, w) for w in weights]  # never zero

        task = self.rng.choices(pool, weights=weights, k=1)[0]
        self.task_history.append(task)
        return task

    def record_outcome(self, task: TaskTemplate, reward: float):
        """Update skill weights based on outcome.

        Low recent reward → higher sampling weight (more training needed).

        Args:
            task: The task that was executed.
            reward: Outcome reward (0.0–1.0).
        """
        area = task.skill_area
        self.skill_rewards[area].append(reward)

        # Keep a sliding window of recent outcomes
        recent = self.skill_rewards[area][-50:]
        if len(recent) >= 5:
            avg = sum(recent) / len(recent)
            # Low avg reward → higher weight (needs more practice)
            # High avg reward → lower weight (mastered, no need to over-train)
            self.skill_weights[area] = 1.0 + (1.0 - avg) ** 2 * 3.0

    def on_stage_change(self, new_stage: Any):
        """Handle stage transition, optionally resetting weights."""
        name = new_stage.value if hasattr(new_stage, "value") else str(new_stage)
        self.stage_name = name
        logger.info(f"Curriculum stage → {name}")

        # In self-play, let all skills compete equally
        if name == "self_play":
            for s in SkillArea:
                self.skill_weights[s] = 1.0

    def _build_pools(self) -> Dict[str, List[TaskTemplate]]:
        """Build task pools for each training stage."""
        return {
            "foundation": [
                TaskTemplate(SkillArea.TOOL_OPERATION,
                    "Read a file and extract specific info", 0.2, tools_required=["file_read", "grep"]),
                TaskTemplate(SkillArea.TOOL_OPERATION,
                    "Create dir structure and write files", 0.3, tools_required=["bash", "file_write"]),
                TaskTemplate(SkillArea.TOOL_OPERATION,
                    "Run a Python script, capture stdout", 0.3, tools_required=["bash"]),
                TaskTemplate(SkillArea.REASONING,
                    "Solve a 3-step math word problem", 0.3),
                TaskTemplate(SkillArea.REASONING,
                    "Trace a logical deduction chain", 0.3),
                TaskTemplate(SkillArea.CODE,
                    "Write a function that sorts a list", 0.3, tools_required=["bash", "file_write"]),
                TaskTemplate(SkillArea.CODE,
                    "Fix a syntax error in supplied code", 0.3, tools_required=["bash", "grep"]),
                TaskTemplate(SkillArea.KNOWLEDGE_RETRIEVAL,
                    "Search and summarize a topic", 0.3, tools_required=["web_search", "web_fetch"]),
                TaskTemplate(SkillArea.ERROR_RECOVERY,
                    "Handle a file-not-found error gracefully", 0.3, tools_required=["file_read"]),
            ],
            "integration": [
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Download file, process it, save results", 0.5,
                    tools_required=["web_fetch", "bash", "file_write"]),
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Search code, read it, suggest improvements", 0.5,
                    tools_required=["grep", "file_read", "file_edit"]),
                TaskTemplate(SkillArea.REASONING,
                    "Analyse a small dataset, draw conclusions", 0.6,
                    tools_required=["bash", "file_read"]),
                TaskTemplate(SkillArea.CODE,
                    "Refactor a function, preserve behaviour", 0.5,
                    tools_required=["file_read", "file_edit", "bash"]),
                TaskTemplate(SkillArea.CODE,
                    "Write a script that transforms JSON→CSV", 0.5,
                    tools_required=["bash", "file_write"]),
                TaskTemplate(SkillArea.ERROR_RECOVERY,
                    "Fix a broken pipeline command", 0.5, tools_required=["bash"]),
                TaskTemplate(SkillArea.KNOWLEDGE_RETRIEVAL,
                    "Cross-reference info from 3 sources", 0.5,
                    tools_required=["web_search", "web_fetch"]),
            ],
            "advanced": [
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Build a small web scraper with error handling", 0.7,
                    tools_required=["bash", "file_write", "web_fetch"]),
                TaskTemplate(SkillArea.CREATIVE,
                    "Write a short story under length + topic constraints", 0.6, requires_creativity=True),
                TaskTemplate(SkillArea.CREATIVE,
                    "Generate 3 distinct solutions to an open problem", 0.7,
                    requires_creativity=True, tools_required=["bash"]),
                TaskTemplate(SkillArea.REASONING,
                    "Plan and execute a 5-step task autonomously", 0.7,
                    tools_required=["bash", "file_read", "file_write", "web_search", "grep"]),
                TaskTemplate(SkillArea.CODE,
                    "Implement a linked list from scratch", 0.7,
                    tools_required=["file_write", "bash"]),
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Automate a repetitive workflow via shell script", 0.7,
                    tools_required=["bash", "file_write"]),
                TaskTemplate(SkillArea.ERROR_RECOVERY,
                    "Recover data from a corrupted JSON file", 0.7,
                    tools_required=["file_read", "bash", "file_write"]),
                TaskTemplate(SkillArea.CREATIVE,
                    "Design a UI layout from a verbal spec", 0.65, requires_creativity=True),
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Scrape a page, extract table, compute stats", 0.75,
                    tools_required=["web_fetch", "bash", "file_write"]),
            ],
            "specialization": [
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Simulate a CI/CD pipeline with build steps", 0.8,
                    tools_required=["bash", "file_write", "git_tool"]),
                TaskTemplate(SkillArea.CODE,
                    "Implement a REST API client with retry logic", 0.8,
                    tools_required=["file_write", "bash", "rest_client"]),
                TaskTemplate(SkillArea.TOOL_CHAIN,
                    "Multi-script data ETL pipeline", 0.8,
                    tools_required=["bash", "file_write", "docker"]),
                TaskTemplate(SkillArea.CREATIVE,
                    "System-design a solution for a given problem", 0.8,
                    requires_creativity=True, tools_required=["file_write"]),
                TaskTemplate(SkillArea.ERROR_RECOVERY,
                    "Debug and fix a race condition in concurrent code", 0.85,
                    tools_required=["file_read", "file_edit", "bash"]),
                TaskTemplate(SkillArea.CREATIVE,
                    "Write a persuasive argument on a controversial topic", 0.75,
                    requires_creativity=True),
                TaskTemplate(SkillArea.REASONING,
                    "Solve a constraint-satisfaction puzzle", 0.8),
                TaskTemplate(SkillArea.KNOWLEDGE_RETRIEVAL,
                    "Multi-hop research: answer requires 3+ sources", 0.8,
                    tools_required=["web_search", "web_fetch"]),
            ],
        }
