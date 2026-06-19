#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.



"""``workflow`` tool -- decompose a goal into a dependency-ordered task graph.

The original implementation depended on ``enta.swarm.planner`` and the
``EncreTaskPlanner`` class, both removed during the EnTA slim-down.  This
module provides a real but self-contained replacement: it accepts a goal,
splits it on blank lines / numbered markers into a flat list of ordered
sub-tasks, registers them with the in-process :class:`EncreTaskStore`,
and returns a human-readable summary.  No LLM is invoked here -- the
model is expected to break the goal down into a JSON list it embeds in
the call.  When no list is provided the tool simply records the goal as
a single task so the workflow is still auditable.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from enta.tools.base import build_tool
from enta.utils.task_store import get_store

logger = logging.getLogger(__name__)


@dataclass
class _WorkflowNode:
    """In-memory description of one sub-task in a workflow."""

    id: str
    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""


@dataclass
class _WorkflowTree:
    goal: str
    nodes: dict[str, _WorkflowNode] = field(default_factory=dict)
    entry_ids: list[str] = field(default_factory=list)


def _parse_inline_steps(goal: str) -> list[str]:
    """Best-effort extractor for a numbered list embedded in ``goal``.

    The model is expected to call this tool with a plain string goal;
    when it includes a newline-separated or numbered list we can split
    it into individual steps without invoking the LLM.
    """
    text = goal.strip()
    if not text:
        return []

    numbered: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(?:\d+[\.\)、]\s+|[-*]\s+)(.+)$", line)
        if m:
            numbered.append(m.group(1).strip())
    if numbered:
        return numbered

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    return [text]


async def _workflow_execute(**kwargs: Any) -> dict[str, Any]:
    goal = kwargs.get("goal", "")
    if not goal:
        return {
            "content": "Error: 'goal' is required.",
            "tasks": [],
        }

    steps = _parse_inline_steps(goal)
    tree = _WorkflowTree(goal=goal)
    store = get_store()
    previous_id: str | None = None

    for idx, step in enumerate(steps, 1):
        node_id = f"n_{idx}_{uuid.uuid4().hex[:6]}"
        node = _WorkflowNode(
            id=node_id,
            name=f"Step {idx}",
            description=step,
            dependencies=[previous_id] if previous_id else [],
        )
        tree.nodes[node_id] = node
        if not node.dependencies:
            tree.entry_ids.append(node_id)

        task_record = store.create(
            name=node.name,
            description=node.description,
            task_type="workflow",
            prompt=step,
        )
        store.update(
            task_record.id,
            status="running",
        )
        store.update(
            task_record.id,
            status="completed",
            result=f"Workflow step recorded for '{goal[:40]}...'",
        )
        previous_id = node_id

    summary_lines = [f"Workflow recorded for: {goal}", ""]
    for node_id in tree.entry_ids + [
        nid for nid in tree.nodes if nid not in tree.entry_ids
    ]:
        node = tree.nodes[node_id]
        deps = ", ".join(node.dependencies) if node.dependencies else "(root)"
        summary_lines.append(f"- [{node.status}] {node.name} ({deps}): {node.description[:80]}")

    summary_lines.append("")
    summary_lines.append(
        f"Total steps: {len(tree.nodes)} | Tasks registered in ledger: {len(steps)}"
    )
    return {
        "content": "\n".join(summary_lines),
        "tasks": [
            {
                "id": n.id,
                "name": n.name,
                "description": n.description,
                "dependencies": n.dependencies,
                "status": n.status,
            }
            for n in tree.nodes.values()
        ],
    }


EncreWorkflowTool = build_tool(
    name="workflow",
    description=(
        "Record a multi-step workflow as a dependency-ordered task graph. "
        "The goal is split on numbered lines or paragraph breaks into "
        "sub-tasks that are registered with the in-process task ledger. "
        "Each sub-task starts in 'running' state and is immediately marked "
        "'completed' after registration; long-running execution should be "
        "performed by the model itself using the specialised tools."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "The complete goal to accomplish. May include a numbered "
                    "list (1. ... 2. ...) or paragraph-separated steps; "
                    "each becomes a sub-task with sequential dependencies."
                ),
            },
        },
        "required": ["goal"],
    },
    execute=_workflow_execute,
    intents=["general", "coding", "system"],
)
