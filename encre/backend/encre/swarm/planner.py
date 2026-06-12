#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from encre.prompts.loader import PromptLoader


@dataclass
class TaskNode:
    id: str
    name: str
    description: str
    assigned_role: str = "general"
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    status: str = "pending"  # pending | ready | running | completed | failed | skipped
    result: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskTree:
    goal: str
    nodes: dict[str, TaskNode]
    entry_nodes: list[str]
    exit_nodes: list[str]

    def get_ready_nodes(self) -> list[TaskNode]:
        ready: list[TaskNode] = []
        for node in self.nodes.values():
            if node.status != "pending":
                continue
            if all(
                self.nodes[dep].status == "completed"
                for dep in node.dependencies
                if dep in self.nodes
            ):
                ready.append(node)
        ready.sort(key=lambda n: (-n.priority, n.name))
        return ready

    def all_done(self) -> bool:
        return all(
            n.status in ("completed", "skipped")
            for n in self.nodes.values()
        )

    def has_failure(self) -> bool:
        return any(n.status == "failed" for n in self.nodes.values())


class EncreTaskPlanner:
    """LLM-driven hierarchical task decomposition.

    Takes a high-level goal, returns a TaskTree (DAG) ready for execution.
    Includes a rule-based fallback (no LLM call) for simple goals.
    """

    def __init__(self) -> None:
        self._known_patterns: dict[str, list[dict[str, Any]]] = {
            "build": [
                {"name": "Design architecture", "role": "architect", "priority": 10},
                {"name": "Implement core logic", "role": "coder", "deps": [0], "priority": 5},
                {"name": "Write tests", "role": "tester", "deps": [1], "priority": 5},
                {"name": "Code review", "role": "reviewer", "deps": [1], "priority": 3},
                {"name": "Integrate & verify", "role": "coder", "deps": [2, 3], "priority": 1},
            ],
            "debug": [
                {"name": "Reproduce the issue", "role": "debugger", "priority": 10},
                {"name": "Identify root cause", "role": "debugger", "deps": [0], "priority": 5},
                {"name": "Implement fix", "role": "coder", "deps": [1], "priority": 5},
                {"name": "Verify fix", "role": "tester", "deps": [2], "priority": 3},
            ],
            "research": [
                {"name": "Gather information", "role": "researcher", "priority": 10},
                {"name": "Analyze findings", "role": "researcher", "deps": [0], "priority": 5},
                {"name": "Synthesize report", "role": "general", "deps": [1], "priority": 3},
                {"name": "Review & refine", "role": "reviewer", "deps": [2], "priority": 1},
            ],
            "refactor": [
                {"name": "Analyze current structure", "role": "architect", "priority": 10},
                {"name": "Plan refactoring", "role": "architect", "deps": [0], "priority": 5},
                {"name": "Execute refactoring", "role": "coder", "deps": [1], "priority": 5},
                {"name": "Run tests", "role": "tester", "deps": [2], "priority": 3},
                {"name": "Review changes", "role": "reviewer", "deps": [2, 3], "priority": 1},
            ],
        }

    async def decompose(self, goal: str) -> TaskTree:
        """Decompose a goal into a task tree (async-compatible)."""
        return self.plan(goal)

    def plan(self, goal: str, context: str = "") -> TaskTree:
        pattern_key = _detect_pattern(goal)
        if pattern_key and pattern_key in self._known_patterns:
            return self._build_from_pattern(goal, pattern_key)

        return self._simple_decompose(goal)

    def plan_with_llm(
        self,
        goal: str,
        context: str,
        plan_prompt: str | None = None,
    ) -> str:
        """Returns a structured prompt for an LLM to generate a task plan.
        The LLM's JSON response can be parsed by plan_from_json().
        """
        if plan_prompt is None:
            plan_prompt = _DEFAULT_PLAN_PROMPT
        return plan_prompt.format(goal=goal, context=context)

    @staticmethod
    def plan_from_json(goal: str, json_str: str) -> TaskTree:
        data = json.loads(json_str)
        nodes: dict[str, TaskNode] = {}
        for item in data.get("tasks", []):
            nid = item.get("id", str(uuid.uuid4())[:8])
            nodes[nid] = TaskNode(
                id=nid,
                name=item.get("name", "Unnamed task"),
                description=item.get("description", ""),
                assigned_role=item.get("role", "general"),
                dependencies=item.get("dependencies", []),
                priority=item.get("priority", 5),
            )
        entry_ids = data.get("entry_tasks", [])
        exit_ids = data.get("exit_tasks", [])
        if not entry_ids and nodes:
            entry_ids = [
                nid for nid, n in nodes.items()
                if not n.dependencies
            ][:1]
        if not exit_ids and nodes:
            all_ids = set(nodes.keys())
            dep_set: set[str] = set()
            for n in nodes.values():
                dep_set.update(n.dependencies)
            exit_ids = list(all_ids - dep_set) or [list(nodes.keys())[-1]]
        return TaskTree(goal=goal, nodes=nodes, entry_nodes=entry_ids, exit_nodes=exit_ids)

    def _build_from_pattern(self, goal: str, pattern_key: str) -> TaskTree:
        pattern = self._known_patterns[pattern_key]
        nodes: dict[str, TaskNode] = {}
        id_map: dict[int, str] = {}
        for i, step in enumerate(pattern):
            nid = str(uuid.uuid4())[:8]
            id_map[i] = nid
            nodes[nid] = TaskNode(
                id=nid,
                name=step["name"],
                description=f"{step['name']} for: {goal}",
                assigned_role=step.get("role", "general"),
                dependencies=[id_map[d] for d in step.get("deps", []) if d in id_map],
                priority=step.get("priority", 5),
            )
        entry_ids = [nid for nid, n in nodes.items() if not n.dependencies]
        all_ids = set(nodes.keys())
        dep_set: set[str] = set()
        for n in nodes.values():
            dep_set.update(n.dependencies)
        exit_ids = list(all_ids - dep_set) or [list(nodes.keys())[-1]]
        return TaskTree(goal=goal, nodes=nodes, entry_nodes=entry_ids, exit_nodes=exit_ids)

    @staticmethod
    def _simple_decompose(goal: str) -> TaskTree:
        nid = str(uuid.uuid4())[:8]
        node = TaskNode(
            id=nid,
            name="Execute",
            description=goal,
            assigned_role="general",
        )
        return TaskTree(
            goal=goal,
            nodes={nid: node},
            entry_nodes=[nid],
            exit_nodes=[nid],
        )


def _detect_pattern(goal: str) -> str | None:
    g = goal.lower()
    if any(kw in g for kw in ("build", "create", "make", "implement", "write a", "develop")):
        return "build"
    if any(kw in g for kw in ("debug", "fix", "bug", "error", "issue", "broken", "wrong")):
        return "debug"
    if any(kw in g for kw in ("research", "investigate", "explore", "study", "analyze", "understand")):
        return "research"
    if any(kw in g for kw in ("refactor", "clean up", "restructure", "reorganize", "improve")):
        return "refactor"
    return None


_loader = PromptLoader()

_DEFAULT_PLAN_PROMPT = _loader.load("planner", category="swarm")
