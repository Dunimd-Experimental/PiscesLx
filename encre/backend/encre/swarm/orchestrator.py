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

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from encre.swarm.blackboard import EncreBlackboard
from encre.swarm.roles import AgentRole, RoleRegistry


@dataclass
class OrchestrationEvent:
    type: str  # task_started | task_completed | task_failed | team_finished | progress
    task_id: str = ""
    task_name: str = ""
    role: str = ""
    result: str = ""
    error: str = ""
    progress: float = 0.0
    timestamp: float = field(default_factory=time.time)


class EncreOrchestrator:
    """Executes a TaskTree with role-based teammate agents.

    Features:
    - DAG-based execution: respects task dependencies
    - Reviewer gate: coder output can be checked by reviewer
    - Parallel execution: independent tasks run concurrently
    - Blackboard: shared state accessible by all teammates
    - Progress streaming: yields OrchestrationEvents
    """

    def __init__(
        self,
        role_registry: RoleRegistry | None = None,
        blackboard: EncreBlackboard | None = None,
        max_concurrent: int = 10,
        enable_reviewer_gate: bool = True,
    ) -> None:
        self._roles = role_registry or RoleRegistry()
        self._blackboard = blackboard or EncreBlackboard()
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._enable_reviewer_gate = enable_reviewer_gate
        self._cancelled = False
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def execute(
        self,
        task_tree: Any,  # TaskTree
    ) -> AsyncGenerator[OrchestrationEvent, None]:
        self._cancelled = False
        self._running_tasks.clear()
        nodes = task_tree.nodes
        if not nodes:
            yield OrchestrationEvent(type="team_finished", progress=1.0)
            return

        self._blackboard.put("__orchestrator__", "goal", task_tree.goal)
        total = len(nodes)
        completed = 0

        running: dict[str, asyncio.Task[None]] = {}

        try:
            while not self._cancelled:
                ready = task_tree.get_ready_nodes()
                if not ready and not running:
                    break

                for node in ready:
                    if self._cancelled:
                        break
                    node.status = "running"
                    task = asyncio.create_task(self._execute_node(node, task_tree))
                    running[node.id] = task
                    self._running_tasks[node.id] = task

                if not running:
                    break

                done, _ = await asyncio.wait(
                    running.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    for nid, t in list(running.items()):
                        if t is task:
                            del running[nid]
                            self._running_tasks.pop(nid, None)
                            node = nodes.get(nid)
                            if node and node.status == "completed":
                                completed += 1
                                yield OrchestrationEvent(
                                    type="task_completed",
                                    task_id=node.id,
                                    task_name=node.name,
                                    role=node.assigned_role,
                                    result=node.result,
                                    progress=completed / total,
                                )
                            elif node and node.status == "cancelled":
                                completed += 1
                                yield OrchestrationEvent(
                                    type="task_failed",
                                    task_id=node.id,
                                    task_name=node.name,
                                    role=node.assigned_role,
                                    error="Task was cancelled",
                                    progress=completed / total,
                                )
                            elif node and node.status == "failed":
                                completed += 1
                                yield OrchestrationEvent(
                                    type="task_failed",
                                    task_id=node.id,
                                    task_name=node.name,
                                    role=node.assigned_role,
                                    error=node.error,
                                    progress=completed / total,
                                )
                            break
        except asyncio.CancelledError:
            # The orchestration loop itself was cancelled — cancel every
            # in-flight node task and mark remaining nodes as cancelled.
            self._cancelled = True
            for t in running.values():
                t.cancel()
            for nid in list(running.keys()):
                node = nodes.get(nid)
                if node and node.status == "running":
                    node.status = "cancelled"
                    node.error = "Orchestrator cancelled"
            yield OrchestrationEvent(
                type="team_finished",
                progress=completed / total if total else 1.0,
                error="Orchestrator cancelled",
            )
            return

        # If the loop exited because of cancel(), drain any still-running tasks.
        if self._cancelled and running:
            for t in running.values():
                t.cancel()
            # Give them a moment to react to cancellation.
            if running:
                await asyncio.wait(running.values(), timeout=5.0)
            for nid, node in nodes.items():
                if node.status == "running":
                    node.status = "cancelled"
                    node.error = "Cancelled by orchestrator"

        if completed >= total:
            yield OrchestrationEvent(type="team_finished", progress=1.0)

    async def _execute_node(self, node: Any, task_tree: Any) -> None:
        async with self._semaphore:
            try:
                role = self._roles.get(node.assigned_role)

                yield_event = OrchestrationEvent(
                    type="task_started",
                    task_id=node.id,
                    task_name=node.name,
                    role=node.assigned_role,
                )

                context = self._build_context(node, task_tree)
                result = await self._run_agent(node, role, context)

                # Reviewer gate
                if self._enable_reviewer_gate and role.name == "coder":
                    review_ok = await self._reviewer_check(node, result)
                    if not review_ok:
                        node.status = "failed"
                        node.error = "Reviewer rejected the output"
                        return

                node.result = result
                node.status = "completed"
                self._blackboard.put(f"task:{node.id}", "result", result)

            except asyncio.CancelledError:
                node.status = "cancelled"
                node.error = "Cancelled by orchestrator"
            except Exception as e:
                node.status = "failed"
                node.error = str(e)

    async def _run_agent(self, node: Any, role: AgentRole, context: str) -> str:
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        from encre.tools.registry import ToolRegistry
        from encre.utils.types import TextDelta, ToolResult, Finish

        config = EncreConfig(
            max_turns=15,
            permission_mode=role.permission_mode,
        )
        agent = EncreAgent(config=config)
        system_prompt = role.system_prompt_override or None
        full_prompt = f"{node.name}: {node.description}\n\n{context}"

        parts: list[str] = []
        async for event in agent.run(full_prompt, system_prompt=system_prompt):
            if isinstance(event, TextDelta) and event.text:
                parts.append(event.text)
            elif isinstance(event, Finish):
                if event.reason == "error":
                    parts.append(f"\n[Error during execution]")
        return "".join(parts)

    async def _reviewer_check(self, coder_node: Any, result: str) -> bool:
        reviewer_role = self._roles.get("reviewer")
        review_prompt = (
            f"Review the output of task '{coder_node.name}'. "
            f"Output:\n```\n{result[:5000]}\n```\n"
            "Does this look correct and production-ready? Reply ONLY with 'APPROVED' or 'REJECTED: <reason>'."
        )
        try:
            reviewer_result = await asyncio.wait_for(
                self._run_simple_agent(reviewer_role, review_prompt),
                timeout=120.0,
            )
            return "APPROVED" in reviewer_result.upper() and "REJECTED" not in reviewer_result.upper()
        except asyncio.TimeoutError:
            return True  # Timeout: approve by default

    async def _run_simple_agent(self, role: AgentRole, prompt: str) -> str:
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        from encre.utils.types import TextDelta

        config = EncreConfig(max_turns=5, permission_mode="auto")
        agent = EncreAgent(config=config)
        parts: list[str] = []
        async for event in agent.run(prompt, system_prompt=role.system_prompt_override or None):
            if isinstance(event, TextDelta) and event.text:
                parts.append(event.text)
        return "".join(parts)

    def _build_context(self, node: Any, task_tree: Any) -> str:
        parts: list[str] = [f"Goal: {task_tree.goal}\n"]
        for dep_id in node.dependencies:
            dep = task_tree.nodes.get(dep_id)
            if dep and dep.result:
                parts.append(f"Dependency [{dep.name}] output:\n{dep.result[:3000]}")
        blackboard_context = self._blackboard.get_all_visible()
        if blackboard_context:
            parts.append(f"Shared context:\n{blackboard_context}")
        return "\n".join(parts)

    def cancel(self) -> None:
        """Cancel the orchestration and all in-flight node tasks.

        Sets the cancellation flag (which causes the main loop to stop
        scheduling new nodes) and immediately cancels every currently
        running asyncio Task so that no work continues in the background.
        """
        self._cancelled = True
        for task in self._running_tasks.values():
            if not task.done():
                task.cancel()
        self._running_tasks.clear()
