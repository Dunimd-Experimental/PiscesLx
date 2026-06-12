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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from encre.logging_config import get_logger
from encre.swarm.blackboard import EncreBlackboard
from encre.swarm.consensus import EncreConsensus, Proposal, ConsensusResult
from encre.swarm.orchestrator import EncreOrchestrator, OrchestrationEvent
from encre.swarm.planner import EncreTaskPlanner, TaskTree
from encre.swarm.roles import RoleRegistry, ROLE_ARCHITECT, ROLE_CODER, ROLE_REVIEWER, ROLE_TESTER, ROLE_RESEARCHER, ROLE_DEBUGGER, ROLE_GENERAL
from encre.swarm.teammate import EncreTeammate, TeammateHandle
from encre.utils.types import AgentEvent, TextDelta, ToolResult, Finish, create_text_delta

logger = get_logger("encre.swarm")


@dataclass
class SwarmEvent:
    """Events emitted during swarm execution."""
    type: str  # planning | task_started | task_completed | task_failed | consensus | team_finished | error
    task_id: str = ""
    task_name: str = ""
    role: str = ""
    result: str = ""
    error: str = ""
    progress: float = 0.0
    consensus: ConsensusResult | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmResult:
    """Final result of a swarm execution."""
    goal: str
    task_tree: TaskTree | None = None
    results: dict[str, str] = field(default_factory=dict)  # task_id → result
    consensus: ConsensusResult | None = None
    blackboard: dict[str, Any] = field(default_factory=dict)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    elapsed_seconds: float = 0.0
    summary: str = ""


class EncreSwarmSession:
    """High-level swarm execution session attached to a EncreAgent.

    Usage:
        agent = EncreAgent(config)
        session = EncreSwarmSession(agent, goal="Build a TODO API")
        result = await session.execute()
    """

    DEFAULT_ROLES = [
        ROLE_ARCHITECT,
        ROLE_CODER,
        ROLE_REVIEWER,
        ROLE_TESTER,
        ROLE_DEBUGGER,
    ]

    def __init__(
        self,
        agent: Any,
        goal: str = "",
        max_concurrent: int = 5,
        enable_reviewer: bool = True,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self._agent = agent
        self._goal = goal
        self._max_concurrent = max_concurrent
        self._enable_reviewer = enable_reviewer
        self._timeout = timeout_seconds

    async def execute(
        self,
        goal: str = "",
        max_concurrent: int = 0,
        on_event: Callable[[SwarmEvent], None] | None = None,
    ) -> SwarmResult:
        goal = goal or self._goal
        max_concurrent = max_concurrent or self._max_concurrent

        if not goal:
            raise ValueError("goal is required for swarm execution")

        start_time = time.time()
        result = SwarmResult(goal=goal)

        # Phase 1: Planning — decompose goal into task tree
        if on_event:
            on_event(SwarmEvent(type="planning", progress=0.0))

        planner = EncreTaskPlanner()
        task_tree = await planner.decompose(goal)
        result.task_tree = task_tree

        # Phase 2: Execution with orchestrator
        blackboard = EncreBlackboard()
        role_registry = RoleRegistry()
        for role in self.DEFAULT_ROLES:
            role_registry.register(role)

        orchestrator = EncreOrchestrator(
            role_registry=role_registry,
            blackboard=blackboard,
            max_concurrent=max_concurrent,
            enable_reviewer_gate=self._enable_reviewer,
        )

        completed = 0
        failed = 0
        try:
            async for orch_event in orchestrator.execute(task_tree):
                if on_event:
                    on_event(SwarmEvent(
                        type=orch_event.type,
                        task_id=orch_event.task_id,
                        task_name=orch_event.task_name,
                        role=orch_event.role,
                        result=orch_event.result,
                        error=orch_event.error,
                        progress=orch_event.progress,
                        timestamp=orch_event.timestamp,
                    ))

                if orch_event.type == "task_completed":
                    completed += 1
                    result.results[orch_event.task_id] = orch_event.result
                elif orch_event.type == "task_failed":
                    failed += 1
                    result.results[orch_event.task_id] = orch_event.error

        except asyncio.TimeoutError:
            orchestrator.cancel()
            result.summary = f"Swarm timed out after {self._timeout}s"
        except Exception as e:
            orchestrator.cancel()
            result.summary = f"Swarm error: {e}"
            if on_event:
                on_event(SwarmEvent(type="error", error=str(e)))

        # Phase 3: Consensus (optional — if multiple agents produced output)
        if len(result.results) >= 2:
            consensus = EncreConsensus()
            # First result is the primary proposal, rest vote on it
            results_list = list(result.results.items())
            primary_id, primary_output = results_list[0]
            proposal = consensus.create_proposal(
                title=f"Swarm result for: {goal[:100]}",
                description=primary_output[:2000],
                options=["APPROVED", "NEEDS_REVISION"],
                proposed_by=primary_id,
            )
            # Collect votes from other results as simple yes/no
            for task_id, output in results_list[1:]:
                choice = "APPROVED" if "error" not in output.lower() else "NEEDS_REVISION"
                consensus.cast_vote(
                    proposal_id=proposal.id,
                    voter_id=task_id,
                    choice=choice,
                    reasoning=output[:500],
                )
            consensus_result = consensus.tally(proposal)
            result.consensus = consensus_result
            if on_event:
                on_event(SwarmEvent(type="consensus", consensus=consensus_result))

        # Collect blackboard state
        result.blackboard = blackboard.get_all("__orchestrator__") or {}

        result.total_tasks = len(task_tree.nodes) if task_tree else 0
        result.completed_tasks = completed
        result.failed_tasks = failed
        result.elapsed_seconds = time.time() - start_time

        if not result.summary:
            result.summary = (
                f"Swarm completed: {completed}/{result.total_tasks} tasks succeeded, "
                f"{failed} failed in {result.elapsed_seconds:.1f}s"
            )

        if on_event:
            on_event(SwarmEvent(
                type="team_finished",
                progress=1.0,
                result=result.summary,
            ))

        return result

    async def execute_streaming(
        self,
        goal: str = "",
    ) -> AsyncGenerator[SwarmEvent, None]:
        """Execute swarm and yield events as they happen."""
        events: list[SwarmEvent] = []

        def collect(event: SwarmEvent) -> None:
            events.append(event)

        result = await self.execute(goal=goal, on_event=collect)
        for event in events:
            yield event
        # Yield final result as last event
        yield SwarmEvent(
            type="team_finished",
            progress=1.0,
            result=result.summary,
        )
