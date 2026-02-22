#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

"""
Agent Orchestrator - Operator-based Multi-Agent Orchestration

This module provides orchestrator classes that integrate with the OPSC operator
infrastructure for unified multi-agent coordination and execution.

Key Components:
    - POPSSOrchestrationStrategy: Orchestration strategy enumeration
    - POPSSOrchestrationPlan: Orchestration plan container
    - POPSSOrchestrationStage: Stage definition for plans
    - POPSSOrchestrationResult: Orchestration result container
    - POPSSOrchestratorConfig: Orchestrator configuration
    - POPSSBaseOrchestrator: Base orchestrator class
    - POPSSDynamicOrchestrator: Dynamic planning orchestrator

Design Principles:
    1. Operator Executor Integration: Use PiscesLxOperatorExecutor for execution
    2. Unified Metrics: Track orchestration metrics via OPSC
    3. Async-Native: Full async support for concurrent agent execution
    4. Event-Driven: Callback system for monitoring orchestration events

Usage:
    config = POPSSOrchestratorConfig(registry=agent_registry)
    orchestrator = POPSSDynamicOrchestrator(config)

    result = await orchestrator.orchestrate("Analyze this data and generate a report")
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from utils.dc import PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing
from utils.opsc.executor import PiscesLxOperatorExecutor
from utils.opsc.interface import PiscesLxOperatorStatus

from .base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentState,
    POPSSAgentCapability,
)
from .registry import POPSSAggregentRegistry, POPSSAggregentType

T = TypeVar('T')


class POPSSOrchestrationStrategy(Enum):
    """
    Orchestration strategy enumeration.

    Strategies:
        SEQUENTIAL: Execute agents one after another
        PARALLEL: Execute agents concurrently
        PIPELINE: Chain agents with data flow
        HIERARCHICAL: Tree-based agent hierarchy
        DYNAMIC: Adaptive strategy based on task
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HIERARCHICAL = "hierarchical"
    DYNAMIC = "dynamic"


@dataclass
class POPSSOrchestrationPlan:
    """
    Orchestration plan container.

    Attributes:
        plan_id: Unique plan identifier
        task: Original task description
        stages: List of orchestration stages
        selected_agents: List of selected agent IDs
        execution_order: Ordered list of agent groups for execution
        dependencies: Stage dependency mapping
        created_at: Creation timestamp
        metadata: Additional metadata
    """
    plan_id: str
    task: str

    stages: List['POPSSOrchestrationStage'] = field(default_factory=list)

    selected_agents: List[str] = field(default_factory=list)
    execution_order: List[List[str]] = field(default_factory=list)

    dependencies: Dict[str, List[str]] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSOrchestrationStage:
    """
    Orchestration stage definition.

    Attributes:
        stage_id: Unique stage identifier
        name: Stage name
        description: Stage description
        agent_ids: Agents assigned to this stage
        strategy: Execution strategy for this stage
        input_requirements: Required inputs
        output_specification: Expected outputs
        parallel_limit: Maximum parallel agents
        timeout_seconds: Stage timeout
        fallback_agents: Fallback agents on failure
        retry_count: Number of retries
    """
    stage_id: str
    name: str
    description: str

    agent_ids: List[str] = field(default_factory=list)
    strategy: POPSSOrchestrationStrategy = POPSSOrchestrationStrategy.SEQUENTIAL

    input_requirements: Dict[str, Any] = field(default_factory=dict)
    output_specification: Dict[str, Any] = field(default_factory=dict)

    parallel_limit: int = 3
    timeout_seconds: float = 300.0

    fallback_agents: List[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class POPSSOrchestrationResult:
    """
    Orchestration result container.

    Attributes:
        success: Whether orchestration succeeded
        plan_id: Associated plan ID
        stage_results: Results from each stage
        aggregated_output: Aggregated output from all agents
        error: Error message if failed
        total_execution_time: Total execution time in seconds
        agent_executions: List of individual agent executions
        metadata: Additional metadata
    """
    success: bool
    plan_id: str

    stage_results: Dict[str, POPSSAgentResult] = field(default_factory=dict)
    aggregated_output: Optional[str] = None
    error: Optional[str] = None

    total_execution_time: float = 0.0
    agent_executions: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSOrchestratorConfig:
    """
    Orchestrator configuration.

    Attributes:
        registry: Agent registry to use
        default_strategy: Default orchestration strategy
        max_parallel_agents: Maximum concurrent agents
        default_timeout: Default timeout in seconds
        enable_adaptive_planning: Enable adaptive planning
        enable_result_aggregation: Enable result aggregation
        enable_error_recovery: Enable error recovery
        planning_model: Model for planning decisions
        planning_temperature: Temperature for planning
    """
    registry: POPSSAggregentRegistry

    default_strategy: POPSSOrchestrationStrategy = POPSSOrchestrationStrategy.DYNAMIC
    max_parallel_agents: int = 5
    default_timeout: float = 600.0

    enable_adaptive_planning: bool = True
    enable_result_aggregation: bool = True
    enable_error_recovery: bool = True

    planning_model: Optional[str] = None
    planning_temperature: float = 0.3


class POPSSBaseOrchestrator(ABC):
    """
    Base orchestrator class integrating with OPSC infrastructure.

    This class provides the foundation for multi-agent orchestration,
    integrating with PiscesLxOperatorExecutor for unified execution.

    Key Features:
        - OPSC executor integration for agent execution
        - Unified metrics and tracing
        - Callback system for event monitoring
        - Async-native design

    Attributes:
        config: Orchestrator configuration
        registry: Agent registry
        executor: OPSC operator executor
        logger: Logger instance
        _metrics: Metrics collector
        _tracing: Tracing system
    """

    def __init__(self, config: POPSSOrchestratorConfig):
        """
        Initialize orchestrator with configuration.

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.registry = config.registry

        self._LOG = self._configure_logging()
        self._metrics = PiscesLxMetrics()
        self._tracing = PiscesLxTracing()

        self.executor = PiscesLxOperatorExecutor(
            registry=self.registry._opsc_registry,
            max_workers=config.max_parallel_agents
        )

        self._execution_history: List[Dict[str, Any]] = []
        self._orchestration_metrics: Dict[str, Any] = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'total_execution_time': 0.0,
        }

        self._callbacks: Dict[str, List[Callable]] = {
            'on_plan_created': [],
            'on_stage_start': [],
            'on_stage_complete': [],
            'on_agent_start': [],
            'on_agent_complete': [],
            'on_error': [],
            'on_complete': [],
        }

        self._LOG.info("orchestrator_initialized",
                        max_parallel=config.max_parallel_agents,
                        strategy=config.default_strategy.value)

    def _configure_logging(self) -> PiscesLxLogger:
        """Configure logging for orchestrator."""
        return get_logger("POPSSOrchestrator")

    @abstractmethod
    async def create_plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> POPSSOrchestrationPlan:
        """Create an orchestration plan for the task."""
        pass

    @abstractmethod
    async def execute_plan(self, plan: POPSSOrchestrationPlan) -> POPSSOrchestrationResult:
        """Execute an orchestration plan."""
        pass

    @abstractmethod
    async def select_agents(self, task: str, requirements: Dict[str, Any]) -> List[str]:
        """Select agents for the task."""
        pass

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit_callback(self, event: str, data: Any) -> None:
        """Emit a callback event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self._LOG.error("callback_error", event=event, error=str(e))

    async def orchestrate(self, task: str, **kwargs) -> POPSSOrchestrationResult:
        """
        Orchestrate agents to complete a task.

        Args:
            task: Task description
            **kwargs: Additional arguments

        Returns:
            Orchestration result
        """
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        span = self._tracing.start_span(
            f"orchestrate_{plan_id}",
            attributes={"task": task[:100], "plan_id": plan_id}
        )

        self._LOG.info("orchestration_started", plan_id=plan_id, task=task[:100])

        context = kwargs.get('context', {})

        plan = await self.create_plan(task, context)
        plan.plan_id = plan_id

        self._emit_callback('on_plan_created', {
            'plan_id': plan_id,
            'task': task,
            'plan': plan,
        })

        result = await self.execute_plan(plan)
        result.plan_id = plan_id

        self._orchestration_metrics['total_plans'] += 1
        self._orchestration_metrics['total_execution_time'] += result.total_execution_time

        if result.success:
            self._orchestration_metrics['successful_plans'] += 1
            self._tracing.end_span(span, status="ok")
        else:
            self._orchestration_metrics['failed_plans'] += 1
            self._tracing.end_span(span, status="error", error_message=result.error)

        self._metrics.counter("orchestration_total")
        self._metrics.gauge("orchestration_success_rate",
                           self._orchestration_metrics['successful_plans'] / max(self._orchestration_metrics['total_plans'], 1))

        self._emit_callback('on_complete', {
            'plan_id': plan_id,
            'result': result,
        })

        self._LOG.info("orchestration_completed",
                        plan_id=plan_id,
                        success=result.success,
                        execution_time=result.total_execution_time)

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        total = self._orchestration_metrics['total_plans']
        return {
            'total_plans': total,
            'successful_plans': self._orchestration_metrics['successful_plans'],
            'failed_plans': self._orchestration_metrics['failed_plans'],
            'success_rate': self._orchestration_metrics['successful_plans'] / max(total, 1),
            'total_execution_time': self._orchestration_metrics['total_execution_time'],
            'average_execution_time': self._orchestration_metrics['total_execution_time'] / max(total, 1),
            'executor_stats': self.executor.get_stats(),
        }

    def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        self.executor.shutdown(wait=True)
        self._LOG.info("orchestrator_shutdown")


class POPSSDynamicOrchestrator(POPSSBaseOrchestrator):
    """
    Dynamic orchestrator with adaptive planning.

    This orchestrator dynamically creates execution plans based on task
    analysis, selecting appropriate agents and execution strategies.

    Key Features:
        - Adaptive planning based on task analysis
        - Capability-based agent selection
        - Multi-stage execution with dependencies
        - Fallback and retry support
    """

    def __init__(self, config: POPSSOrchestratorConfig):
        """Initialize dynamic orchestrator."""
        super().__init__(config)

        self._agent_selectors: Dict[str, Callable] = {}
        self._stage_templates: Dict[str, POPSSOrchestrationStage] = {}

        self._LOG.info("dynamic_orchestrator_initialized")

    def register_agent_selector(self, capability: POPSSAgentCapability, selector: Callable) -> None:
        """Register a custom agent selector for a capability."""
        self._agent_selectors[capability] = selector

    def create_stage_template(self, name: str, stage: POPSSOrchestrationStage) -> None:
        """Create a reusable stage template."""
        self._stage_templates[name] = stage

    async def create_plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> POPSSOrchestrationPlan:
        """
        Create a dynamic orchestration plan.

        Args:
            task: Task description
            context: Optional execution context

        Returns:
            Orchestration plan
        """
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"

        requirements = self._analyze_requirements(task, context)

        stages = await self._design_stages(task, requirements)

        selected_agents = await self.select_agents(task, requirements)

        execution_order = self._plan_execution_order(stages, selected_agents)

        dependencies = self._build_dependencies(stages)

        for i, stage in enumerate(stages):
            stage.agent_ids = execution_order[i] if i < len(execution_order) else []

        plan = POPSSOrchestrationPlan(
            plan_id=plan_id,
            task=task,
            stages=stages,
            selected_agents=selected_agents,
            execution_order=execution_order,
            dependencies=dependencies,
            metadata={
                'requirements': requirements,
                'context': context or {},
            }
        )

        self._LOG.info("plan_created",
                        plan_id=plan_id,
                        stages=len(stages),
                        agents=len(selected_agents))

        return plan

    def _analyze_requirements(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze task requirements."""
        task_lower = task.lower()

        requirements = {
            'capabilities': set(),
            'complexity': 'medium',
            'domains': [],
            'special_requirements': {},
        }

        capability_keywords = {
            POPSSAgentCapability.CODE_GENERATION: ['code', 'program', 'function', 'implement', 'script'],
            POPSSAgentCapability.FILE_OPERATIONS: ['file', 'read', 'write', 'edit', 'save'],
            POPSSAgentCapability.WEB_SEARCH: ['search', 'find', 'web', 'internet', 'lookup'],
            POPSSAgentCapability.DATA_ANALYSIS: ['analyze', 'data', 'statistics', 'chart', 'metrics'],
            POPSSAgentCapability.REASONING: ['reason', 'think', 'logic', 'solve', 'deduce'],
            POPSSAgentCapability.PLANNING: ['plan', 'schedule', 'organize', 'strategy', 'roadmap'],
            POPSSAgentCapability.CREATIVE_WRITING: ['write', 'creative', 'story', 'poem', 'article'],
            POPSSAgentCapability.RESEARCH: ['research', 'study', 'investigate', 'explore', 'survey'],
        }

        for capability, keywords in capability_keywords.items():
            if any(kw in task_lower for kw in keywords):
                requirements['capabilities'].add(capability)

        if len(requirements['capabilities']) >= 3:
            requirements['complexity'] = 'high'
        elif len(requirements['capabilities']) == 0:
            requirements['complexity'] = 'low'

        return requirements

    async def _design_stages(
        self,
        task: str,
        requirements: Dict[str, Any]
    ) -> List[POPSSOrchestrationStage]:
        """Design execution stages based on requirements."""
        stages = []

        stage_counter = 0

        if POPSSAgentCapability.PLANNING in requirements['capabilities']:
            planning_stage = POPSSOrchestrationStage(
                stage_id=f"stage_{stage_counter}",
                name="Planning",
                description="Create execution plan",
                agent_ids=[],
                strategy=POPSSOrchestrationStrategy.SEQUENTIAL,
                timeout_seconds=60.0,
            )
            stages.append(planning_stage)
            stage_counter += 1

        main_capabilities = [c for c in requirements['capabilities']
                          if c not in [POPSSAgentCapability.PLANNING]]

        if len(main_capabilities) <= 2:
            main_stage = POPSSOrchestrationStage(
                stage_id=f"stage_{stage_counter}",
                name="Main Execution",
                description="Execute main task",
                agent_ids=[],
                strategy=POPSSOrchestrationStrategy.SEQUENTIAL,
                parallel_limit=min(len(main_capabilities), self.config.max_parallel_agents),
                timeout_seconds=self.config.default_timeout,
            )
            stages.append(main_stage)
            stage_counter += 1
        else:
            main_stage = POPSSOrchestrationStage(
                stage_id=f"stage_{stage_counter}",
                name="Parallel Execution",
                description="Execute tasks in parallel",
                agent_ids=[],
                strategy=POPSSOrchestrationStrategy.PARALLEL,
                parallel_limit=min(len(main_capabilities), self.config.max_parallel_agents),
                timeout_seconds=self.config.default_timeout,
            )
            stages.append(main_stage)
            stage_counter += 1

        if POPSSAgentCapability.DATA_ANALYSIS in requirements['capabilities']:
            analysis_stage = POPSSOrchestrationStage(
                stage_id=f"stage_{stage_counter}",
                name="Analysis",
                description="Analyze results",
                agent_ids=[],
                strategy=POPSSOrchestrationStrategy.SEQUENTIAL,
                timeout_seconds=120.0,
            )
            stages.append(analysis_stage)

        return stages

    async def select_agents(self, task: str, requirements: Dict[str, Any]) -> List[str]:
        """Select agents based on task requirements."""
        selected = []

        capability_agents: Dict[POPSSAgentCapability, List[str]] = {}

        all_agents = self.registry.list_agents(enabled_only=True)

        for agent_id in all_agents:
            metadata = self.registry.get_metadata(agent_id)
            if metadata:
                for capability in metadata.capabilities:
                    if capability not in capability_agents:
                        capability_agents[capability] = []
                    capability_agents[capability].append(agent_id)

        selected_capabilities: Set[POPSSAgentCapability] = set()

        for capability in requirements.get('capabilities', []):
            if capability in capability_agents:
                agents = capability_agents[capability]
                best_agent = max(
                    agents,
                    key=lambda aid: self.registry.get_metadata(aid).success_rate
                    if self.registry.get_metadata(aid) else 0
                )
                selected.append(best_agent)
                selected_capabilities.add(capability)

        if not selected:
            general_agents = self.registry.list_agents(
                agent_type=POPSSAggregentType.GENERAL,
                enabled_only=True
            )
            if general_agents:
                selected.append(general_agents[0])

        return list(set(selected))

    def _plan_execution_order(
        self,
        stages: List[POPSSOrchestrationStage],
        agents: List[str]
    ) -> List[List[str]]:
        """Plan the execution order of agents across stages."""
        if len(stages) <= 1:
            return [agents]

        execution_order = []

        for stage in stages:
            stage_agents = []

            for agent_id in agents:
                metadata = self.registry.get_metadata(agent_id)
                if metadata and metadata.agent_type in [
                    POPSSAggregentType.GENERAL,
                    POPSSAggregentType.CODE,
                    POPSSAggregentType.RESEARCH,
                ]:
                    stage_agents.append(agent_id)

            if not stage_agents and agents:
                stage_agents = agents[:stage.parallel_limit]

            execution_order.append(stage_agents)

        return execution_order

    def _build_dependencies(self, stages: List[POPSSOrchestrationStage]) -> Dict[str, List[str]]:
        """Build stage dependencies."""
        dependencies = {}

        for i, stage in enumerate(stages):
            if i == 0:
                dependencies[stage.stage_id] = []
            else:
                dependencies[stage.stage_id] = [stages[i - 1].stage_id]

        return dependencies

    async def execute_plan(self, plan: POPSSOrchestrationPlan) -> POPSSOrchestrationResult:
        """
        Execute an orchestration plan.

        Args:
            plan: Orchestration plan to execute

        Returns:
            Orchestration result
        """
        start_time = datetime.now()

        stage_results: Dict[str, POPSSAgentResult] = {}
        agent_executions: List[Dict[str, Any]] = []

        for stage in plan.stages:
            self._emit_callback('on_stage_start', {
                'plan_id': plan.plan_id,
                'stage_id': stage.stage_id,
                'name': stage.name,
            })

            if stage.strategy == POPSSOrchestrationStrategy.PARALLEL:
                result = await self._execute_stage_parallel(stage, plan, stage_results)
            elif stage.strategy == POPSSOrchestrationStrategy.PIPELINE:
                result = await self._execute_stage_pipeline(stage, plan, stage_results)
            else:
                result = await self._execute_stage_sequential(stage, plan, stage_results)

            for agent_id, agent_result in result.items():
                stage_results[f"{stage.stage_id}_{agent_id}"] = agent_result
                agent_executions.append({
                    'agent_id': agent_id,
                    'stage_id': stage.stage_id,
                    'success': agent_result.success,
                    'execution_time': agent_result.execution_time,
                })

            self._emit_callback('on_stage_complete', {
                'plan_id': plan.plan_id,
                'stage_id': stage.stage_id,
                'results': result,
            })

        total_execution_time = (datetime.now() - start_time).total_seconds()

        success = len([r for r in stage_results.values() if r.success]) == len(stage_results)

        aggregated_output = None
        if self.config.enable_result_aggregation and stage_results:
            outputs = [r.output for r in stage_results.values() if r.output]
            if outputs:
                aggregated_output = "\n\n---\n\n".join(outputs)

        return POPSSOrchestrationResult(
            success=success,
            plan_id=plan.plan_id,
            stage_results=stage_results,
            aggregated_output=aggregated_output,
            total_execution_time=total_execution_time,
            agent_executions=agent_executions,
        )

    async def _execute_stage_sequential(
        self,
        stage: POPSSOrchestrationStage,
        plan: POPSSOrchestrationPlan,
        previous_results: Dict[str, POPSSAgentResult]
    ) -> Dict[str, POPSSAgentResult]:
        """Execute stage sequentially."""
        results = {}

        for agent_id in stage.agent_ids:
            agent = self.registry.get_agent(agent_id)
            if agent:
                self._emit_callback('on_agent_start', {
                    'plan_id': plan.plan_id,
                    'stage_id': stage.stage_id,
                    'agent_id': agent_id,
                })

                task_input = self._prepare_task_input(stage, plan, previous_results)

                result = await agent.execute_async({'task': task_input}, context={
                    'plan_id': plan.plan_id,
                    'stage_id': stage.stage_id,
                })

                results[agent_id] = result

                self.registry.update_stats(agent_id, result.success, result.execution_time)

                self._emit_callback('on_agent_complete', {
                    'plan_id': plan.plan_id,
                    'stage_id': stage.stage_id,
                    'agent_id': agent_id,
                    'success': result.success,
                })

                if not result.success and stage.fallback_agents:
                    for fallback_id in stage.fallback_agents:
                        fallback_agent = self.registry.get_agent(fallback_id)
                        if fallback_agent:
                            result = await fallback_agent.execute_async({'task': task_input})
                            results[fallback_id] = result
                            if result.success:
                                break

        return results

    async def _execute_stage_parallel(
        self,
        stage: POPSSOrchestrationStage,
        plan: POPSSOrchestrationPlan,
        previous_results: Dict[str, POPSSAgentResult]
    ) -> Dict[str, POPSSAgentResult]:
        """Execute stage in parallel."""
        results = {}

        semaphore = asyncio.Semaphore(stage.parallel_limit)

        async def execute_with_limit(agent_id: str) -> None:
            async with semaphore:
                agent = self.registry.get_agent(agent_id)
                if agent:
                    self._emit_callback('on_agent_start', {
                        'plan_id': plan.plan_id,
                        'stage_id': stage.stage_id,
                        'agent_id': agent_id,
                    })

                    task_input = self._prepare_task_input(stage, plan, previous_results)

                    result = await agent.execute_async({'task': task_input})

                    results[agent_id] = result

                    self.registry.update_stats(agent_id, result.success, result.execution_time)

                    self._emit_callback('on_agent_complete', {
                        'plan_id': plan.plan_id,
                        'stage_id': stage.stage_id,
                        'agent_id': agent_id,
                        'success': result.success,
                    })

        tasks = [execute_with_limit(agent_id) for agent_id in stage.agent_ids]
        await asyncio.gather(*tasks)

        return results

    async def _execute_stage_pipeline(
        self,
        stage: POPSSOrchestrationStage,
        plan: POPSSOrchestrationPlan,
        previous_results: Dict[str, POPSSAgentResult]
    ) -> Dict[str, POPSSAgentResult]:
        """Execute stage as a pipeline."""
        results = {}
        pipeline_output = None

        for agent_id in stage.agent_ids:
            agent = self.registry.get_agent(agent_id)
            if agent:
                task_input = self._prepare_task_input(stage, plan, previous_results)

                if pipeline_output:
                    task_input += f"\n\nPrevious pipeline output:\n{pipeline_output}"

                result = await agent.execute_async({'task': task_input})

                results[agent_id] = result
                pipeline_output = result.output if result.success else None

                self.registry.update_stats(agent_id, result.success, result.execution_time)

        return results

    def _prepare_task_input(
        self,
        stage: POPSSOrchestrationStage,
        plan: POPSSOrchestrationPlan,
        previous_results: Dict[str, POPSSAgentResult]
    ) -> str:
        """Prepare task input for an agent."""
        task = plan.task

        previous_outputs = []
        for key, result in previous_results.items():
            if result.output:
                previous_outputs.append(result.output)

        if previous_outputs:
            task += f"\n\nConsider the following previous results:\n" + "\n---\n".join(previous_outputs)

        if stage.description:
            task += f"\n\nStage: {stage.name}\n{stage.description}"

        return task

    def shutdown(self) -> None:
        """Shutdown the dynamic orchestrator."""
        super().shutdown()
        self._LOG.info("dynamic_orchestrator_shutdown")
