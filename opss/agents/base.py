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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Agent Base Classes - Operator-based Agent Implementation

This module provides agent base classes that integrate with the OPSC operator
infrastructure, enabling unified execution, monitoring, and resource management.

Key Components:
    - POPSSAgentState: Agent state enumeration
    - POPSSAgentCapability: Agent capability enumeration
    - POPSSAgentConfig: Agent configuration (extends PiscesLxOperatorConfig)
    - POPSSAgentContext: Agent execution context
    - POPSSAgentResult: Agent execution result (extends PiscesLxOperatorResult)
    - POPSSAgentThought: Agent thought/reasoning step
    - POPSSBaseAgent: Base agent class (extends PiscesLxOperatorInterface)
    - POPSSPromptBasedAgent: Prompt-driven agent implementation

Design Principles:
    1. Operator Integration: Agents are operators with reasoning capabilities
    2. Unified Infrastructure: Use OPSC for execution, metrics, tracing
    3. Async Support: Native async execution for agent operations
    4. Callback System: Event-driven architecture for monitoring
    5. Prompt Mode: Support YAML-based prompt configuration

Execution Modes:
    - code: Pure Python implementation (default)
    - prompt: Prompt-driven execution via YAML files
    - hybrid: Combination of code and prompt modes

Usage:
    # Code mode (traditional)
    class MyAgent(POPSSBaseAgent):
        @property
        def name(self) -> str:
            return "my_agent"

        async def think(self, context) -> List[POPSSAgentThought]:
            return [POPSSAgentThought(...)]

    # Prompt mode (YAML-driven)
    class MyPromptAgent(POPSSBaseAgent):
        expert_type = "code_reviewer"
        mode = "prompt"
"""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union

from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)
from utils.opsc.base import PiscesLxBaseOperator

try:
    from .loader import POPSSPromptLoader, POPSSPromptConfig
    PROMPT_LOADER_AVAILABLE = True
except ImportError:
    PROMPT_LOADER_AVAILABLE = False
    POPSSPromptLoader = None
    POPSSPromptConfig = None


class POPSSAgentMode(Enum):
    """
    Agent execution mode enumeration.
    
    Modes:
        CODE: Pure Python implementation
        PROMPT: Prompt-driven execution via YAML files
        HYBRID: Combination of code and prompt modes
    """
    CODE = "code"
    PROMPT = "prompt"
    HYBRID = "hybrid"


class POPSSAgentState(Enum):
    """
    Agent state enumeration for lifecycle tracking.

    States:
        IDLE: Agent is idle and ready for tasks
        REASONING: Agent is thinking/reasoning
        EXECUTING: Agent is executing actions
        WAITING: Agent is waiting for external input
        COMPLETED: Agent has completed successfully
        ERROR: Agent encountered an error
        TERMINATED: Agent has been terminated
    """
    IDLE = "idle"
    REASONING = "reasoning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINATED = "terminated"


class POPSSAgentCapability(Enum):
    """
    Agent capability enumeration for task matching.

    Capabilities:
        CODE_GENERATION: Generate code
        CODE_EXECUTION: Execute code
        FILE_OPERATIONS: File read/write operations
        WEB_SEARCH: Web search capabilities
        DATA_ANALYSIS: Data analysis and visualization
        TEXT_GENERATION: Generate text content
        TRANSLATION: Language translation
        SUMMARIZATION: Text summarization
        REASONING: Logical reasoning
        PLANNING: Task planning
        CREATIVE_WRITING: Creative content generation
        RESEARCH: Research and investigation
        TOOL_USE: Tool usage capabilities
    """
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    TEXT_GENERATION = "text_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    PLANNING = "planning"
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
    TOOL_USE = "tool_use"


@dataclass
class POPSSAgentConfig(PiscesLxOperatorConfig):
    """
    Agent configuration extending operator configuration.

    Attributes:
        agent_id: Unique agent identifier
        capabilities: Set of agent capabilities
        system_prompt: System prompt for the agent
        max_iterations: Maximum reasoning iterations
        memory_limit_mb: Memory limit in MB
        enable_thinking: Enable thinking/reasoning
        enable_self_reflection: Enable self-reflection
        enable_learning: Enable learning from feedback
        tools: List of available tools
        allowed_domains: Allowed domains for operations
        blocked_domains: Blocked domains
        model_name: Model name for LLM operations
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        mode: Execution mode (code/prompt/hybrid)
        expert_type: Expert type identifier for prompt loading
        behavior_prompt: Behavior prompt template
        output_schema: Expected output schema
    """
    agent_id: str = ""
    capabilities: Set[POPSSAgentCapability] = field(default_factory=set)
    system_prompt: str = ""
    max_iterations: int = 10
    memory_limit_mb: int = 512
    enable_thinking: bool = True
    enable_self_reflection: bool = True
    enable_learning: bool = False
    tools: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    mode: str = "code"
    expert_type: str = ""
    behavior_prompt: str = ""
    output_schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        if not self.name:
            self.name = self.agent_id
        
        if self.expert_type and PROMPT_LOADER_AVAILABLE:
            self._load_from_prompt()
    
    def _load_from_prompt(self):
        """Load configuration from prompt file."""
        try:
            prompt_config = POPSSPromptLoader.load(self.expert_type)
            if not self.system_prompt:
                self.system_prompt = prompt_config.system_prompt
            if not self.behavior_prompt:
                self.behavior_prompt = prompt_config.behavior_prompt
            if not self.output_schema:
                self.output_schema = prompt_config.output_schema
        except FileNotFoundError:
            pass


@dataclass
class POPSSAgentContext:
    """
    Agent execution context for tracking state.

    Attributes:
        context_id: Unique context identifier
        task_id: Task identifier
        user_request: Original user request
        session_id: Session identifier
        parent_context_id: Parent context for nested calls
        state: Current agent state
        history: Execution history
        memory: Agent memory storage
        artifacts: Generated artifacts
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
    """
    context_id: str
    task_id: str
    user_request: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_context_id: Optional[str] = None
    state: POPSSAgentState = POPSSAgentState.IDLE
    history: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSAgentResult(PiscesLxOperatorResult):
    """
    Agent execution result extending operator result.

    Attributes:
        reasoning_steps: List of reasoning steps taken
        tool_calls: List of tool calls made
        token_usage: Total token usage
        artifacts: Generated artifacts
    """
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_operator_result(
        cls,
        op_result: PiscesLxOperatorResult,
        reasoning_steps: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> 'POPSSAgentResult':
        """Create agent result from operator result."""
        return cls(
            operator_name=op_result.operator_name,
            status=op_result.status,
            output=op_result.output,
            error=op_result.error,
            execution_time=op_result.execution_time,
            metadata=op_result.metadata,
            reasoning_steps=reasoning_steps or [],
            tool_calls=tool_calls or []
        )


@dataclass
class POPSSAgentThought:
    """
    Agent thought/reasoning step.

    Attributes:
        thought_id: Unique thought identifier
        agent_id: Agent that produced this thought
        context_id: Context this thought belongs to
        thought_type: Type of thought
        content: Thought content
        confidence: Confidence level (0-1)
        reasoning: Reasoning chain
        created_at: Creation timestamp
    """
    thought_id: str
    agent_id: str
    context_id: str
    thought_type: str
    content: str
    confidence: float = 1.0
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class POPSSBaseAgent(PiscesLxOperatorInterface, ABC):
    """
    Base agent class integrating with OPSC operator infrastructure.

    This class extends PiscesLxOperatorInterface to provide agent-specific
    functionality including reasoning, action execution, and observation.

    Key Features:
        - Operator-based execution with metrics and tracing
        - Async-native design for concurrent operations
        - Callback system for event monitoring
        - State management for lifecycle tracking
        - Resource management through OPSC
        - Prompt mode support for YAML-driven agents

    Attributes:
        agent_id: Unique agent identifier
        state: Current agent state
        current_context: Current execution context
        _callbacks: Event callbacks dictionary
        expert_type: Expert type identifier (for prompt mode)
        mode: Execution mode (code/prompt/hybrid)

    Lifecycle:
        1. setup() - Initialize agent resources
        2. execute() - Run agent on a task
        3. teardown() - Clean up resources

    Execution Modes:
        - code: Pure Python implementation (abstract methods required)
        - prompt: Prompt-driven execution via YAML files
        - hybrid: Combination of code and prompt modes

    Usage:
        # Code mode (traditional)
        class MyAgent(POPSSBaseAgent):
            @property
            def name(self) -> str:
                return "my_agent"

            async def think(self, context) -> List[POPSSAgentThought]:
                return [POPSSAgentThought(...)]

        # Prompt mode (YAML-driven)
        class MyPromptAgent(POPSSBaseAgent):
            expert_type = "code_reviewer"
            mode = "prompt"
    """
    
    expert_type: str = ""
    mode: str = "code"

    def __init__(self, config: Optional[POPSSAgentConfig] = None):
        """
        Initialize agent with configuration.

        Args:
            config: Agent configuration
        """
        config = config or POPSSAgentConfig(name="base_agent")
        super().__init__(config)

        self.agent_id = config.agent_id
        self.state = POPSSAgentState.IDLE
        self.current_context: Optional[POPSSAgentContext] = None

        self._execution_count = 0
        self._success_count = 0
        self._total_execution_time = 0.0

        self._callbacks: Dict[str, List[Callable]] = {
            'on_thought': [],
            'on_action': [],
            'on_result': [],
            'on_error': [],
            'on_state_change': [],
        }
        
        self._prompt_config: Optional[Any] = None
        self._model_client: Optional[Any] = None
        
        if self.expert_type and self.mode in ["prompt", "hybrid"]:
            self._load_prompt_config()

        self._LOG.info("agent_initialized", agent_id=self.agent_id, name=self.name)
    
    def _load_prompt_config(self):
        """Load prompt configuration for prompt mode."""
        if not PROMPT_LOADER_AVAILABLE:
            self._LOG.warning("Prompt loader not available")
            return
        
        try:
            self._prompt_config = POPSSPromptLoader.load(self.expert_type)
            if hasattr(self.config, 'system_prompt') and not self.config.system_prompt:
                self.config.system_prompt = self._prompt_config.system_prompt
            if hasattr(self.config, 'behavior_prompt') and not self.config.behavior_prompt:
                self.config.behavior_prompt = self._prompt_config.behavior_prompt
            if hasattr(self.config, 'output_schema') and not self.config.output_schema:
                self.config.output_schema = self._prompt_config.output_schema
        except FileNotFoundError:
            self._LOG.warning(f"Prompt not found: {self.expert_type}")
    
    def set_model_client(self, client: Any):
        """Set the model client for prompt mode execution."""
        self._model_client = client
    
    def _get_mode(self) -> str:
        """Get effective execution mode."""
        if hasattr(self.config, 'mode') and self.config.mode:
            return self.config.mode
        return self.mode

    @property
    def capabilities(self) -> Set[POPSSAgentCapability]:
        """Get agent capabilities."""
        config = self.config
        if isinstance(config, POPSSAgentConfig):
            return config.capabilities.copy()
        return set()

    def has_capability(self, capability: POPSSAgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit_callback(self, event: str, data: Any) -> None:
        """Emit a callback event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(self, data)
                except Exception as e:
                    self._LOG.error("callback_error", event=event, error=str(e))

    def _update_state(self, new_state: POPSSAgentState) -> None:
        """Update agent state and emit callback."""
        old_state = self.state
        self.state = new_state
        self._emit_callback('on_state_change', {
            'agent_id': self.agent_id,
            'old_state': old_state.value,
            'new_state': new_state.value,
        })

    @property
    def version(self) -> str:
        """Get agent version."""
        return getattr(self.config, 'version', '1.0.0')

    @property
    def description(self) -> str:
        """Get agent description."""
        return getattr(self.config, 'description', self.__class__.__name__)

    @property
    def input_schema(self) -> Dict[str, Any]:
        """Get input schema for agent tasks."""
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task to execute"},
                "context": {"type": "object", "description": "Execution context"}
            },
            "required": ["task"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """Get output schema for agent results."""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "output": {"type": "string"},
                "reasoning_steps": {"type": "array"},
                "tool_calls": {"type": "array"}
            }
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate agent inputs."""
        return isinstance(inputs, dict) and 'task' in inputs

    @abstractmethod
    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        """
        Generate thoughts/reasoning for the task.

        Args:
            context: Execution context

        Returns:
            List of thoughts
        """
        mode = self._get_mode()
        
        if mode == "prompt":
            return await self._think_with_prompt(context)
        elif mode == "hybrid":
            return await self._think_hybrid(context)
        else:
            return await self._think_with_code(context)
    
    async def _think_with_code(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        """
        Generate thoughts using code implementation.
        
        Override this method in subclasses for code mode.
        """
        raise NotImplementedError("Agent must implement think() for code mode")
    
    async def _think_with_prompt(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        """
        Generate thoughts using prompt mode.
        
        This method uses the loaded prompt configuration to generate thoughts
        by calling the model with the system and behavior prompts.
        """
        if not self._prompt_config:
            self._load_prompt_config()
        
        if not self._prompt_config:
            raise RuntimeError(f"No prompt config loaded for {self.expert_type}")
        
        formatted_prompt = self._format_behavior_prompt(context)
        
        if self._model_client:
            response = await self._call_model(
                system_prompt=self._prompt_config.system_prompt,
                user_prompt=formatted_prompt
            )
            return self._parse_response_to_thoughts(response, context)
        else:
            thought = POPSSAgentThought(
                thought_id=f"thought_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent_id,
                context_id=context.context_id,
                thought_type="prompt_analysis",
                content=f"[Prompt Mode] {self.expert_type}: {context.user_request[:200]}",
                confidence=0.9,
                reasoning="Generated from prompt configuration"
            )
            return [thought]
    
    async def _think_hybrid(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        """
        Generate thoughts using hybrid mode.
        
        Combines prompt-based reasoning with code logic.
        """
        prompt_thoughts = await self._think_with_prompt(context)
        try:
            code_thoughts = await self._think_with_code(context)
            return prompt_thoughts + code_thoughts
        except NotImplementedError:
            return prompt_thoughts
    
    def _format_behavior_prompt(self, context: POPSSAgentContext) -> str:
        """Format behavior prompt with context variables."""
        if not self._prompt_config:
            return context.user_request
        
        behavior_prompt = self._prompt_config.behavior_prompt
        
        format_vars = {
            "user_request": context.user_request,
            "context_id": context.context_id,
            "task_id": context.task_id,
            **context.metadata
        }
        
        try:
            return behavior_prompt.format(**format_vars)
        except KeyError:
            return behavior_prompt
    
    async def _call_model(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> str:
        """Call the model with prompts."""
        if not self._model_client:
            raise RuntimeError("No model client configured")
        
        if hasattr(self._model_client, 'generate'):
            return await self._model_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        elif hasattr(self._model_client, 'chat'):
            return await self._model_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            raise RuntimeError("Model client does not support generate or chat")
    
    def _parse_response_to_thoughts(
        self, 
        response: str, 
        context: POPSSAgentContext
    ) -> List[POPSSAgentThought]:
        """Parse model response into thoughts."""
        thoughts = []
        
        thought = POPSSAgentThought(
            thought_id=f"thought_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            context_id=context.context_id,
            thought_type="model_response",
            content=response,
            confidence=0.9,
            reasoning="Generated from model response"
        )
        thoughts.append(thought)
        
        return thoughts

    @abstractmethod
    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        """
        Execute an action based on a thought.

        Args:
            thought: Thought to act on
            context: Execution context

        Returns:
            Action result
        """
        mode = self._get_mode()
        
        if mode == "prompt":
            return await self._act_with_prompt(thought, context)
        elif mode == "hybrid":
            return await self._act_hybrid(thought, context)
        else:
            return await self._act_with_code(thought, context)
    
    async def _act_with_code(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        """
        Execute action using code implementation.
        
        Override this method in subclasses for code mode.
        """
        raise NotImplementedError("Agent must implement act() for code mode")
    
    async def _act_with_prompt(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        """
        Execute action using prompt mode.
        
        Returns the thought content as the action result.
        """
        return {
            "action": "prompt_execution",
            "result": thought.content,
            "thought_type": thought.thought_type,
            "confidence": thought.confidence,
        }
    
    async def _act_hybrid(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        """
        Execute action using hybrid mode.
        """
        prompt_result = await self._act_with_prompt(thought, context)
        try:
            code_result = await self._act_with_code(thought, context)
            return {
                "prompt_result": prompt_result,
                "code_result": code_result,
                "action": "hybrid_execution"
            }
        except NotImplementedError:
            return prompt_result

    @abstractmethod
    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        """
        Observe action result and decide whether to continue.

        Args:
            action_result: Result of the action
            context: Execution context

        Returns:
            True to continue, False to stop
        """
        mode = self._get_mode()
        
        if mode == "prompt":
            return self._observe_with_prompt(action_result, context)
        elif mode == "hybrid":
            return self._observe_hybrid(action_result, context)
        else:
            return self._observe_with_code(action_result, context)
    
    def _observe_with_code(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        """
        Observe using code implementation.
        
        Override this method in subclasses for code mode.
        """
        raise NotImplementedError("Agent must implement observe() for code mode")
    
    def _observe_with_prompt(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        """
        Observe using prompt mode.
        
        Default implementation: stop after first action.
        """
        return False
    
    def _observe_hybrid(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        """
        Observe using hybrid mode.
        """
        try:
            return self._observe_with_code(action_result, context)
        except NotImplementedError:
            return self._observe_with_prompt(action_result, context)

    async def need_tool(self, context: POPSSAgentContext) -> bool:
        """Check if agent needs a tool."""
        return False

    async def select_tool(self, context: POPSSAgentContext) -> Optional[str]:
        """Select a tool to use."""
        return None

    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute agent synchronously (wrapper for async execute).

        Args:
            inputs: Input dictionary with 'task' key
            **kwargs: Additional arguments

        Returns:
            Agent execution result
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute_async(inputs, **kwargs))

    async def execute_async(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> POPSSAgentResult:
        """
        Execute agent asynchronously.

        Args:
            inputs: Input dictionary with 'task' key
            **kwargs: Additional arguments

        Returns:
            Agent execution result
        """
        start_time = datetime.now()
        span = self._tracing.start_span(
            f"agent_exec_{self.name}",
            attributes={"agent_id": self.agent_id}
        )

        task = inputs.get('task', '')
        context_id = f"context_{uuid.uuid4().hex[:12]}"

        context = POPSSAgentContext(
            context_id=context_id,
            task_id=kwargs.get('task_id', f"task_{uuid.uuid4().hex[:8]}"),
            user_request=task,
            metadata=kwargs
        )

        self.current_context = context
        self._update_state(POPSSAgentState.REASONING)

        try:
            self._emit_callback('on_thought', {
                'agent_id': self.agent_id,
                'context_id': context_id,
                'event': 'start',
            })

            thoughts = await self.think(context)

            for thought in thoughts:
                thought_entry = {
                    'thought_id': thought.thought_id,
                    'type': thought.thought_type,
                    'content': thought.content,
                    'confidence': thought.confidence,
                }
                context.history.append({
                    'type': 'thought',
                    'data': thought_entry,
                    'timestamp': datetime.now().isoformat(),
                })

                self._emit_callback('on_thought', {
                    'agent_id': self.agent_id,
                    'context_id': context_id,
                    'thought': thought_entry,
                })

            self._update_state(POPSSAgentState.EXECUTING)

            tool_calls = []
            for thought in thoughts:
                action_result = await self.act(thought, context)

                action_entry = {
                    'thought_id': thought.thought_id,
                    'action': action_result,
                }
                context.history.append({
                    'type': 'action',
                    'data': action_entry,
                    'timestamp': datetime.now().isoformat(),
                })

                self._emit_callback('on_action', {
                    'agent_id': self.agent_id,
                    'context_id': context_id,
                    'action': action_entry,
                })

                if 'tool_call' in action_result:
                    tool_calls.append(action_result['tool_call'])

                should_continue = await self.observe(action_result, context)

                if not should_continue:
                    break

            self._update_state(POPSSAgentState.COMPLETED)

            execution_time = (datetime.now() - start_time).total_seconds()

            self._execution_count += 1
            self._success_count += 1
            self._total_execution_time += execution_time

            result = POPSSAgentResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=context.history[-1].get('data', {}).get('action', {}).get('result', '') if context.history else '',
                reasoning_steps=[{
                    'thought_id': t.thought_id,
                    'content': t.content,
                    'confidence': t.confidence,
                } for t in thoughts],
                tool_calls=tool_calls,
                execution_time=execution_time,
            )

            self._metrics.counter("agent_executions_success")
            self._metrics.timer("agent_execution_time_ms", execution_time * 1000)
            self._tracing.end_span(span, status="ok")

            self._emit_callback('on_result', {
                'agent_id': self.agent_id,
                'context_id': context_id,
                'result': result,
            })

            return result

        except Exception as e:
            self._update_state(POPSSAgentState.ERROR)

            execution_time = (datetime.now() - start_time).total_seconds()

            self._execution_count += 1
            self._total_execution_time += execution_time

            result = POPSSAgentResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )

            self._metrics.counter("agent_executions_failed")
            self._tracing.end_span(span, status="error", error_message=str(e))

            self._emit_callback('on_error', {
                'agent_id': self.agent_id,
                'context_id': context_id,
                'error': str(e),
            })

            return result

        finally:
            self.current_context = None

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'state': self.state.value,
            'total_executions': self._execution_count,
            'successful_executions': self._success_count,
            'failed_executions': self._execution_count - self._success_count,
            'success_rate': self._success_count / max(self._execution_count, 1),
            'total_execution_time': self._total_execution_time,
            'average_execution_time': self._total_execution_time / max(self._execution_count, 1),
        }

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        self._execution_count = 0
        self._success_count = 0
        self._total_execution_time = 0.0

    def teardown(self) -> None:
        """Clean up agent resources."""
        self._metrics.counter("agent_teardowns")
        self._LOG.info("agent_teardown", agent_id=self.agent_id, name=self.name)

    def __enter__(self) -> 'POPSSBaseAgent':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.teardown()
        return False


class POPSSPromptBasedAgent(POPSSBaseAgent):
    """
    Prompt-based agent that loads configuration from YAML files.
    
    This class provides a convenient way to create agents that are
    driven entirely by prompt configuration files. No code implementation
    is required - just specify the expert_type and mode.
    
    Usage:
        # Create from prompt file
        agent = POPSSPromptBasedAgent(
            expert_type="code_reviewer",
            config=POPSSAgentConfig(name="code_reviewer")
        )
        
        # Or use factory
        from opss.agents.factory import POPSSExpertFactory
        agent = POPSSExpertFactory.create("code_reviewer")
    """
    
    def __init__(
        self, 
        expert_type: str,
        config: Optional[POPSSAgentConfig] = None,
        model_client: Optional[Any] = None
    ):
        """
        Initialize prompt-based agent.
        
        Args:
            expert_type: Expert type identifier for prompt loading
            config: Optional agent configuration
            model_client: Optional model client for execution
        """
        self.expert_type = expert_type
        self.mode = "prompt"
        
        config = config or POPSSAgentConfig(
            name=expert_type,
            expert_type=expert_type,
            mode="prompt"
        )
        
        super().__init__(config)
        
        if model_client:
            self.set_model_client(model_client)
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.expert_type
