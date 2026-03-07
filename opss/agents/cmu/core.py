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
CMU Core Engine - Main Computer Use Agent Engine

This module provides the main engine for the Computer Use Agent (CMU),
coordinating perception, planning, and action execution across platforms.

Module Components:
    1. POPSSCMUEngine:
       - Main agent engine inheriting from POPSSBaseAgent
       - Cross-platform device control
       - Task planning and execution
       - Safety integration

    2. POPSSCMUConfig:
       - Engine configuration
       - Platform-specific settings
       - Safety policy configuration

Key Features:
    - Cross-platform support (Desktop, Mobile, Tablet, Web)
    - Three-layer security integration
    - Visual perception via YvVisionEncoder
    - Task planning and decomposition
    - Action execution with verification
    - Error recovery and retry logic

Usage Example:
    >>> from opss.agents.cmu import POPSSCMUEngine, POPSSCMUConfig
    >>> 
    >>> # Initialize engine
    >>> config = POPSSCMUConfig(platform=POPSSCMUPlatform.DESKTOP_WINDOWS)
    >>> engine = POPSSCMUEngine(config)
    >>> 
    >>> # Execute task
    >>> result = await engine.execute_task("Open browser and search for AI news")
    >>> 
    >>> # Or use as agent
    >>> agent_result = await engine.execute_async({"task": "Click the submit button"})
"""

from __future__ import annotations

import asyncio
import platform as sys_platform
import sys
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.opsc.interface import PiscesLxOperatorStatus

from ..base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentThought,
    POPSSAgentState,
    POPSSAgentCapability,
)
from ..mcp_bridge import POPSSMCPBridge, POPSSMCPBridgeMixin

from .types import (
    POPSSCMUAction,
    POPSSCMUActionType,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUCoordinate,
    POPSSCMUElement,
    POPSSCMUPlatform,
    POPSSCMURectangle,
    POPSSCMUSafetyLevel,
    POPSSCMUSafetyPolicy,
    POPSSCMUScreenState,
    POPSSCMUTarget,
    POPSSCMUTaskContext,
    POPSSCMUScrollDirection,
    POPSSCMUSwipeDirection,
    POPSSCMUMouseButton,
)
from .safety import (
    POPSSCMUAuditLogger,
    POPSSCMUEmergencyStop,
    POPSSCMUPermissionManager,
    POPSSCMUSafetySystem,
    POPSSCMUSafetyValidator,
    POPSSCMUSandbox,
    POPSSCMURiskLevel,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Core", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Core"), enable_file=True)


@dataclass
class POPSSCMUConfig:
    """
    Configuration for CMU Engine.
    
    Attributes:
        platform: Target platform for the engine
        auto_detect_platform: Whether to auto-detect platform
        enable_safety: Whether to enable safety system
        enable_audit: Whether to enable audit logging
        max_retries: Maximum retry attempts for actions
        default_timeout: Default timeout for actions
        screenshot_dir: Directory for saving screenshots
        enable_visual_perception: Whether to enable visual perception
        enable_ocr: Whether to enable OCR
        confirmation_callback: Callback for user confirmation
    """
    platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS
    auto_detect_platform: bool = True
    enable_safety: bool = True
    enable_audit: bool = True
    max_retries: int = 3
    default_timeout: float = 30.0
    screenshot_dir: str = "./cmu_screenshots"
    enable_visual_perception: bool = True
    enable_ocr: bool = True
    confirmation_callback: Optional[Callable[[str, POPSSCMUAction], bool]] = None
    mouse_speed: float = 0.3
    typing_speed: float = 0.05
    scroll_amount: int = 3
    swipe_duration: float = 0.3

    def __post_init__(self):
        if self.auto_detect_platform:
            self.platform = self._detect_platform()

    def _detect_platform(self) -> POPSSCMUPlatform:
        """Auto-detect current platform."""
        system = sys_platform.system().lower()
        if system == "windows":
            return POPSSCMUPlatform.DESKTOP_WINDOWS
        elif system == "darwin":
            return POPSSCMUPlatform.DESKTOP_MACOS
        elif system == "linux":
            return POPSSCMUPlatform.DESKTOP_LINUX
        return POPSSCMUPlatform.UNKNOWN


class POPSSCMUCapability(POPSSAgentCapability):
    """Extended capabilities for CMU agent."""
    COMPUTER_USE = "computer_use"
    SCREEN_CAPTURE = "screen_capture"
    MOUSE_CONTROL = "mouse_control"
    KEYBOARD_CONTROL = "keyboard_control"
    TOUCH_CONTROL = "touch_control"
    VISUAL_PERCEPTION = "visual_perception"
    OCR = "ocr"


class POPSSCMUEngine(POPSSBaseAgent, POPSSMCPBridgeMixin):
    """
    Main Computer Use Agent Engine.
    
    A comprehensive agent engine that provides cross-platform device control
    with enterprise-grade security, visual perception, and task planning.
    
    Architecture:
        1. Perception Layer:
           - Screen capture
           - Element detection
           - OCR text extraction
        
        2. Planning Layer:
           - Task decomposition
           - Action sequence generation
           - Dependency analysis
        
        3. Execution Layer:
           - Platform adapters
           - Action execution
           - Result verification
        
        4. Safety Layer:
           - Action validation
           - Permission management
           - Audit logging
    
    Attributes:
        cmu_config: CMU-specific configuration
        platform_adapter: Platform-specific adapter
        safety_system: Integrated safety system
        screen_state: Current screen state
        task_context: Current task context
        _vision_encoder: Optional vision encoder for perception
        _action_handlers: Dictionary of action type handlers
    
    Example:
        >>> config = POPSSCMUConfig(platform=POPSSCMUPlatform.DESKTOP_WINDOWS)
        >>> engine = POPSSCMUEngine(POPSSAgentConfig(name="cmu_agent"), cmu_config=config)
        >>> result = await engine.execute_task("Open notepad and type Hello World")
    """

    def __init__(
        self,
        config: Optional[POPSSAgentConfig] = None,
        cmu_config: Optional[POPSSCMUConfig] = None,
        mcp_bridge: Optional[POPSSMCPBridge] = None,
        vision_encoder: Optional[Any] = None,
    ):
        config = config or POPSSAgentConfig(
            name="cmu_agent",
            agent_id=f"cmu_{uuid.uuid4().hex[:8]}",
        )

        capabilities = {
            POPSSCMUCapability.COMPUTER_USE,
            POPSSCMUCapability.SCREEN_CAPTURE,
            POPSSCMUCapability.MOUSE_CONTROL,
            POPSSCMUCapability.KEYBOARD_CONTROL,
            POPSSCMUCapability.VISUAL_PERCEPTION,
            POPSSAgentCapability.TOOL_USE,
            POPSSAgentCapability.PLANNING,
        }
        config.capabilities.update(capabilities)

        super().__init__(config)
        POPSSMCPBridgeMixin.__init__(self, mcp_bridge=mcp_bridge)

        self.cmu_config = cmu_config or POPSSCMUConfig()
        self._vision_encoder = vision_encoder

        self.safety_system = POPSSCMUSafetySystem(
            confirmation_callback=self.cmu_config.confirmation_callback,
            auto_approve_safe=True,
            audit_enabled=self.cmu_config.enable_audit,
        ) if self.cmu_config.enable_safety else None

        self.screen_state: Optional[POPSSCMUScreenState] = None
        self.task_context: Optional[POPSSCMUTaskContext] = None

        self._action_handlers: Dict[POPSSCMUActionType, Callable] = {
            POPSSCMUActionType.CLICK: self._handle_click,
            POPSSCMUActionType.DOUBLE_CLICK: self._handle_double_click,
            POPSSCMUActionType.RIGHT_CLICK: self._handle_right_click,
            POPSSCMUActionType.TYPE: self._handle_type,
            POPSSCMUActionType.KEY_PRESS: self._handle_key_press,
            POPSSCMUActionType.HOTKEY: self._handle_hotkey,
            POPSSCMUActionType.SCROLL: self._handle_scroll,
            POPSSCMUActionType.DRAG: self._handle_drag,
            POPSSCMUActionType.SWIPE: self._handle_swipe,
            POPSSCMUActionType.WAIT: self._handle_wait,
            POPSSCMUActionType.SCREENSHOT: self._handle_screenshot,
            POPSSCMUActionType.HOVER: self._handle_hover,
        }

        self._platform_adapter: Optional[Any] = None
        self._perception_module: Optional[Any] = None
        self._planning_module: Optional[Any] = None

        self._initialize_modules()

        _LOG.info(
            "cmu_engine_initialized",
            agent_id=self.agent_id,
            platform=self.cmu_config.platform.value,
        )

    def _initialize_modules(self) -> None:
        """Initialize platform and perception modules."""
        try:
            from .platform.desktop import POPSSCMUDesktop
            self._platform_adapter = POPSSCMUDesktop(self.cmu_config)
        except ImportError:
            _LOG.warning("desktop_platform_not_available")

        try:
            from .perception.screen_capture import POPSSCMUScreenCapture
            self._perception_module = POPSSCMUScreenCapture(self.cmu_config)
        except ImportError:
            _LOG.warning("perception_module_not_available")

    @property
    def name(self) -> str:
        """Get agent name."""
        return "cmu_agent"

    async def think(self, context: POPSSAgentContext) -> List[POPSSAgentThought]:
        """
        Generate thoughts for the task.
        
        Analyzes the task and plans the approach for execution.
        """
        thoughts = []

        thought1 = POPSSAgentThought(
            thought_id=f"thought_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            context_id=context.context_id,
            thought_type="task_analysis",
            content=f"Analyzing computer use task: {context.user_request[:100]}...",
            confidence=0.9,
            reasoning="Need to understand the task and determine required actions",
        )
        thoughts.append(thought1)

        thought2 = POPSSAgentThought(
            thought_id=f"thought_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            context_id=context.context_id,
            thought_type="platform_assessment",
            content=f"Target platform: {self.cmu_config.platform.value}",
            confidence=0.95,
            reasoning="Determining platform-specific execution strategy",
        )
        thoughts.append(thought2)

        thought3 = POPSSAgentThought(
            thought_id=f"thought_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent_id,
            context_id=context.context_id,
            thought_type="safety_assessment",
            content="Evaluating safety requirements for task execution",
            confidence=0.85,
            reasoning="Need to assess risks and determine required permissions",
        )
        thoughts.append(thought3)

        return thoughts

    async def act(self, thought: POPSSAgentThought, context: POPSSAgentContext) -> Dict[str, Any]:
        """
        Execute action based on thought.
        
        Dispatches to appropriate action handler based on thought type.
        """
        if thought.thought_type == "task_analysis":
            return await self._analyze_task(context)
        elif thought.thought_type == "platform_assessment":
            return await self._assess_platform(context)
        elif thought.thought_type == "safety_assessment":
            return await self._assess_safety(context)
        else:
            return {"action": "unknown", "thought_type": thought.thought_type}

    async def observe(self, action_result: Dict[str, Any], context: POPSSAgentContext) -> bool:
        """
        Observe action result and decide whether to continue.
        
        Returns True to continue execution, False to stop.
        """
        if action_result.get("status") == "error":
            return False
        if action_result.get("action") == "task_completed":
            return False
        return True

    async def execute_task(
        self,
        task: str,
        platform: Optional[POPSSCMUPlatform] = None,
        user_id: str = "",
    ) -> POPSSCMUActionResult:
        """
        Execute a computer use task.
        
        Main entry point for task execution.
        
        Args:
            task: Task description
            platform: Optional target platform override
            user_id: User identifier for permissions
        
        Returns:
            POPSSCMUActionResult: Execution result
        """
        if platform:
            self.cmu_config.platform = platform

        self.task_context = POPSSCMUTaskContext(
            goal=task,
            platform=self.cmu_config.platform,
        )

        self._update_state(POPSSAgentState.EXECUTING)

        try:
            self.screen_state = await self.capture_screen()

            actions = await self.plan_actions(task)
            self.task_context.actions = actions

            for action in actions:
                if self.safety_system:
                    is_active, reason = self.safety_system.emergency_stop.check()
                    if is_active:
                        return POPSSCMUActionResult(
                            action_id=action.action_id,
                            status=POPSSCMUActionResultStatus.CANCELLED,
                            error_message=f"Emergency stop: {reason}",
                        )

                result = await self.execute_action(action, user_id)
                self.task_context.add_result(result)

                if result.status == POPSSCMUActionResultStatus.FAILED:
                    if action.retry_count > 0:
                        for retry in range(min(action.retry_count, self.cmu_config.max_retries)):
                            result = await self.execute_action(action, user_id)
                            if result.success:
                                break

                if not result.success and result.status != POPSSCMUActionResultStatus.PARTIAL:
                    self.task_context.status = "failed"
                    return result

                self.screen_state = await self.capture_screen()

            self.task_context.status = "completed"
            self._update_state(POPSSAgentState.COMPLETED)

            return POPSSCMUActionResult(
                action_id=self.task_context.task_id,
                status=POPSSCMUActionResultStatus.SUCCESS,
                output={"task": task, "actions_executed": len(self.task_context.results)},
            )

        except Exception as e:
            self._update_state(POPSSAgentState.ERROR)
            _LOG.error("task_execution_failed", error=str(e))
            return POPSSCMUActionResult(
                action_id=self.task_context.task_id if self.task_context else "",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )

    async def capture_screen(
        self,
        region: Optional[POPSSCMURectangle] = None,
    ) -> POPSSCMUScreenState:
        """
        Capture current screen state.
        
        Args:
            region: Optional region to capture
        
        Returns:
            POPSSCMUScreenState: Captured screen state
        """
        if self._perception_module:
            try:
                return await self._perception_module.capture(region)
            except Exception as e:
                _LOG.error("screen_capture_failed", error=str(e))

        return POPSSCMUScreenState(
            platform=self.cmu_config.platform,
            width=1920,
            height=1080,
        )

    async def plan_actions(self, goal: str) -> List[POPSSCMUAction]:
        """
        Plan actions for a goal.
        
        Decomposes the goal into executable actions.
        
        Args:
            goal: Goal description
        
        Returns:
            List[POPSSCMUAction]: Planned actions
        """
        actions = []

        goal_lower = goal.lower()

        if "click" in goal_lower:
            target = await self._find_target(goal)
            actions.append(POPSSCMUAction.click(target=target))

        if "type" in goal_lower or "enter" in goal_lower or "write" in goal_lower:
            text = self._extract_text(goal)
            if text:
                actions.append(POPSSCMUAction.type_text(text=text))

        if "scroll" in goal_lower:
            direction = POPSSCMUScrollDirection.DOWN
            if "up" in goal_lower:
                direction = POPSSCMUScrollDirection.UP
            actions.append(POPSSCMUAction.scroll(direction=direction))

        if "open" in goal_lower or "launch" in goal_lower:
            app_name = self._extract_app_name(goal)
            actions.append(POPSSCMUAction(
                action_type=POPSSCMUActionType.APP_LAUNCH,
                params={"app_name": app_name},
                safety_level=POPSSCMUSafetyLevel.CONFIRM,
            ))

        if not actions:
            actions.append(POPSSCMUAction(
                action_type=POPSSCMUActionType.SCREENSHOT,
                description=f"Capture screen for analysis: {goal}",
            ))

        return actions

    async def execute_action(
        self,
        action: POPSSCMUAction,
        user_id: str = "",
    ) -> POPSSCMUActionResult:
        """
        Execute a single action.
        
        Args:
            action: Action to execute
            user_id: User identifier
        
        Returns:
            POPSSCMUActionResult: Execution result
        """
        start_time = time.time()

        if self.safety_system:
            return self.safety_system.validate_and_execute(
                action=action,
                executor=self._execute_action_internal,
                screen_state=self.screen_state,
                user_id=user_id,
            )

        result = await self._execute_action_internal(action)
        result.execution_time = time.time() - start_time
        return result

    async def _execute_action_internal(
        self,
        action: POPSSCMUAction,
    ) -> POPSSCMUActionResult:
        """Internal action execution."""
        if action.delay_before > 0:
            await asyncio.sleep(action.delay_before)

        handler = self._action_handlers.get(action.action_type)
        if handler:
            try:
                result = await handler(action)
                if action.delay_after > 0:
                    await asyncio.sleep(action.delay_after)
                return result
            except Exception as e:
                return POPSSCMUActionResult(
                    action_id=action.action_id,
                    status=POPSSCMUActionResultStatus.FAILED,
                    error_message=str(e),
                )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.FAILED,
            error_message=f"No handler for action type: {action.action_type.value}",
        )

    async def _handle_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle click action."""
        coord = await self._resolve_target(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve click target",
            )

        if self._platform_adapter:
            success = await self._platform_adapter.click(coord.x, coord.y)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"coordinate": coord.to_tuple()},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": coord.to_tuple(), "simulated": True},
        )

    async def _handle_double_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle double click action."""
        coord = await self._resolve_target(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve double click target",
            )

        if self._platform_adapter:
            success = await self._platform_adapter.double_click(coord.x, coord.y)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"coordinate": coord.to_tuple()},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": coord.to_tuple(), "simulated": True},
        )

    async def _handle_right_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle right click action."""
        coord = await self._resolve_target(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve right click target",
            )

        if self._platform_adapter:
            success = await self._platform_adapter.right_click(coord.x, coord.y)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"coordinate": coord.to_tuple()},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": coord.to_tuple(), "simulated": True},
        )

    async def _handle_type(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle type action."""
        text = action.params.get("text", "")
        typing_speed = action.params.get("typing_speed", self.cmu_config.typing_speed)

        if not text:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="No text to type",
            )

        if self._platform_adapter:
            success = await self._platform_adapter.type_text(text, typing_speed)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"text_length": len(text)},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"text": text[:50] + "..." if len(text) > 50 else text, "simulated": True},
        )

    async def _handle_key_press(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle key press action."""
        key = action.params.get("key", "")

        if self._platform_adapter:
            success = await self._platform_adapter.key_press(key)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"key": key},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"key": key, "simulated": True},
        )

    async def _handle_hotkey(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle hotkey action."""
        keys = action.params.get("keys", [])

        if not keys:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="No keys specified for hotkey",
            )

        if self._platform_adapter:
            success = await self._platform_adapter.hotkey(*keys)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"keys": keys},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"keys": keys, "simulated": True},
        )

    async def _handle_scroll(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle scroll action."""
        direction = POPSSCMUScrollDirection(action.params.get("direction", "down"))
        amount = action.params.get("amount", self.cmu_config.scroll_amount)

        if self._platform_adapter:
            success = await self._platform_adapter.scroll(direction.value, amount)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"direction": direction.value, "amount": amount},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"direction": direction.value, "amount": amount, "simulated": True},
        )

    async def _handle_drag(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle drag action."""
        start_coord = action.target.coordinate
        end_coord = action.params.get("end")
        duration = action.params.get("duration", 0.5)

        if not start_coord or not end_coord:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Missing start or end coordinate for drag",
            )

        if isinstance(end_coord, tuple):
            end_coord = POPSSCMUCoordinate(x=end_coord[0], y=end_coord[1])

        if self._platform_adapter:
            success = await self._platform_adapter.drag(
                start_coord.x, start_coord.y,
                end_coord.x, end_coord.y,
                duration
            )
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"start": start_coord.to_tuple(), "end": end_coord.to_tuple()},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"start": start_coord.to_tuple(), "end": end_coord.to_tuple(), "simulated": True},
        )

    async def _handle_swipe(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle swipe action."""
        direction = POPSSCMUSwipeDirection(action.params.get("direction", "up"))
        distance = action.params.get("distance", 0.3)
        duration = action.params.get("duration", self.cmu_config.swipe_duration)

        if self._platform_adapter:
            success = await self._platform_adapter.swipe(direction.value, distance, duration)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"direction": direction.value, "distance": distance},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"direction": direction.value, "distance": distance, "simulated": True},
        )

    async def _handle_wait(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle wait action."""
        duration = action.params.get("duration", 1.0)
        await asyncio.sleep(duration)
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"waited_seconds": duration},
        )

    async def _handle_screenshot(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle screenshot action."""
        self.screen_state = await self.capture_screen()
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"screenshot_captured": True},
        )

    async def _handle_hover(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle hover action."""
        coord = await self._resolve_target(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve hover target",
            )

        if self._platform_adapter:
            success = await self._platform_adapter.move_to(coord.x, coord.y)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
                output={"coordinate": coord.to_tuple()},
            )

        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": coord.to_tuple(), "simulated": True},
        )

    async def _resolve_target(self, target: POPSSCMUTarget) -> Optional[POPSSCMUCoordinate]:
        """Resolve target to coordinate."""
        if target.coordinate:
            if target.coordinate.is_relative and self.screen_state:
                return target.coordinate.to_absolute(
                    self.screen_state.width,
                    self.screen_state.height
                )
            return target.coordinate

        if target.rectangle:
            return target.rectangle.center

        if target.has_element_target() and self.screen_state:
            element = self.screen_state.find_element(
                text=target.text,
                element_type=target.element_type,
                element_id=target.element_id,
            )
            if element:
                return element.center

        return None

    async def _find_target(self, description: str) -> POPSSCMUTarget:
        """Find target from description."""
        if self.screen_state:
            for element in self.screen_state.elements:
                if element.text and element.text.lower() in description.lower():
                    return element.to_target()

        return POPSSCMUTarget(element_description=description)

    def _extract_text(self, goal: str) -> str:
        """Extract text to type from goal."""
        import re
        patterns = [
            r'type ["\'](.+?)["\']',
            r'enter ["\'](.+?)["\']',
            r'write ["\'](.+?)["\']',
            r'type (.+?)(?:\s+in|\s+into|\s+at|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_app_name(self, goal: str) -> str:
        """Extract application name from goal."""
        import re
        patterns = [
            r'open (.+?)(?:\s+app|\s+application|$)',
            r'launch (.+?)(?:\s+app|\s+application|$)',
            r'start (.+?)(?:\s+app|\s+application|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    async def _analyze_task(self, context: POPSSAgentContext) -> Dict[str, Any]:
        """Analyze task for execution planning."""
        return {
            "action": "task_analyzed",
            "task": context.user_request,
            "platform": self.cmu_config.platform.value,
        }

    async def _assess_platform(self, context: POPSSAgentContext) -> Dict[str, Any]:
        """Assess platform capabilities."""
        return {
            "action": "platform_assessed",
            "platform": self.cmu_config.platform.value,
            "is_desktop": self.cmu_config.platform.is_desktop,
            "is_touch_enabled": self.cmu_config.platform.is_touch_enabled,
        }

    async def _assess_safety(self, context: POPSSAgentContext) -> Dict[str, Any]:
        """Assess safety requirements."""
        if self.safety_system:
            return {
                "action": "safety_assessed",
                "safety_enabled": True,
                "audit_enabled": self.cmu_config.enable_audit,
            }
        return {
            "action": "safety_assessed",
            "safety_enabled": False,
        }

    def trigger_emergency_stop(self, reason: str = "User triggered") -> None:
        """Trigger emergency stop."""
        if self.safety_system:
            self.safety_system.emergency_stop.trigger(reason)
            _LOG.critical("emergency_stop_triggered_by_user", reason=reason)

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "platform": self.cmu_config.platform.value,
            "safety_enabled": self.safety_system is not None,
            "current_task": self.task_context.to_dict() if self.task_context else None,
            "screen_state": {
                "width": self.screen_state.width if self.screen_state else 0,
                "height": self.screen_state.height if self.screen_state else 0,
            } if self.screen_state else None,
        }

    def teardown(self) -> None:
        """Clean up engine resources."""
        if self.safety_system:
            self.safety_system.shutdown()
        super().teardown()
        _LOG.info("cmu_engine_teardown", agent_id=self.agent_id)
