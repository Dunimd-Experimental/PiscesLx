#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to Dunimd Team.
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
CMU Action Base - Action Execution Base

This module provides base class for action execution with
validation, retry logic, and result verification.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUAction,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUScreenState,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Action.Base", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Action.Base"), enable_file=True)


@dataclass
class POPSSCMUActionConfig:
    """Action execution configuration."""
    max_retries: int = 3
    retry_delay: float = 0.5
    retry_backoff: float = 2.0
    timeout: float = 30.0
    verify_result: bool = True
    on_failure: Optional[Callable[[POPSSCMUAction, Exception], None]] = None


class POPSSCMUActionExecutor(ABC):
    """
    Base class for action execution.
    
    Provides:
        - Action validation
        - Retry logic with exponential backoff
        - Result verification
        - Timeout handling
        - Error recovery
    
    Attributes:
        config: Execution configuration
        _platform_adapter: Platform adapter for execution
    """
    
    def __init__(
        self,
        config: Optional[POPSSCMUActionConfig] = None,
        platform_adapter: Optional[Any] = None,
    ):
        self.config = config or POPSSCMUActionConfig()
        self._platform_adapter = platform_adapter
        
        _LOG.info("action_executor_initialized")
    
    @abstractmethod
    async def execute_action_internal(
        self,
        action: POPSSCMUAction,
    ) -> POPSSCMUActionResult:
        """Internal action execution implementation."""
        raise NotImplementedError("Subclasses must implement execute_action_internal()")
    
    @abstractmethod
    async def verify_action_result(
        self,
        action: POPSSCMUAction,
        result: POPSSCMUActionResult,
    ) -> bool:
        """Verify action execution result."""
        raise NotImplementedError("Subclasses must implement verify_action_result()")
    
    async def execute(
        self,
        action: POPSSCMUAction,
        screen_state: Optional[POPSSCMUScreenState] = None,
    ) -> POPSSCMUActionResult:
        """
        Execute action with retry and verification.
        
        Args:
            action: Action to execute
            screen_state: Current screen state
        
        Returns:
            POPSSCMUActionResult: Execution result
        """
        validation_result = await self.validate_action(action, screen_state)
        if not validation_result.valid:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.BLOCKED,
                error_message=validation_result.reason,
            )
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            attempt_start = time.time()
            
            try:
                if attempt > 0:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    _LOG.info(
                        "retrying_action",
                        action_id=action.action_id,
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                
                result = await self.execute_action_internal(action)
                result.retry_count = attempt
                
                if result.status == POPSSCMUActionResultStatus.SUCCESS:
                    if self.config.verify_result:
                        is_verified = await self.verify_action_result(action, result)
                        result.verified = is_verified
                        
                        if not is_verified:
                            _LOG.warning(
                                "action_verification_failed",
                                action_id=action.action_id,
                                attempt=attempt + 1,
                            )
                            continue
                    
                    result.execution_time = time.time() - attempt_start
                    return result
                
                elif result.status == POPSSCMUActionResultStatus.RETRY:
                    _LOG.info(
                        "action_needs_retry",
                        action_id=action.action_id,
                        attempt=attempt + 1,
                        reason=result.error_message,
                    )
                    continue
                
                else:
                    _LOG.error(
                        "action_execution_failed",
                        action_id=action.action_id,
                        attempt=attempt + 1,
                        status=result.status.value,
                        error=result.error_message,
                    )
                    last_exception = Exception(result.error_message)
                    break
            
            except asyncio.TimeoutError:
                _LOG.error(
                    "action_timeout",
                    action_id=action.action_id,
                    attempt=attempt + 1,
                )
                last_exception = Exception(f"Action timed out after {self.config.timeout}s")
                break
            
            except Exception as e:
                last_exception = e
                _LOG.error(
                    "action_exception",
                    action_id=action.action_id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                break
        
        if last_exception and self.config.on_failure:
            self.config.on_failure(action, last_exception)
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.FAILED,
            error_message=str(last_exception) if last_exception else "Max retries exceeded",
            retry_count=attempt + 1,
        )
    
    async def validate_action(
        self,
        action: POPSSCMUAction,
        screen_state: Optional[POPSSCMUScreenState] = None,
    ) -> Tuple[bool, str]:
        """
        Validate action before execution.
        
        Args:
            action: Action to validate
            screen_state: Current screen state
        
        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        if not action.target:
            return True, "No target specified"
        
        if action.target.coordinate:
            if screen_state:
                if (
                    action.target.coordinate.x < 0
                    or action.target.coordinate.x > screen_state.width
                    or action.target.coordinate.y < 0
                    or action.target.coordinate.y > screen_state.height
                ):
                    return False, f"Coordinate out of bounds: ({action.target.coordinate.x}, {action.target.coordinate.y})"
        
        if action.action_type.value in ["type", "key_press", "hotkey"]:
            if not action.params.get("text") and not action.params.get("key") and not action.params.get("keys"):
                return False, "Missing required parameter (text/key/keys)"
        
        return True, "Action validated"
    
    async def execute_batch(
        self,
        actions: List[POPSSCMUAction],
        screen_state: Optional[POPSSCMUScreenState] = None,
        ) -> List[POPSSCMUActionResult]:
        """
        Execute multiple actions in sequence.
        
        Args:
            actions: List of actions to execute
            screen_state: Current screen state
        
        Returns:
            List[POPSSCMUActionResult]: Execution results
        """
        results = []
        
        for action in actions:
            result = await self.execute(action, screen_state)
            results.append(result)
            
            if result.status == POPSSCMUActionResultStatus.FAILED:
                _LOG.error(
                    "batch_execution_stopped",
                    action_id=action.action_id,
                    reason="Previous action failed",
                )
                break
            
            if screen_state:
                screen_state = await self.update_screen_state(action, result, screen_state)
        
        return results
    
    async def update_screen_state(
        self,
        action: POPSSCMUAction,
        result: POPSSCMUActionResult,
        current_state: POPSSCMUScreenState,
    ) -> POPSSCMUScreenState:
        """Update screen state after action."""
        return current_state
    
    async def execute_parallel(
        self,
        actions: List[POPSSCMUAction],
        screen_state: Optional[POPSSCMUScreenState] = None,
        max_concurrent: int = 3,
    ) -> List[POPSSCMUActionResult]:
        """
        Execute multiple actions in parallel.
        
        Args:
            actions: List of actions to execute
            screen_state: Current screen state
            max_concurrent: Maximum concurrent actions
        
        Returns:
            List[POPSSCMUActionResult]: Execution results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(action: POPSSCMUAction) -> POPSSCMUActionResult:
            async with semaphore:
                return await self.execute(action, screen_state)
        
        tasks = [execute_with_semaphore(action) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "retry_backoff": self.config.retry_backoff,
            "timeout": self.config.timeout,
            "verify_result": self.config.verify_result,
        }
