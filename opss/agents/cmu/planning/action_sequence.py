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
CMU Action Sequence - Action Sequence Generation and Management

This module provides action sequence generation, pre-condition checking,
post-condition verification, and rollback support.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUAction,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUScreenState,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Planning.ActionSequence", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Planning.ActionSequence"), enable_file=True)


@dataclass
class POPSSCMUSequenceState:
    """State of an action sequence."""
    sequence_id: str
    current_index: int = 0
    total_actions: int = 0
    completed_actions: int = 0
    failed_actions: int = 0
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[POPSSCMUActionResult] = field(default_factory=list)


@dataclass
class POPSSCMUCondition:
    """Condition for pre/post verification."""
    condition_id: str
    description: str
    check_function: Callable[[], bool]
    expected_result: bool = True


class POPSSCMUActionSequence:
    """
    Action sequence manager.
    
    Provides:
        - Action sequence generation
        - Pre-condition checking
        - Post-condition verification
        - Rollback support
        - Parallel action execution
    
    Attributes:
        actions: List of actions in sequence
        pre_conditions: Pre-conditions to check before execution
        post_conditions: Post-conditions to verify after execution
        rollback_actions: Actions to execute on failure
    """
    
    def __init__(
        self,
        actions: Optional[List[POPSSCMUAction]] = None,
        sequence_id: Optional[str] = None,
    ):
        self.sequence_id = sequence_id or f"seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.actions = actions or []
        
        self.pre_conditions: List[POPSSCMUCondition] = []
        self.post_conditions: List[POPSSCMUCondition] = []
        self.rollback_actions: List[POPSSCMUAction] = []
        
        self._state = POPSSCMUSequenceState(
            sequence_id=self.sequence_id,
            total_actions=len(self.actions),
        )
        
        self._snapshots: List[POPSSCMUScreenState] = []
        
        _LOG.info("action_sequence_created", sequence_id=self.sequence_id, action_count=len(self.actions))
    
    def add_action(
        self,
        action: POPSSCMUAction,
        index: Optional[int] = None,
    ) -> None:
        """Add action to sequence."""
        if index is not None:
            self.actions.insert(index, action)
        else:
            self.actions.append(action)
        
        self._state.total_actions = len(self.actions)
    
    def add_pre_condition(
        self,
        description: str,
        check_function: Callable[[], bool],
    ) -> None:
        """Add pre-condition."""
        condition = POPSSCMUCondition(
            condition_id=f"pre_{len(self.pre_conditions)}",
            description=description,
            check_function=check_function,
        )
        self.pre_conditions.append(condition)
    
    def add_post_condition(
        self,
        description: str,
        check_function: Callable[[], bool],
    ) -> None:
        """Add post-condition."""
        condition = POPSSCMUCondition(
            condition_id=f"post_{len(self.post_conditions)}",
            description=description,
            check_function=check_function,
        )
        self.post_conditions.append(condition)
    
    def add_rollback_action(self, action: POPSSCMUAction) -> None:
        """Add rollback action."""
        self.rollback_actions.append(action)
    
    async def check_pre_conditions(self) -> Tuple[bool, List[str]]:
        """Check all pre-conditions."""
        failures = []
        
        for condition in self.pre_conditions:
            try:
                result = condition.check_function()
                if result != condition.expected_result:
                    failures.append(f"Pre-condition failed: {condition.description}")
            except Exception as e:
                failures.append(f"Pre-condition error: {condition.description} - {str(e)}")
        
        return (len(failures) == 0, failures)
    
    async def check_post_conditions(self) -> Tuple[bool, List[str]]:
        """Check all post-conditions."""
        failures = []
        
        for condition in self.post_conditions:
            try:
                result = condition.check_function()
                if result != condition.expected_result:
                    failures.append(f"Post-condition failed: {condition.description}")
            except Exception as e:
                failures.append(f"Post-condition error: {condition.description} - {str(e)}")
        
        return (len(failures) == 0, failures)
    
    async def execute(
        self,
        executor: Callable[[POPSSCMUAction], POPSSCMUActionResult],
        screen_capture: Optional[Callable[[], POPSSCMUScreenState]] = None,
    ) -> List[POPSSCMUActionResult]:
        """
        Execute action sequence.
        
        Args:
            executor: Function to execute individual actions
            screen_capture: Function to capture screen state
        
        Returns:
            List[POPSSCMUActionResult]: Execution results
        """
        self._state.status = "running"
        self._state.started_at = datetime.now()
        
        pre_ok, pre_failures = await self.check_pre_conditions()
        if not pre_ok:
            _LOG.error("pre_conditions_failed", failures=pre_failures)
            self._state.status = "blocked"
            return []
        
        results = []
        
        for i, action in enumerate(self.actions):
            self._state.current_index = i
            
            if screen_capture:
                snapshot = await screen_capture()
                self._snapshots.append(snapshot)
            
            result = await executor(action)
            results.append(result)
            self._state.results.append(result)
            
            if result.status == POPSSCMUActionResultStatus.SUCCESS:
                self._state.completed_actions += 1
            else:
                self._state.failed_actions += 1
                
                if result.status != POPSSCMUActionResultStatus.PARTIAL:
                    _LOG.error(
                        "action_sequence_failed",
                        sequence_id=self.sequence_id,
                        action_index=i,
                        error=result.error_message,
                    )
                    
                    rollback_results = await self._execute_rollback(executor)
                    self._state.status = "failed"
                    self._state.completed_at = datetime.now()
                    
                    return results
        
        post_ok, post_failures = await self.check_post_conditions()
        if not post_ok:
            _LOG.warning("post_conditions_failed", failures=post_failures)
        
        self._state.status = "completed"
        self._state.completed_at = datetime.now()
        
        _LOG.info(
            "action_sequence_completed",
            sequence_id=self.sequence_id,
            completed=self._state.completed_actions,
            failed=self._state.failed_actions,
        )
        
        return results
    
    async def _execute_rollback(
        self,
        executor: Callable[[POPSSCMUAction], POPSSCMUActionResult],
    ) -> List[POPSSCMUActionResult]:
        """Execute rollback actions."""
        if not self.rollback_actions:
            return []
        
        _LOG.info("executing_rollback", sequence_id=self.sequence_id)
        
        results = []
        for action in reversed(self.rollback_actions):
            result = await executor(action)
            results.append(result)
        
        return results
    
    async def execute_parallel(
        self,
        executor: Callable[[POPSSCMUAction], POPSSCMUActionResult],
        groups: List[List[int]],
    ) -> List[POPSSCMUActionResult]:
        """
        Execute actions in parallel groups.
        
        Args:
            executor: Function to execute individual actions
            groups: List of action index groups to execute in parallel
        
        Returns:
            List[POPSSCMUActionResult]: Execution results
        """
        all_results = [None] * len(self.actions)
        
        for group in groups:
            tasks = []
            for idx in group:
                if 0 <= idx < len(self.actions):
                    tasks.append(executor(self.actions[idx]))
            
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, idx in enumerate(group):
                if isinstance(group_results[i], Exception):
                    all_results[idx] = POPSSCMUActionResult(
                        action_id=self.actions[idx].action_id,
                        status=POPSSCMUActionResultStatus.FAILED,
                        error_message=str(group_results[i]),
                    )
                else:
                    all_results[idx] = group_results[i]
        
        return all_results
    
    def get_state(self) -> POPSSCMUSequenceState:
        """Get current sequence state."""
        return self._state
    
    def get_progress(self) -> float:
        """Get execution progress as percentage."""
        if self._state.total_actions == 0:
            return 0.0
        return (self._state.completed_actions / self._state.total_actions) * 100
    
    def get_remaining_actions(self) -> List[POPSSCMUAction]:
        """Get remaining actions."""
        return self.actions[self._state.current_index + 1:]
    
    def get_latest_snapshot(self) -> Optional[POPSSCMUScreenState]:
        """Get latest screen snapshot."""
        if self._snapshots:
            return self._snapshots[-1]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sequence_id": self.sequence_id,
            "total_actions": self._state.total_actions,
            "completed_actions": self._state.completed_actions,
            "failed_actions": self._state.failed_actions,
            "status": self._state.status,
            "progress": self.get_progress(),
            "started_at": self._state.started_at.isoformat() if self._state.started_at else None,
            "completed_at": self._state.completed_at.isoformat() if self._state.completed_at else None,
        }
    
    @classmethod
    def from_actions(
        cls,
        actions: List[POPSSCMUAction],
    ) -> POPSSCMUActionSequence:
        """Create sequence from action list."""
        return cls(actions=actions)
    
    @classmethod
    def merge(
        cls,
        sequences: List[POPSSCMUActionSequence],
    ) -> POPSSCMUActionSequence:
        """Merge multiple sequences into one."""
        merged_actions = []
        for seq in sequences:
            merged_actions.extend(seq.actions)
        
        return cls(actions=merged_actions)
