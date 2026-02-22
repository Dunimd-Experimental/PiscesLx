#!/usr/bin/env/python3
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

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json


class YvAgenticState(Enum):
    IDLE = auto()
    UNDERSTANDING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    OBSERVING = auto()
    REFLECTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    WAITING = auto()
    TERMINATED = auto()


class YvAgenticEvent(Enum):
    START = auto()
    UNDERSTAND_COMPLETE = auto()
    PLAN_CREATED = auto()
    PLAN_COMPLETE = auto()
    ACTION_START = auto()
    ACTION_COMPLETE = auto()
    OBSERVATION_RECEIVED = auto()
    REFLECTION_COMPLETE = auto()
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    INTERRUPT = auto()
    RESUME = auto()
    RESET = auto()
    TERMINATE = auto()
    CHECKPOINT_SAVE = auto()
    CHECKPOINT_RESTORE = auto()


@dataclass
class YvStateTransition:
    from_state: YvAgenticState
    event: YvAgenticEvent
    to_state: YvAgenticState
    guard: Optional[Callable] = None
    action: Optional[Callable] = None


@dataclass
class YvStateHistoryEntry:
    state: YvAgenticState
    event: Optional[YvAgenticEvent]
    timestamp: datetime
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class YvStateSnapshot:
    """State snapshot for checkpoint and recovery.
    
    Captures complete state information for recovery from failures
    and long-running task interruption handling.
    
    Attributes:
        snapshot_id: Unique identifier for the snapshot.
        state: The agentic state at snapshot time.
        metadata: Additional metadata associated with the state.
        timestamp: When the snapshot was created.
        recovery_actions: List of actions to take during recovery.
        execution_context: Context information for resuming execution.
        state_history: Recent state transitions for recovery analysis.
        checkpoint_data: Serializable checkpoint data.
    """
    snapshot_id: str
    state: YvAgenticState
    metadata: Dict[str, Any]
    timestamp: datetime
    recovery_actions: List[str] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "state": self.state.name,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "recovery_actions": self.recovery_actions,
            "execution_context": self.execution_context,
            "state_history": self.state_history,
            "checkpoint_data": self.checkpoint_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'YvStateSnapshot':
        """Create snapshot from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            state=YvAgenticState[data["state"]],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            recovery_actions=data.get("recovery_actions", []),
            execution_context=data.get("execution_context", {}),
            state_history=data.get("state_history", []),
            checkpoint_data=data.get("checkpoint_data", {}),
        )


class YvStateMachine:
    
    _transition_table: List[YvStateTransition] = []
    
    def __init__(self):
        self._current_state = YvAgenticState.IDLE
        self._state_history: List[YvStateHistoryEntry] = []
        self._entry_time = datetime.now()
        self._state_metadata: Dict[str, Any] = {}
        self._transition_callbacks: Dict[YvAgenticEvent, List[Callable]] = {}
        self._snapshots: Dict[str, YvStateSnapshot] = {}
        self._snapshot_stack: List[str] = []
        self._max_snapshots: int = 10
        self._recovery_chain: List[YvAgenticState] = []
        self._setup_transition_table()
    
    def _setup_transition_table(self):
        transitions = [
            (YvAgenticState.IDLE, YvAgenticEvent.START, YvAgenticState.UNDERSTANDING),
            (YvAgenticState.UNDERSTANDING, YvAgenticEvent.UNDERSTAND_COMPLETE, YvAgenticState.PLANNING),
            (YvAgenticState.UNDERSTANDING, YvAgenticEvent.FAILURE, YvAgenticState.FAILED),
            (YvAgenticState.PLANNING, YvAgenticEvent.PLAN_CREATED, YvAgenticState.EXECUTING),
            (YvAgenticState.PLANNING, YvAgenticEvent.FAILURE, YvAgenticState.FAILED),
            (YvAgenticState.EXECUTING, YvAgenticEvent.ACTION_START, YvAgenticState.OBSERVING),
            (YvAgenticState.EXECUTING, YvAgenticEvent.SUCCESS, YvAgenticState.COMPLETED),
            (YvAgenticState.EXECUTING, YvAgenticEvent.FAILURE, YvAgenticState.FAILED),
            (YvAgenticState.EXECUTING, YvAgenticEvent.INTERRUPT, YvAgenticState.WAITING),
            (YvAgenticState.EXECUTING, YvAgenticEvent.CHECKPOINT_SAVE, YvAgenticState.EXECUTING),
            (YvAgenticState.OBSERVING, YvAgenticEvent.OBSERVATION_RECEIVED, YvAgenticState.REFLECTING),
            (YvAgenticState.OBSERVING, YvAgenticEvent.FAILURE, YvAgenticState.FAILED),
            (YvAgenticState.REFLECTING, YvAgenticEvent.REFLECTION_COMPLETE, YvAgenticState.EXECUTING),
            (YvAgenticState.REFLECTING, YvAgenticEvent.SUCCESS, YvAgenticState.COMPLETED),
            (YvAgenticState.REFLECTING, YvAgenticEvent.FAILURE, YvAgenticState.FAILED),
            (YvAgenticState.WAITING, YvAgenticEvent.RESUME, YvAgenticState.EXECUTING),
            (YvAgenticState.WAITING, YvAgenticEvent.TERMINATE, YvAgenticState.TERMINATED),
            (YvAgenticState.FAILED, YvAgenticEvent.RESUME, YvAgenticState.EXECUTING),
            (YvAgenticState.FAILED, YvAgenticEvent.CHECKPOINT_RESTORE, YvAgenticState.EXECUTING),
            (YvAgenticState.COMPLETED, YvAgenticEvent.RESET, YvAgenticState.IDLE),
            (YvAgenticState.FAILED, YvAgenticEvent.RESET, YvAgenticState.IDLE),
            (YvAgenticState.TERMINATED, YvAgenticEvent.RESET, YvAgenticState.IDLE),
            (YvAgenticState.IDLE, YvAgenticEvent.TERMINATE, YvAgenticState.TERMINATED),
        ]
        
        for from_state, event, to_state in transitions:
            self._transition_table.append(
                YvStateTransition(
                    from_state=from_state,
                    event=event,
                    to_state=to_state
                )
            )
    
    @property
    def current_state(self) -> YvAgenticState:
        return self._current_state
    
    def get_available_events(self) -> List[YvAgenticEvent]:
        available = []
        for transition in self._transition_table:
            if transition.from_state == self._current_state:
                available.append(transition.event)
        return available
    
    def can_transition(self, event: YvAgenticEvent) -> bool:
        for transition in self._transition_table:
            if (transition.from_state == self._current_state and 
                transition.event == event):
                if transition.guard is None or transition.guard():
                    return True
        return False
    
    def transition(self, event: YvAgenticEvent, metadata: Dict[str, Any] = None) -> bool:
        for transition in self._transition_table:
            if (transition.from_state == self._current_state and 
                transition.event == event):
                if transition.guard is not None and not transition.guard():
                    return False
                
                old_state = self._current_state
                duration = (datetime.now() - self._entry_time).total_seconds()
                
                self._state_history.append(
                    YvStateHistoryEntry(
                        state=old_state,
                        event=event,
                        timestamp=self._entry_time,
                        duration=duration,
                        metadata=metadata or {}
                    )
                )
                
                if transition.action is not None:
                    transition.action()
                
                self._current_state = transition.to_state
                self._entry_time = datetime.now()
                self._state_metadata = metadata or {}
                
                self._trigger_callbacks(event)
                
                return True
        
        return False
    
    def on_event(self, event: YvAgenticEvent, callback: Callable, metadata: Dict[str, Any] = None) -> str:
        callback_id = str(uuid.uuid4())
        if event not in self._transition_callbacks:
            self._transition_callbacks[event] = []
        self._transition_callbacks[event].append(callback)
        return callback_id
    
    def _trigger_callbacks(self, event: YvAgenticEvent):
        if event in self._transition_callbacks:
            for callback in self._transition_callbacks[event]:
                try:
                    callback(self._current_state, event, self._state_metadata)
                except Exception:
                    pass
    
    def remove_callback(self, callback_id: str) -> bool:
        for event, callbacks in self._transition_callbacks.items():
            for i, callback in enumerate(callbacks):
                if hasattr(callback, '__name__') and callback.__name__ == callback_id:
                    callbacks.pop(i)
                    return True
        return False
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "state": entry.state.name,
                "event": entry.event.name if entry.event else None,
                "timestamp": entry.timestamp.isoformat(),
                "duration": entry.duration,
                "metadata": entry.metadata
            }
            for entry in self._state_history
        ]
    
    def get_state_statistics(self) -> Dict[str, Any]:
        if not self._state_history:
            return {"total_duration": 0.0, "state_durations": {}, "transition_count": 0}
        
        state_durations: Dict[str, float] = {}
        for entry in self._state_history:
            state_name = entry.state.name
            state_durations[state_name] = state_durations.get(state_name, 0.0) + entry.duration
        
        total_duration = sum(state_durations.values())
        
        return {
            "total_duration": total_duration,
            "state_durations": state_durations,
            "transition_count": len(self._state_history),
            "current_state": self._current_state.name,
            "time_in_current_state": (datetime.now() - self._entry_time).total_seconds()
        }
    
    def reset(self):
        self._current_state = YvAgenticState.IDLE
        self._state_history = []
        self._entry_time = datetime.now()
        self._state_metadata = {}
    
    def force_state(self, new_state: YvAgenticState, metadata: Dict[str, Any] = None):
        duration = (datetime.now() - self._entry_time).total_seconds()
        
        self._state_history.append(
            YvStateHistoryEntry(
                state=self._current_state,
                event=None,
                timestamp=self._entry_time,
                duration=duration,
                metadata={"force_transition": True}
            )
        )
        
        self._current_state = new_state
        self._entry_time = datetime.now()
        self._state_metadata = metadata or {}
    
    def is_terminal_state(self) -> bool:
        return self._current_state in [
            YvAgenticState.COMPLETED,
            YvAgenticState.FAILED,
            YvAgenticState.TERMINATED
        ]
    
    def is_active_state(self) -> bool:
        return self._current_state in [
            YvAgenticState.UNDERSTANDING,
            YvAgenticState.PLANNING,
            YvAgenticState.EXECUTING,
            YvAgenticState.OBSERVING,
            YvAgenticState.REFLECTING,
            YvAgenticState.WAITING
        ]
    
    def create_snapshot(self, execution_context: Dict[str, Any] = None) -> YvStateSnapshot:
        """Create a state snapshot for checkpoint and recovery.
        
        Captures the current state, history, and context for later restoration.
        Maintains a maximum number of snapshots to prevent memory bloat.
        
        Args:
            execution_context: Additional context for resuming execution.
            
        Returns:
            YvStateSnapshot: The created snapshot.
        """
        snapshot_id = str(uuid.uuid4())
        
        recent_history = [
            {
                "state": entry.state.name,
                "event": entry.event.name if entry.event else None,
                "timestamp": entry.timestamp.isoformat(),
                "duration": entry.duration,
                "metadata": entry.metadata
            }
            for entry in self._state_history[-20:]
        ]
        
        recovery_actions = self._generate_recovery_actions()
        
        snapshot = YvStateSnapshot(
            snapshot_id=snapshot_id,
            state=self._current_state,
            metadata=self._state_metadata.copy(),
            timestamp=datetime.now(),
            recovery_actions=recovery_actions,
            execution_context=execution_context or {},
            state_history=recent_history,
            checkpoint_data={
                "entry_time": self._entry_time.isoformat(),
                "transition_count": len(self._state_history),
            }
        )
        
        self._snapshots[snapshot_id] = snapshot
        self._snapshot_stack.append(snapshot_id)
        
        while len(self._snapshot_stack) > self._max_snapshots:
            old_id = self._snapshot_stack.pop(0)
            if old_id in self._snapshots:
                del self._snapshots[old_id]
        
        return snapshot
    
    def _generate_recovery_actions(self) -> List[str]:
        """Generate recovery actions based on current state and history."""
        actions = []
        
        if self._current_state == YvAgenticState.EXECUTING:
            actions.extend([
                "resume_execution",
                "retry_last_action",
                "skip_current_step",
                "request_user_input"
            ])
        elif self._current_state == YvAgenticState.OBSERVING:
            actions.extend([
                "re_observe",
                "proceed_with_partial",
                "request_clarification"
            ])
        elif self._current_state == YvAgenticState.REFLECTING:
            actions.extend([
                "continue_reflection",
                "skip_reflection",
                "force_proceed"
            ])
        elif self._current_state == YvAgenticState.FAILED:
            actions.extend([
                "restore_from_checkpoint",
                "retry_from_failure",
                "escalate_to_user",
                "abort_task"
            ])
        elif self._current_state == YvAgenticState.WAITING:
            actions.extend([
                "resume_from_wait",
                "cancel_wait",
                "timeout_and_proceed"
            ])
        else:
            actions.append("restart")
        
        return actions
    
    def restore_from_snapshot(self, snapshot: YvStateSnapshot) -> bool:
        """Restore state from a snapshot.
        
        Args:
            snapshot: The snapshot to restore from.
            
        Returns:
            bool: True if restoration was successful.
        """
        if snapshot.snapshot_id not in self._snapshots:
            return False
        
        self._current_state = snapshot.state
        self._state_metadata = snapshot.metadata.copy()
        self._entry_time = datetime.now()
        
        self._recovery_chain.append(self._current_state)
        
        return True
    
    def get_recovery_point(self) -> Optional[YvStateSnapshot]:
        """Get the most recent recovery point (snapshot).
        
        Returns:
            Optional[YvStateSnapshot]: The most recent snapshot, or None.
        """
        if not self._snapshot_stack:
            return None
        
        latest_id = self._snapshot_stack[-1]
        return self._snapshots.get(latest_id)
    
    def can_recover_from_failure(self) -> bool:
        """Check if recovery from failure is possible.
        
        Returns:
            bool: True if there are snapshots available for recovery.
        """
        return len(self._snapshots) > 0 and self._current_state == YvAgenticState.FAILED
    
    def get_failure_recovery_chain(self) -> List[YvAgenticState]:
        """Get the chain of states for recovery.
        
        Returns:
            List[YvAgenticState]: States to traverse for recovery.
        """
        if not self._recovery_chain:
            return [YvAgenticState.IDLE, YvAgenticState.UNDERSTANDING, 
                    YvAgenticState.PLANNING, YvAgenticState.EXECUTING]
        
        return self._recovery_chain.copy()
    
    def get_snapshot(self, snapshot_id: str) -> Optional[YvStateSnapshot]:
        """Get a specific snapshot by ID.
        
        Args:
            snapshot_id: The snapshot identifier.
            
        Returns:
            Optional[YvStateSnapshot]: The snapshot, or None if not found.
        """
        return self._snapshots.get(snapshot_id)
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots.
        
        Returns:
            List[Dict[str, Any]]: List of snapshot summaries.
        """
        return [
            {
                "snapshot_id": snapshot.snapshot_id,
                "state": snapshot.state.name,
                "timestamp": snapshot.timestamp.isoformat(),
                "recovery_actions": snapshot.recovery_actions,
            }
            for snapshot_id in self._snapshot_stack
            if snapshot_id in self._snapshots
            for snapshot in [self._snapshots[snapshot_id]]
        ]
    
    def clear_snapshots(self):
        """Clear all stored snapshots."""
        self._snapshots.clear()
        self._snapshot_stack.clear()
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery capabilities.
        
        Returns:
            Dict[str, Any]: Recovery statistics.
        """
        return {
            "snapshot_count": len(self._snapshots),
            "max_snapshots": self._max_snapshots,
            "can_recover": self.can_recover_from_failure(),
            "recovery_chain_length": len(self._recovery_chain),
            "current_state": self._current_state.name,
            "last_snapshot_time": (
                self._snapshots[self._snapshot_stack[-1]].timestamp.isoformat()
                if self._snapshot_stack else None
            ),
        }
