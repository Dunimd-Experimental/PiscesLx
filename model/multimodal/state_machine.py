#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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


class RuchbahAgenticState(Enum):
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


class RuchbahAgenticEvent(Enum):
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


@dataclass
class RuchbahStateTransition:
    from_state: RuchbahAgenticState
    event: RuchbahAgenticEvent
    to_state: RuchbahAgenticState
    guard: Optional[Callable] = None
    action: Optional[Callable] = None


@dataclass
class RuchbahStateHistoryEntry:
    state: RuchbahAgenticState
    event: Optional[RuchbahAgenticEvent]
    timestamp: datetime
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuchbahStateMachine:
    
    _transition_table: List[RuchbahStateTransition] = []
    
    def __init__(self):
        self._current_state = RuchbahAgenticState.IDLE
        self._state_history: List[RuchbahStateHistoryEntry] = []
        self._entry_time = datetime.now()
        self._state_metadata: Dict[str, Any] = {}
        self._transition_callbacks: Dict[RuchbahAgenticEvent, List[Callable]] = {}
        self._setup_transition_table()
    
    def _setup_transition_table(self):
        transitions = [
            (RuchbahAgenticState.IDLE, RuchbahAgenticEvent.START, RuchbahAgenticState.UNDERSTANDING),
            (RuchbahAgenticState.UNDERSTANDING, RuchbahAgenticEvent.UNDERSTAND_COMPLETE, RuchbahAgenticState.PLANNING),
            (RuchbahAgenticState.UNDERSTANDING, RuchbahAgenticEvent.FAILURE, RuchbahAgenticState.FAILED),
            (RuchbahAgenticState.PLANNING, RuchbahAgenticEvent.PLAN_CREATED, RuchbahAgenticState.EXECUTING),
            (RuchbahAgenticState.PLANNING, RuchbahAgenticEvent.FAILURE, RuchbahAgenticState.FAILED),
            (RuchbahAgenticState.EXECUTING, RuchbahAgenticEvent.ACTION_START, RuchbahAgenticState.OBSERVING),
            (RuchbahAgenticState.EXECUTING, RuchbahAgenticEvent.SUCCESS, RuchbahAgenticState.COMPLETED),
            (RuchbahAgenticState.EXECUTING, RuchbahAgenticEvent.FAILURE, RuchbahAgenticState.FAILED),
            (RuchbahAgenticState.EXECUTING, RuchbahAgenticEvent.INTERRUPT, RuchbahAgenticState.WAITING),
            (RuchbahAgenticState.OBSERVING, RuchbahAgenticEvent.OBSERVATION_RECEIVED, RuchbahAgenticState.REFLECTING),
            (RuchbahAgenticState.OBSERVING, RuchbahAgenticEvent.FAILURE, RuchbahAgenticState.FAILED),
            (RuchbahAgenticState.REFLECTING, RuchbahAgenticEvent.REFLECTION_COMPLETE, RuchbahAgenticState.EXECUTING),
            (RuchbahAgenticState.REFLECTING, RuchbahAgenticEvent.SUCCESS, RuchbahAgenticState.COMPLETED),
            (RuchbahAgenticState.REFLECTING, RuchbahAgenticEvent.FAILURE, RuchbahAgenticState.FAILED),
            (RuchbahAgenticState.WAITING, RuchbahAgenticEvent.RESUME, RuchbahAgenticState.EXECUTING),
            (RuchbahAgenticState.WAITING, RuchbahAgenticEvent.TERMINATE, RuchbahAgenticState.TERMINATED),
            (RuchbahAgenticState.COMPLETED, RuchbahAgenticEvent.RESET, RuchbahAgenticState.IDLE),
            (RuchbahAgenticState.FAILED, RuchbahAgenticEvent.RESET, RuchbahAgenticState.IDLE),
            (RuchbahAgenticState.TERMINATED, RuchbahAgenticEvent.RESET, RuchbahAgenticState.IDLE),
            (RuchbahAgenticState.IDLE, RuchbahAgenticEvent.TERMINATE, RuchbahAgenticState.TERMINATED),
        ]
        
        for from_state, event, to_state in transitions:
            self._transition_table.append(
                RuchbahStateTransition(
                    from_state=from_state,
                    event=event,
                    to_state=to_state
                )
            )
    
    @property
    def current_state(self) -> RuchbahAgenticState:
        return self._current_state
    
    def get_available_events(self) -> List[RuchbahAgenticEvent]:
        available = []
        for transition in self._transition_table:
            if transition.from_state == self._current_state:
                available.append(transition.event)
        return available
    
    def can_transition(self, event: RuchbahAgenticEvent) -> bool:
        for transition in self._transition_table:
            if (transition.from_state == self._current_state and 
                transition.event == event):
                if transition.guard is None or transition.guard():
                    return True
        return False
    
    def transition(self, event: RuchbahAgenticEvent, metadata: Dict[str, Any] = None) -> bool:
        for transition in self._transition_table:
            if (transition.from_state == self._current_state and 
                transition.event == event):
                if transition.guard is not None and not transition.guard():
                    return False
                
                old_state = self._current_state
                duration = (datetime.now() - self._entry_time).total_seconds()
                
                self._state_history.append(
                    RuchbahStateHistoryEntry(
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
    
    def on_event(self, event: RuchbahAgenticEvent, callback: Callable, metadata: Dict[str, Any] = None) -> str:
        callback_id = str(uuid.uuid4())
        if event not in self._transition_callbacks:
            self._transition_callbacks[event] = []
        self._transition_callbacks[event].append(callback)
        return callback_id
    
    def _trigger_callbacks(self, event: RuchbahAgenticEvent):
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
        self._current_state = RuchbahAgenticState.IDLE
        self._state_history = []
        self._entry_time = datetime.now()
        self._state_metadata = {}
    
    def force_state(self, new_state: RuchbahAgenticState, metadata: Dict[str, Any] = None):
        duration = (datetime.now() - self._entry_time).total_seconds()
        
        self._state_history.append(
            RuchbahStateHistoryEntry(
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
            RuchbahAgenticState.COMPLETED,
            RuchbahAgenticState.FAILED,
            RuchbahAgenticState.TERMINATED
        ]
    
    def is_active_state(self) -> bool:
        return self._current_state in [
            RuchbahAgenticState.UNDERSTANDING,
            RuchbahAgenticState.PLANNING,
            RuchbahAgenticState.EXECUTING,
            RuchbahAgenticState.OBSERVING,
            RuchbahAgenticState.REFLECTING,
            RuchbahAgenticState.WAITING
        ]
