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

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger

class POPSSConflictType(Enum):
    RESOURCE = "resource"
    DATA = "data"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    OUTPUT = "output"
    STATE = "state"
    SCHEDULE = "schedule"

class POPSSConflictSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class POPSSConflict:
    conflict_id: str
    conflict_type: POPSSConflictType
    severity: POPSSConflictSeverity
    
    involved_agents: List[str]
    involved_tasks: List[str]
    
    description: str
    root_cause: Optional[str] = None
    
    resolution_strategy: Optional[str] = None
    resolution_status: str = "unresolved"
    
    timestamp: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSConflictResolution:
    resolution_id: str
    conflict_id: str
    
    strategy: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    success: bool = False
    outcome: str = ""
    
    resolution_time: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSConflictResolverConfig:
    enable_automatic_resolution: bool = True
    max_resolution_attempts: int = 3
    
    severity_thresholds: Dict[POPSSConflictSeverity, int] = field(default_factory=lambda: {
        POPSSConflictSeverity.LOW: 60,
        POPSSConflictSeverity.MEDIUM: 30,
        POPSSConflictSeverity.HIGH: 10,
        POPSSConflictSeverity.CRITICAL: 1,
    })
    
    resolution_strategies: List[str] = field(default_factory=lambda: [
        "priority_based",
        "resource_based",
        "voting",
        "arbitration",
        "escalation",
    ])
    
    enable_escalation: bool = True
    escalation_threshold: POPSSConflictSeverity = POPSSConflictSeverity.HIGH
    
    enable_prevention: bool = True
    prevention_lookahead: int = 5

class POPSSConflictResolver:
    def __init__(self, config: Optional[POPSSConflictResolverConfig] = None):
        self.config = config or POPSSConflictResolverConfig()
        self._LOG = self._configure_logging()
        
        self._conflicts: Dict[str, POPSSConflict] = {}
        self._resolutions: Dict[str, POPSSConflictResolution] = {}
        
        self._conflict_history: List[Dict[str, Any]] = []
        
        self._callbacks: Dict[str, List[Callable]] = {
            'on_conflict_detected': [],
            'on_conflict_resolved': [],
            'on_conflict_escalated': [],
            'on_resolution_failed': [],
        }
        
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="piscesl1_conflict_resolver"
        )
        
        self._LOG.info("POPSSConflictResolver initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        logger = get_logger("PiscesLx.Core.Agents.Collaboration.ConflictResolver")
        return logger
    
    def detect_conflicts(
        self,
        agent_states: Dict[str, Dict[str, Any]],
        task_states: Dict[str, Dict[str, Any]],
        resource_states: Dict[str, Dict[str, Any]]
    ) -> List[POPSSConflict]:
        detected_conflicts = []
        
        resource_conflicts = self._detect_resource_conflicts(resource_states)
        detected_conflicts.extend(resource_conflicts)
        
        task_conflicts = self._detect_task_conflicts(task_states)
        detected_conflicts.extend(task_conflicts)
        
        output_conflicts = self._detect_output_conflicts(task_states)
        detected_conflicts.extend(output_conflicts)
        
        state_conflicts = self._detect_state_conflicts(agent_states, task_states)
        detected_conflicts.extend(state_conflicts)
        
        for conflict in detected_conflicts:
            self._conflicts[conflict.conflict_id] = conflict
            
            self._emit_callback('on_conflict_detected', {
                'conflict_id': conflict.conflict_id,
                'conflict_type': conflict.conflict_type.value,
                'severity': conflict.severity.value,
                'involved_agents': conflict.involved_agents,
            })
        
        self._LOG.info(f"Detected {len(detected_conflicts)} conflicts")
        
        return detected_conflicts
    
    def _detect_resource_conflicts(self, resource_states: Dict[str, Dict[str, Any]]) -> List[POPSSConflict]:
        conflicts = []
        
        resource_agents = {}
        for resource_id, state in resource_states.items():
            agents_using = state.get('agents_using', [])
            for agent_id in agents_using:
                if agent_id not in resource_agents:
                    resource_agents[agent_id] = []
                resource_agents[agent_id].append(resource_id)
        
        for agent_id, resources in resource_agents.items():
            if len(resources) > 1:
                conflict = POPSSConflict(
                    conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
                    conflict_type=POPSSConflictType.RESOURCE,
                    severity=POPSSConflictSeverity.MEDIUM,
                    involved_agents=[agent_id],
                    involved_tasks=[],
                    description=f"Agent {agent_id} accessing multiple resources simultaneously",
                    root_cause="Concurrent resource access",
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_task_conflicts(self, task_states: Dict[str, Dict[str, Any]]) -> List[POPSSConflict]:
        conflicts = []
        
        priority_tasks = {}
        for task_id, state in task_states.items():
            priority = state.get('priority', 5)
            if priority not in priority_tasks:
                priority_tasks[priority] = []
            priority_tasks[priority].append(task_id)
        
        high_priority_tasks = priority_tasks.get(9, []) + priority_tasks.get(10, [])
        for task_id in high_priority_tasks:
            task_state = task_states.get(task_id, {})
            blocked_by = task_state.get('blocked_by', [])
            if blocked_by:
                conflict = POPSSConflict(
                    conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
                    conflict_type=POPSSConflictType.PRIORITY,
                    severity=POPSSConflictSeverity.HIGH,
                    involved_agents=task_state.get('assigned_agents', []),
                    involved_tasks=[task_id] + blocked_by,
                    description=f"High priority task {task_id} blocked by lower priority tasks",
                    root_cause="Task scheduling conflict",
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_output_conflicts(self, task_states: Dict[str, Dict[str, Any]]) -> List[POPSSConflict]:
        conflicts = []
        
        output_signatures = {}
        for task_id, state in task_states.items():
            output = state.get('output', '')
            if output:
                signature = str(output)[:50]
                if signature not in output_signatures:
                    output_signatures[signature] = []
                output_signatures[signature].append(task_id)
        
        for signature, task_ids in output_signatures.items():
            if len(task_ids) > 1:
                conflict = POPSSConflict(
                    conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
                    conflict_type=POPSSConflictType.OUTPUT,
                    severity=POPSSConflictSeverity.LOW,
                    involved_agents=[],
                    involved_tasks=task_ids,
                    description=f"Multiple tasks producing similar outputs",
                    root_cause="Duplicate output generation",
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_state_conflicts(
        self,
        agent_states: Dict[str, Dict[str, Any]],
        task_states: Dict[str, Dict[str, Any]]
    ) -> List[POPSSConflict]:
        conflicts = []
        
        for agent_id, state in agent_states.items():
            current_task = state.get('current_task')
            desired_task = state.get('desired_task')
            
            if current_task and desired_task and current_task != desired_task:
                conflict = POPSSConflict(
                    conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
                    conflict_type=POPSSConflictType.STATE,
                    severity=POPSSConflictSeverity.MEDIUM,
                    involved_agents=[agent_id],
                    involved_tasks=[current_task, desired_task],
                    description=f"Agent {agent_id} state conflict",
                    root_cause="Agent reassignment conflict",
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def resolve_conflict(self, conflict: POPSSConflict) -> POPSSConflictResolution:
        resolution_id = f"res_{uuid.uuid4().hex[:12]}"
        
        self._LOG.info(f"Resolving conflict: {conflict.conflict_id}")
        
        conflict.resolution_status = "resolving"
        
        strategy = self._select_strategy(conflict)
        
        resolution = await self._apply_resolution_strategy(
            resolution_id, conflict, strategy
        )
        
        conflict.resolved_at = datetime.now()
        
        if resolution.success:
            conflict.resolution_status = "resolved"
            conflict.resolution_strategy = strategy
            
            self._emit_callback('on_conflict_resolved', {
                'conflict_id': conflict.conflict_id,
                'resolution_id': resolution_id,
                'strategy': strategy,
                'success': True,
            })
        else:
            conflict.resolution_status = "failed"
            
            self._emit_callback('on_resolution_failed', {
                'conflict_id': conflict.conflict_id,
                'resolution_id': resolution_id,
                'strategy': strategy,
                'attempts': conflict.metadata.get('attempts', 0),
            })
            
            if self.config.enable_escalation and conflict.severity >= self.config.escalation_threshold:
                await self._escalate_conflict(conflict)
                self._emit_callback('on_conflict_escalated', {
                    'conflict_id': conflict.conflict_id,
                    'severity': conflict.severity.value,
                })
        
        self._resolutions[resolution_id] = resolution
        
        self._conflict_history.append({
            'conflict_id': conflict.conflict_id,
            'resolution_id': resolution_id,
            'strategy': strategy,
            'success': resolution.success,
            'resolution_time': resolution.resolution_time,
            'timestamp': datetime.now().isoformat(),
        })
        
        return resolution
    
    def _select_strategy(self, conflict: POPSSConflict) -> str:
        if conflict.conflict_type == POPSSConflictType.RESOURCE:
            return "resource_based"
        elif conflict.conflict_type == POPSSConflictType.PRIORITY:
            return "priority_based"
        elif conflict.conflict_type == POPSSConflictType.DEPENDENCY:
            return "arbitration"
        elif conflict.conflict_type == POPSSConflictType.OUTPUT:
            return "voting"
        else:
            return "priority_based"
    
    async def _apply_resolution_strategy(
        self,
        resolution_id: str,
        conflict: POPSSConflict,
        strategy: str
    ) -> POPSSConflictResolution:
        from datetime import datetime
        start_time = datetime.now()
        
        actions = []
        success = False
        outcome = ""
        
        if strategy == "priority_based":
            agents = conflict.involved_agents
            if agents:
                highest_priority_agent = agents[0]
                actions.append({
                    'action': 'assign_to_agent',
                    'agent_id': highest_priority_agent,
                })
                success = True
                outcome = f"Assigned to highest priority agent: {highest_priority_agent}"
        
        elif strategy == "resource_based":
            actions.append({
                'action': 'sequential_access',
                'description': 'Grant access to one agent at a time',
            })
            success = True
            outcome = "Sequential access policy applied"
        
        elif strategy == "voting":
            actions.append({
                'action': 'majority_vote',
                'description': 'Resolve by majority vote among agents',
            })
            success = True
            outcome = "Majority vote resolution initiated"
        
        elif strategy == "arbitration":
            actions.append({
                'action': 'external_arbitration',
                'description': 'Use external arbiter for resolution',
            })
            success = True
            outcome = "External arbitration initiated"
        
        elif strategy == "escalation":
            actions.append({
                'action': 'escalate_to_human',
                'description': 'Escalate to human operator',
            })
            success = True
            outcome = "Escalation initiated"
        
        else:
            actions.append({
                'action': 'default_resolution',
                'description': 'Apply default resolution strategy',
            })
            success = True
            outcome = "Default resolution applied"
        
        resolution_time = (datetime.now() - start_time).total_seconds()
        
        return POPSSConflictResolution(
            resolution_id=resolution_id,
            conflict_id=conflict.conflict_id,
            strategy=strategy,
            actions=actions,
            success=success,
            outcome=outcome,
            resolution_time=resolution_time,
        )
    
    async def _escalate_conflict(self, conflict: POPSSConflict):
        conflict.severity = POPSSConflictSeverity.CRITICAL
        conflict.metadata['escalated'] = True
        conflict.metadata['escalation_time'] = datetime.now().isoformat()
        
        self._LOG.warning(f"Conflict escalated to critical: {conflict.conflict_id}")
    
    def prevent_conflicts(
        self,
        planned_actions: List[Dict[str, Any]],
        current_states: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.config.enable_prevention:
            return planned_actions
        
        lookahead = self.config.prevention_lookahead
        actions_to_modify = []
        
        for i, action in enumerate(planned_actions[:lookahead]):
            action_type = action.get('type', '')
            
            if action_type == 'resource_access':
                resource = action.get('resource')
                for j, other_action in enumerate(planned_actions[i+1:lookahead]):
                    if other_action.get('resource') == resource:
                        conflict = {
                            'action_index': i,
                            'conflicting_action_index': i + j + 1,
                            'resource': resource,
                            'severity': POPSSConflictSeverity.MEDIUM,
                        }
                        actions_to_modify.append(conflict)
        
        for conflict in actions_to_modify:
            self._LOG.info(f"Preventing potential conflict: {conflict}")
        
        return actions_to_modify
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        active_conflicts = [c for c in self._conflicts.values() if c.resolution_status == "unresolved"]
        resolving_conflicts = [c for c in self._conflicts.values() if c.resolution_status == "resolving"]
        resolved_conflicts = [c for c in self._conflicts.values() if c.resolution_status == "resolved"]
        failed_conflicts = [c for c in self._conflicts.values() if c.resolution_status == "failed"]
        
        severity_distribution = {}
        type_distribution = {}
        
        for conflict in self._conflicts.values():
            severity = conflict.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            
            conflict_type = conflict.conflict_type.value
            type_distribution[conflict_type] = type_distribution.get(conflict_type, 0) + 1
        
        return {
            'total_conflicts': len(self._conflicts),
            'active_conflicts': len(active_conflicts),
            'resolving_conflicts': len(resolving_conflicts),
            'resolved_conflicts': len(resolved_conflicts),
            'failed_conflicts': len(failed_conflicts),
            'resolution_rate': len(resolved_conflicts) / max(len(self._conflicts), 1),
            'severity_distribution': severity_distribution,
            'type_distribution': type_distribution,
        }
    
    def get_active_conflicts(self) -> List[Dict[str, Any]]:
        return [
            {
                'conflict_id': c.conflict_id,
                'conflict_type': c.conflict_type.value,
                'severity': c.severity.value,
                'involved_agents': c.involved_agents,
                'involved_tasks': c.involved_tasks,
                'description': c.description,
                'resolution_status': c.resolution_status,
                'timestamp': c.timestamp.isoformat(),
            }
            for c in self._conflicts.values()
            if c.resolution_status in ['unresolved', 'resolving']
        ]
    
    def get_resolution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [
            {
                'resolution_id': r.resolution_id,
                'conflict_id': r.conflict_id,
                'strategy': r.strategy,
                'success': r.success,
                'outcome': r.outcome,
                'resolution_time': r.resolution_time,
                'timestamp': r.timestamp.isoformat(),
            }
            for r in list(self._resolutions.values())[-limit:]
        ]
    
    def register_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit_callback(self, event: str, data: Any):
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self._LOG.error(f"Error in callback for {event}: {e}")
    
    def shutdown(self):
        self._async_executor.shutdown(wait=True)
        self._LOG.info("POPSSConflictResolver shutdown")
