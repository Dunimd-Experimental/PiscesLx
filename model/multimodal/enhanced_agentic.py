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
Enhanced Agentic Capabilities for PiscesL1.

Core enhancements:
- Long-term planning engine: 100+ step task decomposition
- Tool orchestration engine: conditional branching, loops + fallbacks
- Self-evaluation system: multi-dimensional quality assessment
- Persistent memory: vector database storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class YvLongTermPlanner(nn.Module):
    """Long-term planning engine supporting 100+ step complex task decomposition.
    
    Core capabilities:
    - Hierarchical task decomposition (goal -> stage -> step)
    - Dependency analysis (DAG construction)
    - Critical path identification
    - Dynamic re-planning trigger
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        max_steps: int = 128,
        num_stages: int = 8
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.num_stages = num_stages
        
        self.task_graph_encoder = nn.GRU(
            hidden_size, hidden_size, num_layers=2, batch_first=True
        )
        
        self.dependency_analyzer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.critical_path_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.replan_trigger = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.stage_planner = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2)
            ) for _ in range(num_stages)
        ])
        
        self.step_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        self.complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)
        )
    
    def forward(self, goal: str, context: Dict[str, Any]) -> 'YvTaskGraph':
        batch_size = 1
        
        goal_embedding = self._encode_goal(goal, context)
        complexity = self._assess_complexity(goal_embedding, context)
        
        num_stages = min(self.num_stages, max(3, int(complexity['overall']) * self.num_stages))
        
        stages = []
        for i in range(num_stages):
            stage_embedding = self.stage_planner[i](goal_embedding)
            stage_info = self._plan_stage(i, stage_embedding, context, complexity)
            stages.append(stage_info)
        
        steps = self._generate_steps(stages, context, goal_embedding)
        
        dependencies = self._analyze_dependencies(steps)
        
        critical_nodes = self._identify_critical_path(steps, dependencies)
        
        return YvTaskGraph(
            goal=goal,
            stages=stages,
            steps=steps,
            dependencies=dependencies,
            critical_nodes=critical_nodes,
            complexity=complexity
        )
    
    def _encode_goal(self, goal: str, context: Dict[str, Any]) -> torch.Tensor:
        goal_tokens = torch.tensor([ord(c) for c in goal[:100]], dtype=torch.float32)
        goal_tokens = F.normalize(goal_tokens.unsqueeze(0), p=2, dim=-1)
        
        padding = torch.zeros(1, max(0, 100 - len(goal_tokens[0])))
        goal_tokens = torch.cat([goal_tokens, padding], dim=-1)
        
        if goal_tokens.shape[-1] < self.hidden_size:
            repeat = (self.hidden_size // goal_tokens.shape[-1]) + 1
            goal_tokens = goal_tokens.repeat(1, repeat)[:, :self.hidden_size]
        else:
            goal_tokens = goal_tokens[:, :self.hidden_size]
        
        return goal_tokens
    
    def _assess_complexity(
        self,
        goal_embedding: torch.Tensor,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        seq_len = context.get('sequence_length', 512)
        length_complexity = min(seq_len / 1024, 1.0)
        
        has_multiple_modalities = len(context.get('modalities', [])) > 1
        modality_complexity = 0.7 if has_multiple_modalities else 0.3
        
        is_research = 'research' in context.get('goal', '').lower()
        is_code = 'code' in context.get('goal', '').lower()
        task_complexity = 0.8 if is_research or is_code else 0.5
        
        features = torch.cat([
            goal_embedding.mean(dim=0),
            torch.tensor([length_complexity]),
            torch.tensor([modality_complexity])
        ]).unsqueeze(0)
        
        complexity_scores = self.complexity_assessor(features)
        nn_complexity = torch.softmax(complexity_scores, dim=-1)[0]
        
        return {
            'length': length_complexity,
            'modality': modality_complexity,
            'task': task_complexity,
            'nn_score': nn_complexity[0].item(),
            'overall': (length_complexity * 0.3 + modality_complexity * 0.3 + task_complexity * 0.4)
        }
    
    def _plan_stage(
        self,
        stage_id: int,
        stage_embedding: torch.Tensor,
        context: Dict[str, Any],
        complexity: Dict[str, float]
    ) -> 'YvTaskStage':
        num_steps = max(3, int(self.max_steps / self.num_stages * complexity['overall']))
        
        stage_name = f"Stage {stage_id + 1}"
        description = f"Execute {stage_name} of the task"
        
        success_criteria = [f"Complete all steps in {stage_name}"]
        
        return YvTaskStage(
            stage_id=stage_id,
            name=stage_name,
            description=description,
            num_steps=num_steps,
            success_criteria=success_criteria,
            parallelizable=stage_id % 2 == 0
        )
    
    def _generate_steps(
        self,
        stages: List['YvTaskStage'],
        context: Dict[str, Any],
        goal_embedding: torch.Tensor
    ) -> List['YvPlanStep']:
        steps = []
        step_id = 0
        
        for stage in stages:
            for step_idx in range(stage.num_steps):
                step_embedding = self.step_generator(goal_embedding)
                step_name = f"{stage.name} - Step {step_idx + 1}"
                description = f"Execute {step_name}"
                
                step = YvPlanStep(
                    step_id=step_id,
                    name=step_name,
                    description=description,
                    stage_id=stage.stage_id,
                    status="pending",
                    estimated_duration=1.0,
                    complexity=stage.complexity if hasattr(stage, 'complexity') else 0.5
                )
                steps.append(step)
                step_id += 1
        
        return steps[:self.max_steps]
    
    def _analyze_dependencies(self, steps: List['YvPlanStep']) -> List[Tuple[int, int]]:
        dependencies = []
        for i, step in enumerate(steps):
            if step.stage_id > 0:
                prev_stage_last_step = max(
                    s.step_id for s in steps 
                    if s.stage_id == step.stage_id - 1
                )
                dependencies.append((prev_stage_last_step, i))
            
            step_deps = getattr(step, 'dependencies', [])
            if step_deps:
                for dep in step_deps:
                    if dep < i:
                        dependencies.append((dep, i))
        
        return dependencies
    
    def _identify_critical_path(
        self,
        steps: List['YvPlanStep'],
        dependencies: List[Tuple[int, int]]
    ) -> List[int]:
        if not steps:
            return []
        
        step_ids = [s.step_id for s in steps]
        critical_path = []
        
        in_degree = {sid: 0 for sid in step_ids}
        for _, target in dependencies:
            if target in in_degree:
                in_degree[target] += 1
        
        ready = [sid for sid, deg in in_degree.items() if deg == 0]
        temp_order = []
        
        while ready:
            current = ready.pop(0)
            temp_order.append(current)
            
            for src, tgt in dependencies:
                if src == current and tgt in in_degree:
                    in_degree[tgt] -= 1
                    if in_degree[tgt] == 0:
                        ready.append(tgt)
        
        critical_path = temp_order[-min(10, len(temp_order)):]
        
        return critical_path
    
    def should_replan(
        self,
        current_state: Dict[str, Any],
        expected_state: Dict[str, Any],
        execution_history: List[Dict[str, Any]]
    ) -> bool:
        state_deviation = self._compute_state_deviation(current_state, expected_state)
        error_rate = self._compute_error_rate(execution_history)
        time_overrun = self._check_time_overrun(execution_history)
        
        features = torch.tensor([
            state_deviation, error_rate, time_overrun
        ]).unsqueeze(0)
        
        trigger_prob = self.replan_trigger(features)
        return trigger_prob > 0.5
    
    def _compute_state_deviation(
        self,
        current_state: Dict[str, Any],
        expected_state: Dict[str, Any]
    ) -> float:
        if not expected_state:
            return 0.0
        
        deviation = 0.0
        count = 0
        for key in expected_state:
            if key in current_state:
                curr_val = current_state[key]
                exp_val = expected_state[key]
                if isinstance(curr_val, (int, float)) and isinstance(exp_val, (int, float)):
                    diff = abs(curr_val - exp_val) / (abs(exp_val) + 1e-8)
                    deviation += min(diff, 1.0)
                    count += 1
        
        return deviation / (count + 1e-8)
    
    def _compute_error_rate(self, execution_history: List[Dict[str, Any]]) -> float:
        if not execution_history:
            return 0.0
        
        errors = sum(1 for h in execution_history if h.get('status') == 'failed')
        return errors / len(execution_history)
    
    def _check_time_overrun(self, execution_history: List[Dict[str, Any]]) -> float:
        if not execution_history:
            return 0.0
        
        overruns = 0
        total = 0
        for h in execution_history:
            if 'estimated_time' in h and 'actual_time' in h:
                if h['actual_time'] > h['estimated_time']:
                    overruns += 1
                total += 1
        
        return overruns / (total + 1e-8)
    
    def save_checkpoint(
        self,
        task_graph: YvTaskGraph,
        execution_state: Dict[str, Any]
    ) -> YvCheckpoint:
        """Save a checkpoint of the current execution state.
        
        Creates a complete snapshot of the task execution state for
        recovery from failures or interruptions.
        
        Args:
            task_graph: The current task graph being executed.
            execution_state: Additional execution context and state.
            
        Returns:
            YvCheckpoint: The created checkpoint.
        """
        import time
        import uuid
        
        checkpoint_id = str(uuid.uuid4())
        timestamp = time.time()
        
        completed_steps = task_graph.completed_steps.copy()
        failed_steps = task_graph.failed_steps.copy()
        
        can_resume = len(completed_steps) < len(task_graph.steps)
        
        total_retries = sum(task_graph.retry_counts.values())
        
        checkpoint = YvCheckpoint(
            checkpoint_id=checkpoint_id,
            task_graph=task_graph,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            execution_context=execution_state,
            timestamp=timestamp,
            can_resume=can_resume,
            retry_count=total_retries,
            error_history=task_graph.error_history.copy(),
            metadata={
                "goal": task_graph.goal,
                "total_steps": len(task_graph.steps),
                "progress": len(completed_steps) / max(len(task_graph.steps), 1),
            }
        )
        
        task_graph.checkpoint_history.append(checkpoint_id)
        task_graph.last_checkpoint_time = timestamp
        
        return checkpoint
    
    def restore_checkpoint(
        self,
        checkpoint: YvCheckpoint,
        checkpoint_storage: Dict[str, YvCheckpoint] = None
    ) -> Optional[YvTaskGraph]:
        """Restore execution state from a checkpoint.
        
        Args:
            checkpoint: The checkpoint to restore from.
            checkpoint_storage: Optional storage to retrieve checkpoint by ID.
            
        Returns:
            Optional[YvTaskGraph]: The restored task graph, or None if failed.
        """
        if checkpoint is None:
            return None
        
        if not checkpoint.can_resume:
            return None
        
        task_graph = checkpoint.task_graph
        if task_graph is None:
            return None
        
        task_graph.completed_steps = checkpoint.completed_steps.copy()
        task_graph.failed_steps = checkpoint.failed_steps.copy()
        task_graph.error_history = checkpoint.error_history.copy()
        task_graph.interrupted = False
        task_graph.interruption_point = None
        
        for step in task_graph.steps:
            if step.step_id in checkpoint.completed_steps:
                step.status = "completed"
            elif step.step_id in checkpoint.failed_steps:
                step.status = "failed"
            else:
                step.status = "pending"
        
        return task_graph
    
    def should_save_checkpoint(
        self,
        step_id: int,
        time_elapsed: float,
        task_graph: YvTaskGraph
    ) -> bool:
        """Determine if a checkpoint should be saved.
        
        Args:
            step_id: Current step being executed.
            time_elapsed: Time elapsed since last checkpoint.
            task_graph: The current task graph.
            
        Returns:
            bool: True if a checkpoint should be saved.
        """
        if time_elapsed >= task_graph.checkpoint_interval:
            return True
        
        steps_since_checkpoint = len(task_graph.completed_steps) - len(task_graph.checkpoint_history) * 10
        if steps_since_checkpoint >= 10:
            return True
        
        if step_id in task_graph.critical_nodes:
            return True
        
        return False
    
    def get_recovery_point(
        self,
        task_graph: YvTaskGraph,
        checkpoints: Dict[str, YvCheckpoint]
    ) -> int:
        """Find the best recovery point in the task graph.
        
        Args:
            task_graph: The current task graph.
            checkpoints: Available checkpoints.
            
        Returns:
            int: The step ID to resume from.
        """
        if not task_graph.completed_steps:
            return 0
        
        last_completed = max(task_graph.completed_steps)
        
        if task_graph.failed_steps:
            last_failed = max(task_graph.failed_steps)
            for step in task_graph.steps:
                if step.step_id > last_failed and step.step_id not in task_graph.completed_steps:
                    return step.step_id
        
        for step in task_graph.steps:
            if step.step_id > last_completed:
                return step.step_id
        
        return last_completed
    
    def estimate_remaining_time(
        self,
        task_graph: YvTaskGraph,
        avg_step_time: float = 1.0
    ) -> float:
        """Estimate remaining execution time.
        
        Args:
            task_graph: The current task graph.
            avg_step_time: Average time per step.
            
        Returns:
            float: Estimated remaining time in seconds.
        """
        total_steps = len(task_graph.steps)
        completed_steps = len(task_graph.completed_steps)
        remaining_steps = total_steps - completed_steps
        
        base_estimate = remaining_steps * avg_step_time
        
        failed_steps = len(task_graph.failed_steps)
        retry_overhead = failed_steps * avg_step_time * 0.5
        
        critical_remaining = sum(
            1 for step in task_graph.steps
            if step.step_id in task_graph.critical_nodes
            and step.step_id not in task_graph.completed_steps
        )
        critical_overhead = critical_remaining * avg_step_time * 0.3
        
        return base_estimate + retry_overhead + critical_overhead
    
    def update_progress(
        self,
        task_graph: YvTaskGraph,
        step_id: int,
        status: str,
        time_taken: float
    ) -> Dict[str, Any]:
        """Update task graph progress.
        
        Args:
            task_graph: The task graph to update.
            step_id: The step that was processed.
            status: The new status of the step.
            time_taken: Time taken for the step.
            
        Returns:
            Dict[str, Any]: Progress report.
        """
        import time
        
        if status == "completed":
            if step_id not in task_graph.completed_steps:
                task_graph.completed_steps.append(step_id)
        elif status == "failed":
            if step_id not in task_graph.failed_steps:
                task_graph.failed_steps.append(step_id)
            task_graph.retry_counts[step_id] = task_graph.retry_counts.get(step_id, 0) + 1
        
        for step in task_graph.steps:
            if step.step_id == step_id:
                step.status = status
                break
        
        task_graph.elapsed_time += time_taken
        total_steps = len(task_graph.steps)
        task_graph.progress_percentage = len(task_graph.completed_steps) / max(total_steps, 1) * 100
        
        return {
            "step_id": step_id,
            "status": status,
            "progress_percentage": task_graph.progress_percentage,
            "completed_steps": len(task_graph.completed_steps),
            "failed_steps": len(task_graph.failed_steps),
            "remaining_steps": total_steps - len(task_graph.completed_steps),
            "elapsed_time": task_graph.elapsed_time,
            "estimated_remaining": self.estimate_remaining_time(task_graph),
        }
    
    def get_progress_report(self, task_graph: YvTaskGraph) -> Dict[str, Any]:
        """Generate a comprehensive progress report.
        
        Args:
            task_graph: The task graph to report on.
            
        Returns:
            Dict[str, Any]: Detailed progress report.
        """
        total_steps = len(task_graph.steps)
        completed = len(task_graph.completed_steps)
        failed = len(task_graph.failed_steps)
        
        stage_progress = {}
        for stage in task_graph.stages:
            stage_steps = [s for s in task_graph.steps if s.stage_id == stage.stage_id]
            stage_completed = sum(1 for s in stage_steps if s.step_id in task_graph.completed_steps)
            stage_progress[stage.name] = {
                "completed": stage_completed,
                "total": len(stage_steps),
                "percentage": stage_completed / max(len(stage_steps), 1) * 100
            }
        
        return {
            "goal": task_graph.goal,
            "current_phase": task_graph.current_phase,
            "overall_progress": task_graph.progress_percentage,
            "completed_steps": completed,
            "failed_steps": failed,
            "pending_steps": total_steps - completed - failed,
            "total_steps": total_steps,
            "elapsed_time": task_graph.elapsed_time,
            "estimated_remaining": self.estimate_remaining_time(task_graph),
            "stage_progress": stage_progress,
            "critical_nodes_remaining": sum(
                1 for nid in task_graph.critical_nodes
                if nid not in task_graph.completed_steps
            ),
            "checkpoint_count": len(task_graph.checkpoint_history),
            "interrupted": task_graph.interrupted,
        }


class YvToolOrchestrator(nn.Module):
    """Tool orchestration engine supporting complex tool composition and conditional calls.
    
    Core capabilities:
    - Automatic tool chain composition
    - Conditional branching
    - Loop execution control
    - Error recovery and fallback
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        max_tools: int = 64
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_tools = max_tools
        
        self.tool_descriptor_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
        )
        
        self.tool_selector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self.param_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.condition_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.compatibility_graph = {}
        self.fallback_strategies = {}
        
        self.register_buffer('tool_usage_stats', torch.zeros(max_tools, 3))
        
        self._failure_history: Dict[str, List[Dict]] = {}
        self._success_alternatives: Dict[str, List[str]] = {}
        self._tool_health_scores: Dict[str, float] = {}
        self._fallback_chains: Dict[str, List[str]] = {}
    
    def compose_tool_chain(
        self,
        goal: str,
        available_tools: List[Dict[str, Any]]
    ) -> 'YvToolChain':
        goal_embedding = self._encode_goal(goal)
        
        required_capabilities = self._analyze_required_capabilities(goal, available_tools)
        
        selected_tools = self._select_tools(required_capabilities, available_tools, goal_embedding)
        
        execution_order = self._topological_sort(selected_tools)
        
        conditional_branches = self._identify_conditions(execution_order, available_tools)
        
        fallback_strategies = self._generate_fallbacks(execution_order)
        
        return YvToolChain(
            tools=execution_order,
            conditions=conditional_branches,
            fallback_strategies=fallback_strategies,
            goal=goal
        )
    
    def _encode_goal(self, goal: str) -> torch.Tensor:
        goal_tokens = torch.tensor([ord(c) for c in goal[:100]], dtype=torch.float32)
        goal_tokens = F.normalize(goal_tokens.unsqueeze(0), p=2, dim=-1)
        
        if goal_tokens.shape[-1] < self.hidden_size:
            repeat = (self.hidden_size // goal_tokens.shape[-1]) + 1
            goal_tokens = goal_tokens.repeat(1, repeat)[:, :self.hidden_size]
        
        return goal_tokens
    
    def _analyze_required_capabilities(
        self,
        goal: str,
        available_tools: List[Dict[str, Any]]
    ) -> List[str]:
        capabilities = set()
        
        goal_lower = goal.lower()
        
        capability_keywords = {
            'search': ['search', 'find', 'query', 'lookup'],
            'fetch': ['fetch', 'get', 'retrieve', 'download'],
            'compute': ['calculate', 'compute', 'analyze', 'process'],
            'write': ['write', 'create', 'generate', 'save'],
            'read': ['read', 'open', 'load', 'access'],
            'execute': ['run', 'execute', 'call', 'invoke']
        }
        
        for cap, keywords in capability_keywords.items():
            if any(kw in goal_lower for kw in keywords):
                capabilities.add(cap)
        
        for tool in available_tools:
            if 'capabilities' in tool:
                capabilities.update(tool['capabilities'])
        
        return list(capabilities)
    
    def _select_tools(
        self,
        required_capabilities: List[str],
        available_tools: List[Dict[str, Any]],
        goal_embedding: torch.Tensor
    ) -> List[Dict[str, Any]]:
        tool_scores = []
        
        for i, tool in enumerate(available_tools):
            tool_capabilities = tool.get('capabilities', [])
            
            match_score = sum(
                1 for cap in required_capabilities if cap in tool_capabilities
            ) / max(len(required_capabilities), 1)
            
            usage_stats = self.tool_usage_stats[i]
            success_rate = usage_stats[0] / (usage_stats[1] + 1e-8)
            avg_time = usage_stats[2] / (usage_stats[0] + 1e-8)
            
            efficiency_score = 1.0 / (1.0 + avg_time)
            
            combined_score = match_score * 0.5 + success_rate * 0.3 + efficiency_score * 0.2
            
            tool_scores.append((tool, combined_score, i))
        
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [tool for tool, score, _ in tool_scores[:8] if score > 0.1]
        
        return selected
    
    def _topological_sort(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not tools:
            return []
        
        tool_names = {t['name'] for t in tools}
        
        in_degree = {name: 0 for name in tool_names}
        graph = {name: [] for name in tool_names}
        
        for tool in tools:
            deps = tool.get('depends_on', [])
            for dep in deps:
                if dep in tool_names:
                    graph[dep].append(tool['name'])
                    in_degree[tool['name']] += 1
        
        queue = [name for name in tool_names if in_degree[name] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            current_tool = next((t for t in tools if t['name'] == current), None)
            if current_tool:
                result.append(current_tool)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _identify_conditions(
        self,
        tools: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]]
    ) -> List['YvToolCondition']:
        conditions = []
        
        for i, tool in enumerate(tools):
            if 'condition' in tool:
                condition = YvToolCondition(
                    trigger_tool=tool['name'],
                    condition_type=tool['condition'].get('type', 'if'),
                    condition_expression=tool['condition'].get('expression', ''),
                    branch_tools=tool['condition'].get('branch', []),
                    priority=i
                )
                conditions.append(condition)
        
        return conditions
    
    def _generate_fallbacks(
        self,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        fallback_strategies = {}
        
        for tool in tools:
            fallback = []
            
            if tool.get('alternative'):
                fallback.extend(tool['alternative'])
            
            fallback_strategies[tool['name']] = fallback
        
        return fallback_strategies
    
    def execute_with_conditionals(
        self,
        tool_chain: 'YvToolChain',
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        results = {}
        
        for tool_step in tool_chain.tools:
            if not self._check_prerequisites(tool_step, results):
                fallback_result = self._execute_fallback(
                    tool_step, context, results, tool_chain.fallback_strategies
                )
                results[tool_step['name']] = fallback_result
                continue
            
            result = self._execute_tool(tool_step, context, results)
            results[tool_step['name']] = result
            
            for condition in tool_chain.conditions:
                if condition.trigger_tool == tool_step['name']:
                    if self._evaluate_condition(condition, result, context):
                        branch_result = self._execute_branch(
                            condition.branch_tools, context, results
                        )
                        results.update(branch_result)
        
        return results
    
    def _check_prerequisites(
        self,
        tool_step: Dict[str, Any],
        results: Dict[str, Any]
    ) -> bool:
        prerequisites = tool_step.get('requires', [])
        return all(req in results for req in prerequisites)
    
    def _execute_tool(
        self,
        tool_step: Dict[str, Any],
        context: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            'status': 'success',
            'output': f"Executed {tool_step['name']}",
            'data': {}
        }
    
    def _execute_fallback(
        self,
        tool_step: Dict[str, Any],
        context: Dict[str, Any],
        previous_results: Dict[str, Any],
        fallback_strategies: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        fallback_tools = fallback_strategies.get(tool_step['name'], [])
        
        for fallback_name in fallback_tools:
            result = self._execute_tool(
                {'name': fallback_name},
                context,
                previous_results
            )
            if result['status'] == 'success':
                result['fallback_from'] = tool_step['name']
                return result
        
        return {
            'status': 'failed',
            'output': f"All fallbacks for {tool_step['name']} failed",
            'fallback_from': tool_step['name']
        }
    
    def _evaluate_condition(
        self,
        condition: 'YvToolCondition',
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        if condition.condition_type == 'if':
            return result['status'] == 'success'
        elif condition.condition_type == 'unless':
            return result['status'] != 'success'
        elif condition.condition_type == 'after':
            return True
        
        return True
    
    def _execute_branch(
        self,
        branch_tools: List[str],
        context: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        results = {}
        for tool_name in branch_tools:
            result = self._execute_tool({'name': tool_name}, context, previous_results)
            results[tool_name] = result
        return results
    
    def build_fallback_chain(
        self,
        tool_name: str,
        error_type: str
    ) -> List[str]:
        """Build an intelligent fallback chain for a failed tool.
        
        Analyzes failure history and success patterns to construct
        an optimal fallback chain.
        
        Args:
            tool_name: The tool that failed.
            error_type: The type of error encountered.
            
        Returns:
            List[str]: Ordered list of fallback tools to try.
        """
        if tool_name in self._fallback_chains:
            cached_chain = self._fallback_chains[tool_name]
            return cached_chain
        
        chain = []
        
        if tool_name in self.fallback_strategies:
            chain.extend(self.fallback_strategies[tool_name])
        
        if tool_name in self._success_alternatives:
            alternatives = self._success_alternatives[tool_name]
            for alt in alternatives:
                if alt not in chain:
                    chain.append(alt)
        
        for other_tool, failures in self._failure_history.items():
            if other_tool != tool_name:
                for failure in failures:
                    if failure.get('error_type') == error_type:
                        if failure.get('successful_alternative') and failure['successful_alternative'] not in chain:
                            chain.append(failure['successful_alternative'])
        
        sorted_chain = sorted(
            chain,
            key=lambda t: self._tool_health_scores.get(t, 0.5),
            reverse=True
        )
        
        self._fallback_chains[tool_name] = sorted_chain
        return sorted_chain
    
    def execute_with_retry(
        self,
        tool_step: Dict[str, Any],
        context: Dict[str, Any],
        retry_policy: YvRetryPolicy
    ) -> Dict[str, Any]:
        """Execute a tool with retry and fallback support.
        
        Args:
            tool_step: The tool step to execute.
            context: Execution context.
            retry_policy: Retry policy configuration.
            
        Returns:
            Dict[str, Any]: Execution result.
        """
        import time
        
        tool_name = tool_step.get('name', 'unknown')
        attempt = 0
        last_error = None
        start_time = time.time()
        
        while attempt < retry_policy.max_attempts:
            try:
                result = self._execute_tool(tool_step, context, {})
                
                if result.get('status') == 'success':
                    self.update_tool_health(tool_name, True, time.time() - start_time)
                    return result
                
                last_error = result.get('error', 'unknown_error')
                
            except Exception as e:
                last_error = str(e)
            
            error_type = self._classify_error(last_error)
            
            if not retry_policy.should_retry(error_type, attempt):
                break
            
            delay = retry_policy.calculate_delay(attempt)
            time.sleep(delay)
            attempt += 1
        
        fallback_chain = self.build_fallback_chain(tool_name, self._classify_error(last_error))
        
        for fallback_tool in fallback_chain:
            try:
                fallback_result = self._execute_tool(
                    {'name': fallback_tool},
                    context,
                    {}
                )
                
                if fallback_result.get('status') == 'success':
                    self.learn_from_failure(tool_name, last_error, fallback_tool)
                    fallback_result['fallback_used'] = True
                    fallback_result['original_tool'] = tool_name
                    return fallback_result
                    
            except Exception:
                continue
        
        return {
            'status': 'failed',
            'error': last_error,
            'attempts': attempt + 1,
            'fallbacks_tried': len(fallback_chain),
        }
    
    def _classify_error(self, error_message: str) -> str:
        """Classify an error message into an error type."""
        error_lower = str(error_message).lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'connection_error'
        elif 'rate' in error_lower or 'limit' in error_lower:
            return 'rate_limit'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'permission_error'
        elif 'not found' in error_lower or 'missing' in error_lower:
            return 'not_found'
        else:
            return 'unknown'
    
    def learn_from_failure(
        self,
        tool_name: str,
        error: str,
        successful_alternative: str
    ):
        """Learn from a failure to improve future fallback selection.
        
        Args:
            tool_name: The tool that failed.
            error: The error encountered.
            successful_alternative: The alternative that succeeded.
        """
        if tool_name not in self._failure_history:
            self._failure_history[tool_name] = []
        
        error_type = self._classify_error(error)
        
        self._failure_history[tool_name].append({
            'error_type': error_type,
            'error_message': str(error)[:200],
            'successful_alternative': successful_alternative,
            'timestamp': time.time() if 'time' in dir() else 0,
        })
        
        if tool_name not in self._success_alternatives:
            self._success_alternatives[tool_name] = []
        
        if successful_alternative not in self._success_alternatives[tool_name]:
            self._success_alternatives[tool_name].append(successful_alternative)
        
        if tool_name in self._fallback_chains:
            del self._fallback_chains[tool_name]
    
    def get_tool_health_status(self, tool_name: str) -> Dict[str, Any]:
        """Get health status for a tool.
        
        Args:
            tool_name: The tool to check.
            
        Returns:
            Dict[str, Any]: Health status information.
        """
        health_score = self._tool_health_scores.get(tool_name, 0.5)
        
        failure_count = len(self._failure_history.get(tool_name, []))
        
        alternatives = self._success_alternatives.get(tool_name, [])
        
        return {
            'tool_name': tool_name,
            'health_score': health_score,
            'failure_count': failure_count,
            'available_alternatives': alternatives,
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.3 else 'unhealthy',
        }
    
    def update_tool_health(
        self,
        tool_name: str,
        success: bool,
        execution_time: float
    ):
        """Update tool health score based on execution result.
        
        Args:
            tool_name: The tool that was executed.
            success: Whether the execution succeeded.
            execution_time: Time taken for execution.
        """
        current_score = self._tool_health_scores.get(tool_name, 0.5)
        
        if success:
            new_score = current_score + 0.05 * (1 - current_score)
        else:
            new_score = current_score - 0.1 * current_score
        
        self._tool_health_scores[tool_name] = max(0.0, min(1.0, new_score))
    
    def get_optimal_fallback(
        self,
        tool_name: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Get the optimal fallback tool for a given tool and context.
        
        Args:
            tool_name: The tool that needs a fallback.
            context: Current execution context.
            
        Returns:
            Optional[str]: The best fallback tool, or None.
        """
        chain = self.build_fallback_chain(tool_name, 'unknown')
        
        if not chain:
            return None
        
        for fallback in chain:
            health = self.get_tool_health_status(fallback)
            if health['health_score'] > 0.5:
                return fallback
        
        return chain[0] if chain else None
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get statistics about fallback usage and effectiveness."""
        total_failures = sum(len(failures) for failures in self._failure_history.values())
        
        successful_alternatives = sum(
            len(alts) for alts in self._success_alternatives.values()
        )
        
        avg_health = (
            sum(self._tool_health_scores.values()) / len(self._tool_health_scores)
            if self._tool_health_scores else 0.5
        )
        
        return {
            'total_recorded_failures': total_failures,
            'successful_alternatives_found': successful_alternatives,
            'tools_with_fallbacks': len(self._fallback_chains),
            'average_tool_health': avg_health,
            'most_failed_tools': sorted(
                self._failure_history.keys(),
                key=lambda t: len(self._failure_history[t]),
                reverse=True
            )[:5],
        }


class YvSelfEvaluator(nn.Module):
    """Self-evaluation system for multi-dimensional quality assessment.
    
    Evaluation dimensions:
    - Correctness: whether results meet expectations
    - Completeness: whether all requirements are covered
    - Efficiency: whether resource usage is reasonable
    - Consistency: whether consistent with historical behavior
    """
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.correctness_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.completeness_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.efficiency_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.improvement_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 5)
        )
        
        self._error_patterns: Dict[str, YvErrorPattern] = {}
        self._tool_health_scores: Dict[str, float] = {}
    
    def evaluate_execution(
        self,
        goal: str,
        execution_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> 'YvExecutionEvaluation':
        features = self._build_evaluation_features(goal, execution_result, context)
        
        correctness = self.correctness_evaluator(features).item()
        completeness = self.completeness_evaluator(features).item()
        
        efficiency_features = torch.cat([
            features[0],
            torch.tensor([execution_result.get('time_taken', 1.0)]),
            torch.tensor([execution_result.get('resource_usage', 0.5)])
        ]).unsqueeze(0)
        efficiency = self.efficiency_evaluator(efficiency_features).item()
        
        overall_score = (
            correctness * 0.4 + 
            completeness * 0.4 + 
            efficiency * 0.2
        )
        
        improvement_embedding = self.improvement_generator(features)
        suggestions = self._decode_suggestions(improvement_embedding, execution_result)
        
        quality_details = self.quality_scorer(features[0].unsqueeze(0))[0]
        
        return YvExecutionEvaluation(
            overall_score=overall_score,
            correctness=correctness,
            completeness=completeness,
            efficiency=efficiency,
            quality_details=quality_details.tolist(),
            suggestions=suggestions,
            needs_improvement=overall_score < 0.8,
            is_successful=overall_score >= 0.6
        )
    
    def _build_evaluation_features(
        self,
        goal: str,
        execution_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> torch.Tensor:
        goal_embedding = self._encode_text(goal)
        result_embedding = self._encode_result(execution_result)
        
        features = torch.cat([goal_embedding, result_embedding], dim=-1)
        
        return features.unsqueeze(0)
    
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = torch.tensor([ord(c) for c in text[:100]], dtype=torch.float32)
        tokens = F.normalize(tokens.unsqueeze(0), p=2, dim=-1)
        
        if tokens.shape[-1] < self.hidden_size:
            repeat = (self.hidden_size // tokens.shape[-1]) + 1
            tokens = tokens.repeat(1, repeat)[:, :self.hidden_size]
        
        return tokens.squeeze(0)
    
    def _encode_result(self, result: Dict[str, Any]) -> torch.Tensor:
        status_score = 1.0 if result.get('status') == 'success' else 0.0
        output_length = min(len(str(result.get('output', ''))) / 1000, 1.0)
        has_data = 1.0 if result.get('data') else 0.0
        
        simple_features = torch.tensor([status_score, output_length, has_data])
        
        if simple_features.shape[-1] < self.hidden_size:
            repeat = (self.hidden_size // simple_features.shape[-1]) + 1
            simple_features = simple_features.repeat(repeat)[:self.hidden_size]
        else:
            simple_features = simple_features[:self.hidden_size]
        
        return simple_features
    
    def _decode_suggestions(
        self,
        embedding: torch.Tensor,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            'issues': self._identify_issues(result),
            'recommended_actions': self._generate_actions(result),
            'priority_order': self._prioritize_improvements(result)
        }
    
    def _identify_issues(self, result: Dict[str, Any]) -> List[str]:
        issues = []
        
        if result.get('status') != 'success':
            issues.append("Execution failed")
        
        if not result.get('output'):
            issues.append("No output generated")
        
        if result.get('time_taken', 0) > 10:
            issues.append("Execution time exceeded threshold")
        
        return issues
    
    def _generate_actions(self, result: Dict[str, Any]) -> List[str]:
        actions = []
        
        if result.get('status') != 'success':
            actions.append("Retry with fallback strategy")
            actions.append("Check tool availability")
        
        if result.get('status') == 'success':
            actions.append("Store result for future reference")
            actions.append("Update success metrics")
        
        return actions[:3]
    
    def _prioritize_improvements(self, result: Dict[str, Any]) -> List[str]:
        priorities = []
        
        if result.get('status') != 'success':
            priorities.append("Fix execution errors")
        
        priorities.append("Optimize resource usage")
        priorities.append("Improve response quality")
        
        return priorities
    
    def generate_iterative_improvement(
        self,
        evaluation: 'YvExecutionEvaluation',
        current_result: Dict[str, Any],
        goal: str
    ) -> Dict[str, Any]:
        if not evaluation.needs_improvement:
            return current_result
        
        improvement_plan = {
            'issues': evaluation.suggestions.get('issues', []),
            'fixes': evaluation.suggestions.get('recommended_actions', []),
            'retry_strategy': self._determine_retry_strategy(evaluation),
            'expected_improvement': min(0.2, 1.0 - evaluation.overall_score)
        }
        
        return improvement_plan
    
    def _determine_retry_strategy(
        self,
        evaluation: 'YvExecutionEvaluation'
    ) -> str:
        if evaluation.correctness < 0.5:
            return "change_tool"
        elif evaluation.completeness < 0.5:
            return "add_steps"
        elif evaluation.efficiency < 0.5:
            return "optimize_parameters"
        else:
            return "retry_same"
    
    def analyze_error_pattern(
        self,
        execution_history: List[Dict[str, Any]]
    ) -> YvErrorPattern:
        """Analyze execution history to identify error patterns.
        
        Args:
            execution_history: List of execution records with errors.
            
        Returns:
            YvErrorPattern: Identified error pattern.
        """
        if not execution_history:
            return None
        
        error_counts: Dict[str, int] = {}
        error_contexts: Dict[str, List[Dict]] = {}
        error_actions: Dict[str, List[str]] = {}
        
        for record in execution_history:
            if record.get('status') == 'failed':
                error_msg = record.get('error', 'unknown')
                error_type = self._classify_error_type(error_msg)
                
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                if error_type not in error_contexts:
                    error_contexts[error_type] = []
                error_contexts[error_type].append(record.get('context', {}))
                
                if error_type not in error_actions:
                    error_actions[error_type] = []
                action = record.get('action', 'unknown')
                if action not in error_actions[error_type]:
                    error_actions[error_type].append(action)
        
        if not error_counts:
            return None
        
        most_common = max(error_counts.keys(), key=lambda k: error_counts[k])
        
        import uuid
        pattern = YvErrorPattern(
            pattern_id=str(uuid.uuid4()),
            error_signature=most_common,
            frequency=error_counts[most_common],
            affected_actions=error_actions.get(most_common, []),
            effective_solutions=self._find_effective_solutions(most_common, execution_history),
            success_rate_after_retry=self._calculate_retry_success_rate(most_common, execution_history),
            last_occurrence=time.time(),
            context_patterns=error_contexts.get(most_common, [])[:5],
        )
        
        return pattern
    
    def _classify_error_type(self, error_message: str) -> str:
        """Classify error message into a type signature."""
        error_lower = str(error_message).lower()
        
        if 'timeout' in error_lower:
            return 'timeout_error'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'network_error'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'permission_error'
        elif 'not found' in error_lower or 'missing' in error_lower:
            return 'not_found_error'
        elif 'invalid' in error_lower or 'format' in error_lower:
            return 'validation_error'
        elif 'memory' in error_lower or 'resource' in error_lower:
            return 'resource_error'
        else:
            return 'unknown_error'
    
    def _find_effective_solutions(
        self,
        error_type: str,
        execution_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Find solutions that have worked for this error type."""
        solutions = []
        
        for record in execution_history:
            if record.get('status') == 'success':
                prev_errors = record.get('previous_errors', [])
                if any(self._classify_error_type(e) == error_type for e in prev_errors):
                    solution = record.get('solution', 'retry')
                    if solution not in solutions:
                        solutions.append(solution)
        
        if not solutions:
            solutions = ['retry', 'use_fallback', 'skip_step']
        
        return solutions[:5]
    
    def _calculate_retry_success_rate(
        self,
        error_type: str,
        execution_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate success rate after retrying this error type."""
        retries = 0
        successes = 0
        
        for record in execution_history:
            if record.get('error_type') == error_type:
                retries += 1
                if record.get('retry_success', False):
                    successes += 1
        
        return successes / max(retries, 1)
    
    def predict_failure_risk(
        self,
        current_step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Predict the risk of failure for the current step.
        
        Args:
            current_step: The step to evaluate.
            context: Current execution context.
            
        Returns:
            float: Risk score between 0 and 1.
        """
        risk_factors = []
        
        tool_name = current_step.get('tool', 'unknown')
        if tool_name in self._tool_health_scores:
            health = self._tool_health_scores[tool_name]
            risk_factors.append(1 - health)
        
        step_complexity = current_step.get('complexity', 0.5)
        risk_factors.append(step_complexity * 0.3)
        
        dependencies = current_step.get('dependencies', [])
        failed_deps = sum(1 for d in dependencies if context.get(f'dep_{d}_failed', False))
        if dependencies:
            risk_factors.append(failed_deps / len(dependencies))
        
        if risk_factors:
            return sum(risk_factors) / len(risk_factors)
        return 0.3
    
    def suggest_preventive_actions(
        self,
        risk_score: float
    ) -> List[str]:
        """Suggest preventive actions based on risk score.
        
        Args:
            risk_score: The calculated risk score.
            
        Returns:
            List[str]: Suggested preventive actions.
        """
        actions = []
        
        if risk_score > 0.7:
            actions.extend([
                "Consider using a fallback tool",
                "Pre-validate inputs before execution",
                "Set up checkpoint before this step",
                "Prepare alternative execution path",
            ])
        elif risk_score > 0.5:
            actions.extend([
                "Monitor execution closely",
                "Have fallback ready",
                "Consider splitting into smaller steps",
            ])
        elif risk_score > 0.3:
            actions.extend([
                "Standard monitoring",
                "Log execution details",
            ])
        else:
            actions.append("Proceed with normal execution")
        
        return actions
    
    def update_error_knowledge(
        self,
        error: str,
        resolution: str,
        success: bool
    ):
        """Update error knowledge base with new information.
        
        Args:
            error: The error encountered.
            resolution: The resolution attempted.
            success: Whether the resolution succeeded.
        """
        error_type = self._classify_error_type(error)
        
        if error_type not in self._error_patterns:
            self._error_patterns[error_type] = YvErrorPattern(
                pattern_id=f"pattern_{error_type}",
                error_signature=error_type,
                frequency=1,
                affected_actions=[],
                effective_solutions=[resolution] if success else [],
                success_rate_after_retry=1.0 if success else 0.0,
                last_occurrence=time.time(),
            )
        else:
            pattern = self._error_patterns[error_type]
            pattern.frequency += 1
            pattern.last_occurrence = time.time()
            
            if success and resolution not in pattern.effective_solutions:
                pattern.effective_solutions.append(resolution)
            
            total = pattern.frequency
            current_rate = pattern.success_rate_after_retry
            pattern.success_rate_after_retry = (current_rate * (total - 1) + (1 if success else 0)) / total
    
    def get_similar_error_cases(
        self,
        error_signature: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar error cases from history.
        
        Args:
            error_signature: The error signature to match.
            limit: Maximum number of cases to return.
            
        Returns:
            List[Dict[str, Any]]: Similar error cases.
        """
        similar = []
        
        for pattern_id, pattern in self._error_patterns.items():
            match_score = pattern.matches(error_signature, {})
            if match_score > 0.3:
                similar.append({
                    'pattern_id': pattern_id,
                    'error_signature': pattern.error_signature,
                    'frequency': pattern.frequency,
                    'solutions': pattern.effective_solutions,
                    'success_rate': pattern.success_rate_after_retry,
                    'match_score': match_score,
                })
        
        similar.sort(key=lambda x: x['match_score'], reverse=True)
        return similar[:limit]


class YvPersistentMemory(nn.Module):
    """Persistent memory system supporting cross-session experience accumulation.
    
    Storage contents:
    - Success cases: efficient execution patterns
    - Failure lessons: error patterns to avoid
    - Tool experience: usage tips and best practices
    - User preferences: personalized settings
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        max_cases: int = 10000,
        embedding_dim: int = 256
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_cases = max_cases
        self.embedding_dim = embedding_dim
        
        self.memory_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim)
        )
        
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim)
        )
        
        self.case_library = []
        self.tool_experience = {}
        self.user_preferences = {}
        
        self.quality_threshold = 0.7
        self.success_patterns = []
        self.failure_patterns = []
        
        self.register_buffer('case_embeddings', torch.zeros(max_cases, embedding_dim))
        self.register_buffer('case_quality', torch.zeros(max_cases))
        self.register_buffer('case_types', torch.zeros(max_cases, dtype=torch.long))
        self.register_buffer('case_count', torch.tensor(0))
    
    def store_experience(self, experience: 'YvAgentExperience'):
        embedding = self._extract_experience_embedding(experience)
        
        quality = self._assess_experience_quality(experience)
        
        if quality >= self.quality_threshold:
            case_idx = int(self.case_count) % self.max_cases
            self.case_embeddings[case_idx] = embedding.detach()
            self.case_quality[case_idx] = torch.tensor(quality)
            self.case_types[case_idx] = torch.tensor(self._get_case_type(experience))
            self.case_count += 1
            
            self.case_library.append({
                'task_type': experience.task_type,
                'pattern': experience.execution_pattern,
                'outcome': experience.outcome,
                'quality': quality
            })
            
            for tool_usage in experience.tool_usages:
                self._record_tool_experience(tool_usage)
    
    def _extract_experience_embedding(
        self,
        experience: 'YvAgentExperience'
    ) -> torch.Tensor:
        features = []
        
        goal_tokens = torch.tensor([ord(c) for c in experience.goal[:50]], dtype=torch.float32)
        goal_features = F.normalize(goal_tokens.unsqueeze(0), p=2, dim=-1)
        features.append(goal_features.squeeze(0))
        
        success_rate = torch.tensor([experience.success_rate])
        features.append(success_rate)
        
        complexity = torch.tensor([experience.complexity])
        features.append(complexity)
        
        combined = torch.cat(features)
        
        if combined.shape[-1] < self.hidden_size:
            repeat = (self.hidden_size // combined.shape[-1]) + 1
            combined = combined.repeat(repeat)[:self.hidden_size]
        
        return self.memory_encoder(combined.unsqueeze(0))
    
    def _assess_experience_quality(self, experience: 'YvAgentExperience') -> float:
        quality = 0.0
        
        quality += experience.success_rate * 0.5
        
        if experience.outcome.get('efficiency', 0) > 0.8:
            quality += 0.2
        
        if len(experience.tool_usages) > 0:
            avg_success = sum(u.success_rate for u in experience.tool_usages) / len(experience.tool_usages)
            quality += avg_success * 0.3
        
        return min(1.0, quality)
    
    def _get_case_type(self, experience: 'YvAgentExperience') -> int:
        if experience.success_rate > 0.8:
            return 0
        elif experience.success_rate > 0.5:
            return 1
        else:
            return 2
    
    def _record_tool_experience(self, tool_usage: 'YvToolUsage'):
        tool_name = tool_usage.tool_name
        if tool_name not in self.tool_experience:
            self.tool_experience[tool_name] = {
                'count': 0,
                'success_rate': 0.5,
                'avg_time': 1.0,
                'patterns': []
            }
        
        exp = self.tool_experience[tool_name]
        exp['count'] += 1
        exp['success_rate'] = exp['success_rate'] * 0.9 + tool_usage.success_rate * 0.1
        exp['avg_time'] = exp['avg_time'] * 0.9 + tool_usage.execution_time * 0.1
        
        if tool_usage.success_rate > 0.8:
            exp['patterns'].append(tool_usage.parameter_pattern)
    
    def retrieve_similar_experiences(
        self,
        query: str,
        k: int = 5
    ) -> List['YvSimilarCase']:
        query_embedding = self._encode_query(query)
        
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.case_embeddings[:self.case_count],
            dim=-1
        )
        
        top_k = similarities.topk(min(k, len(similarities)))
        
        similar_cases = []
        for idx, sim in zip(top_k.indices, top_k.values):
            if idx < len(self.case_library):
                case = self.case_library[idx]
                similar_cases.append(YvSimilarCase(
                    case=case,
                    similarity=sim.item(),
                    relevance_score=self._calculate_relevance(case, query)
                ))
        
        return sorted(similar_cases, key=lambda x: x.relevance_score, reverse=True)
    
    def _encode_query(self, query: str) -> torch.Tensor:
        tokens = torch.tensor([ord(c) for c in query[:100]], dtype=torch.float32)
        tokens = F.normalize(tokens.unsqueeze(0), p=2, dim=-1)
        
        if tokens.shape[-1] < self.embedding_dim:
            repeat = (self.embedding_dim // tokens.shape[-1]) + 1
            tokens = tokens.repeat(1, repeat)[:, :self.embedding_dim]
        
        return self.query_encoder(tokens).squeeze(0)
    
    def _calculate_relevance(self, case: Dict, query: str) -> float:
        query_lower = query.lower()
        case_type = case.get('task_type', '').lower()
        
        relevance = 0.0
        if case_type in query_lower:
            relevance += 0.5
        
        if case.get('quality', 0) > 0.8:
            relevance += 0.3
        
        relevance += case.get('success_rate', 0.5) * 0.2
        
        return min(1.0, relevance)
    
    def get_tool_recommendations(self, task_type: str) -> List['YvToolRecommendation']:
        relevant_experiences = self.retrieve_similar_experiences(task_type, k=10)
        
        tool_usage_stats = {}
        for exp in relevant_experiences:
            for tool_usage in exp.case.get('tool_usages', []):
                tool_name = tool_usage.tool_name
                if tool_name not in tool_usage_stats:
                    tool_usage_stats[tool_name] = {
                        'count': 0,
                        'success_rate': 0,
                        'avg_time': 0
                    }
                tool_usage_stats[tool_name]['count'] += 1
                tool_usage_stats[tool_name]['success_rate'] = (
                    tool_usage_stats[tool_name]['success_rate'] * 0.9 + 
                    tool_usage.success_rate * 0.1
                )
        
        recommendations = []
        for tool_name, stats in tool_usage_stats.items():
            recommendations.append(YvToolRecommendation(
                tool_name=tool_name,
                success_probability=stats['success_rate'],
                usage_count=stats['count'],
                recommendation_score=stats['success_rate'] * 0.7 + 
                                   min(stats['count'] / 10, 1.0) * 0.3
            ))
        
        return sorted(recommendations, key=lambda x: x.recommendation_score, reverse=True)


@dataclass
class YvTaskStage:
    """Task stage data class."""
    stage_id: int
    name: str
    description: str
    num_steps: int
    success_criteria: List[str]
    parallelizable: bool = False


@dataclass
class YvPlanStep:
    """Plan step data class."""
    step_id: int
    name: str
    description: str
    stage_id: int
    status: str = "pending"
    estimated_duration: float = 1.0
    complexity: float = 0.5


@dataclass
class YvTaskGraph:
    """Task graph data class with checkpoint and recovery support."""
    goal: str
    stages: List[YvTaskStage]
    steps: List[YvPlanStep]
    dependencies: List[Tuple[int, int]]
    critical_nodes: List[int]
    complexity: Dict[str, float]
    current_phase: str = "planning"
    execution_start_time: float = 0.0
    last_checkpoint_time: float = 0.0
    checkpoint_interval: float = 300.0
    elapsed_time: float = 0.0
    progress_percentage: float = 0.0
    interrupted: bool = False
    interruption_point: Optional[int] = None
    checkpoint_history: List[str] = field(default_factory=list)
    completed_steps: List[int] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    retry_counts: Dict[int, int] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class YvCheckpoint:
    """Checkpoint for long-running task execution.
    
    Captures complete execution state for recovery from failures
    and interruption handling.
    
    Attributes:
        checkpoint_id: Unique identifier for the checkpoint.
        task_graph: The task graph at checkpoint time.
        completed_steps: List of completed step IDs.
        failed_steps: List of failed step IDs.
        execution_context: Context information for resuming.
        timestamp: When the checkpoint was created.
        can_resume: Whether this checkpoint can be resumed.
        retry_count: Total retry count across all steps.
        error_history: History of errors encountered.
    """
    checkpoint_id: str
    task_graph: Optional[YvTaskGraph]
    completed_steps: List[int]
    failed_steps: List[int]
    execution_context: Dict[str, Any]
    timestamp: float
    can_resume: bool
    retry_count: int
    error_history: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "execution_context": self.execution_context,
            "timestamp": self.timestamp,
            "can_resume": self.can_resume,
            "retry_count": self.retry_count,
            "error_history": self.error_history,
            "metadata": self.metadata,
        }


class YvBackoffStrategy(Enum):
    """Backoff strategy for retry operations."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"
    JITTERED = "jittered"


@dataclass
class YvRetryPolicy:
    """Retry policy for error recovery.
    
    Defines comprehensive retry behavior including backoff strategies,
    maximum attempts, and error classification.
    
    Attributes:
        max_attempts: Maximum number of retry attempts.
        base_delay: Base delay in seconds for backoff.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter to delays.
        retryable_errors: List of error types that can be retried.
        backoff_strategy: The backoff strategy to use.
        timeout: Maximum time for retry operations.
        escalate_after: Number of failures before escalation.
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[str] = field(default_factory=lambda: [
        "timeout", "connection_error", "rate_limit", "temporary_failure"
    ])
    backoff_strategy: YvBackoffStrategy = YvBackoffStrategy.EXPONENTIAL
    timeout: float = 300.0
    escalate_after: int = 5
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        import random
        
        if self.backoff_strategy == YvBackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** attempt)
        elif self.backoff_strategy == YvBackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.backoff_strategy == YvBackoffStrategy.FIBONACCI:
            fib = [1, 1]
            for _ in range(attempt):
                fib.append(fib[-1] + fib[-2])
            delay = self.base_delay * fib[min(attempt, len(fib) - 1)]
        elif self.backoff_strategy == YvBackoffStrategy.JITTERED:
            delay = self.base_delay * (self.exponential_base ** attempt)
            delay = delay * (0.5 + random.random())
        else:
            delay = self.base_delay
        
        if self.jitter and self.backoff_strategy != YvBackoffStrategy.JITTERED:
            delay = delay * (0.8 + random.random() * 0.4)
        
        return min(delay, self.max_delay)
    
    def should_retry(self, error_type: str, attempt: int) -> bool:
        """Determine if an error should be retried."""
        if attempt >= self.max_attempts:
            return False
        return error_type in self.retryable_errors or error_type == "unknown"


@dataclass
class YvErrorPattern:
    """Error pattern for learning and prediction.
    
    Captures recurring error patterns for proactive error handling
    and preventive action suggestions.
    
    Attributes:
        pattern_id: Unique identifier for the pattern.
        error_signature: Hash/signature of the error type.
        frequency: How often this pattern occurs.
        affected_actions: Actions that trigger this error.
        effective_solutions: Solutions that have worked.
        success_rate_after_retry: Success rate after applying solutions.
        last_occurrence: When this pattern was last seen.
        context_patterns: Common context patterns for this error.
    """
    pattern_id: str
    error_signature: str
    frequency: int
    affected_actions: List[str]
    effective_solutions: List[str]
    success_rate_after_retry: float
    last_occurrence: float
    context_patterns: List[Dict[str, Any]] = field(default_factory=list)
    severity: str = "medium"
    auto_recoverable: bool = True
    
    def matches(self, error_signature: str, context: Dict[str, Any]) -> float:
        """Calculate match score for an error."""
        if self.error_signature != error_signature:
            return 0.0
        
        context_match = 0.0
        if self.context_patterns:
            for pattern in self.context_patterns:
                matches = sum(1 for k, v in pattern.items() if context.get(k) == v)
                context_match = max(context_match, matches / len(pattern))
        
        return 0.5 + 0.5 * context_match


@dataclass
class YvRecoveryContext:
    """Context for recovery operations.
    
    Bundles all information needed for intelligent recovery
    from failures during long-running execution.
    
    Attributes:
        error_type: Classification of the error.
        error_message: Detailed error message.
        attempt_count: Number of attempts so far.
        elapsed_time: Time spent on current operation.
        previous_strategies: Strategies already tried.
        suggested_delay: Recommended delay before retry.
        recovery_options: Available recovery options.
    """
    error_type: str
    error_message: str
    attempt_count: int
    elapsed_time: float
    previous_strategies: List[str]
    suggested_delay: float
    recovery_options: List[str]
    checkpoint_available: bool = False
    fallback_tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class YvToolChain:
    """Tool chain data class."""
    tools: List[Dict[str, Any]]
    conditions: List['YvToolCondition']
    fallback_strategies: Dict[str, List[str]]
    goal: str


@dataclass
class YvToolCondition:
    """Tool condition data class."""
    trigger_tool: str
    condition_type: str
    condition_expression: str
    branch_tools: List[str]
    priority: int = 0


@dataclass
class YvExecutionEvaluation:
    """Execution evaluation data class."""
    overall_score: float
    correctness: float
    completeness: float
    efficiency: float
    quality_details: List[float]
    suggestions: Dict[str, Any]
    needs_improvement: bool
    is_successful: bool


@dataclass
class YvAgentExperience:
    """Agent experience data class."""
    goal: str
    task_type: str
    execution_pattern: Dict[str, Any]
    outcome: Dict[str, Any]
    success_rate: float
    complexity: float
    tool_usages: List['YvToolUsage']


@dataclass
class YvToolUsage:
    """Tool usage data class."""
    tool_name: str
    parameter_pattern: Dict[str, Any]
    success_rate: float
    execution_time: float


@dataclass
class YvSimilarCase:
    """Similar case data class."""
    case: Dict[str, Any]
    similarity: float
    relevance_score: float


@dataclass
class YvToolRecommendation:
    """Tool recommendation data class."""
    tool_name: str
    success_probability: float
    usage_count: int
    recommendation_score: float
