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
ReAct (Reasoning + Acting) Agentic for PiscesL1.

Implements the complete Goal -> Plan -> Execute -> Reflect closed loop with:
- Goal Understanding: Understand user's true intent
- Task Decomposition: Break down complex tasks
- ReAct Loop: Think -> Act -> Observe -> Reflect cycle
- Self-Correction: Automatic error recovery
- Memory Integration: Persistent context

Key Papers:
- ReAct: Synergizing Reasoning and Acting in Language Models (2022)
- Chain-of-Thought Prompting Elicits Reasoning in LLMs (2023)
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn

from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Multimodal", file_path=get_log_file("Yv.Multimodal"), enable_file=True)


class AgenticState(Enum):
    """Agentic execution states."""
    IDLE = "idle"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


class ActionType(Enum):
    """Supported action types."""
    ANSWER = "answer"
    SEARCH = "search"
    TOOL_CALL = "tool_call"
    CODE_EXEC = "code_exec"
    WRITE_FILE = "write_file"
    READ_FILE = "read_file"
    ASK_HUMAN = "ask_human"
    PLAN_TASK = "plan_task"
    REFLECT = "reflect"
    FINISH = "finish"


class RetryStrategy(Enum):
    """Retry strategy for error recovery."""
    IMMEDIATE = "immediate"
    BACKOFF = "backoff"
    ALTERNATE = "alternate"
    DEGRADE = "degrade"
    SKIP = "skip"
    ABORT = "abort"
    ESCALATE = "escalate"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass
class Goal:
    """User goal/intent representation."""
    original_text: str
    intent: str = ""
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    domain: str = "general"
    complexity: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "intent": self.intent,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "domain": self.domain,
            "complexity": self.complexity,
        }


@dataclass
class PlanStep:
    """Single step in a plan."""
    step_id: int
    description: str
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None
    reflection: Optional[str] = None


@dataclass
class Plan:
    """Task plan with multiple steps."""
    goal: Goal
    steps: List[PlanStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "created"
    
    def add_step(self, description: str, action_type: ActionType, 
                 params: Dict[str, Any] = None, depends_on: List[int] = None) -> int:
        """Add a step to the plan."""
        step_id = len(self.steps)
        step = PlanStep(
            step_id=step_id,
            description=description,
            action_type=action_type,
            params=params or {},
            depends_on=depends_on or [],
        )
        self.steps.append(step)
        return step_id
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next executable step."""
        for step in self.steps[self.current_step:]:
            if step.status == "pending":
                deps_done = all(
                    self.steps[dep].status == "completed" 
                    for dep in step.depends_on
                )
                if deps_done:
                    return step
        return None
    
    def update_step(self, step_id: int, status: str, result: str = None):
        """Update step status and result."""
        if 0 <= step_id < len(self.steps):
            self.steps[step_id].status = status
            if result:
                self.steps[step_id].result = result


@dataclass
class Action:
    """Agent action representation."""
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    confidence: float = 1.0


@dataclass
class Observation:
    """Action observation/result."""
    action: Action
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class Reflection:
    """Self-reflection on recent actions."""
    thought: str
    is_valid: bool = True
    suggestions: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)


class YvGoalUnderstanding(nn.Module):
    """Goal understanding module.
    
    Analyzes user input to understand true intent, constraints, and success criteria.
    """
    
    def __init__(self, hidden_size: int = 4096):
        """Initialize goal understanding module.
        
        Args:
            hidden_size: Hidden size for neural modules
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 10),
        )
        
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 8),
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Understand user goal from hidden states.
        
        Args:
            hidden_states: [B, T, H] hidden states
            
        Returns:
            Goal understanding dict
        """
        pooled = hidden_states[:, -1, :]
        
        intent_logits = self.intent_classifier(pooled)
        intent = intent_logits.argmax(dim=-1).item()
        
        complexity = self.complexity_estimator(pooled).item()
        
        domain_logits = self.domain_classifier(pooled)
        domain = domain_logits.argmax(dim=-1).item()
        
        return {
            "intent_class": intent,
            "complexity_score": complexity,
            "domain_class": domain,
        }
    
    def understand(self, text: str, model, tokenizer) -> Goal:
        """Understand user goal from text.
        
        Args:
            text: User input text
            model: Language model
            tokenizer: Tokenizer
            
        Returns:
            Goal object
        """
        prompt = f"""Analyze the user's goal and extract:
1. Core intent (what they really want)
2. Constraints (any limitations mentioned)
3. Success criteria (how to know if successful)
4. Domain (coding, writing, analysis, etc.)

User input: {text}

Output as JSON:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            return Goal(
                original_text=text,
                intent=parsed.get("intent", ""),
                constraints=parsed.get("constraints", []),
                success_criteria=parsed.get("success_criteria", []),
                domain=parsed.get("domain", "general"),
                complexity=int(parsed.get("complexity", 1)),
            )
        except (json.JSONDecodeError, ValueError):
            return Goal(
                original_text=text,
                intent=text[:100],
                domain="general",
                complexity=1,
            )


class YvTaskDecomposition(nn.Module):
    """Task decomposition module.
    
    Breaks down complex goals into executable steps.
    """
    
    def __init__(self, hidden_size: int = 4096):
        """Initialize task decomposition module."""
        super().__init__()
        self.hidden_size = hidden_size
        
        self.step_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.dependency_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        
    def decompose(self, goal: Goal, model, tokenizer) -> Plan:
        """Decompose goal into executable plan.
        
        Args:
            goal: User goal
            model: Language model
            tokenizer: Tokenizer
            
        Returns:
            Plan with decomposed steps
        """
        prompt = f"""Break down this goal into clear, executable steps:

Goal: {goal.intent}
Constraints: {', '.join(goal.constraints) if goal.constraints else 'None'}
Success criteria: {', '.join(goal.success_criteria) if goal.success_criteria else 'None'}

For each step, specify:
- Description (what to do)
- Action type (search, code_exec, write_file, answer, etc.)
- Parameters needed

Output as JSON array:
[{{"description": "...", "action_type": "...", "params": {{}}}}]
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        plan = Plan(goal=goal)
        
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            json_str = response[json_start:json_end]
            steps_data = json.loads(json_str)
            
            action_map = {
                "search": ActionType.SEARCH,
                "code": ActionType.CODE_EXEC,
                "code_exec": ActionType.CODE_EXEC,
                "write": ActionType.WRITE_FILE,
                "write_file": ActionType.WRITE_FILE,
                "read": ActionType.READ_FILE,
                "read_file": ActionType.READ_FILE,
                "answer": ActionType.ANSWER,
                "ask": ActionType.ASK_HUMAN,
                "finish": ActionType.FINISH,
            }
            
            for i, step_data in enumerate(steps_data):
                action_type = action_map.get(
                    step_data.get("action_type", "answer").lower(),
                    ActionType.ANSWER
                )
                plan.add_step(
                    description=step_data.get("description", f"Step {i+1}"),
                    action_type=action_type,
                    params=step_data.get("params", {}),
                )
                
        except (json.JSONDecodeError, ValueError) as e:
            _LOG.warning(f"Failed to parse steps: {e}")
            plan.add_step(
                description=f"Complete the goal: {goal.intent}",
                action_type=ActionType.ANSWER,
            )
        
        plan.status = "ready"
        
        return plan


class YvReActEngine(nn.Module):
    """ReAct (Reasoning + Acting) engine.
    
    Implements the Think -> Act -> Observe -> Reflect cycle.
    """
    
    def __init__(self, hidden_size: int = 4096, max_history: int = 20):
        """Initialize ReAct engine.
        
        Args:
            hidden_size: Hidden size for neural modules
            max_history: Maximum history to keep
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_history = max_history
        
        self.thought_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.action_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(ActionType)),
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        
        self.reflector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def think(self, goal: Goal, plan: Plan, history: List[Dict[str, Any]], 
              context: str = "") -> str:
        """Generate reasoning thought.
        
        Args:
            goal: User goal
            plan: Current plan
            history: Action-observation history
            context: Current context
            
        Returns:
            Thought string
        """
        next_step = plan.get_next_step()
        
        history_text = ""
        for h in history[-5:]:
            history_text += f"Thought: {h.get('thought', '')}\n"
            history_text += f"Action: {h.get('action', '')}\n"
            history_text += f"Observation: {h.get('observation', '')}\n\n"
        
        prompt = f"""You are reasoning through how to complete this goal.

Goal: {goal.intent}
Constraints: {', '.join(goal.constraints) if goal.constraints else 'None'}

Current plan progress: Step {plan.current_step + 1}/{len(plan.steps)}
{"Next step: " + next_step.description if next_step else "All steps completed"}

Recent history:
{history_text}

Current context: {context}

Think about:
1. What is the current state?
2. What should I do next?
3. What might go wrong?

Thought:"""
        
        return prompt
    
    def select_action(self, goal: Goal, plan: Plan, thought: str, 
                      context: str = "") -> Action:
        """Select next action based on reasoning.
        
        Args:
            goal: User goal
            plan: Current plan
            thought: Reasoning thought
            context: Current context
            
        Returns:
            Selected action
        """
        next_step = plan.get_next_step()
        
        if next_step is None:
            return Action(
                action_type=ActionType.FINISH,
                params={},
                thought="All steps completed",
                confidence=1.0,
            )
        
        return Action(
            action_type=next_step.action_type,
            params=next_step.params,
            thought=thought,
            confidence=0.9,
        )
    
    def observe(self, action: Action, result: Any, 
                execution_time: float = 0.0) -> Observation:
        """Process action observation.
        
        Args:
            action: Executed action
            result: Action result
            execution_time: Time taken to execute
            
        Returns:
            Observation object
        """
        success = not (
            isinstance(result, dict) and result.get("error") or
            isinstance(result, str) and "error" in result.lower()
        )
        
        return Observation(
            action=action,
            result=result,
            success=success,
            error=None if success else str(result),
            execution_time=execution_time,
        )
    
    def reflect(self, goal: Goal, plan: Plan, 
                observation: Observation) -> Reflection:
        """Reflect on action result and plan progress.
        
        Args:
            goal: User goal
            plan: Current plan
            observation: Action observation
            
        Returns:
            Reflection object
        """
        if observation.success:
            return Reflection(
                thought="Action completed successfully",
                is_valid=True,
                suggestions=[],
                corrections=[],
            )
        
        suggestions = []
        corrections = []
        
        if "tool" in str(observation.error).lower():
            suggestions.append("Try a different tool or approach")
            corrections.append("Check tool parameters and availability")
        
        if "timeout" in str(observation.error).lower():
            suggestions.append("Break task into smaller parts")
            corrections.append("Reduce scope or increase timeout")
        
        return Reflection(
            thought=f"Action failed: {observation.error}",
            is_valid=False,
            suggestions=suggestions,
            corrections=corrections,
        )


class YvSelfCorrection(nn.Module):
    """Self-correction module for error recovery with multi-level retry strategies."""
    
    def __init__(self, hidden_size: int = 4096):
        """Initialize self-correction module."""
        super().__init__()
        self.hidden_size = hidden_size
        
        self.error_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 5),
        )
        
        self.correction_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self._retry_history: Dict[str, List[Dict]] = {}
        self._error_patterns: Dict[str, Dict] = {}
        self._max_retry_attempts: int = 5
        self._base_backoff_delay: float = 1.0
        self._max_backoff_delay: float = 60.0
    
    def analyze_error(self, error: str, action_type: ActionType) -> Dict[str, Any]:
        """Analyze error type and suggest recovery.
        
        Args:
            error: Error message
            action_type: Failed action type
            
        Returns:
            Error analysis dict
        """
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            error_type = "timeout"
        elif "not found" in error_lower or "missing" in error_lower:
            error_type = "resource_not_found"
        elif "permission" in error_lower or "access" in error_lower:
            error_type = "permission_denied"
        elif "invalid" in error_lower or "wrong" in error_lower:
            error_type = "invalid_input"
        elif "connection" in error_lower or "network" in error_lower:
            error_type = "network_error"
        elif "rate" in error_lower or "limit" in error_lower:
            error_type = "rate_limit"
        else:
            error_type = "unknown"
        
        return {
            "error_type": error_type,
            "action_type": action_type.value,
            "should_retry": error_type in ["timeout", "resource_not_found", "network_error", "rate_limit"],
            "should_alternate": error_type in ["permission_denied", "invalid_input"],
            "should_abort": error_type == "unknown" and len(error) > 100,
        }
    
    def generate_correction(self, goal: Goal, plan: Plan, 
                            error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate correction strategy.
        
        Args:
            goal: User goal
            plan: Current plan
            error_info: Error analysis result
            
        Returns:
            Correction strategy dict
        """
        if error_info["should_retry"]:
            return {
                "strategy": "retry",
                "message": "Retrying with same parameters",
                "changes": [],
            }
        elif error_info["should_alternate"]:
            return {
                "strategy": "alternate",
                "message": "Trying alternative approach",
                "changes": ["Use different parameters", "Simplify task"],
            }
        else:
            return {
                "strategy": "skip",
                "message": "Skipping this step",
                "changes": ["Mark step as optional", "Continue with next step"],
            }
    
    def determine_retry_strategy(
        self,
        error: str,
        attempt_count: int,
        context: Dict[str, Any]
    ) -> RetryStrategy:
        """Determine the optimal retry strategy based on error and context.
        
        Args:
            error: The error message.
            attempt_count: Number of attempts so far.
            context: Current execution context.
            
        Returns:
            RetryStrategy: The recommended retry strategy.
        """
        error_analysis = self.analyze_error(error, ActionType.TOOL_CALL)
        error_type = error_analysis["error_type"]
        
        if attempt_count >= self._max_retry_attempts:
            return RetryStrategy.ESCALATE
        
        if error_type in ["timeout", "network_error", "rate_limit"]:
            if attempt_count == 0:
                return RetryStrategy.IMMEDIATE
            elif attempt_count < 3:
                return RetryStrategy.BACKOFF
            else:
                return RetryStrategy.ALTERNATE
        
        elif error_type in ["permission_denied"]:
            return RetryStrategy.ESCALATE
        
        elif error_type in ["invalid_input", "resource_not_found"]:
            if attempt_count < 2:
                return RetryStrategy.ALTERNATE
            else:
                return RetryStrategy.SKIP
        
        else:
            if attempt_count < 2:
                return RetryStrategy.BACKOFF
            elif attempt_count < 4:
                return RetryStrategy.ALTERNATE
            else:
                return RetryStrategy.ESCALATE
    
    def calculate_backoff_time(
        self,
        attempt_count: int,
        base_delay: float = None,
        max_delay: float = None
    ) -> float:
        """Calculate backoff time using exponential strategy with jitter.
        
        Args:
            attempt_count: Current attempt number.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay cap.
            
        Returns:
            float: Delay time in seconds.
        """
        import random
        
        base = base_delay or self._base_backoff_delay
        maximum = max_delay or self._max_backoff_delay
        
        exponential_delay = base * (2 ** attempt_count)
        
        jitter = random.uniform(0.8, 1.2)
        delay = exponential_delay * jitter
        
        return min(delay, maximum)
    
    def generate_alternate_actions(
        self,
        failed_action: Dict[str, Any],
        error_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alternative actions for a failed action.
        
        Args:
            failed_action: The action that failed.
            error_info: Information about the error.
            
        Returns:
            List[Dict[str, Any]]: List of alternative actions.
        """
        alternatives = []
        action_type = failed_action.get("type", "unknown")
        error_type = error_info.get("error_type", "unknown")
        
        if action_type == "tool_call":
            tool_name = failed_action.get("tool", "")
            
            if error_type == "timeout":
                alternatives.append({
                    "type": "tool_call",
                    "tool": tool_name,
                    "params": {**failed_action.get("params", {}), "timeout": 60},
                    "reason": "Increased timeout",
                })
            
            alternatives.append({
                "type": "tool_call",
                "tool": f"{tool_name}_fallback",
                "params": failed_action.get("params", {}),
                "reason": "Using fallback tool",
            })
        
        elif action_type == "search":
            alternatives.append({
                "type": "search",
                "query": failed_action.get("query", "") + " simplified",
                "reason": "Simplified search query",
            })
        
        elif action_type == "code_exec":
            alternatives.append({
                "type": "code_exec",
                "code": f"# Simplified version\n{failed_action.get('code', '')}",
                "reason": "Simplified code execution",
            })
        
        alternatives.append({
            "type": "ask_human",
            "question": f"The action '{action_type}' failed. How should I proceed?",
            "reason": "Escalate to user",
        })
        
        return alternatives[:3]
    
    def should_escalate(
        self,
        consecutive_failures: int,
        error_pattern: str
    ) -> bool:
        """Determine if the situation should be escalated.
        
        Args:
            consecutive_failures: Number of consecutive failures.
            error_pattern: Pattern of errors encountered.
            
        Returns:
            bool: True if should escalate.
        """
        if consecutive_failures >= self._max_retry_attempts:
            return True
        
        critical_errors = ["permission_denied", "security_violation", "data_corruption"]
        if error_pattern in critical_errors:
            return True
        
        pattern_data = self._error_patterns.get(error_pattern, {})
        if pattern_data.get("escalation_rate", 0) > 0.5:
            return True
        
        return False
    
    def learn_from_error(
        self,
        error: str,
        resolution: str,
        success: bool
    ):
        """Learn from an error for future improvement.
        
        Args:
            error: The error encountered.
            resolution: The resolution attempted.
            success: Whether the resolution succeeded.
        """
        error_analysis = self.analyze_error(error, ActionType.TOOL_CALL)
        error_type = error_analysis["error_type"]
        
        if error_type not in self._error_patterns:
            self._error_patterns[error_type] = {
                "count": 0,
                "resolutions": {},
                "success_rate": 0.0,
            }
        
        pattern = self._error_patterns[error_type]
        pattern["count"] += 1
        
        if resolution not in pattern["resolutions"]:
            pattern["resolutions"][resolution] = {"attempts": 0, "successes": 0}
        
        pattern["resolutions"][resolution]["attempts"] += 1
        if success:
            pattern["resolutions"][resolution]["successes"] += 1
        
        total_successes = sum(r["successes"] for r in pattern["resolutions"].values())
        total_attempts = sum(r["attempts"] for r in pattern["resolutions"].values())
        pattern["success_rate"] = total_successes / max(total_attempts, 1)
    
    def get_error_pattern_prediction(
        self,
        action_type: ActionType
    ) -> float:
        """Predict the likelihood of error for an action type.
        
        Args:
            action_type: The type of action to evaluate.
            
        Returns:
            float: Predicted error probability.
        """
        action_errors = 0
        total_errors = 0
        
        for error_type, pattern in self._error_patterns.items():
            if action_type.value in str(pattern.get("affected_actions", [])):
                action_errors += pattern["count"]
            total_errors += pattern["count"]
        
        if total_errors == 0:
            return 0.1
        
        return action_errors / total_errors
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get statistics about retry attempts and outcomes."""
        total_retries = sum(len(history) for history in self._retry_history.values())
        
        successful_retries = sum(
            sum(1 for attempt in history if attempt.get("success"))
            for history in self._retry_history.values()
        )
        
        return {
            "total_retry_attempts": total_retries,
            "successful_retries": successful_retries,
            "success_rate": successful_retries / max(total_retries, 1),
            "error_patterns": len(self._error_patterns),
            "most_common_errors": sorted(
                self._error_patterns.keys(),
                key=lambda k: self._error_patterns[k]["count"],
                reverse=True
            )[:5],
        }


class YvReActAgentic(nn.Module):
    """Complete ReAct Agentic for PiscesL1.
    
    Integrates Goal Understanding, Task Decomposition, ReAct Engine,
    and Self-Correction for autonomous task completion.
    """
    
    def __init__(self, hidden_size: int = 4096, max_steps: int = 20):
        """Initialize ReAct Agentic.
        
        Args:
            hidden_size: Hidden size for neural modules
            max_steps: Maximum ReAct steps per goal
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        
        self.goal_understanding = YvGoalUnderstanding(hidden_size)
        self.task_decomposition = YvTaskDecomposition(hidden_size)
        self.react_engine = YvReActEngine(hidden_size)
        self.self_correction = YvSelfCorrection(hidden_size)
        
        self.goal_encoder = nn.Linear(hidden_size, hidden_size)
        self.plan_encoder = nn.Linear(hidden_size, hidden_size)
        self.history_encoder = nn.Linear(hidden_size, hidden_size)
        
        self._checkpoint_storage: Dict[str, Dict] = {}
        self._retry_policy: Dict[str, Any] = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
        }
        self._execution_context: Dict[str, Any] = {}
        self._interruption_flag: bool = False
        self._last_checkpoint_time: float = 0.0
        self._checkpoint_interval: float = 300.0
        
    def run(self, user_input: str, model, tokenizer, 
            tools: Dict[str, Any] = None, verbose: bool = False) -> Dict[str, Any]:
        """Run agentic task completion.
        
        Args:
            user_input: User's request
            model: Language model
            tokenizer: Tokenizer
            tools: Available tools dict
            verbose: Print detailed progress
            
        Returns:
            Agentic result dict
        """
        start_time = time.time()
        state = AgenticState.IDLE
        
        goal = self.goal_understanding.understand(user_input, model, tokenizer)
        state = AgenticState.UNDERSTANDING
        
        if verbose:
            _LOG.info(f"Goal: {goal.intent}")
            _LOG.info(f"Constraints: {goal.constraints}")
        
        plan = self.task_decomposition.decompose(goal, model, tokenizer)
        state = AgenticState.PLANNING
        
        if verbose:
            _LOG.info(f"Plan ({len(plan.steps)} steps):")
            for i, step in enumerate(plan.steps):
                _LOG.info(f"  {i+1}. [{step.action_type.value}] {step.description}")
        
        history = []
        context = ""
        
        for step_idx in range(self.max_steps):
            state = AgenticState.EXECUTING
            
            thought = self.react_engine.think(goal, plan, history, context)
            
            action = self.react_engine.select_action(goal, plan, thought, context)
            
            if verbose:
                _LOG.debug(f"Thought: {thought[:100]}...")
                _LOG.debug(f"Action: {action.action_type.value}")
            
            if action.action_type == ActionType.FINISH:
                state = AgenticState.COMPLETED
                break
            
            state = AgenticState.OBSERVING
            
            action_start = time.time()
            result = self._execute_action(action, tools, model, tokenizer)
            execution_time = time.time() - action_start
            
            observation = self.react_engine.observe(action, result, execution_time)
            
            if verbose:
                status = "[OK]" if observation.success else "[FAIL]"
                print(f"{status} Result: {str(result)[:100]}...")
            
            history.append({
                "thought": thought,
                "action": action.action_type.value,
                "observation": str(result)[:200],
            })
            
            state = AgenticState.REFLECTING
            
            reflection = self.react_engine.reflect(goal, plan, observation)
            
            if not observation.success:
                state = AgenticState.EXECUTING
                error_info = self.self_correction.analyze_error(
                    str(observation.error), action.action_type
                )
                correction = self.self_correction.generate_correction(
                    goal, plan, error_info
                )
                
                if verbose:
                    print(f"🔧 Correction: {correction['strategy']}")
                
                if correction["strategy"] == "skip":
                    plan.update_step(
                        plan.current_step, "skipped", 
                        f"Skipped due to error: {observation.error}"
                    )
                else:
                    plan.update_step(
                        plan.current_step, "failed", 
                        f"Failed: {observation.error}"
                    )
            
            next_step = plan.get_next_step()
            
            if next_step:
                plan.current_step = next_step.step_id
                next_step.status = "executing"
            else:
                state = AgenticState.COMPLETED
                break
        
        total_time = time.time() - start_time
        
        return {
            "goal": goal.to_dict(),
            "plan_status": plan.status,
            "steps_completed": sum(1 for s in plan.steps if s.status == "completed"),
            "steps_failed": sum(1 for s in plan.steps if s.status == "failed"),
            "steps_skipped": sum(1 for s in plan.steps if s.status == "skipped"),
            "total_steps": len(plan.steps),
            "history_length": len(history),
            "state": state.value,
            "total_time": total_time,
            "success": state == AgenticState.COMPLETED,
        }
    
    def _execute_action(self, action: Action, tools: Dict[str, Any],
                        model, tokenizer) -> Any:
        """Execute action using available tools.
        
        Args:
            action: Action to execute
            tools: Available tools
            model: Language model
            tokenizer: Tokenizer
            
        Returns:
            Action result
        """
        if action.action_type == ActionType.ANSWER:
            return action.params.get("content", "Answer generated")
        
        elif action.action_type == ActionType.SEARCH:
            query = action.params.get("query", "")
            if tools and "search" in tools:
                return tools["search"](query)
            return f"Search query: {query}"
        
        elif action.action_type == ActionType.CODE_EXEC:
            code = action.params.get("code", "")
            if tools and "execute" in tools:
                return tools["execute"](code)
            return f"Code: {code[:100]}..."
        
        elif action.action_type == ActionType.WRITE_FILE:
            path = action.params.get("path", "")
            content = action.params.get("content", "")
            if tools and "write_file" in tools:
                return tools["write_file"](path, content)
            return f"File: {path}"
        
        elif action.action_type == ActionType.READ_FILE:
            path = action.params.get("path", "")
            if tools and "read_file" in tools:
                return tools["read_file"](path)
            return f"Read: {path}"
        
        elif action.action_type == ActionType.ASK_HUMAN:
            return "Waiting for human input"
        
        return f"Action {action.action_type.value} completed"
    
    def run_with_persistence(
        self,
        user_input: str,
        model,
        tokenizer,
        tools: Dict[str, Any] = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run agentic task with checkpoint persistence for long-running execution.
        
        Args:
            user_input: User's request.
            model: Language model.
            tokenizer: Tokenizer.
            tools: Available tools dict.
            config: Execution configuration.
            
        Returns:
            Agentic result dict with checkpoint info.
        """
        config = config or {}
        checkpoint_interval = config.get("checkpoint_interval", self._checkpoint_interval)
        enable_recovery = config.get("enable_recovery", True)
        
        self._execution_context = {
            "user_input": user_input,
            "start_time": time.time(),
            "config": config,
        }
        
        try:
            result = self.run(user_input, model, tokenizer, tools, config.get("verbose", False))
            
            checkpoint_id = self.save_execution_state()
            result["checkpoint_id"] = checkpoint_id
            
            return result
            
        except Exception as e:
            if enable_recovery:
                checkpoint_id = self.save_execution_state()
                return {
                    "success": False,
                    "error": str(e),
                    "checkpoint_id": checkpoint_id,
                    "can_resume": True,
                }
            raise
    
    def resume_from_checkpoint(
        self,
        checkpoint_id: str,
        model,
        tokenizer,
        tools: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Resume execution from a saved checkpoint.
        
        Args:
            checkpoint_id: The checkpoint to resume from.
            model: Language model.
            tokenizer: Tokenizer.
            tools: Available tools dict.
            
        Returns:
            Agentic result dict.
        """
        if checkpoint_id not in self._checkpoint_storage:
            return {
                "success": False,
                "error": f"Checkpoint {checkpoint_id} not found",
            }
        
        checkpoint = self._checkpoint_storage[checkpoint_id]
        
        self._execution_context = checkpoint.get("execution_context", {})
        user_input = self._execution_context.get("user_input", "")
        
        result = self.run(user_input, model, tokenizer, tools)
        
        result["resumed_from"] = checkpoint_id
        
        return result
    
    def handle_interruption(self, signal: str = None) -> Dict[str, Any]:
        """Handle execution interruption and save checkpoint.
        
        Args:
            signal: Interruption signal (optional).
            
        Returns:
            Dict with checkpoint info.
        """
        self._interruption_flag = True
        
        checkpoint_id = self.save_execution_state()
        
        return {
            "interrupted": True,
            "checkpoint_id": checkpoint_id,
            "can_resume": True,
            "signal": signal,
        }
    
    def execute_with_recovery(
        self,
        action: 'Action',
        tools: Dict[str, Any],
        model,
        tokenizer,
        retry_policy: Dict[str, Any] = None
    ) -> Any:
        """Execute action with automatic recovery on failure.
        
        Args:
            action: Action to execute.
            tools: Available tools.
            model: Language model.
            tokenizer: Tokenizer.
            retry_policy: Custom retry policy.
            
        Returns:
            Action result.
        """
        policy = retry_policy or self._retry_policy
        max_attempts = policy.get("max_attempts", 3)
        
        attempt = 0
        last_error = None
        
        while attempt < max_attempts:
            try:
                result = self._execute_action(action, tools, model, tokenizer)
                return result
                
            except Exception as e:
                last_error = str(e)
                
                strategy = self.self_correction.determine_retry_strategy(
                    last_error, attempt, self._execution_context
                )
                
                if strategy == RetryStrategy.ABORT:
                    break
                elif strategy == RetryStrategy.SKIP:
                    return {"skipped": True, "reason": last_error}
                elif strategy == RetryStrategy.ESCALATE:
                    return {"escalate": True, "error": last_error}
                
                if strategy == RetryStrategy.BACKOFF:
                    delay = self.self_correction.calculate_backoff_time(
                        attempt,
                        policy.get("base_delay", 1.0),
                        policy.get("max_delay", 60.0)
                    )
                    time.sleep(delay)
                
                attempt += 1
        
        return {"failed": True, "error": last_error, "attempts": attempt}
    
    def save_execution_state(self) -> str:
        """Save current execution state as a checkpoint.
        
        Returns:
            str: Checkpoint ID.
        """
        import uuid
        
        checkpoint_id = str(uuid.uuid4())
        
        self._checkpoint_storage[checkpoint_id] = {
            "execution_context": self._execution_context.copy(),
            "timestamp": time.time(),
            "status": "interrupted" if self._interruption_flag else "saved",
        }
        
        self._last_checkpoint_time = time.time()
        
        return checkpoint_id
    
    def restore_execution_state(self, checkpoint_id: str) -> bool:
        """Restore execution state from a checkpoint.
        
        Args:
            checkpoint_id: The checkpoint to restore.
            
        Returns:
            bool: True if restoration succeeded.
        """
        if checkpoint_id not in self._checkpoint_storage:
            return False
        
        checkpoint = self._checkpoint_storage[checkpoint_id]
        self._execution_context = checkpoint.get("execution_context", {}).copy()
        self._interruption_flag = False
        
        return True
    
    def get_execution_progress(self) -> Dict[str, Any]:
        """Get current execution progress.
        
        Returns:
            Dict[str, Any]: Progress information.
        """
        start_time = self._execution_context.get("start_time", time.time())
        elapsed = time.time() - start_time
        
        return {
            "elapsed_time": elapsed,
            "checkpoint_count": len(self._checkpoint_storage),
            "last_checkpoint": self._last_checkpoint_time,
            "interrupted": self._interruption_flag,
            "has_context": bool(self._execution_context),
        }
    
    def estimate_completion_time(self) -> float:
        """Estimate remaining execution time.
        
        Returns:
            float: Estimated remaining time in seconds.
        """
        start_time = self._execution_context.get("start_time", time.time())
        elapsed = time.time() - start_time
        
        estimated_total = elapsed * 1.5
        
        return max(0, estimated_total - elapsed)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List[Dict[str, Any]]: List of checkpoint summaries.
        """
        return [
            {
                "checkpoint_id": cp_id,
                "timestamp": cp["timestamp"],
                "status": cp["status"],
            }
            for cp_id, cp in self._checkpoint_storage.items()
        ]
    
    def clear_checkpoints(self):
        """Clear all stored checkpoints."""
        self._checkpoint_storage.clear()


def create_react_agentic(hidden_size: int = 4096, 
                         max_steps: int = 20) -> YvReActAgentic:
    """Factory function to create ReAct Agentic.
    
    Args:
        hidden_size: Hidden size for neural modules
        max_steps: Maximum ReAct steps
        
    Returns:
        ReAct Agentic instance
    """
    return YvReActAgentic(hidden_size, max_steps)
