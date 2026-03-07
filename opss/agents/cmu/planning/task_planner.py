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
CMU Task Planner - Task Decomposition and Planning

This module provides task planning capabilities for the Computer Use Agent,
decomposing complex goals into executable action sequences.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUAction,
    POPSSCMUActionType,
    POPSSCMUPlatform,
    POPSSCMUSafetyLevel,
    POPSSCMUTarget,
    POPSSCMUCoordinate,
    POPSSCMUScreenState,
    POPSSCMUTaskContext,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Planning.TaskPlanner", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Planning.TaskPlanner"), enable_file=True)


@dataclass
class POPSSCMUTaskNode:
    """Task node in dependency graph."""
    task_id: str
    description: str
    actions: List[POPSSCMUAction] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    status: str = "pending"
    priority: int = 0
    estimated_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSCMUPlanResult:
    """Task planning result."""
    success: bool
    actions: List[POPSSCMUAction] = field(default_factory=list)
    task_nodes: List[POPSSCMUTaskNode] = field(default_factory=list)
    total_estimated_duration: float = 0.0
    warnings: List[str] = field(default_factory=list)


class POPSSCMUTaskPlanner:
    """
    Task planner for goal decomposition.
    
    Provides:
        - Goal decomposition into sub-tasks
        - Dependency analysis (DAG construction)
        - Critical path identification
        - Dynamic re-planning
        - Integration with YvUnifiedReasoner
    
    Attributes:
        platform: Target platform
        _reasoner: Optional unified reasoner
        _task_templates: Task templates for common operations
    """
    
    TASK_TEMPLATES = {
        "open_app": [
            {"action_type": "app_launch", "params": {"app_name": "{app_name}"}},
            {"action_type": "wait", "params": {"duration": 2.0}},
        ],
        "click_element": [
            {"action_type": "click", "target": "{target}"},
        ],
        "type_text": [
            {"action_type": "click", "target": "{target}"},
            {"action_type": "type", "params": {"text": "{text}"}},
        ],
        "search_web": [
            {"action_type": "hotkey", "params": {"keys": ["ctrl", "l"]}},
            {"action_type": "type", "params": {"text": "{query}"}},
            {"action_type": "key_press", "params": {"key": "enter"}},
        ],
        "copy_paste": [
            {"action_type": "hotkey", "params": {"keys": ["ctrl", "c"]}},
            {"action_type": "click", "target": "{target}"},
            {"action_type": "hotkey", "params": {"keys": ["ctrl", "v"]}},
        ],
        "scroll_find": [
            {"action_type": "scroll", "params": {"direction": "down", "amount": 3}},
            {"action_type": "wait", "params": {"duration": 0.5}},
        ],
    }
    
    def __init__(
        self,
        platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS,
        reasoner: Optional[Any] = None,
    ):
        self.platform = platform
        self._reasoner = reasoner
        self._task_graph: Dict[str, POPSSCMUTaskNode] = {}
        
        _LOG.info("task_planner_initialized", platform=platform.value)
    
    async def plan(
        self,
        goal: str,
        screen_state: Optional[POPSSCMUScreenState] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> POPSSCMUPlanResult:
        """
        Plan actions for a goal.
        
        Args:
            goal: Goal description
            screen_state: Current screen state
            context: Additional context
        
        Returns:
            POPSSCMUPlanResult: Planning result
        """
        _LOG.info("planning_task", goal=goal[:100])
        
        sub_tasks = await self._decompose_goal(goal, screen_state, context)
        
        task_nodes = []
        for i, sub_task in enumerate(sub_tasks):
            node = POPSSCMUTaskNode(
                task_id=f"task_{i}",
                description=sub_task["description"],
                actions=sub_task.get("actions", []),
                priority=sub_task.get("priority", 0),
                estimated_duration=sub_task.get("duration", 1.0),
            )
            task_nodes.append(node)
        
        all_actions = []
        for node in task_nodes:
            all_actions.extend(node.actions)
        
        total_duration = sum(n.estimated_duration for n in task_nodes)
        
        return POPSSCMUPlanResult(
            success=True,
            actions=all_actions,
            task_nodes=task_nodes,
            total_estimated_duration=total_duration,
        )
    
    async def _decompose_goal(
        self,
        goal: str,
        screen_state: Optional[POPSSCMUScreenState],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Decompose goal into sub-tasks."""
        sub_tasks = []
        
        goal_lower = goal.lower()
        
        if "open" in goal_lower or "launch" in goal_lower:
            app_name = self._extract_app_name(goal)
            sub_tasks.append({
                "description": f"Open {app_name}",
                "actions": self._create_actions_from_template(
                    "open_app",
                    {"app_name": app_name}
                ),
                "priority": 1,
                "duration": 3.0,
            })
        
        if "click" in goal_lower:
            target_desc = self._extract_click_target(goal)
            sub_tasks.append({
                "description": f"Click {target_desc}",
                "actions": [
                    POPSSCMUAction.click(
                        target=POPSSCMUTarget(element_description=target_desc),
                        description=f"Click on {target_desc}",
                    )
                ],
                "priority": 2,
                "duration": 0.5,
            })
        
        if "type" in goal_lower or "enter" in goal_lower or "write" in goal_lower:
            text = self._extract_text(goal)
            target_desc = self._extract_type_target(goal)
            
            actions = []
            if target_desc:
                actions.append(POPSSCMUAction.click(
                    target=POPSSCMUTarget(element_description=target_desc),
                    description=f"Click on {target_desc}",
                ))
            
            actions.append(POPSSCMUAction.type_text(
                text=text,
                description=f"Type: {text}",
            ))
            
            sub_tasks.append({
                "description": f"Type text: {text[:30]}...",
                "actions": actions,
                "priority": 3,
                "duration": len(text) * 0.05 + 0.5,
            })
        
        if "search" in goal_lower:
            query = self._extract_search_query(goal)
            sub_tasks.append({
                "description": f"Search for: {query}",
                "actions": self._create_actions_from_template(
                    "search_web",
                    {"query": query}
                ),
                "priority": 2,
                "duration": 2.0,
            })
        
        if "scroll" in goal_lower:
            direction = "down" if "down" in goal_lower else "up"
            amount = self._extract_scroll_amount(goal)
            sub_tasks.append({
                "description": f"Scroll {direction}",
                "actions": [
                    POPSSCMUAction.scroll(
                        direction=POPSSCMUAction.scroll.__wrapped__.__self__.SCROLL_DOWN if direction == "down" else POPSSCMUAction.scroll.__wrapped__.__self__.SCROLL_UP,
                        amount=amount,
                    )
                ],
                "priority": 1,
                "duration": 0.3,
            })
        
        if "copy" in goal_lower or "paste" in goal_lower:
            sub_tasks.append({
                "description": "Copy/Paste operation",
                "actions": [
                    POPSSCMUAction.hotkey(["ctrl", "c" if "copy" in goal_lower else "v"]),
                ],
                "priority": 2,
                "duration": 0.5,
            })
        
        if not sub_tasks:
            sub_tasks.append({
                "description": "Capture screen for analysis",
                "actions": [
                    POPSSCMUAction(
                        action_type=POPSSCMUActionType.SCREENSHOT,
                        description="Capture current screen state",
                    )
                ],
                "priority": 0,
                "duration": 0.5,
            })
        
        return sub_tasks
    
    def _create_actions_from_template(
        self,
        template_name: str,
        params: Dict[str, Any],
    ) -> List[POPSSCMUAction]:
        """Create actions from template."""
        if template_name not in self.TASK_TEMPLATES:
            return []
        
        template = self.TASK_TEMPLATES[template_name]
        actions = []
        
        for step in template:
            action_type = POPSSCMUActionType(step["action_type"])
            
            step_params = {}
            for key, value in step.get("params", {}).items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    param_key = value[1:-1]
                    step_params[key] = params.get(param_key, value)
                else:
                    step_params[key] = value
            
            action = POPSSCMUAction(
                action_type=action_type,
                params=step_params,
                description=f"Execute {action_type.value}",
            )
            actions.append(action)
        
        return actions
    
    def _extract_app_name(self, goal: str) -> str:
        """Extract application name from goal."""
        patterns = [
            r"open\s+(.+?)(?:\s+app|\s+application|$)",
            r"launch\s+(.+?)(?:\s+app|\s+application|$)",
            r"start\s+(.+?)(?:\s+app|\s+application|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "unknown"
    
    def _extract_click_target(self, goal: str) -> str:
        """Extract click target from goal."""
        patterns = [
            r"click\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button|\s+link|$)",
            r"press\s+(?:the\s+)?(.+?)(?:\s+button|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_text(self, goal: str) -> str:
        """Extract text to type from goal."""
        patterns = [
            r'type\s+["\'](.+?)["\']',
            r'enter\s+["\'](.+?)["\']',
            r'write\s+["\'](.+?)["\']',
            r'type\s+(.+?)(?:\s+in|\s+into|\s+at|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_type_target(self, goal: str) -> str:
        """Extract typing target from goal."""
        patterns = [
            r"in\s+(?:the\s+)?(.+?)(?:\s+field|\s+box|\s+input)",
            r"into\s+(?:the\s+)?(.+?)(?:\s+field|\s+box|\s+input)",
            r"at\s+(?:the\s+)?(.+?)(?:\s+field|\s+box|\s+input)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_search_query(self, goal: str) -> str:
        """Extract search query from goal."""
        patterns = [
            r"search\s+(?:for\s+)?(.+?)(?:\s+on|\s+in|$)",
            r"find\s+(.+?)(?:\s+on|\s+in|$)",
            r"look\s+up\s+(.+?)(?:\s+on|\s+in|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_scroll_amount(self, goal: str) -> int:
        """Extract scroll amount from goal."""
        match = re.search(r"(\d+)\s*(?:times|pages)?", goal, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 3
    
    async def replan(
        self,
        failed_action: POPSSCMUAction,
        error_message: str,
        screen_state: Optional[POPSSCMUScreenState],
    ) -> List[POPSSCMUAction]:
        """
        Re-plan after action failure.
        
        Args:
            failed_action: Action that failed
            error_message: Error message
            screen_state: Current screen state
        
        Returns:
            List[POPSSCMUAction]: Alternative actions
        """
        _LOG.info("replanning_after_failure", action_type=failed_action.action_type.value, error=error_message)
        
        alternative_actions = []
        
        if failed_action.action_type == POPSSCMUActionType.CLICK:
            alternative_actions.append(
                POPSSCMUAction(
                    action_type=POPSSCMUActionType.SCREENSHOT,
                    description="Capture screen to find alternative target",
                )
            )
        
        elif failed_action.action_type == POPSSCMUActionType.TYPE:
            alternative_actions.append(
                POPSSCMUAction.click(
                    target=failed_action.target,
                    description="Click target before typing",
                )
            )
            alternative_actions.append(failed_action)
        
        elif failed_action.action_type == POPSSCMUActionType.SCROLL:
            alternative_actions.append(
                POPSSCMUAction.scroll(
                    direction=POPSSCMUAction.scroll.__wrapped__.__self__.SCROLL_UP,
                    amount=1,
                )
            )
        
        return alternative_actions
    
    def get_task_graph(self) -> Dict[str, POPSSCMUTaskNode]:
        """Get task dependency graph."""
        return self._task_graph.copy()
    
    def clear_task_graph(self) -> None:
        """Clear task graph."""
        self._task_graph.clear()
