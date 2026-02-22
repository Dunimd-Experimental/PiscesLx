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

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger

class POPSSTaskComplexity(Enum):
    TRIVIAL = "trivial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    COMPLEX = "complex"

class POPSSTaskDependencyType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BLOCKING = "blocking"
    CONDITIONAL = "conditional"

@dataclass
class POPSSSubtask:
    subtask_id: str
    name: str
    description: str
    
    complexity: POPSSTaskComplexity = POPSSTaskComplexity.MEDIUM
    priority: int = 5
    
    task_type: str = ""
    expected_output: str = ""
    
    dependencies: List[str] = field(default_factory=list)
    blocking_tasks: List[str] = field(default_factory=list)
    
    assigned_agent: Optional[str] = None
    status: str = "pending"
    
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    estimated_effort: float = 1.0
    actual_effort: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class POPSSTaskDecompositionResult:
    decomposition_id: str
    original_task: str
    
    subtasks: List[POPSSSubtask] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[List[str]] = field(default_factory=list)
    
    estimated_total_effort: float = 0.0
    parallelizable_groups: List[List[str]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSTaskDecomposerConfig:
    max_subtasks: int = 10
    min_subtask_complexity: POPSSTaskComplexity = POPSSTaskComplexity.TRIVIAL
    max_subtask_complexity: POPSSTaskComplexity = POPSSTaskComplexity.HIGH
    
    enable_parallel_analysis: bool = True
    enable_dependency_detection: bool = True
    enable_effort_estimation: bool = True
    
    complexity_keywords: Dict[str, POPSSTaskComplexity] = field(default_factory=lambda: {
        "simple": POPSSTaskComplexity.TRIVIAL,
        "basic": POPSSTaskComplexity.LOW,
        "standard": POPSSTaskComplexity.MEDIUM,
        "advanced": POPSSTaskComplexity.HIGH,
        "complex": POPSSTaskComplexity.COMPLEX,
    })

class POPSSTaskDecomposer:
    def __init__(self, config: Optional[POPSSTaskDecomposerConfig] = None):
        self.config = config or POPSSTaskDecomposerConfig()
        self._LOG = self._configure_logging()
        
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="piscesl1_task_decomposer"
        )
        
        self._LOG.info("POPSSTaskDecomposer initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        logger = get_logger("PiscesLx.Core.Agents.Collaboration.TaskDecomposer")
        return logger
    
    async def decompose(self, task: str, context: Optional[Dict[str, Any]] = None) -> POPSSTaskDecompositionResult:
        decomposition_id = f"decomp_{uuid.uuid4().hex[:12]}"
        
        self._LOG.info(f"Decomposing task: {task[:100]}...")
        
        task_analysis = await self._analyze_task(task, context)
        
        complexity = self._assess_complexity(task, task_analysis)
        
        subtasks = await self._generate_subtasks(task, task_analysis, complexity)
        
        subtasks = self._prune_subtasks(subtasks)
        
        dependency_graph = self._build_dependency_graph(subtasks)
        
        execution_order = self._plan_execution_order(subtasks, dependency_graph)
        
        parallel_groups = self._identify_parallel_groups(subtasks, dependency_graph)
        
        estimated_effort = sum(st.estimated_effort for st in subtasks)
        
        result = POPSSTaskDecompositionResult(
            decomposition_id=decomposition_id,
            original_task=task,
            subtasks=subtasks,
            dependency_graph=dependency_graph,
            execution_order=execution_order,
            estimated_total_effort=estimated_effort,
            parallelizable_groups=parallel_groups,
            metadata={
                'analysis': task_analysis,
                'complexity': complexity.value,
                'context': context or {},
            }
        )
        
        self._LOG.info(f"Task decomposed into {len(subtasks)} subtasks")
        
        return result
    
    async def _analyze_task(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        task_lower = task.lower()
        
        analysis = {
            'intent': self._classify_intent(task_lower),
            'entities': self._extract_entities(task),
            'actions': self._identify_actions(task_lower),
            'domains': self._identify_domains(task_lower),
            'constraints': self._extract_constraints(task),
            'requirements': [],
        }
        
        action_keywords = {
            'create': ['write', 'generate', 'make', 'build', 'develop'],
            'analyze': ['analyze', 'examine', 'review', 'evaluate', 'assess'],
            'search': ['find', 'search', 'lookup', 'query', 'get'],
            'modify': ['update', 'change', 'modify', 'edit', 'revise'],
            'delete': ['remove', 'delete', 'clear', 'eliminate'],
            'organize': ['organize', 'sort', 'group', 'categorize', 'structure'],
        }
        
        for action, keywords in action_keywords.items():
            if any(kw in task_lower for kw in keywords):
                analysis['actions'].append(action)
        
        return analysis
    
    def _classify_intent(self, task: str) -> str:
        intent_patterns = {
            'generation': ['write', 'create', 'generate', 'make', 'produce'],
            'analysis': ['analyze', 'examine', 'investigate', 'study'],
            'retrieval': ['find', 'search', 'lookup', 'retrieve', 'get'],
            'modification': ['update', 'change', 'modify', 'edit'],
            'computation': ['calculate', 'compute', 'solve', 'determine'],
            'planning': ['plan', 'schedule', 'organize', 'design'],
        }
        
        for intent, patterns in intent_patterns.items():
            if any(p in task for p in patterns):
                return intent
        
        return 'general'
    
    def _extract_entities(self, task: str) -> List[str]:
        import re
        entity_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            r'\b\d+\b',
        ]
        
        entities = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, task)
            entities.extend(matches)
        
        return list(set(entities))[:10]
    
    def _identify_actions(self, task: str) -> List[str]:
        return []
    
    def _identify_domains(self, task: str) -> List[str]:
        domain_keywords = {
            'programming': ['code', 'program', 'function', 'class', 'api', 'software'],
            'data': ['data', 'database', 'query', 'sql', 'table', 'record'],
            'web': ['website', 'page', 'url', 'http', 'html', 'css', 'javascript'],
            'file': ['file', 'document', 'folder', 'directory', 'path'],
            'research': ['research', 'study', 'paper', 'article', 'analysis'],
            'math': ['calculate', 'compute', 'math', 'number', 'formula'],
        }
        
        domains = []
        task_lower = task.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in task_lower for kw in keywords):
                domains.append(domain)
        
        return domains
    
    def _extract_constraints(self, task: str) -> List[str]:
        import re
        constraint_patterns = [
            r'must be (.+?)[\.,]',
            r'should be (.+?)[\.,]',
            r'require[s]? (.+?)[\.,]',
            r'within \d+ [\w]+',
            r'before \w+',
        ]
        
        constraints = []
        for pattern in constraint_patterns:
            matches = re.findall(pattern, task, re.IGNORECASE)
            constraints.extend(matches)
        
        return constraints[:5]
    
    def _assess_complexity(self, task: str, analysis: Dict[str, Any]) -> POPSSTaskComplexity:
        complexity_score = 0
        
        if len(task) > 200:
            complexity_score += 2
        elif len(task) > 100:
            complexity_score += 1
        
        action_count = len(analysis.get('actions', []))
        if action_count >= 3:
            complexity_score += 2
        elif action_count >= 2:
            complexity_score += 1
        
        domain_count = len(analysis.get('domains', []))
        if domain_count >= 3:
            complexity_score += 2
        elif domain_count >= 2:
            complexity_score += 1
        
        if len(analysis.get('constraints', [])) >= 3:
            complexity_score += 1
        
        if complexity_score >= 6:
            return POPSSTaskComplexity.COMPLEX
        elif complexity_score >= 4:
            return POPSSTaskComplexity.HIGH
        elif complexity_score >= 2:
            return POPSSTaskComplexity.MEDIUM
        elif complexity_score >= 1:
            return POPSSTaskComplexity.LOW
        else:
            return POPSSTaskComplexity.TRIVIAL
    
    async def _generate_subtasks(
        self,
        task: str,
        analysis: Dict[str, Any],
        complexity: POPSSTaskComplexity
    ) -> List[POPSSSubtask]:
        subtasks = []
        
        subtask_id = f"subtask_{uuid.uuid4().hex[:8]}"
        
        if analysis['intent'] == 'generation':
            subtask = POPSSSubtask(
                subtask_id=subtask_id,
                name="Content Generation",
                description=f"Generate content for: {task[:50]}",
                complexity=complexity,
                task_type="generation",
                expected_output="Generated content",
                estimated_effort=1.0,
            )
            subtasks.append(subtask)
        
        elif analysis['intent'] == 'analysis':
            subtask = POPSSSubtask(
                subtask_id=subtask_id,
                name="Data Analysis",
                description=f"Analyze: {task[:50]}",
                complexity=complexity,
                task_type="analysis",
                expected_output="Analysis results",
                estimated_effort=1.5,
            )
            subtasks.append(subtask)
        
        elif analysis['intent'] == 'retrieval':
            subtask = POPSSSubtask(
                subtask_id=subtask_id,
                name="Information Retrieval",
                description=f"Find information: {task[:50]}",
                complexity=complexity,
                task_type="retrieval",
                expected_output="Retrieved information",
                estimated_effort=0.8,
            )
            subtasks.append(subtask)
        
        else:
            subtask = POPSSSubtask(
                subtask_id=subtask_id,
                name="Task Execution",
                description=f"Execute: {task[:50]}",
                complexity=complexity,
                task_type="general",
                expected_output="Task result",
                estimated_effort=1.0,
            )
            subtasks.append(subtask)
        
        if len(analysis.get('domains', [])) > 1:
            for domain in analysis['domains'][:3]:
                domain_subtask = POPSSSubtask(
                    subtask_id=f"subtask_{uuid.uuid4().hex[:8]}",
                    name=f"{domain.title()} Processing",
                    description=f"Process {domain} requirements",
                    complexity=POPSSTaskComplexity.MEDIUM,
                    task_type=f"{domain}_processing",
                    expected_output=f"{domain} results",
                    estimated_effort=1.0,
                )
                subtasks.append(domain_subtask)
        
        if analysis.get('constraints'):
            constraint_subtask = POPSSSubtask(
                subtask_id=f"subtask_{uuid.uuid4().hex[:8]}",
                name="Validation",
                description="Validate results against constraints",
                complexity=POPSSTaskComplexity.LOW,
                task_type="validation",
                expected_output="Validation report",
                estimated_effort=0.5,
            )
            subtasks.append(constraint_subtask)
        
        return subtasks
    
    def _prune_subtasks(self, subtasks: List[POPSSSubtask]) -> List[POPSSSubtask]:
        max_subtasks = self.config.max_subtask_complexity.value * 2
        if len(subtasks) > self.config.max_subtasks:
            subtasks = subtasks[:self.config.max_subtasks]
        
        filtered = []
        for subtask in subtasks:
            if subtask.complexity.value >= self.config.min_subtask_complexity.value:
                filtered.append(subtask)
            elif subtask.task_type in ['validation', 'general']:
                filtered.append(subtask)
        
        return filtered
    
    def _build_dependency_graph(self, subtasks: List[POPSSSubtask]) -> Dict[str, List[str]]:
        graph = {}
        
        for subtask in subtasks:
            graph[subtask.subtask_id] = subtask.dependencies.copy()
        
        return graph
    
    def _plan_execution_order(
        self,
        subtasks: List[POPSSSubtask],
        dependency_graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        if not subtasks:
            return []
        
        subtask_ids = [st.subtask_id for st in subtasks]
        in_degree = {sid: 0 for sid in subtask_ids}
        
        for sid, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[sid] += 1
        
        queue = [sid for sid in subtask_ids if in_degree[sid] == 0]
        result = []
        
        while queue:
            current_level = sorted(queue)
            result.append(current_level)
            
            next_queue = []
            for sid in current_level:
                for dependent, deps in dependency_graph.items():
                    if sid in deps and dependent in in_degree:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            next_queue.append(dependent)
            
            queue = next_queue
        
        return result
    
    def _identify_parallel_groups(
        self,
        subtasks: List[POPSSSubtask],
        dependency_graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        parallel_groups = []
        
        task_types = {}
        for subtask in subtasks:
            if subtask.task_type not in task_types:
                task_types[subtask.task_type] = []
            task_types[subtask.task_type].append(subtask.subtask_id)
        
        independent_types = [tids for ttype, tids in task_types.items() 
                         if all(len(dependency_graph.get(tid, [])) == 0 for tid in tids)]
        
        for group in independent_types:
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def shutdown(self):
        self._async_executor.shutdown(wait=True)
        self._LOG.info("POPSSTaskDecomposer shutdown")
