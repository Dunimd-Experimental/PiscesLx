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
Sequential Thinking Tool - Structured reasoning and problem solving
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json

from .base import POPSSMCPToolBase, POPSSMCPToolResult


@dataclass
class Thought:
    step: int
    content: str
    thought_type: str = "reasoning"
    confidence: float = 0.8
    next_action: Optional[str] = None


class SequentialThinkingTool(POPSSMCPToolBase):
    name = "sequential_thinking"
    description = "Perform structured sequential reasoning for complex problems"
    category = "reasoning"
    tags = ["thinking", "reasoning", "problem-solving", "chain-of-thought"]
    
    parameters = {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": "The problem or question to reason about"
            },
            "context": {
                "type": "string",
                "description": "Additional context or constraints"
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum number of reasoning steps",
                "default": 10
            },
            "approach": {
                "type": "string",
                "description": "Reasoning approach: 'analytical', 'creative', 'systematic'",
                "default": "analytical"
            }
        },
        "required": ["problem"]
    }
    
    _sessions: Dict[str, List[Thought]] = {}
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        problem = arguments.get("problem", "")
        context = arguments.get("context", "")
        max_steps = arguments.get("max_steps", 10)
        approach = arguments.get("approach", "analytical")
        
        if not problem:
            return self._create_error_result("problem is required", "ValidationError")
        
        thoughts = self._generate_thoughts(problem, context, max_steps, approach)
        
        return self._create_success_result({
            "problem": problem,
            "approach": approach,
            "steps": len(thoughts),
            "thoughts": [
                {
                    "step": t.step,
                    "type": t.thought_type,
                    "content": t.content,
                    "confidence": t.confidence,
                }
                for t in thoughts
            ],
            "conclusion": thoughts[-1].content if thoughts else "",
        })
    
    def _generate_thoughts(
        self, 
        problem: str, 
        context: str, 
        max_steps: int,
        approach: str
    ) -> List[Thought]:
        thoughts = []
        
        thoughts.append(Thought(
            step=1,
            content=f"Understanding the problem: {self._extract_key_elements(problem)}",
            thought_type="understanding",
            confidence=0.9
        ))
        
        if approach == "analytical":
            thoughts.extend(self._analytical_approach(problem, context, max_steps))
        elif approach == "creative":
            thoughts.extend(self._creative_approach(problem, context, max_steps))
        else:
            thoughts.extend(self._systematic_approach(problem, context, max_steps))
        
        thoughts.append(Thought(
            step=len(thoughts) + 1,
            content=self._synthesize_conclusion(thoughts),
            thought_type="conclusion",
            confidence=0.85
        ))
        
        return thoughts[:max_steps]
    
    def _extract_key_elements(self, problem: str) -> str:
        words = problem.split()
        if len(words) > 20:
            return " ".join(words[:20]) + "..."
        return problem
    
    def _analytical_approach(
        self, 
        problem: str, 
        context: str, 
        max_steps: int
    ) -> List[Thought]:
        thoughts = []
        
        thoughts.append(Thought(
            step=2,
            content="Breaking down the problem into components",
            thought_type="decomposition",
            confidence=0.85
        ))
        
        thoughts.append(Thought(
            step=3,
            content="Identifying relationships between components",
            thought_type="analysis",
            confidence=0.8
        ))
        
        thoughts.append(Thought(
            step=4,
            content="Applying logical reasoning to each component",
            thought_type="reasoning",
            confidence=0.8
        ))
        
        thoughts.append(Thought(
            step=5,
            content="Evaluating potential solutions",
            thought_type="evaluation",
            confidence=0.75
        ))
        
        return thoughts
    
    def _creative_approach(
        self, 
        problem: str, 
        context: str, 
        max_steps: int
    ) -> List[Thought]:
        thoughts = []
        
        thoughts.append(Thought(
            step=2,
            content="Exploring alternative perspectives",
            thought_type="exploration",
            confidence=0.75
        ))
        
        thoughts.append(Thought(
            step=3,
            content="Generating novel connections",
            thought_type="ideation",
            confidence=0.7
        ))
        
        thoughts.append(Thought(
            step=4,
            content="Synthesizing creative solutions",
            thought_type="synthesis",
            confidence=0.75
        ))
        
        return thoughts
    
    def _systematic_approach(
        self, 
        problem: str, 
        context: str, 
        max_steps: int
    ) -> List[Thought]:
        thoughts = []
        
        thoughts.append(Thought(
            step=2,
            content="Defining clear objectives and constraints",
            thought_type="definition",
            confidence=0.9
        ))
        
        thoughts.append(Thought(
            step=3,
            content="Gathering relevant information",
            thought_type="research",
            confidence=0.85
        ))
        
        thoughts.append(Thought(
            step=4,
            content="Developing systematic solution steps",
            thought_type="planning",
            confidence=0.85
        ))
        
        thoughts.append(Thought(
            step=5,
            content="Implementing and verifying solution",
            thought_type="execution",
            confidence=0.8
        ))
        
        return thoughts
    
    def _synthesize_conclusion(self, thoughts: List[Thought]) -> str:
        reasoning_steps = [t for t in thoughts if t.thought_type in ["reasoning", "analysis", "evaluation"]]
        
        if reasoning_steps:
            return f"Based on {len(reasoning_steps)} reasoning steps, a structured solution approach has been developed."
        return "A systematic approach to the problem has been established."


class ProblemDecompositionTool(POPSSMCPToolBase):
    name = "problem_decompose"
    description = "Decompose complex problems into smaller sub-problems"
    category = "reasoning"
    tags = ["decomposition", "problem-solving", "analysis"]
    
    parameters = {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": "The problem to decompose"
            },
            "depth": {
                "type": "integer",
                "description": "Decomposition depth (1-3)",
                "default": 2
            }
        },
        "required": ["problem"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        problem = arguments.get("problem", "")
        depth = min(3, max(1, arguments.get("depth", 2)))
        
        if not problem:
            return self._create_error_result("problem is required", "ValidationError")
        
        sub_problems = self._decompose(problem, depth)
        
        return self._create_success_result({
            "original_problem": problem,
            "depth": depth,
            "sub_problems": sub_problems,
            "total_sub_problems": len(sub_problems),
        })
    
    def _decompose(self, problem: str, depth: int) -> List[Dict[str, Any]]:
        sub_problems = []
        
        aspects = [
            ("technical", "Technical implementation aspects"),
            ("logical", "Logical reasoning aspects"),
            ("resource", "Resource and constraint aspects"),
            ("temporal", "Timeline and sequencing aspects"),
        ]
        
        for i, (aspect_name, aspect_desc) in enumerate(aspects[:depth + 1]):
            sub_problems.append({
                "id": f"sub_{i+1}",
                "aspect": aspect_name,
                "description": f"{aspect_desc} of: {problem[:50]}...",
                "priority": "high" if i < 2 else "medium",
                "dependencies": [],
            })
        
        return sub_problems
