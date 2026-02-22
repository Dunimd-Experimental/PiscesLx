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

"""
Agent Interceptor - Intercept and Route Agent-Specific Requests

This module provides functionality to detect and process agent-specific
XML patterns in user requests, routing them to appropriate handlers.

Supported XML Patterns:
    - <agentic>...</agentic>: Agent execution block
    - <ag>agent_name</ag>: Single agent invocation
    - <swarm mode="...">...</swarm>: Swarm execution
    - <orchestrate strategy="...">...</orchestrate>: Orchestration
    - <tool name="...">...</tool>: Tool invocation
    - <args>...</args>: Tool arguments

Agent Modes:
    - NONE: No agent pattern detected, standard chat
    - SINGLE: Single agent invocation
    - SWARM: Multi-agent swarm execution
    - ORCHESTRATED: Dynamic orchestration
    - TOOL_CALL: Direct tool invocation

Architecture:
    PiscesLxAgentInterceptor
    ├── Pattern Detection
    │   ├── Agentic block detection
    │   ├── Agent invocation detection
    │   ├── Swarm mode detection
    │   ├── Orchestration detection
    │   └── Tool call detection
    ├── Content Extraction
    │   ├── Agent name extraction
    │   ├── Task description extraction
    │   └── Argument parsing
    └── Result Generation
        ├── Mode determination
        ├── Request structuring
        └── Metadata collection

Usage:
    >>> interceptor = PiscesLxAgentInterceptor()
    >>> result = interceptor.intercept("请帮我分析数据 <ag>data_analyzer</ag>")
    >>> print(result.mode)  # PiscesLxAgentMode.SINGLE
    >>> print(result.agent_requests)  # [{"type": "single", "agent_name": "data_analyzer"}]
"""

import re
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging


class PiscesLxAgentMode(Enum):
    """Agent execution mode enumeration."""
    NONE = "none"
    SINGLE = "single"
    SWARM = "swarm"
    ORCHESTRATED = "orchestrated"
    TOOL_CALL = "tool_call"
    HYBRID = "hybrid"


@dataclass
class PiscesLxInterceptResult:
    """
    Result of agent interception.
    
    Attributes:
        mode: Detected agent mode
        original_content: Original content before processing
        processed_content: Content after removing agent patterns
        agent_requests: List of agent request specifications
        tool_calls: List of tool call specifications
        metadata: Additional metadata about the interception
    """
    mode: PiscesLxAgentMode
    original_content: str
    processed_content: str
    agent_requests: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PiscesLxAgentInterceptor:
    """
    Intercept and Route Agent-Specific Requests.
    
    This class detects XML-based agent patterns in user messages and
    routes them to appropriate handlers (single agent, swarm, orchestrator, tool).
    
    Features:
        - Multiple pattern detection
        - Nested pattern support
        - Argument parsing
        - Mode classification
        - Metadata extraction
    
    Pattern Examples:
        Single Agent:
            <ag>data_analyzer</ag>请分析这份数据
        
        Swarm Mode:
            <agentic>
                <swarm mode="hierarchical">分析市场数据并生成报告</swarm>
            </agentic>
        
        Orchestration:
            <agentic>
                <orchestrate strategy="dynamic">研究AI发展趋势</orchestrate>
            </agentic>
        
        Tool Call:
            <tool name="web_search">
                <args>{"query": "latest AI news"}</args>
            </tool>
    
    Example:
        >>> interceptor = PiscesLxAgentInterceptor()
        >>> 
        >>> # Single agent
        >>> result = interceptor.intercept("<ag>code_gen</ag>写一个排序算法")
        >>> print(result.mode)  # PiscesLxAgentMode.SINGLE
        >>> 
        >>> # Swarm
        >>> result = interceptor.intercept(
        ...     "<agentic><swarm mode='mesh'>分析数据</swarm></agentic>"
        ... )
        >>> print(result.mode)  # PiscesLxAgentMode.SWARM
    """
    
    PATTERNS = {
        "agentic_block": re.compile(
            r'<agentic[^>]*>(.*?)</agentic>',
            re.DOTALL | re.IGNORECASE
        ),
        "agent_invoke": re.compile(
            r'<ag[^>]*>(.*?)</ag>',
            re.DOTALL | re.IGNORECASE
        ),
        "swarm_invoke": re.compile(
            r'<swarm[^>]*mode=["\']?(\w+)["\']?[^>]*>(.*?)</swarm>',
            re.DOTALL | re.IGNORECASE
        ),
        "orchestrate_invoke": re.compile(
            r'<orchestrate[^>]*strategy=["\']?(\w+)["\']?[^>]*>(.*?)</orchestrate>',
            re.DOTALL | re.IGNORECASE
        ),
        "tool_invoke": re.compile(
            r'<tool[^>]*name=["\']?([\w_\-]+)["\']?[^>]*>(.*?)</tool>',
            re.DOTALL | re.IGNORECASE
        ),
        "tool_args": re.compile(
            r'<args>(.*?)</args>',
            re.DOTALL | re.IGNORECASE
        ),
        "agent_with_params": re.compile(
            r'<ag[^>]*type=["\']?(\w+)["\']?[^>]*>(.*?)</ag>',
            re.DOTALL | re.IGNORECASE
        ),
    }
    
    def __init__(self):
        """Initialize the agent interceptor."""
        self._LOG = logging.getLogger(self.__class__.__name__)
        self._intercept_count = 0
        self._mode_counts: Dict[str, int] = {mode.value: 0 for mode in PiscesLxAgentMode}
    
    def intercept(self, content: str) -> PiscesLxInterceptResult:
        """
        Intercept and analyze content for agent patterns.
        
        This method performs comprehensive pattern detection to determine
        the appropriate agent mode and extract relevant request data.
        
        Args:
            content: The content to analyze
        
        Returns:
            PiscesLxInterceptResult with mode and extracted requests
        """
        self._intercept_count += 1
        
        agent_requests: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        processed_content = content
        metadata: Dict[str, Any] = {"patterns_found": []}
        
        agentic_match = self.PATTERNS["agentic_block"].search(content)
        if agentic_match:
            metadata["patterns_found"].append("agentic_block")
            agentic_content = agentic_match.group(1)
            
            swarm_match = self.PATTERNS["swarm_invoke"].search(agentic_content)
            if swarm_match:
                mode = swarm_match.group(1)
                task = swarm_match.group(2).strip()
                metadata["swarm_mode"] = mode
                metadata["patterns_found"].append("swarm_invoke")
                
                self._update_mode_counts(PiscesLxAgentMode.SWARM)
                return PiscesLxInterceptResult(
                    mode=PiscesLxAgentMode.SWARM,
                    original_content=content,
                    processed_content=task,
                    agent_requests=[{
                        "type": "swarm",
                        "mode": mode,
                        "task": task,
                    }],
                    tool_calls=[],
                    metadata=metadata
                )
            
            orchestrate_match = self.PATTERNS["orchestrate_invoke"].search(agentic_content)
            if orchestrate_match:
                strategy = orchestrate_match.group(1)
                task = orchestrate_match.group(2).strip()
                metadata["orchestrate_strategy"] = strategy
                metadata["patterns_found"].append("orchestrate_invoke")
                
                self._update_mode_counts(PiscesLxAgentMode.ORCHESTRATED)
                return PiscesLxInterceptResult(
                    mode=PiscesLxAgentMode.ORCHESTRATED,
                    original_content=content,
                    processed_content=task,
                    agent_requests=[{
                        "type": "orchestrate",
                        "strategy": strategy,
                        "task": task,
                    }],
                    tool_calls=[],
                    metadata=metadata
                )
        
        agent_matches = list(self.PATTERNS["agent_invoke"].finditer(content))
        if agent_matches:
            metadata["patterns_found"].append("agent_invoke")
            
            for match in agent_matches:
                agent_name = match.group(1).strip()
                agent_requests.append({
                    "type": "single",
                    "agent_name": agent_name,
                })
            
            processed_content = self.PATTERNS["agent_invoke"].sub("", content).strip()
            
            self._update_mode_counts(PiscesLxAgentMode.SINGLE)
            return PiscesLxInterceptResult(
                mode=PiscesLxAgentMode.SINGLE,
                original_content=content,
                processed_content=processed_content,
                agent_requests=agent_requests,
                tool_calls=[],
                metadata=metadata
            )
        
        tool_matches = list(self.PATTERNS["tool_invoke"].finditer(content))
        if tool_matches:
            metadata["patterns_found"].append("tool_invoke")
            
            for match in tool_matches:
                tool_name = match.group(1)
                tool_content = match.group(2)
                
                args = self._parse_tool_args(tool_content)
                
                tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": args,
                })
            
            processed_content = self.PATTERNS["tool_invoke"].sub("", content).strip()
            
            self._update_mode_counts(PiscesLxAgentMode.TOOL_CALL)
            return PiscesLxInterceptResult(
                mode=PiscesLxAgentMode.TOOL_CALL,
                original_content=content,
                processed_content=processed_content,
                agent_requests=[],
                tool_calls=tool_calls,
                metadata=metadata
            )
        
        self._update_mode_counts(PiscesLxAgentMode.NONE)
        return PiscesLxInterceptResult(
            mode=PiscesLxAgentMode.NONE,
            original_content=content,
            processed_content=content,
            agent_requests=[],
            tool_calls=[],
            metadata=metadata
        )
    
    def _parse_tool_args(self, content: str) -> Dict[str, Any]:
        """
        Parse tool arguments from content.
        
        Args:
            content: Content containing tool arguments
        
        Returns:
            Parsed arguments dictionary
        """
        args_match = self.PATTERNS["tool_args"].search(content)
        if args_match:
            args_str = args_match.group(1).strip()
            try:
                return json.loads(args_str)
            except json.JSONDecodeError:
                return {"raw": args_str}
        
        content = content.strip()
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"input": content}
        
        return {}
    
    def _update_mode_counts(self, mode: PiscesLxAgentMode):
        """Update mode statistics."""
        self._mode_counts[mode.value] = self._mode_counts.get(mode.value, 0) + 1
    
    def has_agent_pattern(self, content: str) -> bool:
        """
        Check if content contains any agent patterns.
        
        Args:
            content: Content to check
        
        Returns:
            True if any agent pattern is found
        """
        for pattern in self.PATTERNS.values():
            if pattern.search(content):
                return True
        return False
    
    def extract_agent_names(self, content: str) -> List[str]:
        """
        Extract all agent names from content.
        
        Args:
            content: Content to extract from
        
        Returns:
            List of agent names
        """
        matches = self.PATTERNS["agent_invoke"].findall(content)
        return [m.strip() for m in matches if m.strip()]
    
    def extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract all tool calls from content.
        
        Args:
            content: Content to extract from
        
        Returns:
            List of tool call specifications
        """
        tool_calls = []
        tool_matches = self.PATTERNS["tool_invoke"].finditer(content)
        
        for match in tool_matches:
            tool_name = match.group(1)
            tool_content = match.group(2)
            args = self._parse_tool_args(tool_content)
            
            tool_calls.append({
                "tool_name": tool_name,
                "arguments": args,
            })
        
        return tool_calls
    
    def extract_swarm_config(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract swarm configuration from content.
        
        Args:
            content: Content to extract from
        
        Returns:
            Swarm configuration or None
        """
        agentic_match = self.PATTERNS["agentic_block"].search(content)
        if not agentic_match:
            return None
        
        agentic_content = agentic_match.group(1)
        swarm_match = self.PATTERNS["swarm_invoke"].search(agentic_content)
        
        if swarm_match:
            return {
                "mode": swarm_match.group(1),
                "task": swarm_match.group(2).strip(),
            }
        
        return None
    
    def extract_orchestrate_config(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract orchestration configuration from content.
        
        Args:
            content: Content to extract from
        
        Returns:
            Orchestration configuration or None
        """
        agentic_match = self.PATTERNS["agentic_block"].search(content)
        if not agentic_match:
            return None
        
        agentic_content = agentic_match.group(1)
        orchestrate_match = self.PATTERNS["orchestrate_invoke"].search(agentic_content)
        
        if orchestrate_match:
            return {
                "strategy": orchestrate_match.group(1),
                "task": orchestrate_match.group(2).strip(),
            }
        
        return None
    
    def strip_agent_patterns(self, content: str) -> str:
        """
        Remove all agent patterns from content.
        
        Args:
            content: Content to strip
        
        Returns:
            Content with agent patterns removed
        """
        result = content
        
        for pattern in self.PATTERNS.values():
            result = pattern.sub("", result)
        
        return result.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get interception statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_intercepts": self._intercept_count,
            "mode_counts": dict(self._mode_counts),
        }
    
    def reset_statistics(self):
        """Reset interception statistics."""
        self._intercept_count = 0
        self._mode_counts = {mode.value: 0 for mode in PiscesLxAgentMode}
    
    def __repr__(self) -> str:
        return f"PiscesLxAgentInterceptor(intercepts={self._intercept_count})"
