#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
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
Smart Routing Engine for Remote MCP Client Execution.

This module implements intelligent routing decisions for tool execution,
determining whether tools should be executed locally, remotely, or through
specific user-local MCP clients based on various criteria.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
from dataclasses import dataclass

from .types import (
    RemoteExecutionMode, RemoteExecutionResult, RemoteToolCall,
    RemoteClientConfig, RemoteConnectionError
)
from .manager import ArcticRemoteMCPManager
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Remote.MCP.Router")

class ToolCategory(Enum):
    """Categories of tools for routing decisions."""
    PRIVACY_SENSITIVE = "privacy_sensitive"
    SECURITY_CRITICAL = "security_critical"
    USER_LOCAL = "user_local"
    SERVER_LOCAL = "server_local"
    NETWORK_DEPENDENT = "network_dependent"
    PERFORMANCE_CRITICAL = "performance_critical"
    PUBLIC_UTILITY = "public_utility"
    UNKNOWN = "unknown"

@dataclass
class RoutingRule:
    """Represents a routing rule for tool execution."""
    tool_patterns: List[str]  # Tool name patterns (supports wildcards)
    categories: List[ToolCategory]
    preferred_mode: RemoteExecutionMode
    client_requirements: Dict[str, Any]  # Requirements for client selection
    priority: int = 0  # Higher priority rules take precedence
    enabled: bool = True
    description: str = ""

@dataclass
class RoutingDecision:
    """Represents a routing decision for tool execution."""
    execution_mode: RemoteExecutionMode
    target_client_id: Optional[str]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    alternative_modes: List[RemoteExecutionMode]

class ArcticRemoteMCPRouter:
    """
    Smart routing engine for MCP tool execution.
    
    Implements intelligent routing decisions based on tool characteristics,
    client availability, performance metrics, and user preferences.
    """
    
    def __init__(self, manager: ArcticRemoteMCPManager):
        """
        Initialize the routing engine.
        
        Args:
            manager: Remote MCP client manager
        """
        self.manager = manager
        self._routing_rules: List[RoutingRule] = []
        self._tool_categories: Dict[str, Set[ToolCategory]] = {}
        self._client_performance_cache: Dict[str, float] = {}
        self._execution_history: List[Tuple[str, str, RemoteExecutionMode, bool, float]] = []
        self._router_lock = asyncio.Lock()
        
        # Initialize default routing rules
        self._initialize_default_rules()
        
        logger.info("Initialized Remote MCP Router")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default routing rules."""
        default_rules = [
            RoutingRule(
                tool_patterns=["file_*", "shell", "python", "system"],
                categories=[ToolCategory.PRIVACY_SENSITIVE, ToolCategory.USER_LOCAL],
                preferred_mode=RemoteExecutionMode.REMOTE,
                client_requirements={"capabilities": ["local_execution"]},
                priority=100,
                description="Privacy-sensitive tools should run on user-local clients"
            ),
            
            RoutingRule(
                tool_patterns=["calculator", "text_processor", "data_transformer"],
                categories=[ToolCategory.PUBLIC_UTILITY, ToolCategory.PERFORMANCE_CRITICAL],
                preferred_mode=RemoteExecutionMode.NATIVE,
                client_requirements={},
                priority=90,
                description="Simple utility tools can run natively for better performance"
            ),
            
            RoutingRule(
                tool_patterns=["web_*", "api_*", "network_*"],
                categories=[ToolCategory.NETWORK_DEPENDENT],
                preferred_mode=RemoteExecutionMode.REMOTE,
                client_requirements={"capabilities": ["network_access"]},
                priority=80,
                description="Network-dependent tools can run on user-local clients"
            ),
            
            RoutingRule(
                tool_patterns=["*"],
                categories=[ToolCategory.UNKNOWN],
                preferred_mode=RemoteExecutionMode.AUTO,
                client_requirements={},
                priority=0,
                description="Default rule for unknown tools"
            )
        ]
        
        self._routing_rules.extend(default_rules)
        logger.debug(f"Initialized {len(default_rules)} default routing rules")
    
    async def add_routing_rule(self, rule: RoutingRule) -> None:
        """
        Add a custom routing rule.
        
        Args:
            rule: Routing rule to add
        """
        async with self._router_lock:
            self._routing_rules.append(rule)
            # Sort by priority (descending)
            self._routing_rules.sort(key=lambda r: r.priority, reverse=True)
            
        logger.info(f"Added routing rule: {rule.description}")
    
    async def categorize_tool(self, tool_name: str, 
                            tool_metadata: Optional[Dict[str, Any]] = None) -> Set[ToolCategory]:
        """
        Categorize a tool based on its name and metadata.
        
        Args:
            tool_name: Tool name
            tool_metadata: Optional tool metadata
            
        Returns:
            Set of tool categories
        """
        categories = set()
        
        # Check cached categories
        if tool_name in self._tool_categories:
            categories.update(self._tool_categories[tool_name])
        
        # Analyze tool name patterns
        tool_name_lower = tool_name.lower()
        
        if any(keyword in tool_name_lower for keyword in ['file', 'read', 'write', 'delete']):
            categories.add(ToolCategory.USER_LOCAL)
            categories.add(ToolCategory.PRIVACY_SENSITIVE)
        
        if any(keyword in tool_name_lower for keyword in ['shell', 'command', 'execute', 'system']):
            categories.add(ToolCategory.SECURITY_CRITICAL)
            categories.add(ToolCategory.USER_LOCAL)
        
        if any(keyword in tool_name_lower for keyword in ['web', 'api', 'http', 'network']):
            categories.add(ToolCategory.NETWORK_DEPENDENT)
        
        if any(keyword in tool_name_lower for keyword in ['calculator', 'math', 'transform']):
            categories.add(ToolCategory.PUBLIC_UTILITY)
        
        if tool_metadata:
            # Analyze metadata
            if tool_metadata.get('requires_user_data', False):
                categories.add(ToolCategory.PRIVACY_SENSITIVE)
            
            if tool_metadata.get('performance_critical', False):
                categories.add(ToolCategory.PERFORMANCE_CRITICAL)
            
            if tool_metadata.get('security_level', 'low') == 'high':
                categories.add(ToolCategory.SECURITY_CRITICAL)
        
        if not categories:
            categories.add(ToolCategory.UNKNOWN)
        
        # Cache for future use
        self._tool_categories[tool_name] = categories
        
        return categories
    
    async def make_routing_decision(self, tool_name: str, 
                                  parameters: Dict[str, Any],
                                  user_preference: Optional[str] = None) -> RoutingDecision:
        """
        Make a routing decision for tool execution.
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            user_preference: Optional user preference ("local", "remote", "auto")
            
        Returns:
            Routing decision
        """
        async with self._router_lock:
            # Categorize tool
            categories = await self.categorize_tool(tool_name)
            
            # Check user preference
            if user_preference:
                preference_mode = self._parse_user_preference(user_preference)
                if preference_mode:
                    return RoutingDecision(
                        execution_mode=preference_mode,
                        target_client_id=None,
                        reasoning=f"User preference: {user_preference}",
                        confidence=1.0,
                        alternative_modes=[]
                    )
            
            # Apply routing rules
            for rule in self._routing_rules:
                if not rule.enabled:
                    continue
                
                if self._matches_rule(tool_name, categories, rule):
                    return await self._create_routing_decision(rule, tool_name, parameters)
            
            # Default decision
            return RoutingDecision(
                execution_mode=RemoteExecutionMode.AUTO,
                target_client_id=None,
                reasoning="No matching rules, using auto mode",
                confidence=0.5,
                alternative_modes=[RemoteExecutionMode.NATIVE, RemoteExecutionMode.REMOTE]
            )
    
    def _parse_user_preference(self, preference: str) -> Optional[RemoteExecutionMode]:
        """Parse user preference string."""
        preference_lower = preference.lower()
        
        if preference_lower in ('local', 'native', 'server'):
            return RemoteExecutionMode.NATIVE
        elif preference_lower in ('remote', 'client', 'user'):
            return RemoteExecutionMode.REMOTE
        elif preference_lower == 'auto':
            return RemoteExecutionMode.AUTO
        
        return None
    
    def _matches_rule(self, tool_name: str, categories: Set[ToolCategory], 
                     rule: RoutingRule) -> bool:
        """Check if a tool matches a routing rule."""
        # Check tool name patterns
        name_match = any(
            self._matches_pattern(tool_name, pattern) 
            for pattern in rule.tool_patterns
        )
        
        # Check categories
        category_match = any(category in categories for category in rule.categories)
        
        return name_match or category_match
    
    def _matches_pattern(self, tool_name: str, pattern: str) -> bool:
        """Check if tool name matches a pattern (supports wildcards)."""
        if pattern == "*":
            return True
        
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(tool_name.lower(), pattern.lower())
        
        return tool_name.lower() == pattern.lower()
    
    async def _create_routing_decision(self, rule: RoutingRule, 
                                     tool_name: str, 
                                     parameters: Dict[str, Any]) -> RoutingDecision:
        """Create a routing decision based on a rule."""
        if rule.preferred_mode == RemoteExecutionMode.REMOTE:
            # Find suitable client
            client_id = await self._select_best_client(tool_name, rule.client_requirements)
            
            if client_id:
                return RoutingDecision(
                    execution_mode=RemoteExecutionMode.REMOTE,
                    target_client_id=client_id,
                    reasoning=f"Rule match: {rule.description}",
                    confidence=0.9,
                    alternative_modes=[RemoteExecutionMode.NATIVE]
                )
            else:
                # Fallback to native if no suitable client
                return RoutingDecision(
                    execution_mode=RemoteExecutionMode.NATIVE,
                    target_client_id=None,
                    reasoning=f"Rule match but no suitable client: {rule.description}",
                    confidence=0.7,
                    alternative_modes=[]
                )
        
        else:
            return RoutingDecision(
                execution_mode=rule.preferred_mode,
                target_client_id=None,
                reasoning=f"Rule match: {rule.description}",
                confidence=0.8,
                alternative_modes=[RemoteExecutionMode.REMOTE, RemoteExecutionMode.NATIVE]
            )
    
    async def _select_best_client(self, tool_name: str, 
                                requirements: Dict[str, Any]) -> Optional[str]:
        """Select the best client for tool execution."""
        try:
            # Get all clients status
            clients_status = await self.manager.get_all_clients_status()
            
            suitable_clients = []
            
            for client_id, client_info in clients_status["clients"].items():
                # Skip unhealthy clients
                if not client_info.get("is_healthy", False):
                    continue
                
                # Skip disconnected clients
                if not client_info.get("is_connected", False):
                    continue
                
                # Check requirements
                if self._meets_requirements(client_info, requirements):
                    # Calculate score based on performance and usage
                    score = self._calculate_client_score(client_info)
                    suitable_clients.append((client_id, score))
            
            # Sort by score (descending)
            suitable_clients.sort(key=lambda x: x[1], reverse=True)
            
            if suitable_clients:
                best_client_id = suitable_clients[0][0]
                logger.debug(f"Selected client {best_client_id} for tool {tool_name}")
                return best_client_id
            
            logger.warning(f"No suitable client found for tool {tool_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error selecting best client: {e}")
            return None
    
    def _meets_requirements(self, client_info: Dict[str, Any], 
                           requirements: Dict[str, Any]) -> bool:
        """Check if client meets requirements."""
        if not requirements:
            return True
        
        # Check capabilities
        required_capabilities = requirements.get("capabilities", [])
        if required_capabilities:
            client_capabilities = client_info.get("connection_status", {}).get("capabilities", {})
            if not all(cap in client_capabilities for cap in required_capabilities):
                return False
        
        return True
    
    def _calculate_client_score(self, client_info: Dict[str, Any]) -> float:
        """Calculate a score for client selection."""
        score = 0.5  # Base score
        
        # Bonus for being connected
        if client_info.get("is_connected", False):
            score += 0.2
        
        # Bonus for being healthy
        if client_info.get("is_healthy", False):
            score += 0.2
        
        # Penalty for recent errors
        consecutive_errors = client_info.get("consecutive_errors", 0)
        score -= min(0.3, consecutive_errors * 0.1)
        
        # Bonus for low usage (load balancing)
        usage_count = client_info.get("usage_count", 0)
        score += max(0.0, (10 - usage_count) * 0.01)
        
        return max(0.0, min(1.0, score))
    
    async def record_execution_result(self, tool_name: str, 
                                    execution_mode: RemoteExecutionMode,
                                    success: bool, 
                                    execution_time: float) -> None:
        """
        Record execution result for learning and optimization.
        
        Args:
            tool_name: Tool name
            execution_mode: Execution mode used
            success: Whether execution was successful
            execution_time: Execution time in seconds
        """
        async with self._router_lock:
            self._execution_history.append(
                (tool_name, str(execution_mode), success, execution_time)
            )
            
            # Keep only recent history (last 1000 executions)
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        async with self._router_lock:
            if not self._execution_history:
                return {"total_executions": 0}
            
            total_executions = len(self._execution_history)
            successful_executions = sum(1 for _, _, success, _ in self._execution_history if success)
            
            # Statistics by execution mode
            mode_stats = {}
            for tool_name, mode_str, success, exec_time in self._execution_history:
                if mode_str not in mode_stats:
                    mode_stats[mode_str] = {
                        "count": 0,
                        "successes": 0,
                        "total_time": 0.0,
                        "avg_time": 0.0
                    }
                
                mode_stats[mode_str]["count"] += 1
                if success:
                    mode_stats[mode_str]["successes"] += 1
                mode_stats[mode_str]["total_time"] += exec_time
            
            # Calculate averages
            for mode_stats_data in mode_stats.values():
                if mode_stats_data["count"] > 0:
                    mode_stats_data["avg_time"] = (
                        mode_stats_data["total_time"] / mode_stats_data["count"]
                    )
                    mode_stats_data["success_rate"] = (
                        mode_stats_data["successes"] / mode_stats_data["count"]
                    )
            
            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "overall_success_rate": successful_executions / total_executions,
                "mode_statistics": mode_stats,
                "routing_rules_count": len(self._routing_rules),
                "tool_categories_count": len(self._tool_categories)
            }
    
    def __repr__(self) -> str:
        """String representation of the router."""
        return f"ArcticRemoteMCPRouter(rules={len(self._routing_rules)}, history={len(self._execution_history)})"