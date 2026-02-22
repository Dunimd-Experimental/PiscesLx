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
OPSS Integration Module - Unified Interface for OPSS Components

This module provides a unified interface for integrating OPSS (Operator-based
Production Service System) components into the inference engine.

OPSS Components:
    - POPSSMCPPlaza: MCP tool plaza for tool management
    - POPSSSwarmCoordinator: Multi-agent swarm coordination
    - POPSSDynamicOrchestrator: Dynamic task orchestration
    - POPSSMCPBridge: Agent-MCP bridge for tool access
    - POPSSAggregentRegistry: Agent registration and discovery
    - POPSSToolRegistry: Unified tool registry

Architecture:
    PiscesLxOPSSIntegration
    ├── MCP Plaza Integration
    │   ├── Tool registration and discovery
    │   ├── Tool execution (sync/async)
    │   └── Session management
    ├── Swarm Coordinator Integration
    │   ├── Multi-agent task distribution
    │   ├── Topology management
    │   └── Agent registration
    ├── Orchestrator Integration
    │   ├── Dynamic task planning
    │   ├── Agent selection
    │   └── Result aggregation
    └── MCP Bridge Integration
        ├── Tool discovery and caching
        ├── Tool validation
        └── Statistics tracking

Usage:
    >>> from tools.infer.opss_integration import PiscesLxOPSSIntegration
    >>> from tools.infer.config import OPSSConfig
    >>> 
    >>> config = OPSSConfig(enable_mcp_plaza=True, enable_swarm_coordinator=True)
    >>> opss = PiscesLxOPSSIntegration(config)
    >>> opss.initialize()
    >>> 
    >>> # Execute a tool
    >>> result = await opss.execute_tool_async("web_search", {"query": "AI news"})
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import logging
import asyncio

from .config import OPSSConfig


@dataclass
class PiscesLxToolInfo:
    """Tool information structure."""
    name: str
    description: str
    category: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class PiscesLxAgentInfo:
    """Agent information structure."""
    agent_id: str
    agent_type: str
    name: str
    capabilities: List[str]
    status: str = "idle"


class PiscesLxOPSSIntegration:
    """
    Unified OPSS Integration for PiscesLx Inference Engine.
    
    This class provides a single point of access to all OPSS components,
    handling initialization, lifecycle management, and unified APIs.
    
    Features:
        - Unified interface for all OPSS components
        - Lazy initialization of components
        - Graceful degradation when components unavailable
        - Comprehensive error handling
        - Statistics and monitoring
    
    Example:
        >>> config = OPSSConfig()
        >>> opss = PiscesLxOPSSIntegration(config)
        >>> opss.initialize()
        >>> 
        >>> # List available tools
        >>> tools = opss.list_tools()
        >>> 
        >>> # Execute a tool
        >>> result = await opss.execute_tool_async("web_search", {"query": "test"})
    """
    
    def __init__(self, config: OPSSConfig):
        """
        Initialize OPSS integration.
        
        Args:
            config: OPSS configuration
        """
        self.config = config
        self._LOG = logging.getLogger(self.__class__.__name__)
        
        self._mcp_plaza = None
        self._agent_registry = None
        self._swarm_coordinator = None
        self._orchestrator = None
        self._mcp_bridge = None
        self._tool_registry = None
        
        self._initialized = False
        self._init_errors: List[str] = []
    
    def initialize(self) -> bool:
        """
        Initialize all enabled OPSS components.
        
        Returns:
            True if initialization succeeded for all enabled components
        """
        if self._initialized:
            return True
        
        self._LOG.info("Initializing OPSS integration...")
        
        try:
            if self.config.enable_mcp_plaza:
                self._init_mcp_plaza()
            
            if self.config.enable_swarm_coordinator or self.config.enable_orchestrator:
                self._init_agent_registry()
            
            if self.config.enable_swarm_coordinator:
                self._init_swarm_coordinator()
            
            if self.config.enable_orchestrator:
                self._init_orchestrator()
            
            if self.config.enable_mcp_bridge and self._mcp_plaza:
                self._init_mcp_bridge()
            
            self._init_tool_registry()
            
            self._initialized = True
            self._LOG.info("OPSS integration initialized successfully")
            
            if self._init_errors:
                self._LOG.warning(f"Initialization warnings: {self._init_errors}")
            
            return True
            
        except Exception as e:
            self._LOG.error(f"Failed to initialize OPSS integration: {e}")
            return False
    
    def _init_mcp_plaza(self):
        """Initialize MCP Plaza."""
        try:
            from opss.mcp.core import POPSSMCPPlaza
            from opss.mcp.types import POPSSMCPConfiguration
            
            mcp_config = POPSSMCPConfiguration(
                load_default_tools=True,
                max_workers=self.config.mcp_max_workers,
                session_timeout=self.config.mcp_session_timeout,
            )
            
            self._mcp_plaza = POPSSMCPPlaza(mcp_config)
            self._mcp_plaza.initialize()
            self._LOG.info("MCP Plaza initialized")
            
        except ImportError as e:
            self._init_errors.append(f"MCP Plaza import failed: {e}")
            self._LOG.warning(f"MCP Plaza not available: {e}")
        except Exception as e:
            self._init_errors.append(f"MCP Plaza init failed: {e}")
            self._LOG.error(f"Failed to initialize MCP Plaza: {e}")
    
    def _init_agent_registry(self):
        """Initialize Agent Registry."""
        try:
            from opss.agents.registry import POPSSAggregentRegistry
            
            self._agent_registry = POPSSAggregentRegistry()
            self._LOG.info("Agent Registry initialized")
            
        except ImportError as e:
            self._init_errors.append(f"Agent Registry import failed: {e}")
            self._LOG.warning(f"Agent Registry not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Agent Registry init failed: {e}")
            self._LOG.error(f"Failed to initialize Agent Registry: {e}")
    
    def _init_swarm_coordinator(self):
        """Initialize Swarm Coordinator."""
        try:
            from opss.agents.swarm_coordinator import (
                POPSSSwarmCoordinator,
                POPSSSwarmConfig,
                POPSSSwarmTopology,
            )
            
            topology_map = {
                "hierarchical": POPSSSwarmTopology.HIERARCHICAL,
                "flat": POPSSSwarmTopology.FLAT,
                "mesh": POPSSSwarmTopology.MESH,
                "star": POPSSSwarmTopology.STAR,
                "ring": POPSSSwarmTopology.RING,
            }
            
            topology = topology_map.get(
                self.config.swarm_topology.lower(),
                POPSSSwarmTopology.HIERARCHICAL
            )
            
            swarm_config = POPSSSwarmConfig(
                registry=self._agent_registry,
                topology=topology,
                max_agents=self.config.swarm_max_agents,
                enable_collaboration=True,
                enable_task_decomposition=True,
            )
            
            self._swarm_coordinator = POPSSSwarmCoordinator(swarm_config)
            self._swarm_coordinator.initialize()
            self._LOG.info(f"Swarm Coordinator initialized with {self.config.swarm_topology} topology")
            
        except ImportError as e:
            self._init_errors.append(f"Swarm Coordinator import failed: {e}")
            self._LOG.warning(f"Swarm Coordinator not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Swarm Coordinator init failed: {e}")
            self._LOG.error(f"Failed to initialize Swarm Coordinator: {e}")
    
    def _init_orchestrator(self):
        """Initialize Dynamic Orchestrator."""
        try:
            from opss.agents.orchestrator import (
                POPSSDynamicOrchestrator,
                POPSSOrchestratorConfig,
                POPSSOrchestrationStrategy,
            )
            
            orchestrator_config = POPSSOrchestratorConfig(
                registry=self._agent_registry,
                default_strategy=POPSSOrchestrationStrategy.DYNAMIC,
                max_parallel_agents=self.config.orchestrator_max_parallel,
                enable_adaptive_planning=True,
                enable_result_aggregation=True,
            )
            
            self._orchestrator = POPSSDynamicOrchestrator(orchestrator_config)
            self._LOG.info("Dynamic Orchestrator initialized")
            
        except ImportError as e:
            self._init_errors.append(f"Orchestrator import failed: {e}")
            self._LOG.warning(f"Orchestrator not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Orchestrator init failed: {e}")
            self._LOG.error(f"Failed to initialize Orchestrator: {e}")
    
    def _init_mcp_bridge(self):
        """Initialize MCP Bridge."""
        try:
            from opss.agents.mcp_bridge import (
                POPSSMCPBridge,
                POPSSMCPBridgeConfig,
            )
            
            bridge_config = POPSSMCPBridgeConfig(
                mcp_plaza=self._mcp_plaza,
                auto_discover_tools=True,
                cache_ttl_seconds=self.config.bridge_cache_ttl,
                enable_tool_validation=True,
                retry_on_failure=True,
                retry_count=self.config.bridge_retry_count,
            )
            
            self._mcp_bridge = POPSSMCPBridge(bridge_config)
            self._LOG.info("MCP Bridge initialized")
            
        except ImportError as e:
            self._init_errors.append(f"MCP Bridge import failed: {e}")
            self._LOG.warning(f"MCP Bridge not available: {e}")
        except Exception as e:
            self._init_errors.append(f"MCP Bridge init failed: {e}")
            self._LOG.error(f"Failed to initialize MCP Bridge: {e}")
    
    def _init_tool_registry(self):
        """Initialize Tool Registry."""
        try:
            from opss.am.registry import POPSSToolRegistry
            
            self._tool_registry = POPSSToolRegistry()
            self._LOG.info("Tool Registry initialized")
            
        except ImportError as e:
            self._init_errors.append(f"Tool Registry import failed: {e}")
            self._LOG.warning(f"Tool Registry not available: {e}")
        except Exception as e:
            self._init_errors.append(f"Tool Registry init failed: {e}")
            self._LOG.error(f"Failed to initialize Tool Registry: {e}")
    
    # === MCP Plaza Methods ===
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool synchronously through MCP Plaza.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        
        Raises:
            RuntimeError: If MCP Plaza is not initialized
        """
        if not self._mcp_plaza:
            raise RuntimeError("MCP Plaza not initialized")
        
        return self._mcp_plaza.execute_tool(tool_name, arguments)
    
    async def execute_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool asynchronously through MCP Plaza.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        
        Raises:
            RuntimeError: If MCP Plaza is not initialized
        """
        if not self._mcp_plaza:
            raise RuntimeError("MCP Plaza not initialized")
        
        return await self._mcp_plaza.execute_tool_async(tool_name, arguments)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List available tools.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of tool names
        """
        if not self._mcp_plaza:
            return []
        
        try:
            return self._mcp_plaza.list_tools(category=category)
        except Exception as e:
            self._LOG.error(f"Failed to list tools: {e}")
            return []
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Tool information dictionary or None
        """
        if not self._mcp_plaza:
            return None
        
        try:
            return self._mcp_plaza.get_tool(tool_name)
        except Exception as e:
            self._LOG.error(f"Failed to get tool info for {tool_name}: {e}")
            return None
    
    # === Swarm Coordinator Methods ===
    
    async def submit_swarm_task(
        self,
        task_type: str,
        description: str,
        input_data: Dict[str, Any]
    ) -> str:
        """
        Submit a task to the swarm.
        
        Args:
            task_type: Type of task
            description: Task description
            input_data: Input data for the task
        
        Returns:
            Task ID
        
        Raises:
            RuntimeError: If Swarm Coordinator is not initialized
        """
        if not self._swarm_coordinator:
            raise RuntimeError("Swarm Coordinator not initialized")
        
        return await self._swarm_coordinator.submit_task(
            task_type=task_type,
            description=description,
            input_data=input_data
        )
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get swarm status.
        
        Returns:
            Swarm status dictionary
        """
        if not self._swarm_coordinator:
            return {"status": "unavailable"}
        
        try:
            return self._swarm_coordinator.get_swarm_status()
        except Exception as e:
            self._LOG.error(f"Failed to get swarm status: {e}")
            return {"status": "error", "error": str(e)}
    
    def register_swarm_agent(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        capabilities: set
    ) -> bool:
        """
        Register an agent in the swarm.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            name: Human-readable name
            capabilities: Set of agent capabilities
        
        Returns:
            True if registration succeeded
        """
        if not self._swarm_coordinator:
            return False
        
        try:
            from opss.agents.registry import POPSSAggregentType
            
            type_map = {
                "general": POPSSAggregentType.GENERAL,
                "code": POPSSAggregentType.CODE,
                "research": POPSSAggregentType.RESEARCH,
                "analysis": POPSSAggregentType.ANALYSIS,
                "creative": POPSSAggregentType.CREATIVE,
                "tool": POPSSAggregentType.TOOL,
            }
            
            agent_type_enum = type_map.get(agent_type.lower(), POPSSAggregentType.GENERAL)
            
            return self._swarm_coordinator.register_agent(
                agent_id=agent_id,
                agent_type=agent_type_enum,
                name=name,
                capabilities=capabilities
            )
        except Exception as e:
            self._LOG.error(f"Failed to register swarm agent: {e}")
            return False
    
    # === Orchestrator Methods ===
    
    async def orchestrate_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate a task using dynamic orchestrator.
        
        Args:
            task: Task description
            context: Optional execution context
        
        Returns:
            Orchestration result dictionary
        
        Raises:
            RuntimeError: If Orchestrator is not initialized
        """
        if not self._orchestrator:
            raise RuntimeError("Orchestrator not initialized")
        
        try:
            result = await self._orchestrator.orchestrate(task, context=context)
            
            return {
                "success": result.success,
                "output": result.aggregated_output,
                "execution_time": result.total_execution_time,
                "agent_executions": result.agent_executions,
                "metadata": result.metadata if hasattr(result, 'metadata') else {},
            }
        except Exception as e:
            self._LOG.error(f"Orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator metrics.
        
        Returns:
            Metrics dictionary
        """
        if not self._orchestrator:
            return {}
        
        try:
            return self._orchestrator.get_metrics()
        except Exception as e:
            self._LOG.error(f"Failed to get orchestrator metrics: {e}")
            return {}
    
    # === MCP Bridge Methods ===
    
    def get_available_tools_info(self, category: Optional[str] = None) -> List[PiscesLxToolInfo]:
        """
        Get available tools with metadata.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of tool information
        """
        if not self._mcp_bridge:
            return []
        
        try:
            tools = self._mcp_bridge.get_available_tools(category=category)
            return [
                PiscesLxToolInfo(
                    name=t.tool_name,
                    description=t.description,
                    category=t.category if hasattr(t, 'category') else None,
                    parameters=t.parameters if hasattr(t, 'parameters') else None,
                )
                for t in tools
            ]
        except Exception as e:
            self._LOG.error(f"Failed to get available tools: {e}")
            return []
    
    async def call_tool_via_bridge(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Any:
        """
        Call a tool through MCP Bridge.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            session_id: Optional session ID
        
        Returns:
            Tool result
        
        Raises:
            RuntimeError: If MCP Bridge is not initialized
        """
        if not self._mcp_bridge:
            raise RuntimeError("MCP Bridge not initialized")
        
        return await self._mcp_bridge.call_tool(tool_name, arguments, session_id)
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Get tool usage statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self._mcp_bridge:
            return {}
        
        try:
            return self._mcp_bridge.get_tool_statistics()
        except Exception as e:
            self._LOG.error(f"Failed to get tool statistics: {e}")
            return {}
    
    # === Tool Registry Methods ===
    
    def register_native_function(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Register a native function as a tool.
        
        Args:
            name: Tool name
            function: Python function
            description: Tool description
            parameters: Parameter schema
        """
        if not self._tool_registry:
            self._LOG.warning("Tool Registry not initialized")
            return
        
        try:
            self._tool_registry.register_native_function(
                name=name,
                function=function,
                description=description,
                parameters=parameters or {}
            )
        except Exception as e:
            self._LOG.error(f"Failed to register native function: {e}")
    
    async def execute_registered_tool(
        self,
        tool_id: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a registered tool.
        
        Args:
            tool_id: Tool identifier
            arguments: Tool arguments
        
        Returns:
            Tool result
        
        Raises:
            RuntimeError: If Tool Registry is not initialized
        """
        if not self._tool_registry:
            raise RuntimeError("Tool Registry not initialized")
        
        return await self._tool_registry.execute(tool_id, arguments)
    
    def list_registered_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Returns:
            List of tool information
        """
        if not self._tool_registry:
            return []
        
        try:
            return self._tool_registry.list_tools()
        except Exception as e:
            self._LOG.error(f"Failed to list registered tools: {e}")
            return []
    
    # === Agent Registry Methods ===
    
    def list_agents(self) -> List[PiscesLxAgentInfo]:
        """
        List all registered agents.
        
        Returns:
            List of agent information
        """
        if not self._agent_registry:
            return []
        
        try:
            agents = self._agent_registry.list_agents()
            return [
                PiscesLxAgentInfo(
                    agent_id=a.get("id", ""),
                    agent_type=a.get("type", "general"),
                    name=a.get("name", ""),
                    capabilities=a.get("capabilities", []),
                    status=a.get("status", "idle"),
                )
                for a in agents
            ]
        except Exception as e:
            self._LOG.error(f"Failed to list agents: {e}")
            return []
    
    def get_agent(self, agent_id: str) -> Optional[PiscesLxAgentInfo]:
        """
        Get agent information by ID.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Agent information or None
        """
        if not self._agent_registry:
            return None
        
        try:
            agent = self._agent_registry.get_agent(agent_id)
            if agent:
                return PiscesLxAgentInfo(
                    agent_id=agent.get("id", ""),
                    agent_type=agent.get("type", "general"),
                    name=agent.get("name", ""),
                    capabilities=agent.get("capabilities", []),
                    status=agent.get("status", "idle"),
                )
            return None
        except Exception as e:
            self._LOG.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    # === Status and Lifecycle ===
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get OPSS integration status.
        
        Returns:
            Status dictionary
        """
        return {
            "initialized": self._initialized,
            "components": {
                "mcp_plaza": self._mcp_plaza is not None,
                "agent_registry": self._agent_registry is not None,
                "swarm_coordinator": self._swarm_coordinator is not None,
                "orchestrator": self._orchestrator is not None,
                "mcp_bridge": self._mcp_bridge is not None,
                "tool_registry": self._tool_registry is not None,
            },
            "config": {
                "enable_mcp_plaza": self.config.enable_mcp_plaza,
                "enable_swarm_coordinator": self.config.enable_swarm_coordinator,
                "enable_orchestrator": self.config.enable_orchestrator,
                "enable_mcp_bridge": self.config.enable_mcp_bridge,
            },
            "init_errors": self._init_errors,
        }
    
    def shutdown(self):
        """Shutdown all OPSS components."""
        self._LOG.info("Shutting down OPSS integration...")
        
        if self._swarm_coordinator:
            try:
                self._swarm_coordinator.shutdown()
            except Exception as e:
                self._LOG.error(f"Error shutting down Swarm Coordinator: {e}")
        
        if self._orchestrator:
            try:
                self._orchestrator.shutdown()
            except Exception as e:
                self._LOG.error(f"Error shutting down Orchestrator: {e}")
        
        if self._mcp_bridge:
            try:
                self._mcp_bridge.shutdown()
            except Exception as e:
                self._LOG.error(f"Error shutting down MCP Bridge: {e}")
        
        if self._mcp_plaza:
            try:
                self._mcp_plaza.shutdown()
            except Exception as e:
                self._LOG.error(f"Error shutting down MCP Plaza: {e}")
        
        self._initialized = False
        self._LOG.info("OPSS integration shutdown complete")
    
    def __repr__(self) -> str:
        return (
            f"PiscesLxOPSSIntegration("
            f"initialized={self._initialized}, "
            f"components={sum([
                self._mcp_plaza is not None,
                self._agent_registry is not None,
                self._swarm_coordinator is not None,
                self._orchestrator is not None,
                self._mcp_bridge is not None,
                self._tool_registry is not None,
            ])})"
            f")"
        )
