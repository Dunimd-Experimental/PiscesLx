#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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
Unified tool executor for MCP system.

This module provides unified execution of native tools, internal tools,
and external tools with intelligent routing and fallback mechanisms.
"""

import asyncio
import time
import importlib
import inspect
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Use dms_core logging exclusively
import dms_core
logger = dms_core.log.get_logger("Ruchbah.Utils.MCP.ToolExecutor")

# Import modules with fallback for standalone testing
try:
    from .execution import PiscesLxCoreMCPExecutionResult, PiscesLxCoreMCPExecutionMode, PiscesLxCoreMCPExecutionStatus, PiscesLxCoreMCPExecutionManager
    from .remote_client import PiscesLxCoreMCPRemoteClient, _RemoteToolMetadata, execute_remote_tool
    from .xml_utils import PiscesLxCoreMCPXMLParser, validate_xml
except ImportError:
    # Fallback for standalone testing
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from execution import PiscesLxCoreMCPExecutionResult, PiscesLxCoreMCPExecutionMode, PiscesLxCoreMCPExecutionStatus, PiscesLxCoreMCPExecutionManager
    from remote_client import PiscesLxCoreMCPRemoteClient, _RemoteToolMetadata, execute_remote_tool
    from xml_utils import PiscesLxCoreMCPXMLParser


class PiscesLxCoreMCPToolType(Enum):
    """Types of tools supported by the executor."""
    NATIVE = "native"      # Built-in tools optimized for performance
    INTERNAL = "internal"  # Internal tools for system operations
    EXTERNAL = "external" # External tools requiring remote execution
    PYTHON = "python"     # Python-based tools
    EXECUTABLE = "executable" # Executable tools


@dataclass
class PiscesLxCoreMCPToolMetadata:
    """Metadata for tools."""
    name: str
    description: str
    tool_type: PiscesLxCoreMCPToolType
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    module_path: Optional[str] = None
    executable_path: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    priority: int = 0  # Higher priority tools are tried first
    fallback_enabled: bool = True
    timeout: float = 30.0


@dataclass
class PiscesLxCoreMCPExecutionContext:
    """Context for tool execution."""
    execution_id: str
    tool_name: str
    parameters: Dict[str, Any]
    preferred_types: List[PiscesLxCoreMCPToolType]
    allow_fallback: bool
    timeout: float
    metadata: Optional[Dict[str, Any]] = None


class ToolExecutor(ABC):
    """Abstract base class for tool executors."""
    
    @abstractmethod
    async def can_execute(self, tool_name: str, context: PiscesLxCoreMCPExecutionContext) -> bool:
        """Check if this executor can handle the tool."""
        pass
    
    @abstractmethod
    async def execute(self, tool_name: str, parameters: Dict[str, Any], context: PiscesLxCoreMCPExecutionContext) -> PiscesLxCoreMCPExecutionResult:
        """Execute the tool."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[PiscesLxCoreMCPToolType]:
        """Get list of tool types this executor supports."""
        pass


class PiscesLxCoreMCPNativeToolExecutor(ToolExecutor):
    """Executor for native tools with optimized performance."""
    
    def __init__(self):
        """Initialize native tool executor."""
        self.native_tools: Dict[str, PiscesLxCoreMCPToolMetadata] = {}
        self.execution_manager = PiscesLxCoreMCPExecutionManager()
        
        logger.info("NativeToolExecutor initialized")
    
    def register_native_tool(self, metadata: PiscesLxCoreMCPToolMetadata):
        """Register a native tool."""
        if metadata.tool_type != PiscesLxCoreMCPToolType.NATIVE:
            raise ValueError(f"Tool {metadata.name} is not a native tool")
        
        self.native_tools[metadata.name] = metadata
        logger.debug(f"Registered native tool: {metadata.name}")
    
    async def can_execute(self, tool_name: str, context: PiscesLxCoreMCPExecutionContext) -> bool:
        """Check if this executor can handle the tool."""
        return tool_name in self.native_tools
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any], context: PiscesLxCoreMCPExecutionContext) -> PiscesLxCoreMCPExecutionResult:
        """Execute the native tool."""
        metadata = self.native_tools.get(tool_name)
        if not metadata:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"Native tool {tool_name} not found",
                error_code="TOOL_NOT_FOUND",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
        
        if not metadata.function:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"Native tool {tool_name} has no function",
                error_code="NO_FUNCTION",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
        
        try:
            # Execute with performance optimization
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(metadata.function):
                result = await metadata.function(**parameters)
            else:
                result = metadata.function(**parameters)
            
            execution_time = time.time() - start_time
            
            return PiscesLxCoreMCPExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.COMPLETED,
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Native tool {tool_name} execution failed: {e}")
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="NATIVE_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
    
    def get_supported_types(self) -> List[PiscesLxCoreMCPToolType]:
        """Get supported tool types."""
        return [PiscesLxCoreMCPToolType.NATIVE]


class PiscesLxCoreMCPInternalToolExecutor(ToolExecutor):
    """Executor for internal system tools."""
    
    def __init__(self):
        """Initialize internal tool executor."""
        self.internal_tools: Dict[str, PiscesLxCoreMCPToolMetadata] = {}
        self.execution_manager = PiscesLxCoreMCPExecutionManager()
        
        logger.info("InternalToolExecutor initialized")
    
    def register_internal_tool(self, metadata: PiscesLxCoreMCPToolMetadata):
        """Register an internal tool."""
        if metadata.tool_type != PiscesLxCoreMCPToolType.INTERNAL:
            raise ValueError(f"Tool {metadata.name} is not an internal tool")
        
        self.internal_tools[metadata.name] = metadata
        logger.debug(f"Registered internal tool: {metadata.name}")
    
    async def can_execute(self, tool_name: str, context: PiscesLxCoreMCPExecutionContext) -> bool:
        """Check if this executor can handle the tool."""
        return tool_name in self.internal_tools
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any], context: PiscesLxCoreMCPExecutionContext) -> PiscesLxCoreMCPExecutionResult:
        """Execute the internal tool."""
        metadata = self.internal_tools.get(tool_name)
        if not metadata:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"Internal tool {tool_name} not found",
                error_code="TOOL_NOT_FOUND",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
        
        if not metadata.function:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"Internal tool {tool_name} has no function",
                error_code="NO_FUNCTION",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
        
        try:
            # Execute with system-level privileges
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(metadata.function):
                result = await metadata.function(**parameters)
            else:
                result = metadata.function(**parameters)
            
            execution_time = time.time() - start_time
            
            return PiscesLxCoreMCPExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.COMPLETED,
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Internal tool {tool_name} execution failed: {e}")
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="INTERNAL_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
    
    def get_supported_types(self) -> List[PiscesLxCoreMCPToolType]:
        """Get supported tool types."""
        return [PiscesLxCoreMCPToolType.INTERNAL]


class PiscesLxCoreMCPExternalToolExecutor(ToolExecutor):
    """Executor for external tools using remote execution."""
    
    def __init__(self):
        """Initialize external tool executor."""
        self.external_tools: Dict[str, PiscesLxCoreMCPToolMetadata] = {}
        
        logger.info("ExternalToolExecutor initialized")
    
    def register_external_tool(self, metadata: PiscesLxCoreMCPToolMetadata):
        """Register an external tool."""
        if metadata.tool_type != PiscesLxCoreMCPToolType.EXTERNAL:
            raise ValueError(f"Tool {metadata.name} is not an external tool")
        
        self.external_tools[metadata.name] = metadata
        logger.debug(f"Registered external tool: {metadata.name}")
    
    async def can_execute(self, tool_name: str, context: PiscesLxCoreMCPExecutionContext) -> bool:
        """Check if this executor can handle the tool."""
        return tool_name in self.external_tools
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any], context: PiscesLxCoreMCPExecutionContext) -> PiscesLxCoreMCPExecutionResult:
        """Execute the external tool remotely."""
        metadata = self.external_tools.get(tool_name)
        if not metadata:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"External tool {tool_name} not found",
                error_code="TOOL_NOT_FOUND",
                mode=PiscesLxCoreMCPExecutionMode.REMOTE
            )
        
        try:
            # Execute remotely using the remote client
            client_id = context.metadata.get("client_id", "default") if context.metadata else "default"
            
            result = await execute_remote_tool(
                client_id=client_id,
                tool_name=tool_name,
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            logger.error(f"External tool {tool_name} execution failed: {e}")
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="EXTERNAL_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.REMOTE
            )
    
    def get_supported_types(self) -> List[PiscesLxCoreMCPToolType]:
        """Get supported tool types."""
        return [PiscesLxCoreMCPToolType.EXTERNAL]


class PiscesLxCoreMCPUnifiedToolExecutor:
    """Unified executor that intelligently routes tool execution."""
    
    def __init__(self):
        """Initialize unified tool executor."""
        self.executors: List[ToolExecutor] = [
            PiscesLxCoreMCPNativeToolExecutor(),
            PiscesLxCoreMCPInternalToolExecutor(),
            PiscesLxCoreMCPExternalToolExecutor()
        ]
        
        self.tool_registry: Dict[str, List[PiscesLxCoreMCPToolMetadata]] = {}  # tool_name -> [metadata1, metadata2, ...]
        self.execution_stats: Dict[str, Dict[str, Any]] = {}  # tool_name -> stats
        
        logger.info("UnifiedToolExecutor initialized")
    
    def register_tool(self, metadata: PiscesLxCoreMCPToolMetadata):
        """Register a tool with the unified executor."""
        if metadata.name not in self.tool_registry:
            self.tool_registry[metadata.name] = []
        
        self.tool_registry[metadata.name].append(metadata)
        
        # Register with appropriate executor
        for executor in self.executors:
            if metadata.tool_type in executor.get_supported_types():
                if isinstance(executor, PiscesLxCoreMCPNativeToolExecutor) and metadata.tool_type == PiscesLxCoreMCPToolType.NATIVE:
                    executor.register_native_tool(metadata)
                elif isinstance(executor, PiscesLxCoreMCPInternalToolExecutor) and metadata.tool_type == PiscesLxCoreMCPToolType.INTERNAL:
                    executor.register_internal_tool(metadata)
                elif isinstance(executor, PiscesLxCoreMCPExternalToolExecutor) and metadata.tool_type == PiscesLxCoreMCPToolType.EXTERNAL:
                    executor.register_external_tool(metadata)
        
        logger.debug(f"Registered tool {metadata.name} of type {metadata.tool_type.value}")
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        execution_id: str,
        preferred_types: Optional[List[PiscesLxCoreMCPToolType]] = None,
        allow_fallback: bool = True,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PiscesLxCoreMCPExecutionResult:
        """
        Execute a tool with intelligent routing and fallback.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            execution_id: Unique execution identifier
            preferred_types: Preferred tool types to try first
            allow_fallback: Whether to allow fallback to other types
            timeout: Execution timeout
            metadata: Additional metadata for execution
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        # Get available tool variants
        tool_variants = self.tool_registry.get(tool_name, [])
        if not tool_variants:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"Tool {tool_name} not found in registry",
                error_code="TOOL_NOT_REGISTERED",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
        
        # Create execution context
        context = PiscesLxCoreMCPExecutionContext(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters,
            preferred_types=preferred_types or [],
            allow_fallback=allow_fallback,
            timeout=timeout,
            metadata=metadata
        )
        
        # Try preferred types first
        if preferred_types:
            for tool_type in preferred_types:
                for variant in tool_variants:
                    if variant.tool_type == tool_type:
                        result = await self._execute_variant(variant, context)
                        if result.success or not allow_fallback:
                            return result
        
        # Try all available variants if fallback is enabled
        if allow_fallback:
            # Sort by priority (higher priority first)
            sorted_variants = sorted(tool_variants, key=lambda x: x.priority, reverse=True)
            
            for variant in sorted_variants:
                result = await self._execute_variant(variant, context)
                if result.success:
                    return result
        
        # All attempts failed
        return PiscesLxCoreMCPExecutionResult(
            success=False,
            result=None,
            execution_time=time.time() - start_time,
            status=PiscesLxCoreMCPExecutionStatus.FAILED,
            error_message=f"All execution attempts failed for tool {tool_name}",
            error_code="ALL_ATTEMPTS_FAILED",
            mode=PiscesLxCoreMCPExecutionMode.SYNC
        )
    
    async def _execute_variant(self, metadata: PiscesLxCoreMCPToolMetadata, context: PiscesLxCoreMCPExecutionContext) -> PiscesLxCoreMCPExecutionResult:
        """Execute a specific tool variant."""
        # Find appropriate executor
        for executor in self.executors:
            if metadata.tool_type in executor.get_supported_types():
                try:
                    result = await executor.execute(context.tool_name, context.parameters, context)
                    
                    # Update execution statistics
                    self._update_stats(metadata.name, result)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Executor {executor.__class__.__name__} failed for tool {metadata.name}: {e}")
                    continue
        
        return PiscesLxCoreMCPExecutionResult(
            success=False,
            result=None,
            execution_time=0.0,
            status=PiscesLxCoreMCPExecutionStatus.FAILED,
            error_message=f"No executor found for tool type {metadata.tool_type.value}",
            error_code="NO_EXECUTOR",
            mode=PiscesLxCoreMCPExecutionMode.SYNC
        )
    
    def _update_stats(self, tool_name: str, result: PiscesLxCoreMCPExecutionResult):
        """Update execution statistics."""
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }
        
        stats = self.execution_stats[tool_name]
        stats["total_executions"] += 1
        stats["total_time"] += result.execution_time
        
        if result.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
        
        stats["average_time"] = stats["total_time"] / stats["total_executions"]
    
    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for a tool."""
        return self.execution_stats.get(tool_name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools."""
        return self.execution_stats.copy()


# Global unified executor instance
_unified_executor: Optional[PiscesLxCoreMCPUnifiedToolExecutor] = None


class PiscesLxCoreMCPUnifiedToolExecutor:
    """Unified executor that intelligently routes tool execution."""
    
    def __init__(self):
        """Initialize unified tool executor."""
        self.executors: List[ToolExecutor] = [
            PiscesLxCoreMCPNativeToolExecutor(),
            PiscesLxCoreMCPInternalToolExecutor(),
            PiscesLxCoreMCPExternalToolExecutor()
        ]
        
        self.tool_registry: Dict[str, List[PiscesLxCoreMCPToolMetadata]] = {}  # tool_name -> [metadata1, metadata2, ...]
        self.execution_stats: Dict[str, Dict[str, Any]] = {}  # tool_name -> stats
        
        logger.info("UnifiedToolExecutor initialized")
    
    def register_tool(self, metadata: PiscesLxCoreMCPToolMetadata):
        """Register a tool with the unified executor."""
        if metadata.name not in self.tool_registry:
            self.tool_registry[metadata.name] = []
        
        self.tool_registry[metadata.name].append(metadata)
        
        # Register with appropriate executor
        for executor in self.executors:
            if metadata.tool_type in executor.get_supported_types():
                if isinstance(executor, PiscesLxCoreMCPNativeToolExecutor) and metadata.tool_type == PiscesLxCoreMCPToolType.NATIVE:
                    executor.register_native_tool(metadata)
                elif isinstance(executor, PiscesLxCoreMCPInternalToolExecutor) and metadata.tool_type == PiscesLxCoreMCPToolType.INTERNAL:
                    executor.register_internal_tool(metadata)
                elif isinstance(executor, PiscesLxCoreMCPExternalToolExecutor) and metadata.tool_type == PiscesLxCoreMCPToolType.EXTERNAL:
                    executor.register_external_tool(metadata)
        
        logger.debug(f"Registered tool {metadata.name} of type {metadata.tool_type.value}")
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        execution_id: str,
        preferred_types: Optional[List[PiscesLxCoreMCPToolType]] = None,
        allow_fallback: bool = True,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PiscesLxCoreMCPExecutionResult:
        """
        Execute a tool with intelligent routing and fallback.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            execution_id: Unique execution identifier
            preferred_types: Preferred tool types to try first
            allow_fallback: Whether to allow fallback to other types
            timeout: Execution timeout
            metadata: Additional metadata for execution
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        # Get available tool variants
        tool_variants = self.tool_registry.get(tool_name, [])
        if not tool_variants:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=f"Tool {tool_name} not found in registry",
                error_code="TOOL_NOT_REGISTERED",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
        
        # Create execution context
        context = PiscesLxCoreMCPExecutionContext(
            execution_id=execution_id,
            tool_name=tool_name,
            parameters=parameters,
            preferred_types=preferred_types or [],
            allow_fallback=allow_fallback,
            timeout=timeout,
            metadata=metadata
        )
        
        # Try preferred types first
        if preferred_types:
            for tool_type in preferred_types:
                for variant in tool_variants:
                    if variant.tool_type == tool_type:
                        result = await self._execute_variant(variant, context)
                        if result.success or not allow_fallback:
                            return result
        
        # Try all available variants if fallback is enabled
        if allow_fallback:
            # Sort by priority (higher priority first)
            sorted_variants = sorted(tool_variants, key=lambda x: x.priority, reverse=True)
            
            for variant in sorted_variants:
                result = await self._execute_variant(variant, context)
                if result.success:
                    return result
        
        # All attempts failed
        return PiscesLxCoreMCPExecutionResult(
            success=False,
            result=None,
            execution_time=time.time() - start_time,
            status=PiscesLxCoreMCPExecutionStatus.FAILED,
            error_message=f"All execution attempts failed for tool {tool_name}",
            error_code="ALL_ATTEMPTS_FAILED",
            mode=PiscesLxCoreMCPExecutionMode.SYNC
        )
    
    async def _execute_variant(self, metadata: PiscesLxCoreMCPToolMetadata, context: PiscesLxCoreMCPExecutionContext) -> PiscesLxCoreMCPExecutionResult:
        """Execute a specific tool variant."""
        # Find appropriate executor
        for executor in self.executors:
            if metadata.tool_type in executor.get_supported_types():
                try:
                    result = await executor.execute(context.tool_name, context.parameters, context)
                    
                    # Update execution statistics
                    self._update_stats(metadata.name, result)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Executor {executor.__class__.__name__} failed for tool {metadata.name}: {e}")
                    continue
        
        return PiscesLxCoreMCPExecutionResult(
            success=False,
            result=None,
            execution_time=0.0,
            status=PiscesLxCoreMCPExecutionStatus.FAILED,
            error_message=f"No executor found for tool type {metadata.tool_type.value}",
            error_code="NO_EXECUTOR",
            mode=PiscesLxCoreMCPExecutionMode.SYNC
        )
    
    def _update_stats(self, tool_name: str, result: PiscesLxCoreMCPExecutionResult):
        """Update execution statistics."""
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }
        
        stats = self.execution_stats[tool_name]
        stats["total_executions"] += 1
        stats["total_time"] += result.execution_time
        
        if result.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
        
        stats["average_time"] = stats["total_time"] / stats["total_executions"]
    
    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for a tool."""
        return self.execution_stats.get(tool_name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools."""
        return self.execution_stats.copy()
    
    @staticmethod
    def get_unified_tool_executor() -> "PiscesLxCoreMCPUnifiedToolExecutor":
        """Get the global unified tool executor instance."""
        global _unified_executor
        if _unified_executor is None:
            _unified_executor = PiscesLxCoreMCPUnifiedToolExecutor()
        return _unified_executor
    
    @staticmethod
    async def execute_tool_unified(
        tool_name: str,
        parameters: Dict[str, Any],
        execution_id: str,
        preferred_types: Optional[List[PiscesLxCoreMCPToolType]] = None,
        allow_fallback: bool = True,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PiscesLxCoreMCPExecutionResult:
        """
        Convenience function for unified tool execution.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            execution_id: Unique execution identifier
            preferred_types: Preferred tool types to try first
            allow_fallback: Whether to allow fallback to other types
            timeout: Execution timeout
            metadata: Additional metadata for execution
            
        Returns:
            Execution result
        """
        executor = PiscesLxCoreMCPUnifiedToolExecutor.get_unified_tool_executor()
        return await executor.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            execution_id=execution_id,
            preferred_types=preferred_types,
            allow_fallback=allow_fallback,
            timeout=timeout,
            metadata=metadata
        )