#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import os
import sys
import json
import time
import asyncio
import importlib
import threading
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Set
import logging

# Ensure the project root directory is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core functionality from model/mcp
from model.mcp.server import OptimizedMCPServer as CoreMCPServer
from model.mcp.translator import MCPTranslationLayer

@dataclass
class ToolMetadata:
    """
    Structure for tool metadata.
    
    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool.
        category (str): Category of the tool.
        version (str): Version of the tool.
        author (str): Author of the tool.
        last_updated (datetime): Last updated time of the tool.
        dependencies (List[str]): List of dependencies.
        performance_score (float): Performance score of the tool, default is 1.0.
        usage_count (int): Number of times the tool has been used, default is 0.
        error_rate (float): Error rate of the tool, default is 0.0.
    """
    name: str
    description: str
    category: str
    version: str
    author: str
    last_updated: datetime
    dependencies: List[str]
    performance_score: float = 1.0
    usage_count: int = 0
    error_rate: float = 0.0

@dataclass
class ToolStats:
    """
    Structure for tool statistics.
    
    Attributes:
        name (str): Name of the module.
        load_time (float): Module loading time.
        tool_count (int): Number of tools in the module.
        status (str): Loading status of the module.
        error (Optional[str]): Error information, default is None.
        memory_usage (Optional[int]): Memory usage, default is None.
        last_used (Optional[datetime]): Last used time, default is None.
    """
    name: str
    load_time: float
    tool_count: int
    status: str
    error: Optional[str] = None
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None

class PiscesL1MCPPlaza:
    """
    Pisces L1 MCP Plaza - An advanced integrated system.
    """
    
    def __init__(self):
        self.core_server = CoreMCPServer()
        self.translation_layer = MCPTranslationLayer()
        
        # Tool management
        self.tools: Dict[str, Any] = {}
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self.tool_stats: List[ToolStats] = []
        self.blacklisted_tools: Set[str] = {
            'calculator',  # Duplicates with PiscesL1's math module
            'translator',  # Duplicates with PiscesL1's multilingual module
            'weather',     # Duplicates with PiscesL1's real-time data module
        }
        
        # New: Tool-level session memory (decoupled from model memory)
        self.tool_session_memory = {}  # Memory used only by tools, does not affect model memory
        self.context_suggestions = {}  # Tool-based suggestions
        self.tool_workflows = {}  # Tool workflow orchestration
        
        # Performance optimization
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._last_discovery = 0
        self.discovery_interval = 5  # Seconds
        
        # Initialization
        self._setup_logging()
        self._discover_and_register_tools()
    
    def _setup_logging(self):
        """
        Set up the logging system.
        """
        self.logger = logging.getLogger("PiscesL1.MCPPlaza")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _is_tool_compatible(self, module_name: str, module) -> bool:
        """
        Check if a tool is compatible with the system.
        
        Args:
            module_name (str): Name of the module.
            module: Module object.
            
        Returns:
            bool: True if the tool is compatible, False otherwise.
        """
        if module_name in self.blacklisted_tools:
            self.logger.info(f"Skipping blacklisted tool: {module_name}")
            return False
        
        # Check if the necessary interfaces exist
        has_tool = hasattr(module, 'pisces_tool') or hasattr(module, 'mcp')
        if not has_tool:
            return False
            
        return True
    
    def _load_module_with_stats(self, py_file: Path) -> ToolStats:
        """
        Load a module with statistics.
        
        Args:
            py_file (Path): Path to the Python file.
            
        Returns:
            ToolStats: Statistics of the module loading.
        """
        start_time = time.time()
        module_name = py_file.stem
        
        if module_name in ["__init__", "__pycache__"]:
            return ToolStats(module_name, 0, 0, "skipped")
        
        try:
            # Dynamic import
            if module_name in sys.modules:
                importlib.reload(sys.modules[f"MCP.{module_name}"])
            
            module = importlib.import_module(f"MCP.{module_name}")
            
            # Compatibility check
            if not self._is_tool_compatible(module_name, module):
                return ToolStats(module_name, time.time() - start_time, 0, "incompatible")
            
            # Register tools
            tool_count = 0
            if hasattr(module, 'pisces_tool'):
                tool = module.pisces_tool
                self.tools[tool.name] = tool
                tool_count += 1
                
                # Record metadata
                self.tool_metadata[tool.name] = ToolMetadata(
                    name=tool.name,
                    description=tool.description,
                    category="PiscesL1 Extension",
                    version="1.0.0",
                    author="PiscesL1 Team",
                    last_updated=datetime.now(),
                    dependencies=[]
                )
            
            # Integrate FastMCP tools
            if hasattr(module, 'mcp') and hasattr(module.mcp, 'tools'):
                for tool_name, tool_func in module.mcp.tools.items():
                    self.core_server.mcp_server.tools[tool_name] = tool_func
                    tool_count += 1
            
            load_time = time.time() - start_time
            return ToolStats(module_name, load_time, tool_count, "success")
            
        except Exception as e:
            return ToolStats(module_name, time.time() - start_time, 0, "error", str(e))
    
    def _discover_and_register_tools(self):
        """
        Discover and register all compatible tools.
        """
        current_time = time.time()
        
        # Avoid frequent discovery
        if current_time - self._last_discovery < self.discovery_interval:
            return
        
        self._last_discovery = current_time
        
        mcp_dir = Path(__file__).parent
        py_files = [f for f in mcp_dir.glob("*.py") if f.name != "__init__.py"]
        
        # Concurrent loading
        self.tool_stats.clear()
        with ThreadPoolExecutor(max_workers=min(4, len(py_files))) as executor:
            futures = [executor.submit(self._load_module_with_stats, f) for f in py_files]
            
            for future in futures:
                stats = future.result()
                self.tool_stats.append(stats)
        
        # Log statistics
        successful = [s for s in self.tool_stats if s.status == "success"]
        total_tools = sum(s.tool_count for s in successful)
        
        self.logger.info(
            f"🚀 MCP Plaza loaded: {total_tools} tools, "
            f"{len(successful)}/{len(py_files)} modules loaded successfully"
        )
    
    def get_tool_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the tool-level session memory (used only by tools, does not affect model memory).
        
        Args:
            session_id (str): Session ID.
            
        Returns:
            Dict[str, Any]: Tool-level session memory.
        """
        return self.tool_session_memory.get(session_id, {})
    
    def update_tool_session_context(self, session_id: str, tool_name: str, context: Dict[str, Any]):
        """
        Update the tool-level session memory (decoupled from the model memory system).
        
        Args:
            session_id (str): Session ID.
            tool_name (str): Name of the tool.
            context (Dict[str, Any]): Context to be updated.
        """
        if session_id not in self.tool_session_memory:
            self.tool_session_memory[session_id] = {}
        self.tool_session_memory[session_id][tool_name] = context
    
    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get all available tools.
        
        Returns:
            Dict[str, Any]: Information about all available tools.
        """
        self._discover_and_register_tools()
        
        tools_info = {}
        
        # PiscesL1 native tools
        for name, tool in self.tools.items():
            tools_info[name] = {
                "type": "PiscesL1 Extension",
                "description": tool.description,
                "parameters": tool.parameters,
                "metadata": asdict(self.tool_metadata.get(name, ToolMetadata(
                    name=name, description="", category="", version="", 
                    author="", last_updated=datetime.now(), dependencies=[]
                )))
            }
        
        # FastMCP integrated tools
        for tool_name in self.core_server.mcp_server.tools.keys():
            if tool_name not in tools_info:
                tools_info[tool_name] = {
                    "type": "FastMCP Integrated",
                    "description": f"FastMCP tool: {tool_name}",
                    "parameters": {},
                    "metadata": asdict(ToolMetadata(
                        name=tool_name, description="", category="FastMCP", 
                        version="1.0.0", author="MCP", last_updated=datetime.now(), dependencies=[]
                    ))
                }
        
        return tools_info
    
    def get_relevant_resources(self, query: str, session_id: str) -> List[str]:
        """
        Provide relevant resource suggestions based on the query.
        
        Args:
            query (str): Query string.
            session_id (str): Session ID.
            
        Returns:
            List[str]: List of relevant resource suggestions.
        """
        suggestions = []
        
        # Code-related suggestions
        if any(word in query.lower() for word in ['python', 'code', '文件']):
            if 'filesystem' in self.tools:
                suggestions.append("filesystem: Helps you operate files and directories")
            if 'git' in self.tools:
                suggestions.append("git: Helps you manage code versions")
        
        # Network-related suggestions
        if any(word in query.lower() for word in ['搜索', '网页', 'url']):
            if 'fetch' in self.tools:
                suggestions.append("fetch: Can get web page content")
        
        # Encryption-related suggestions
        if any(word in query.lower() for word in ['加密', '安全', '密码']):
            if 'crypto' in self.tools:
                suggestions.append("crypto: Provides encryption and decryption functions")
        
        return suggestions

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        """
        Execute a specified tool.
        
        Args:
            tool_name (str): Name of the tool.
            parameters (Dict[str, Any]): Parameters for the tool.
            session_id (str, optional): Session ID, default is "default".
            
        Returns:
            Dict[str, Any]: Execution result or error information.
        """
        self._discover_and_register_tools()
        
        # Prioritize executing PiscesL1 native tools
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                result = tool.function(parameters)
                
                # Update statistics
                if tool_name in self.tool_metadata:
                    meta = self.tool_metadata[tool_name]
                    meta.usage_count += 1
                    meta.last_used = datetime.now()
                
                return result
            except Exception as e:
                self.logger.error(f"Tool execution failed {tool_name}: {e}")
                return {"error": str(e)}
        
        # Execute FastMCP integrated tools
        if tool_name in self.core_server.mcp_server.tools:
            try:
                tool_func = self.core_server.mcp_server.tools[tool_name]
                return tool_func(parameters)
            except Exception as e:
                self.logger.error(f"FastMCP tool execution failed {tool_name}: {e}")
                return {"error": str(e)}
    
    def execute_tool_workflow(self, tasks: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Execute a tool-level workflow (isolated from model memory).
        
        Args:
            tasks (List[Dict[str, Any]]): List of tasks.
            session_id (str): Session ID.
            
        Returns:
            List[Dict[str, Any]]: List of execution results.
        """
        results = []
        
        for task in tasks:
            tool_name = task.get("tool")
            parameters = task.get("parameters", {})
            
            # Only add tool-level context, does not affect model memory
            tool_context = self.get_tool_session_context(session_id)
            parameters["_tool_context"] = tool_context
            
            # Execute the tool
            result = self.execute_tool(tool_name, parameters, session_id)
            results.append(result)
            
            # Only update tool-level session memory, decoupled from the model memory system
            self.update_tool_session_context(session_id, tool_name, {
                "parameters": parameters,
                "result": result,
                "timestamp": str(datetime.now()),
                "type": "tool_operation"  # Clearly marked as a tool operation
            })
        
        return results
        
        return {"error": f"Tool '{tool_name}' not found"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the system status.
        
        Returns:
            Dict[str, Any]: System status information.
        """
        self._discover_and_register_tools()
        
        total_tools = len(self.tools) + len(self.core_server.mcp_server.tools)
        
        return {
            "status": "running",
            "total_tools": total_tools,
            "piscesl1_tools": len(self.tools),
            "fastmcp_tools": len(self.core_server.mcp_server.tools),
            "blacklisted_tools": list(self.blacklisted_tools),
            "tool_stats": [
                {
                    "name": stat.name,
                    "load_time": stat.load_time,
                    "tool_count": stat.tool_count,
                    "status": stat.status,
                    "error": stat.error
                }
                for stat in self.tool_stats
            ],
            "last_discovery": self._last_discovery,
            "compatibility": "PiscesL1 native integration"
        }
    
    def reload_tools(self):
        """
        Reload all tools.
        """
        with self._lock:
            self.tools.clear()
            self.tool_metadata.clear()
            self._discover_and_register_tools()
            self.logger.info("🔄 Tool reloading completed")
    
    def register_custom_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                           function: Callable, category: str = "Custom") -> bool:
        """
        Register a custom tool.
        
        Args:
            name (str): Name of the custom tool.
            description (str): Description of the custom tool.
            parameters (Dict[str, Any]): Parameters of the custom tool.
            function (Callable): Execution function of the custom tool.
            category (str, optional): Category of the custom tool, default is "Custom".
            
        Returns:
            bool: True if registration is successful, False otherwise.
        """
        try:
            from dataclasses import dataclass
            
            @dataclass
            class CustomTool:
                name: str
                description: str
                parameters: Dict[str, Any]
                function: Callable
            
            tool = CustomTool(name, description, parameters, function)
            self.tools[name] = tool
            
            self.tool_metadata[name] = ToolMetadata(
                name=name,
                description=description,
                category=category,
                version="1.0.0",
                author="Custom",
                last_updated=datetime.now(),
                dependencies=[]
            )
            
            self.logger.info(f"📝 Custom tool registered: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register custom tool: {e}")
            return False

# Global instance
plaza = PiscesL1MCPPlaza()

# Convenient interfaces
def get_available_tools() -> Dict[str, Any]:
    """
    Get all available tools.
    
    Returns:
        Dict[str, Any]: Information about all available tools.
    """
    return plaza.get_available_tools()

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Execute a tool.
    
    Args:
        tool_name (str): Name of the tool.
        arguments (Dict[str, Any]): Arguments for the tool.
        
    Returns:
        Any: Execution result of the tool.
    """
    return plaza.execute_tool(tool_name, arguments)

def get_system_status() -> Dict[str, Any]:
    """
    Get the system status.
    
    Returns:
        Dict[str, Any]: System status information.
    """
    return plaza.get_system_status()

def reload_tools():
    """
    Reload tools.
    """
    plaza.reload_tools()

def register_custom_tool(name: str, description: str, parameters: Dict[str, Any], 
                        function: Callable, category: str = "Custom") -> bool:
    """
    Register a custom tool.
    
    Args:
        name (str): Name of the custom tool.
        description (str): Description of the custom tool.
        parameters (Dict[str, Any]): Parameters of the custom tool.
        function (Callable): Execution function of the custom tool.
        category (str, optional): Category of the custom tool, default is "Custom".
        
    Returns:
        bool: True if registration is successful, False otherwise.
    """
    return plaza.register_custom_tool(name, description, parameters, function, category)

# Automatically discover tools (lazy loading)
plaza._discover_and_register_tools()

# Export interfaces
__all__ = [
    'get_available_tools',
    'execute_tool', 
    'get_system_status',
    'reload_tools',
    'register_custom_tool',
    'plaza'
]