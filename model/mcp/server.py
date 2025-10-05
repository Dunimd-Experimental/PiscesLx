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
import time
import asyncio
import importlib
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import RIGHT, DEBUG, ERROR

@dataclass
class ToolStats:
    """
    Data class for storing tool loading statistics.

    Attributes:
        name (str): Name of the tool module.
        load_time (float): Time taken to load the module.
        tool_count (int): Number of tools loaded from the module.
        status (str): Loading status, e.g., "success", "no_tools", etc.
        error (Optional[str]): Error message if loading failed, None otherwise.
    """
    name: str
    load_time: float
    tool_count: int
    status: str
    error: Optional[str] = None

class OptimizedMCPServer:
    """
    An optimized MCP server with advanced features, designed to efficiently manage and load MCP tools.
    """
    
    def __init__(self):
        """
        Initialize the OptimizedMCPServer instance.
        
        Creates a FastMCP server, initializes tool statistics and loaded modules sets,
        sets up a thread pool executor, enables hot-reload, and records the initial check time.
        """
        self.mcp_server = FastMCP("PiscesL1 Optimized MCP Server")
        self.tool_stats: List[ToolStats] = []
        self.loaded_modules: Set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._hot_reload_enabled = True
        self._last_check_time = 0
        
    def _load_module_async(self, py_file: Path) -> ToolStats:
        """
        Asynchronously load a single Python module and update the MCP server with its tools.

        Args:
            py_file (Path): Path to the Python file to load.

        Returns:
            ToolStats: Statistics about the module loading process.
        """
        start_time = time.time()
        module_name = py_file.stem
        
        # Skip __init__.py file as it's usually used for package initialization
        if module_name == "__init__":
            return ToolStats(module_name, 0, 0, "skipped")
            
        try:
            # Invalidate cache for hot-reload to ensure the latest module version is loaded
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(f"MCP.{module_name}")
                
            # Check if the module has an 'mcp' attribute to determine if it contains MCP tools
            if hasattr(module, 'mcp'):
                tool_mcp = getattr(module, 'mcp')
                # Check if the 'mcp' attribute has a 'tools' attribute to get available tools
                if hasattr(tool_mcp, 'tools'):
                    self.mcp_server.tools.update(tool_mcp.tools)
                    tool_count = len(tool_mcp.tools)
                    load_time = time.time() - start_time
                    return ToolStats(module_name, load_time, tool_count, "success")
                else:
                    return ToolStats(module_name, time.time() - start_time, 0, "no_tools")
            else:
                return ToolStats(module_name, time.time() - start_time, 0, "no_mcp")
                
        except Exception as e:
            return ToolStats(module_name, time.time() - start_time, 0, "error", str(e))
    
    def auto_discover_tools(self, force_reload: bool = False):
        """
        Optimized tool discovery with concurrent module loading.
        
        Scans the MCP directory for Python files, loads them concurrently,
        and updates the server with the discovered tools.

        Args:
            force_reload (bool, optional): If True, force reload all modules. Defaults to False.
        """
        # Get the path to the MCP directory
        mcp_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "MCP"
        
        # Check if the MCP directory exists. If not, log an error and exit the function
        if not mcp_dir.exists():
            ERROR(f"MCP directory not found: {mcp_dir}")
            return
        
        # Skip discovery if no changes and not forced to reload to improve performance
        if not force_reload and time.time() - self._last_check_time < 1:
            return
            
        self._last_check_time = time.time()
        
        # Get all Python files except __init__.py in the MCP directory
        py_files = [f for f in mcp_dir.glob("*.py") if f.name != "__init__.py"]
        
        # Clear previous tool statistics to ensure fresh data
        self.tool_stats.clear()
        # Load modules concurrently using a thread pool to speed up the process
        with ThreadPoolExecutor(max_workers=min(4, len(py_files))) as executor:
            future_to_file = {executor.submit(self._load_module_async, f): f for f in py_files}
            
            for future in as_completed(future_to_file):
                stats = future.result()
                self.tool_stats.append(stats)
        
        # Filter successful module loading statistics
        successful = [s for s in self.tool_stats if s.status == "success"]
        total_tools = sum(s.tool_count for s in successful)
        
        RIGHT(f"Optimized auto-discovery: {total_tools} tools from {len(successful)}/{len(py_files)} modules")
        
        # Log detailed loading statistics for debugging and monitoring
        for stats in sorted(self.tool_stats, key=lambda x: x.load_time, reverse=True):
            if stats.status == "success":
                DEBUG(f"{stats.name}: {stats.tool_count} tools ({stats.load_time:.3f}s)")
            elif stats.error:
                ERROR(f"{stats.name}: {stats.error}")

# Global optimized server instance
optimized_server = OptimizedMCPServer()

# Background tool discovery with timeout
import threading
import time

def _async_tool_discovery():
    """Run tool discovery in background thread"""
    try:
        optimized_server.auto_discover_tools()
    except Exception as e:
        ERROR(f"Background tool discovery failed: {e}")

# Start background discovery
discovery_thread = threading.Thread(target=_async_tool_discovery, daemon=True)
discovery_thread.start()

def wait_for_tools(timeout=5):
    """
    Wait for tool discovery to complete.

    Args:
        timeout (int, optional): Maximum waiting time in seconds. Defaults to 5.

    Returns:
        bool: True if the discovery thread is still alive after waiting, False otherwise.
    """
    discovery_thread.join(timeout=timeout)
    return discovery_thread.is_alive()

# Backward compatibility function
def auto_discover_tools():
    """
    Legacy interface for backward compatibility. Calls the auto_discover_tools method of the global optimized server.
    """
    optimized_server.auto_discover_tools()

mcp_server = optimized_server.mcp_server

# Additional server configuration
@mcp_server.resource("status://server")
def server_status() -> str:
    """
    Get the current status information of the MCP server.

    Returns:
        str: A JSON-formatted string containing server status details.
    """
    return f"""{{
    "name": "PiscesL1 MCP Server",
    "version": "1.0.0",
    "protocol": "Official MCP SDK",
    "tools_count": {len(mcp_server.tools)},
    "xml_compatibility": true,
    "auto_discovery": true,
    "status": "running"
}}"""

@mcp_server.prompt()
def system_prompt() -> str:
    """
    Generate a system prompt for PiscesL1 model interaction.
    
    Includes instructions on how to use tools and lists available tools.

    Returns:
        str: System prompt string.
    """
    tool_list = ", ".join(mcp_server.tools.keys()) if mcp_server.tools else "No tools available"
    
    return f"""You are PiscesL1, a powerful multimodal AI model. You can use tools by outputting XML tags in this format:

            <agent><an>tool_name</an><ap1>parameter1</ap1><ap2>parameter2</ap2></agent>

            Available tools: {tool_list}

            Always use XML tags for tool calls. The translation layer will handle the conversion to MCP protocol."""

def get_server():
    """
    Get the MCP server instance.

    Returns:
        FastMCP: The MCP server instance.
    """
    return mcp_server
