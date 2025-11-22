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

import os
import json
import threading
import importlib
import importlib.util
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

from utils.mcp.types import (
    PiscesLxCoreMCPToolMetadata,
    PiscesLxCoreMCPModuleStats,
    PiscesLxCoreMCPModuleStatus,
    PiscesLxCoreMCPExecutionContext
)

from utils.log.core import PiscesLxCoreLog

class PiscesLxCoreMCPTools:
    """Tool management system for PiscesLxCoreMCP.
    
    Manages tool discovery, loading, validation, and lifecycle operations
    for both native Python tools and external tool executables.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the tools manager.
        
        Args:
            config: Optional configuration for tool management
        """
        self.config = config or {}
        self.logger = self._configure_logging()
        
        # Tool storage
        self.tools: Dict[str, PiscesLxCoreMCPToolMetadata] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.tool_modules: Dict[str, Any] = {}
        
        # Tool categories and discovery
        self.tool_categories: Dict[str, List[str]] = {}
        self.discovery_paths: List[str] = []
        
        # Tool statistics
        self.tool_stats: Dict[str, PiscesLxCoreMCPModuleStats] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration
        self.auto_reload = getattr(self.config, 'auto_reload', False)
        self.strict_validation = getattr(self.config, 'strict_validation', True)
        self.max_tool_cache_size = getattr(self.config, 'max_tool_cache_size', 100)
        
        # Initialize discovery paths
        self._initialize_discovery_paths()
        
        self.logger.info("PiscesLxCoreMCPTools initialized")
    
    def _configure_logging(self) -> PiscesLxCoreLog:
        """Configure structured logging for tools manager.
        
        Returns:
            Configured logger instance
        """
        logger = PiscesLxCoreLog("PiscesLx.Core.MCP.Tools")
        return logger
    
    def _initialize_discovery_paths(self):
        """Initialize tool discovery paths from configuration."""
        # Add default discovery paths
        default_paths = [
            "MCP",  # MCP tools directory
            "tools",  # Local tools directory
            os.path.join(os.path.dirname(__file__), "tools"),  # Package tools
            os.path.expanduser("~/.piscesl1/tools"),  # User tools
        ]
        
        # Add configured paths
        config_paths = getattr(self.config, 'tool_paths', [])
        
        self.discovery_paths = config_paths + default_paths
        
        # Ensure paths exist
        for path in self.discovery_paths:
            if os.path.exists(path):
                self.logger.debug(f"Tool discovery path: {path}")
            else:
                self.logger.debug(f"Tool discovery path (not found): {path}")
    
    def discover_tools(self, path: Optional[str] = None) -> List[str]:
        """Discover available tools in the specified path.
        
        Args:
            path: Optional specific path to search, or use discovery paths if None
            
        Returns:
            List of discovered tool names
        """
        discovered_tools = []
        search_paths = [path] if path else self.discovery_paths
        
        for search_path in search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
            
            try:
                # Discover Python tools
                python_tools = self._discover_python_tools(search_path)
                discovered_tools.extend(python_tools)
                
                # Discover executable tools
                executable_tools = self._discover_executable_tools(search_path)
                discovered_tools.extend(executable_tools)
                
                # Discover JSON tool definitions
                json_tools = self._discover_json_tools(search_path)
                discovered_tools.extend(json_tools)
                
            except Exception as e:
                self.logger.error(f"Error discovering tools in {search_path}: {e}")
        
        self.logger.info(f"Discovered {len(discovered_tools)} tools: {discovered_tools}")
        return discovered_tools
    
    def _discover_python_tools(self, path: str) -> List[str]:
        """Discover Python-based tools.
        
        Args:
            path: Directory path to search
            
        Returns:
            List of discovered Python tool names
        """
        python_tools = []
        path_obj = Path(path)
        
        # Look for Python files
        for py_file in path_obj.glob("*.py"):
            try:
                tool_name = py_file.stem
                if self._validate_python_tool(py_file):
                    python_tools.append(tool_name)
                    self.logger.debug(f"Discovered Python tool: {tool_name}")
            except Exception as e:
                self.logger.warning(f"Failed to validate Python tool {py_file}: {e}")
        
        # Look for tool packages
        for pkg_dir in path_obj.iterdir():
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                try:
                    tool_name = pkg_dir.name
                    if self._validate_python_tool(pkg_dir):
                        python_tools.append(tool_name)
                        self.logger.debug(f"Discovered Python tool package: {tool_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to validate Python tool package {pkg_dir}: {e}")
        
        return python_tools
    
    def _discover_executable_tools(self, path: str) -> List[str]:
        """Discover executable tools.
        
        Args:
            path: Directory path to search
            
        Returns:
            List of discovered executable tool names
        """
        executable_tools = []
        path_obj = Path(path)
        
        # Look for executable files (skip Python files as they are handled by Python tool discovery)
        for exec_file in path_obj.iterdir():
            if exec_file.is_file() and os.access(exec_file, os.X_OK) and exec_file.suffix != '.py':
                try:
                    tool_name = exec_file.stem
                    if self._validate_executable_tool(exec_file):
                        executable_tools.append(tool_name)
                        self.logger.debug(f"Discovered executable tool: {tool_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to validate executable tool {exec_file}: {e}")
        
        return executable_tools
    
    def _discover_json_tools(self, path: str) -> List[str]:
        """Discover JSON-defined tools.
        
        Args:
            path: Directory path to search
            
        Returns:
            List of discovered JSON tool names
        """
        json_tools = []
        path_obj = Path(path)
        
        # Look for JSON files
        for json_file in path_obj.glob("*.json"):
            try:
                tool_name = json_file.stem
                if self._validate_json_tool(json_file):
                    json_tools.append(tool_name)
                    self.logger.debug(f"Discovered JSON tool: {tool_name}")
            except Exception as e:
                self.logger.warning(f"Failed to validate JSON tool {json_file}: {e}")
        
        return json_tools
    
    def _validate_python_tool(self, tool_path: Path) -> bool:
        """Validate a Python tool.
        
        Args:
            tool_path: Path to the Python tool
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # For validation, we just check if the file exists and is readable
            # We don't try to import it here to avoid dependency issues during discovery
            if tool_path.is_file():
                # Check if it's a valid Python file by trying to read it
                with open(tool_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Basic Python file validation
                    if len(content) > 0 and ('import ' in content or 'def ' in content or 'class ' in content):
                        return True
                    return False
            else:  # Package
                # Check if package has __init__.py
                init_file = tool_path / "__init__.py"
                return init_file.exists() and init_file.is_file()
                
        except Exception as e:
            self.logger.warning(f"Python tool validation failed for {tool_path}: {e}")
            return False
    
    def _validate_executable_tool(self, tool_path: Path) -> bool:
        """Validate an executable tool.
        
        Args:
            tool_path: Path to the executable tool
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if file is executable
            if not os.access(tool_path, os.X_OK):
                return False
            
            # Try to get tool metadata (if supported)
            try:
                result = subprocess.run(
                    [str(tool_path), "--metadata"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Tool supports metadata query
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Basic validation passed
            return True
            
        except Exception as e:
            self.logger.error(f"Executable tool validation failed for {tool_path}: {e}")
            return False
    
    def _validate_json_tool(self, tool_path: Path) -> bool:
        """Validate a JSON tool definition.
        
        Args:
            tool_path: Path to the JSON tool definition
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Skip MCP.json as it's a configuration file, not a tool definition
            if tool_path.name == 'MCP.json':
                return False
                
            with open(tool_path, 'r', encoding='utf-8') as f:
                tool_def = json.load(f)
            
            # Check required fields
            required_fields = ['name', 'description', 'command']
            for field in required_fields:
                if field not in tool_def:
                    return False
            
            return True
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"JSON tool validation failed for {tool_path}: {e}")
            return False
    
    def load_tool(self, tool_name: str, tool_path: Optional[str] = None) -> bool:
        """Load a tool into the manager.
        
        Args:
            tool_name: Name of the tool to load
            tool_path: Optional specific path to the tool
            
        Returns:
            True if loaded successfully, False otherwise
        """
        with self._lock:
            try:
                if tool_name in self.tools:
                    self.logger.debug(f"Tool already loaded: {tool_name}")
                    return True
                
                # Try to find and load the tool
                if tool_path:
                    loaded = self._load_tool_from_path(tool_name, tool_path)
                else:
                    loaded = self._load_tool_from_discovery(tool_name)
                
                if loaded:
                    self.logger.info(f"Tool loaded successfully: {tool_name}")
                    return True
                else:
                    self.logger.error(f"Failed to load tool: {tool_name}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error loading tool {tool_name}: {e}")
                return False
    
    def _load_tool_from_path(self, tool_name: str, tool_path: str) -> bool:
        """Load a tool from a specific path.
        
        Args:
            tool_name: Name of the tool
            tool_path: Path to the tool
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path_obj = Path(tool_path)
        
        if not path_obj.exists():
            return False
        
        # Try different loading strategies
        if path_obj.suffix == '.py':
            return self._load_python_tool(tool_name, path_obj)
        elif path_obj.suffix == '.json':
            return self._load_json_tool(tool_name, path_obj)
        elif path_obj.is_file() and os.access(path_obj, os.X_OK):
            return self._load_executable_tool(tool_name, path_obj)
        
        return False
    
    def _load_tool_from_discovery(self, tool_name: str) -> bool:
        """Load a tool using discovery paths.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if loaded successfully, False otherwise
        """
        for search_path in self.discovery_paths:
            if not os.path.exists(search_path):
                continue
            
            path_obj = Path(search_path)
            
            # Try Python tool
            py_path = path_obj / f"{tool_name}.py"
            if py_path.exists():
                if self._load_python_tool(tool_name, py_path):
                    return True
            
            # Try tool package
            pkg_path = path_obj / tool_name
            if pkg_path.exists() and (pkg_path / "__init__.py").exists():
                if self._load_python_tool(tool_name, pkg_path):
                    return True
            
            # Try JSON tool
            json_path = path_obj / f"{tool_name}.json"
            if json_path.exists():
                if self._load_json_tool(tool_name, json_path):
                    return True
            
            # Try executable tool
            for exec_file in path_obj.iterdir():
                if exec_file.stem == tool_name and os.access(exec_file, os.X_OK):
                    if self._load_executable_tool(tool_name, exec_file):
                        return True
        
        return False
    
    def _load_python_tool(self, tool_name: str, tool_path: Path) -> bool:
        """Load a Python-based tool.
        
        Args:
            tool_name: Name of the tool
            tool_path: Path to the Python tool
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Import the module
            if tool_path.is_file():
                spec = importlib.util.spec_from_file_location(tool_name, tool_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:  # Package
                module = importlib.import_module(tool_path.name)
            
            # Get tool metadata
            if hasattr(module, 'get_metadata'):
                metadata = module.get_metadata()
            else:
                # Create basic metadata for MCP tools
                if tool_name in ['crypto', 'time', 'web_search', 'fetch', 'document_processor', 'sequentialthinking', 'template']:
                    # MCP tools have special structure with mcp instance
                    metadata = PiscesLxCoreMCPToolMetadata(
                        name=tool_name,
                        description=f"MCP tool: {tool_name}",
                        version="1.0.0",
                        author="PiscesL1",
                        category="mcp",
                        tags=["mcp", "python"],
                        parameters={},
                        return_type="any",
                        examples=[],
                        requirements=[],
                        is_native=True,
                        is_enabled=True,
                        load_time=datetime.now()
                    )
                else:
                    # Create basic metadata for regular tools
                    metadata = PiscesLxCoreMCPToolMetadata(
                        name=tool_name,
                        description=f"Python tool: {tool_name}",
                        version="1.0.0",
                        author="Unknown",
                        category="general",
                        tags=["python"],
                        parameters={},
                        return_type="any",
                        examples=[],
                        requirements=[],
                        is_native=True,
                        is_enabled=True,
                        load_time=datetime.now()
                    )
            
            # Store tool information
            self.tools[tool_name] = metadata
            
            # Handle MCP tools specially
            if tool_name in ['crypto', 'time', 'web_search', 'fetch', 'document_processor', 'sequentialthinking', 'template']:
                # For MCP tools, we need to find the actual function
                if hasattr(module, 'mcp'):
                    mcp_instance = getattr(module, 'mcp')
                    # Look for functions decorated with @mcp.tool()
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if callable(attr) and not attr_name.startswith('_'):
                            # Store the first non-private function as the main function
                            self.tool_functions[tool_name] = attr
                            break
                    else:
                        # If no suitable function found, use a wrapper
                        def mcp_wrapper(args):
                            return f"MCP tool {tool_name} loaded successfully"
                        self.tool_functions[tool_name] = mcp_wrapper
                else:
                    # Fallback for regular execute function
                    if hasattr(module, 'execute'):
                        self.tool_functions[tool_name] = getattr(module, 'execute')
                    else:
                        def fallback_wrapper(args):
                            return f"Tool {tool_name} loaded successfully"
                        self.tool_functions[tool_name] = fallback_wrapper
            else:
                # Regular tools use execute function
                if hasattr(module, 'execute'):
                    self.tool_functions[tool_name] = getattr(module, 'execute')
                else:
                    def fallback_wrapper(args):
                        return f"Tool {tool_name} loaded successfully"
                    self.tool_functions[tool_name] = fallback_wrapper
            
            self.tool_modules[tool_name] = module
            
            # Initialize statistics
            self.tool_stats[tool_name] = PiscesLxCoreMCPModuleStats(
                module_name=tool_name,
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                average_execution_time=0.0,
                last_used=None
            )
            
            # Add to category
            category = metadata.category
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Python tool {tool_name}: {e}")
            return False
    
    def _load_executable_tool(self, tool_name: str, tool_path: Path) -> bool:
        """Load an executable tool.
        
        Args:
            tool_name: Name of the tool
            tool_path: Path to the executable tool
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Try to get metadata
            metadata = None
            try:
                result = subprocess.run(
                    [str(tool_path), "--metadata"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    metadata_data = json.loads(result.stdout)
                    metadata = PiscesLxCoreMCPToolMetadata(**metadata_data)
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
            
            # Create basic metadata if not available
            if metadata is None:
                metadata = PiscesLxCoreMCPToolMetadata(
                    name=tool_name,
                    description=f"Executable tool: {tool_name}",
                    version="1.0.0",
                    author="Unknown",
                    category="external",
                    tags=["executable", "external"],
                    parameters={},
                    return_type="any",
                    examples=[],
                    requirements=[],
                    is_native=False,
                    is_enabled=True,
                    load_time=datetime.now()
                )
            
            # Create execution wrapper
            def execute_wrapper(arguments: Dict[str, Any]) -> Any:
                return self._execute_external_tool(tool_path, arguments)
            
            # Store tool information
            self.tools[tool_name] = metadata
            self.tool_functions[tool_name] = execute_wrapper
            
            # Initialize statistics
            self.tool_stats[tool_name] = PiscesLxCoreMCPModuleStats(
                module_name=tool_name,
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                average_execution_time=0.0,
                last_used=None
            )
            
            # Add to category
            category = metadata.category
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load executable tool {tool_name}: {e}")
            return False
    
    def _load_json_tool(self, tool_name: str, tool_path: Path) -> bool:
        """Load a JSON-defined tool.
        
        Args:
            tool_name: Name of the tool
            tool_path: Path to the JSON tool definition
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(tool_path, 'r', encoding='utf-8') as f:
                tool_def = json.load(f)
            
            # Create metadata from JSON definition
            metadata = PiscesLxCoreMCPToolMetadata(
                name=tool_def['name'],
                description=tool_def['description'],
                version=tool_def.get('version', '1.0.0'),
                author=tool_def.get('author', 'Unknown'),
                category=tool_def.get('category', 'external'),
                tags=tool_def.get('tags', ['json']),
                parameters=tool_def.get('parameters', {}),
                return_type=tool_def.get('return_type', 'any'),
                examples=tool_def.get('examples', []),
                requirements=tool_def.get('requirements', []),
                is_native=False,
                is_enabled=tool_def.get('enabled', True),
                load_time=datetime.now()
            )
            
            # Create execution wrapper
            command = tool_def['command']
            
            def execute_wrapper(arguments: Dict[str, Any]) -> Any:
                return self._execute_json_tool(command, arguments)
            
            # Store tool information
            self.tools[tool_name] = metadata
            self.tool_functions[tool_name] = execute_wrapper
            
            # Initialize statistics
            self.tool_stats[tool_name] = PiscesLxCoreMCPModuleStats(
                module_name=tool_name,
                total_calls=0,
                successful_calls=0,
                failed_calls=0,
                average_execution_time=0.0,
                last_used=None
            )
            
            # Add to category
            category = metadata.category
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON tool {tool_name}: {e}")
            return False
    
    def _execute_external_tool(self, tool_path: Path, arguments: Dict[str, Any]) -> Any:
        """Execute an external tool.
        
        Args:
            tool_path: Path to the external tool
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        try:
            # Convert arguments to JSON string
            args_json = json.dumps(arguments)
            
            # Execute the tool
            result = subprocess.run(
                [str(tool_path), args_json],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Tool execution failed: {result.stderr}")
            
            # Parse result
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Tool execution timed out")
        except Exception as e:
            raise RuntimeError(f"Tool execution error: {e}")
    
    def _execute_json_tool(self, command: str, arguments: Dict[str, Any]) -> Any:
        """Execute a JSON-defined tool.
        
        Args:
            command: Command template with placeholders
            arguments: Arguments to substitute in command
            
        Returns:
            Tool execution result
        """
        try:
            # Substitute arguments in command
            formatted_command = command.format(**arguments)
            
            # Execute the command
            result = subprocess.run(
                formatted_command.split(),
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Command execution failed: {result.stderr}")
            
            # Parse result
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Command execution timed out")
        except Exception as e:
            raise RuntimeError(f"Command execution error: {e}")
    
    def get_tool(self, tool_name: str) -> Optional[PiscesLxCoreMCPToolMetadata]:
        """Get tool metadata.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool metadata or None if not found
        """
        with self._lock:
            return self.tools.get(tool_name)
    
    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Get tool execution function.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool function or None if not found
        """
        with self._lock:
            return self.tool_functions.get(tool_name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        with self._lock:
            if category:
                return self.tool_categories.get(category, [])
            return list(self.tools.keys())
    
    def list_categories(self) -> List[str]:
        """List available tool categories.
        
        Returns:
            List of category names
        """
        with self._lock:
            return list(self.tool_categories.keys())
    
    def get_tool_stats(self, tool_name: Optional[str] = None) -> Dict[str, PiscesLxCoreMCPModuleStats]:
        """Get tool usage statistics.
        
        Args:
            tool_name: Optional specific tool name
            
        Returns:
            Dictionary of tool statistics
        """
        with self._lock:
            if tool_name:
                return {tool_name: self.tool_stats.get(tool_name)} if tool_name in self.tool_stats else {}
            return self.tool_stats.copy()
    
    def unload_tool(self, tool_name: str) -> bool:
        """Unload a tool from the manager.
        
        Args:
            tool_name: Name of the tool to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        with self._lock:
            if tool_name not in self.tools:
                return False
            
            try:
                # Remove from all storage
                metadata = self.tools.pop(tool_name)
                self.tool_functions.pop(tool_name, None)
                self.tool_modules.pop(tool_name, None)
                self.tool_stats.pop(tool_name, None)
                
                # Remove from category
                category = metadata.category
                if category in self.tool_categories:
                    if tool_name in self.tool_categories[category]:
                        self.tool_categories[category].remove(tool_name)
                    if not self.tool_categories[category]:
                        del self.tool_categories[category]
                
                self.logger.info(f"Tool unloaded: {tool_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error unloading tool {tool_name}: {e}")
                return False
    
    def reload_tool(self, tool_name: str) -> bool:
        """Reload a tool.
        
        Args:
            tool_name: Name of the tool to reload
            
        Returns:
            True if reloaded successfully, False otherwise
        """
        with self._lock:
            # First unload the tool
            self.unload_tool(tool_name)
            
            # Then load it again
            return self.load_tool(tool_name)
    
    def validate_tool(self, tool_name: str) -> bool:
        """Validate a loaded tool.
        
        Args:
            tool_name: Name of the tool to validate
            
        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            if tool_name not in self.tools:
                return False
            
            try:
                tool_function = self.tool_functions.get(tool_name)
                if not tool_function:
                    return False
                
                # Test execution with empty arguments
                # This is a basic validation - tools should handle empty args gracefully
                try:
                    result = tool_function({})
                    return True  # Tool executed without error
                except Exception as e:
                    self.logger.warning(f"Tool validation failed for {tool_name}: {e}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error validating tool {tool_name}: {e}")
                return False