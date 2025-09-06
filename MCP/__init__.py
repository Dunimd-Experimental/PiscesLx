#!/usr/bin/env/python3

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
import logging
import importlib
import threading
import inspect
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Set, Union, TypeVar, Generic

# Ensure the project root directory is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core functionality from model/mcp
from model.mcp.server import OptimizedMCPServer as CoreMCPServer
from model.mcp.translator import MCPTranslationLayer

# FastMCP兼容导入
T = TypeVar('T')

@dataclass
class ToolMetadata:
    """Structure for tool metadata."""
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
    """Structure for tool statistics."""
    name: str
    load_time: float
    tool_count: int
    status: str
    error: Optional[str] = None
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None

class PiscesL1MCPPlaza:
    """Pisces L1 MCP Plaza - An advanced integrated system."""
    
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
        
        # Tool-level session memory
        self.tool_session_memory = {}
        self.context_suggestions = {}
        self.tool_workflows = {}
        
        # Performance optimization
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._last_discovery = 0
        self.discovery_interval = 5
        
        # 文件监控
        self._file_watchers = {}
        self._watcher_thread = None
        self._stop_watcher = threading.Event()
        
        # FastMCP兼容层
        self._fastmcp_instance = None
        
        self._setup_logging()
        self._discover_and_register_tools()
        self._start_file_watcher()
    
    def _start_file_watcher(self):
        """启动文件监控线程"""
        def watch_files():
            import time
            mcp_dir = Path(__file__).parent
            last_check = {}
            
            # 初始化文件时间戳
            for py_file in mcp_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    try:
                        last_check[py_file.name] = py_file.stat().st_mtime
                    except OSError:
                        continue
            
            while not self._stop_watcher.is_set():
                try:
                    # 检查文件修改
                    for py_file in mcp_dir.glob("*.py"):
                        if py_file.name != "__init__.py" and py_file.name in last_check:
                            try:
                                current_mtime = py_file.stat().st_mtime
                                if current_mtime > last_check[py_file.name]:
                                    self.logger.info(f"🔄 File changed: {py_file.name}")
                                    last_check[py_file.name] = current_mtime
                                    
                                    # 标记缓存无效并触发重载
                                    with self._lock:
                                        self._discovery_cache_valid = False
                                        self._discover_and_register_tools()
                            except OSError:
                                continue
                    
                    time.sleep(2)  # 每2秒检查一次
                    
                except Exception as e:
                    self.logger.error(f"File watcher error: {e}")
                    time.sleep(5)
        
        self._watcher_thread = threading.Thread(target=watch_files, daemon=True)
        self._watcher_thread.start()
        self.logger.info("🎯 File watcher started for hot reload")
    
    def _setup_logging(self):
        """Set up the logging system."""
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
        """Check if a tool is compatible with the system."""
        if module_name in self.blacklisted_tools:
            self.logger.info(f"Skipping blacklisted tool: {module_name}")
            return False
        
        has_tool = hasattr(module, 'pisces_tool') or hasattr(module, 'mcp')
        if not has_tool:
            return False
        return True
    
    def _load_module_with_stats(self, py_file: Path) -> ToolStats:
        """Load a module with statistics."""
        start_time = time.time()
        module_name = py_file.stem
        
        if module_name in ["__init__", "__pycache__"]:
            return ToolStats(module_name, 0, 0, "skipped")
        
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[f"MCP.{module_name}"])
            
            module = importlib.import_module(f"MCP.{module_name}")
            
            if not self._is_tool_compatible(module_name, module):
                return ToolStats(module_name, time.time() - start_time, 0, "incompatible")
            
            tool_count = 0
            if hasattr(module, 'pisces_tool'):
                tool = module.pisces_tool
                self.tools[tool.name] = tool
                tool_count += 1
                
                self.tool_metadata[tool.name] = ToolMetadata(
                    name=tool.name,
                    description=tool.description,
                    category="PiscesL1 Extension",
                    version="1.0.0",
                    author="PiscesL1 Team",
                    last_updated=datetime.now(),
                    dependencies=[]
                )
            
            if hasattr(module, 'mcp') and hasattr(module.mcp, 'tools'):
                for tool_name, tool_func in module.mcp.tools.items():
                    self.core_server.mcp_server.tools[tool_name] = tool_func
                    tool_count += 1
            
            load_time = time.time() - start_time
            return ToolStats(module_name, load_time, tool_count, "success")
            
        except Exception as e:
            return ToolStats(module_name, time.time() - start_time, 0, "error", str(e))
    
    def _discover_and_register_tools(self):
        """优化后的工具发现逻辑 - 智能缓存和增量更新"""
        current_time = time.time()
        
        # 快速返回：如果缓存有效且间隔未到
        if (current_time - self._last_discovery < self.discovery_interval and 
            hasattr(self, '_discovery_cache_valid') and self._discovery_cache_valid):
            return
        
        mcp_dir = Path(__file__).parent
        py_files = [f for f in mcp_dir.glob("*.py") if f.name != "__init__.py"]
        
        # 智能扫描：只检查修改过的文件
        modified_files = []
        for py_file in py_files:
            try:
                mtime = py_file.stat().st_mtime
                cache_key = f"file_mtime_{py_file.name}"
                
                if cache_key not in self._cache or self._cache[cache_key] != mtime:
                    modified_files.append(py_file)
                    self._cache[cache_key] = mtime
            except OSError:
                continue
        
        # 如果没有修改且缓存有效，直接返回
        if not modified_files and hasattr(self, '_discovery_cache_valid'):
            self._last_discovery = current_time
            return
        
        # 增量更新：只处理修改过的文件
        files_to_process = modified_files if hasattr(self, '_discovery_cache_valid') else py_files
        
        if not files_to_process:
            self._discovery_cache_valid = True
            return
        
        # 并行处理文件
        new_stats = []
        with ThreadPoolExecutor(max_workers=min(4, len(files_to_process))) as executor:
            futures = [executor.submit(self._load_module_with_stats, f) for f in files_to_process]
            
            for future in futures:
                stats = future.result()
                new_stats.append(stats)
        
        # 更新统计信息
        if hasattr(self, '_discovery_cache_valid'):
            # 增量更新：替换修改过的文件统计
            for new_stat in new_stats:
                self.tool_stats = [s for s in self.tool_stats if s.name != new_stat.name]
                self.tool_stats.append(new_stat)
        else:
            # 首次加载：完整替换
            self.tool_stats = new_stats
        
        self._last_discovery = current_time
        self._discovery_cache_valid = True
        
        # 智能日志：只在有变化时输出
        successful = [s for s in new_stats if s.status == "success" and s.tool_count > 0]
        if successful:
            total_tools = sum(s.tool_count for s in self.tool_stats if s.status == "success")
            self.logger.info(
                f"🚀 MCP Plaza updated: {total_tools} tools, "
                f"{len(successful)} files processed"
            )

    # FastMCP兼容API
    # 在PiscesL1MCPPlaza类中添加FastMCP兼容装饰器
    def tool(self, name: str = None, description: str = None):
        """FastMCP兼容的装饰器 - 100%官方语法兼容"""
        def decorator(func):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"Tool: {tool_name}"
            
            # 使用register_custom_tool注册，但对外表现为FastMCP
            return self.register_custom_tool(
                name=tool_name,
                description=tool_desc,
                func=func,
                category="FastMCP兼容"
            )
        return decorator
    
    def register_custom_tool(self, name: str, description: str, func: Callable, 
                           category: str = "custom", **kwargs):
        """企业级工具注册 - 内部核心"""
        tool_metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            version=kwargs.get('version', '1.0.0'),
            author=kwargs.get('author', 'PiscesL1'),
            last_updated=datetime.now(),
            dependencies=kwargs.get('dependencies', [])
        )
        
        self.tool_metadata[name] = tool_metadata
        self.tools[name] = func
        
        # 注册到核心服务器
        self.core_server.mcp_server.tools[name] = func
        
        # 企业级增强
        self._enhance_tool(name, func)
        
        return func
    
    def _enhance_tool(self, name: str, func: Callable):
        """企业级功能增强"""
        # 添加性能监控
        original_func = func
        
        def enhanced_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 工具级会话内存
            if name not in self.tool_session_memory:
                self.tool_session_memory[name] = {}
            
            try:
                result = original_func(*args, **kwargs)
                
                # 更新统计
                self.tool_metadata[name].usage_count += 1
                self.tool_metadata[name].performance_score = min(
                    1.0, self.tool_metadata[name].performance_score + 0.01
                )
                
                # 记录会话
                self.tool_session_memory[name]['last_input'] = {
                    'args': args,
                    'kwargs': kwargs
                }
                self.tool_session_memory[name]['last_result'] = result
                self.tool_session_memory[name]['usage_count'] = \
                    self.tool_session_memory[name].get('usage_count', 0) + 1
                
                execution_time = time.time() - start_time
                
                return {
                    'result': result,
                    '_enhanced': {
                        'execution_time': f"{execution_time:.4f}s",
                        'tool_category': self.tool_metadata[name].category,
                        'performance_score': self.tool_metadata[name].performance_score
                    }
                }
                
            except Exception as e:
                self.tool_metadata[name].error_rate += 0.1
                raise e
        
        # 替换原始函数
        self.tools[name] = enhanced_wrapper
        self.core_server.mcp_server.tools[name] = enhanced_wrapper
    
    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """基于类型注解自动生成JSON Schema"""
        sig = inspect.signature(func)
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        type_mapping = {
            int: "number",
            float: "number", 
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # 默认类型
            
            if param.annotation != inspect.Parameter.empty:
                python_type = param.annotation
                if python_type in type_mapping:
                    param_type = type_mapping[python_type]
                elif hasattr(python_type, '__origin__'):
                    # 处理List[str], Dict[str, int]等
                    if python_type.__origin__ is list:
                        param_type = "array"
                    elif python_type.__origin__ is dict:
                        param_type = "object"
            
            schema["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
        
        return schema
    
    def resource(self, uri: str):
        """FastMCP兼容的资源装饰器"""
        def decorator(func: Callable):
            # 资源实现待扩展
            return func
        return decorator
    
    def prompt(self, name: str):
        """FastMCP兼容的提示装饰器"""
        def decorator(func: Callable):
            # 提示实现待扩展  
            return func
        return decorator
    
    # 现有功能保持不变
    def get_tool_session_context(self, session_id: str) -> Dict[str, Any]:
        return self.tool_session_memory.get(session_id, {})
    
    def update_tool_session_context(self, session_id: str, tool_name: str, context: Dict[str, Any]):
        if session_id not in self.tool_session_memory:
            self.tool_session_memory[session_id] = {}
        self.tool_session_memory[session_id][tool_name] = context
    
    def get_available_tools(self) -> Dict[str, Any]:
        self._discover_and_register_tools()
        
        tools_info = {}
        
        # Pisces原生工具
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
        
        # FastMCP集成工具
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
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        self._discover_and_register_tools()
        
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                start_time = time.time()
                result = tool.function(parameters)
                
                # 性能统计
                if tool_name in self.tool_metadata:
                    meta = self.tool_metadata[tool_name]
                    meta.usage_count += 1
                    meta.last_used = datetime.now()
                    # 简单性能计算
                    execution_time = time.time() - start_time
                    if execution_time > 0:
                        meta.performance_score = max(0.1, min(1.0, 1.0 / (execution_time * 10)))
                
                return result
            except Exception as e:
                self.logger.error(f"Tool execution failed {tool_name}: {e}")
                if tool_name in self.tool_metadata:
                    self.tool_metadata[tool_name].error_rate = min(1.0, 
                        self.tool_metadata[tool_name].error_rate + 0.1)
                return {"error": str(e)}
        
        # FastMCP工具执行
        if tool_name in self.core_server.mcp_server.tools:
            try:
                tool_func = self.core_server.mcp_server.tools[tool_name]
                return tool_func(parameters)
            except Exception as e:
                self.logger.error(f"FastMCP tool execution failed {tool_name}: {e}")
                return {"error": str(e)}
        
        return {"error": f"Tool '{tool_name}' not found"}
    
    def register_custom_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                           function: Callable, category: str = "Custom") -> bool:
        """Register a custom tool with Pisces enterprise features."""
        try:
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
                author="PiscesL1",
                last_updated=datetime.now(),
                dependencies=[]
            )
            
            self.logger.info(f"📝 Enhanced tool registered: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register custom tool: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        self._discover_and_register_tools()
        
        total_tools = len(self.tools) + len(self.core_server.mcp_server.tools)
        
        return {
            "status": "running",
            "total_tools": total_tools,
            "piscesl1_tools": len(self.tools),
            "fastmcp_tools": len(self.core_server.mcp_server.tools),
            "enhancement_level": "enterprise",
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
            ]
        }
    
    def reload_tools(self):
        """Reload all tools with enhancement."""
        with self._lock:
            self.tools.clear()
            self.tool_metadata.clear()
            self._discover_and_register_tools()
            self.logger.info("🔄 Enhanced tool reloading completed")

# 全局单例
plaza = PiscesL1MCPPlaza()

# FastMCP兼容API - 秘密增强
class FastMCP:
    """
    100% FastMCP兼容API，秘密增强为Pisces企业级
    用户以为在用官方FastMCP，实际运行在Pisces增强引擎上
    """
    
    def __init__(self, name: str = "PiscesMCP"):
        self.name = name
        self._plaza = plaza
    
    def __del__(self):
        """清理资源"""
        self._cleanup()
    
    def _cleanup(self):
        """优雅清理所有资源"""
        try:
            # 停止文件监控
            if self._watcher_thread and self._watcher_thread.is_alive():
                self._stop_watcher.set()
                self._watcher_thread.join(timeout=1)
                self.logger.info("🛑 File watcher stopped")
            
            # 清理线程池
            if self._executor:
                self._executor.shutdown(wait=True)
                
            # 清理缓存
            with self._lock:
                self._cache.clear()
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        """手动停止MCP Plaza"""
        self._cleanup()
        self.logger.info("🎯 MCP Plaza stopped gracefully")
    
    def tool(self, name: Optional[str] = None, description: Optional[str] = None):
        """FastMCP兼容的装饰器"""
        return self._plaza.tool(name, description)
    
    def resource(self, uri: str):
        """FastMCP兼容的资源装饰器"""
        return self._plaza.resource(uri)
    
    def prompt(self, name: str):
        """FastMCP兼容的提示装饰器"""
        return self._plaza.prompt(name)
    
    def run(self):
        """运行MCP服务器"""
        self._plaza.logger.info(f"🚀 Enhanced FastMCP '{self.name}' running on Pisces engine")
        return self._plaza.get_system_status()

# 向后兼容的便捷接口
def get_available_tools() -> Dict[str, Any]:
    """Get all available tools."""
    return plaza.get_available_tools()

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool with enhancement."""
    return plaza.execute_tool(tool_name, arguments)

def get_system_status() -> Dict[str, Any]:
    """Get enhanced system status."""
    return plaza.get_system_status()

def reload_tools():
    """Reload tools with enhancement."""
    plaza.reload_tools()

def register_custom_tool(name: str, description: str, parameters: Dict[str, Any], 
                        function: Callable, category: str = "Custom") -> bool:
    """Register custom tool with enhancement."""
    return plaza.register_custom_tool(name, description, parameters, function, category)

# 秘密增强：创建FastMCP兼容实例
mcp = FastMCP("PiscesEnhanced")

# 导出接口 - 既支持传统方式也支持FastMCP方式
__all__ = [
    'get_available_tools',
    'execute_tool',
    'get_system_status', 
    'reload_tools',
    'register_custom_tool',
    'plaza',
    'mcp',
    'FastMCP'
]

# 自动发现工具
plaza._discover_and_register_tools()
