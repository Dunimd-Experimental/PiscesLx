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
import inspect
import importlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Set, Union, TypeVar, Generic

# Ensure the project root directory is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core functionality from model/mcp
from model.mcp.server import ArcticOptimizedMCPServer as CoreMCPServer
from model.mcp.translator import ArcticMCPTranslationLayer

@dataclass
class PiscesLxMCPToolMetadata:
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
    last_used: Optional[datetime] = None

@dataclass
class PiscesLxMCPToolStats:
    """Structure for tool statistics."""
    name: str
    load_time: float
    tool_count: int
    status: str
    error: Optional[str] = None
    memory_usage: Optional[int] = None
    last_used: Optional[datetime] = None

class PiscesLxMCPPlaza:
    """Pisces L1 MCP Plaza - An advanced integrated system."""
    
    def __init__(self):
        self.core_server = CoreMCPServer()
        self.translation_layer = ArcticMCPTranslationLayer()
        
        # Tool management
        self.tools: Dict[str, Any] = {}
        self.tool_metadata: Dict[str, PiscesLxMCPToolMetadata] = {}
        self.tool_stats: List[PiscesLxMCPToolStats] = []
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
        
        # 文档处理器集成
        self.document_processor = None
        self._init_document_processor()
        
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
    
    def _init_document_processor(self):
        """Initialize document processor for PDF/DOCX/PPTX handling."""
        try:
            from MCP import document_processor
            self.document_processor = document_processor.document_processor
            self.logger.info("📄 Document processor initialized")
        except ImportError as e:
            self.logger.warning(f"Document processor not available: {e}")
            self.document_processor = None
    
    def _is_tool_compatible(self, module_name: str, module) -> bool:
        """Check if a tool is compatible with the system."""
        if module_name in self.blacklisted_tools:
            self.logger.info(f"Skipping blacklisted tool: {module_name}")
            return False
        
        has_tool = hasattr(module, 'pisces_tool') or hasattr(module, 'mcp')
        if not has_tool:
            return False
        return True
    
    def _load_module_with_stats(self, py_file: Path) -> PiscesLxMCPToolStats:
        """Load a module with statistics."""
        start_time = time.time()
        module_name = py_file.stem
        
        if module_name in ["__init__", "__pycache__"]:
            return PiscesLxMCPToolStats(module_name, 0, 0, "skipped")
        
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[f"MCP.{module_name}"])
            
            module = importlib.import_module(f"MCP.{module_name}")
            
            if not self._is_tool_compatible(module_name, module):
                return PiscesLxMCPToolStats(module_name, time.time() - start_time, 0, "incompatible")
            
            tool_count = 0
            if hasattr(module, 'pisces_tool'):
                tool = module.pisces_tool
                self.tools[tool.name] = tool
                tool_count += 1
                
                self.tool_metadata[tool.name] = PiscesLxMCPToolMetadata(
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
                    self.core_server.mcp_server._tool_manager._tools[tool_name] = tool_func
                    tool_count += 1
            
            load_time = time.time() - start_time
            return PiscesLxMCPToolStats(module_name, load_time, tool_count, "success")
            
        except Exception as e:
            return PiscesLxMCPToolStats(module_name, time.time() - start_time, 0, "error", str(e))
    
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

    def register_custom_tool(self, name: str, description: str, func: Callable, 
                           category: str = "custom", **kwargs):
        """企业级工具注册 - 内部核心"""
        tool_metadata = PiscesLxMCPToolMetadata(
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
        self.core_server.mcp_server.add_tool(func, name=name)
        
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
                
                # 更新最后使用时间
                self.tool_metadata[name].last_used = datetime.now()
                
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
        self.core_server.mcp_server.add_tool(enhanced_wrapper, name=name)
    
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
                "description": getattr(tool, '__doc__', ''),
                "parameters": {},
                "metadata": asdict(self.tool_metadata.get(name, PiscesLxMCPToolMetadata(
                    name=name, description="", category="", version="", 
                    author="", last_updated=datetime.now(), dependencies=[]
                )))
            }
        
        # FastMCP集成工具
        for tool_name in self.core_server.mcp_server._tool_manager._tools.keys():
            if tool_name not in tools_info:
                tools_info[tool_name] = {
                    "type": "FastMCP Integrated",
                    "description": f"FastMCP tool: {tool_name}",
                    "parameters": {},
                    "metadata": asdict(PiscesLxMCPToolMetadata(
                        name=tool_name, description="", category="FastMCP", 
                        version="1.0.0", author="MCP", last_updated=datetime.now(), dependencies=[]
                    ))
                }
        
        # 文档处理工具
        if self.document_processor:
            tools_info.update({
                "extract_document_content": {
                    "type": "Document Processor",
                    "description": "Extract content from PDF, DOCX, PPTX files",
                    "parameters": {
                        "file_path": {"type": "string", "description": "Path to document file"},
                        "include_full_content": {"type": "boolean", "description": "Include full content or summary"}
                    },
                    "metadata": asdict(PiscesLxMCPToolMetadata(
                        name="extract_document_content", 
                        description="Advanced document content extraction", 
                        category="Document Processing", 
                        version="1.0.0", 
                        author="PiscesL1", 
                        last_updated=datetime.now(), 
                        dependencies=["PyMuPDF", "python-docx", "python-pptx"]
                    ))
                },
                "list_supported_formats": {
                    "type": "Document Processor",
                    "description": "List all supported document formats",
                    "parameters": {},
                    "metadata": asdict(PiscesLxMCPToolMetadata(
                        name="list_supported_formats", 
                        description="Document format support information", 
                        category="Document Processing", 
                        version="1.0.0", 
                        author="PiscesL1", 
                        last_updated=datetime.now(), 
                        dependencies=[]
                    ))
                },
                "batch_process_documents": {
                    "type": "Document Processor", 
                    "description": "Process multiple documents in batch",
                    "parameters": {
                        "directory_path": {"type": "string", "description": "Directory containing documents"},
                        "recursive": {"type": "boolean", "description": "Search subdirectories"}
                    },
                    "metadata": asdict(PiscesLxMCPToolMetadata(
                        name="batch_process_documents", 
                        description="Batch document processing", 
                        category="Document Processing", 
                        version="1.0.0", 
                        author="PiscesL1", 
                        last_updated=datetime.now(), 
                        dependencies=["PyMuPDF", "python-docx", "python-pptx"]
                    ))
                }
            })
        
        return tools_info
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        self._discover_and_register_tools()
        
        # 文档处理工具
        if self.document_processor:
            if tool_name in ["extract_document_content", "list_supported_formats", "batch_process_documents"]:
                try:
                    start_time = time.time()
                    result = None
                    
                    if tool_name == "extract_document_content":
                        from MCP.document_processor import extract_document_content
                        result = extract_document_content(
                            parameters.get("file_path", ""),
                            parameters.get("include_full_content", True)
                        )
                    elif tool_name == "list_supported_formats":
                        from MCP.document_processor import list_supported_formats
                        result = list_supported_formats()
                    elif tool_name == "batch_process_documents":
                        from MCP.document_processor import batch_process_documents
                        result = batch_process_documents(
                            parameters.get("directory_path", ""),
                            parameters.get("recursive", False)
                        )
                    
                    # 性能统计
                    execution_time = time.time() - start_time
                    if tool_name in self.tool_metadata:
                        meta = self.tool_metadata[tool_name]
                        meta.usage_count += 1
                        meta.last_used = datetime.now()
                        if execution_time > 0:
                            meta.performance_score = max(0.1, min(1.0, 1.0 / (execution_time * 10)))
                    
                    # 确保返回值是字典类型
                    if result is not None:
                        if isinstance(result, dict):
                            return result
                        else:
                            return {"result": result}
                    else:
                        return {"error": "Tool execution failed"}
                    
                except Exception as e:
                    self.logger.error(f"Document tool execution failed {tool_name}: {e}")
                    if tool_name in self.tool_metadata:
                        self.tool_metadata[tool_name].error_rate = min(1.0, 
                            self.tool_metadata[tool_name].error_rate + 0.1)
                    return {"error": str(e)}
        
        # Pisces原生工具
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                start_time = time.time()
                result = None
                if callable(tool):
                    result = tool(parameters)
                else:
                    result = getattr(tool, 'function', lambda x: {})(parameters)
                
                # 性能统计
                if tool_name in self.tool_metadata:
                    meta = self.tool_metadata[tool_name]
                    meta.usage_count += 1
                    meta.last_used = datetime.now()
                    # 简单性能计算
                    execution_time = time.time() - start_time
                    if execution_time > 0:
                        meta.performance_score = max(0.1, min(1.0, 1.0 / (execution_time * 10)))
                
                # 确保返回值是字典类型
                if result is not None:
                    if isinstance(result, dict):
                        return result
                    else:
                        return {"result": result}
                else:
                    return {"error": "Tool execution failed"}
            except Exception as e:
                self.logger.error(f"Tool execution failed {tool_name}: {e}")
                if tool_name in self.tool_metadata:
                    self.tool_metadata[tool_name].error_rate = min(1.0, 
                        self.tool_metadata[tool_name].error_rate + 0.1)
                return {"error": str(e)}
        
        # FastMCP工具执行
        if tool_name in self.core_server.mcp_server._tool_manager._tools:
            try:
                tool_func = self.core_server.mcp_server._tool_manager._tools[tool_name]
                # 直接调用工具函数而不是Tool对象
                try:
                    if hasattr(tool_func, 'function'):
                        # 尝试直接调用tool_func
                        if callable(tool_func):
                            return tool_func(parameters)
                        else:
                            return {"error": "Tool is not callable"}
                    else:
                        # 如果tool_func是一个可调用对象，直接调用它
                        if callable(tool_func):
                            return tool_func(parameters)
                        else:
                            return {"error": "Tool is not callable"}
                except Exception as e:
                    self.logger.error(f"FastMCP tool execution failed {tool_name}: {e}")
                    return {"error": str(e)}
            except Exception as e:
                self.logger.error(f"FastMCP tool execution failed {tool_name}: {e}")
                return {"error": str(e)}
        
        return {"error": f"Tool '{tool_name}' not found"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        self._discover_and_register_tools()
        
        total_tools = len(self.tools) + len(self.core_server.mcp_server._tool_manager._tools)
        doc_tools_count = 3 if self.document_processor else 0
        
        return {
            "status": "running",
            "total_tools": total_tools + doc_tools_count,
            "piscesl1_tools": len(self.tools),
            "fastmcp_tools": len(self.core_server.mcp_server._tool_manager._tools),
            "document_processor_tools": doc_tools_count,
            "document_processor_available": self.document_processor is not None,
            "enhancement_level": "enterprise",
            "blacklisted_tools": list(self.blacklisted_tools),
            "supported_document_formats": list(self.document_processor.supported_formats.keys()) if self.document_processor else [],
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
plaza = PiscesLxMCPPlaza()

# 导出接口
__all__ = [
    'plaza',
    'PiscesLxMCPPlaza',
    'PiscesLxMCPToolMetadata',
    'PiscesLxMCPToolStats'
]

# 自动发现工具
plaza._discover_and_register_tools()