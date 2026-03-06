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

import asyncio
import json
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import socket

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from configs.version import VERSION

from .types import (
    POPSSMCPConfiguration,
    POPSSMCPHealthStatus,
    POPSSMCPPerformanceMetrics,
    POPSSMCPFileWatcherConfig
)
from .registry import POPSSMCPToolRegistry
from .execution import POPSSMCPExecutionManager, POPSSMCPExecutionMode
from .monitor import POPSSMCPMonitor

T = TypeVar('T')

@dataclass
class POPSSMCPServerInfo:
    name: str = "POPSSMCPPlaza"
    version: str = VERSION
    protocol_version: str = "2024-11-05"
    capabilities: List[str] = field(default_factory=lambda: ["tools", "resources", "prompts"])
    server_id: str = field(default_factory=lambda: f"popss_mcp_{uuid.uuid4().hex[:8]}")

@dataclass
class POPSSMCPToolProvider:
    provider_id: str
    name: str
    tools: List[str]
    connection_info: Dict[str, Any] = field(default_factory=dict)
    status: str = "connected"

class POPSSMCPPlaza:
    def __init__(self, config: Optional[Union[POPSSMCPConfiguration, Dict[str, Any]]] = None):
        if config is None:
            self.config = POPSSMCPConfiguration()
        elif isinstance(config, dict):
            self.config = POPSSMCPConfiguration(**config)
        else:
            self.config = config
        
        self._LOG = self._configure_logging()
        
        self.server_info = POPSSMCPServerInfo()
        
        self._lock = threading.RLock()
        
        self._registry = POPSSMCPToolRegistry({
            'auto_register': self.config.load_default_tools,
            'strict_validation': self.config.strict_validation
        })
        
        self._execution_manager = POPSSMCPExecutionManager(self.config)
        
        self._monitor = POPSSMCPMonitor(self.config)
        
        self._sessions: Dict[str, 'POPSSMCPSession'] = {}
        
        self._tool_providers: Dict[str, POPSSMCPToolProvider] = {}
        self._provider_connections: Dict[str, Any] = {}
        
        self._server: Optional[threading.Thread] = None
        self._server_stop_event = threading.Event()
        self._server_running = False
        
        self._tool_discovery_lock = threading.Lock()
        self._discovery_in_progress = False
        self._last_discovery_time: Optional[datetime] = None
        
        self._tool_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._cache_timestamp: Optional[datetime] = None
        
        self._callbacks: Dict[str, Callable] = {
            'on_tool_registered': [],
            'on_tool_unregistered': [],
            'on_execution_started': [],
            'on_execution_completed': [],
            'on_error': [],
        }
        
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="pisceslx_mcp_async"
        )
        
        self._initialized = False
        self._shutdown = False
        
        self._LOG.info(f"POPSSMCPPlaza initialized with config: max_workers={self.config.max_workers}")
    
    def _configure_logging(self) -> PiscesLxLogger:
        logger = PiscesLxLogger("PiscesLx.Opss.MCP",file_path=get_log_file("PiscesLx.Opss.MCP"), enable_file=True)
        return logger
    
    def initialize(self) -> bool:
        if self._initialized:
            self._LOG.warning("POPSSMCPPlaza already initialized")
            return True
        
        try:
            self._monitor.start_monitoring()
            
            if self.config.load_default_tools:
                self._discover_and_register_tools()
            
            self._initialized = True
            self._LOG.info("POPSSMCPPlaza initialized successfully")
            return True
            
        except Exception as e:
            self._LOG.error(f"Failed to initialize POPSSMCPPlaza: {e}")
            return False
    
    def _discover_and_register_tools(self):
        discovery_paths = self._get_default_discovery_paths()
        
        for path in discovery_paths:
            self._discover_tools_in_path(path)
    
    def _get_default_discovery_paths(self) -> List[str]:
        paths = []
        
        mcp_dir = Path(__file__).parent.parent.parent / "MCP"
        if mcp_dir.exists():
            paths.append(str(mcp_dir))
        
        user_config_path = os.path.expanduser("~/.pisceslx/mcp_tools")
        if os.path.exists(user_config_path):
            paths.append(user_config_path)
        
        project_mcp_path = os.path.expanduser("~/piscesl1/MCP")
        if os.path.exists(project_mcp_path):
            paths.append(str(project_mcp_path))
        
        return paths
    
    def _discover_tools_in_path(self, path: str):
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return
            
            for item in path_obj.iterdir():
                if item.is_file() and item.suffix == '.py':
                    tool_name = item.stem
                    if self._is_valid_tool_name(tool_name):
                        self._register_tool_from_file(item)
                
                elif item.is_dir():
                    init_file = item / "__init__.py"
                    if init_file.exists():
                        self._register_tool_from_package(item)
                        
        except Exception as e:
            self._LOG.error(f"Error discovering tools in {path}: {e}")
    
    def _is_valid_tool_name(self, name: str) -> bool:
        if not name:
            return False
        if name.startswith('_'):
            return False
        if not name.isidentifier():
            return False
        return True
    
    def _register_tool_from_file(self, file_path: Path):
        try:
            tool_name = file_path.stem
            
            self._registry.register_tool(
                tool_name=tool_name,
                description=f"Tool from {file_path.name}",
                category="discovered",
                tags={"discovered", "file"}
            )
            
            self._LOG.debug(f"Discovered tool: {tool_name} from {file_path}")
            
        except Exception as e:
            self._LOG.warning(f"Failed to register tool from {file_path}: {e}")
    
    def _register_tool_from_package(self, package_path: Path):
        try:
            package_name = package_path.name
            
            self._registry.register_tool(
                tool_name=package_name,
                description=f"Tool package: {package_name}",
                category="discovered",
                tags={"discovered", "package"}
            )
            
            self._LOG.debug(f"Discovered tool package: {package_name}")
            
        except Exception as e:
            self._LOG.warning(f"Failed to register tool package {package_path}: {e}")
    
    def create_session(self, session_id: Optional[str] = None) -> 'POPSSMCPSession':
        session = POPSSMCPSession(
            plaza=self,
            session_id=session_id,
            config={
                'max_executions': self.config.max_executions,
                'session_timeout': self.config.session_timeout,
                'memory_limit': self.config.memory_limit
            }
        )
        
        with self._lock:
            self._sessions[session.session_id] = session
        
        return session
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        mode: POPSSMCPExecutionMode = POPSSMCPExecutionMode.AUTO,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        arguments = arguments or {}
        
        if self._shutdown:
            raise RuntimeError("Cannot execute tool: MCP Plaza is shutdown")
        
        if not self._initialized:
            raise RuntimeError("Cannot execute tool: MCP Plaza not initialized")
        
        self._LOG.info(f"Executing tool: {tool_name} with arguments: {list(arguments.keys())}")
        
        execution_result = self._execution_manager.execute(
            tool_name=tool_name,
            arguments=arguments,
            mode=mode,
            session_id=session_id,
            **kwargs
        )
        
        if execution_result.success:
            return execution_result.result
        else:
            raise RuntimeError(f"Tool execution failed: {execution_result.error}")
    
    async def execute_tool_async(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Any:
        arguments = arguments or {}
        
        loop = asyncio.get_event_loop()
        
        def run_execution():
            return self.execute_tool(tool_name, arguments, session_id=session_id)
        
        return await loop.run_in_executor(self._async_executor, run_execution)
    
    def register_tool(
        self,
        tool_name: str,
        description: str,
        function: Optional[Callable] = None,
        category: str = "custom",
        tags: Optional[Set[str]] = None,
        **kwargs
    ) -> bool:
        result = self._registry.register_tool(
            tool_name=tool_name,
            description=description,
            category=category,
            function=function,
            tags=tags,
            **kwargs
        )
        
        if result:
            self._invalidate_cache()
            self._notify_callbacks('on_tool_registered', {'tool_name': tool_name})
        
        return result
    
    def unregister_tool(self, tool_name: str) -> bool:
        result = self._registry.unregister_tool(tool_name)
        
        if result:
            self._invalidate_cache()
            self._notify_callbacks('on_tool_unregistered', {'tool_name': tool_name})
        
        return result
    
    def list_tools(
        self,
        category: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        use_cache: bool = True
    ) -> List[str]:
        cache_key = f"tools_{category}_{','.join(sorted(tags)) if tags else 'all'}"
        
        if use_cache and self._cache_timestamp:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
            if cache_age < self.config.cache_timeout:
                return self._tool_cache.get(cache_key, [])
        
        tools = self._registry.list_tools(category=category, tags=tags)
        
        if use_cache:
            with self._cache_lock:
                self._tool_cache[cache_key] = tools
                self._cache_timestamp = datetime.now()
        
        return tools
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        cache_key = f"tool_{tool_name}"
        
        if self._cache_timestamp:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
            if cache_age < self.config.cache_timeout:
                if cache_key in self._tool_cache:
                    return self._tool_cache[cache_key]
        
        metadata = self._registry.get_tool(tool_name)
        
        if metadata:
            result = {
                'name': metadata.name,
                'description': metadata.description,
                'category': metadata.category,
                'version': metadata.version,
                'author': metadata.author,
                'dependencies': metadata.dependencies,
                'performance_score': metadata.performance_score,
                'usage_count': metadata.usage_count,
                'error_rate': metadata.error_rate,
                'last_used': metadata.last_used.isoformat() if metadata.last_used else None
            }
            
            with self._cache_lock:
                self._tool_cache[cache_key] = result
                self._cache_timestamp = datetime.now()
            
            return result
        
        return None
    
    def search_tools(self, query: str) -> List[str]:
        return self._registry.search_tools(query)
    
    def get_tool_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        return self._registry.get_tool_stats(tool_name)
    
    def execute_tool_batch(
        self,
        executions: List[Dict[str, Any]],
        mode: POPSSMCPExecutionMode = POPSSMCPExecutionMode.THREADED
    ) -> List[Any]:
        results = []
        
        for exec_config in executions:
            tool_name = exec_config.get('tool_name')
            arguments = exec_config.get('arguments', {})
            
            try:
                result = self.execute_tool(tool_name, arguments, mode=mode)
                results.append({'success': True, 'result': result})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        
        return results
    
    def execute_tool_parallel(
        self,
        executions: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[Any]:
        return self._execution_manager.execute_parallel(executions, max_concurrent=max_concurrent)
    
    def get_health_status(self) -> POPSSMCPHealthStatus:
        return self._monitor.get_health_status()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            'execution': self._execution_manager.get_metrics(),
            'monitor': self._monitor.get_performance_summary()
        }
    
    def run_health_check(self) -> POPSSMCPHealthStatus:
        return self._monitor.run_health_check()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        return self._monitor.get_active_alerts()
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: Optional[str] = None) -> bool:
        return self._monitor.acknowledge_alert(alert_id, acknowledged_by)
    
    def register_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self._LOG.error(f"Error in callback for {event}: {e}")
    
    def _invalidate_cache(self):
        with self._cache_lock:
            self._tool_cache.clear()
            self._cache_timestamp = None
    
    def create_tool_provider(
        self,
        provider_id: str,
        name: str,
        tools: List[str],
        connection_info: Optional[Dict[str, Any]] = None
    ) -> POPSSMCPToolProvider:
        provider = POPSSMCPToolProvider(
            provider_id=provider_id,
            name=name,
            tools=tools,
            connection_info=connection_info or {}
        )
        
        with self._lock:
            self._tool_providers[provider_id] = provider
        
        return provider
    
    def connect_tool_provider(
        self,
        provider_id: str,
        connection_type: str = "socket",
        **connection_params
    ) -> bool:
        provider = self._tool_providers.get(provider_id)
        if not provider:
            self._LOG.error(f"Tool provider not found: {provider_id}")
            return False
        
        try:
            if connection_type == "socket":
                return self._connect_socket_provider(provider, **connection_params)
            elif connection_type == "http":
                return self._connect_http_provider(provider, **connection_params)
            else:
                self._LOG.error(f"Unknown connection type: {connection_type}")
                return False
                
        except Exception as e:
            self._LOG.error(f"Failed to connect provider {provider_id}: {e}")
            return False
    
    def _connect_socket_provider(self, provider: POPSSMCPToolProvider, **params) -> bool:
        host = params.get('host', 'localhost')
        port = params.get('port', 8080)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))
            
            provider.connection_info = {
                'type': 'socket',
                'host': host,
                'port': port,
                'socket': sock
            }
            provider.status = "connected"
            
            self._provider_connections[provider.provider_id] = sock
            
            return True
            
        except Exception as e:
            self._LOG.error(f"Socket connection failed: {e}")
            provider.status = "error"
            return False
    
    def _connect_http_provider(self, provider: POPSSMCPToolProvider, **params) -> bool:
        base_url = params.get('base_url', f"http://localhost:8080")
        
        try:
            provider.connection_info = {
                'type': 'http',
                'base_url': base_url,
                'session': None
            }
            provider.status = "connected"
            
            self._provider_connections[provider.provider_id] = provider
            
            return True
            
        except Exception as e:
            self._LOG.error(f"HTTP connection failed: {e}")
            provider.status = "error"
            return False
    
    def disconnect_tool_provider(self, provider_id: str) -> bool:
        if provider_id in self._provider_connections:
            conn = self._provider_connections[provider_id]
            
            if hasattr(conn, 'close'):
                conn.close()
            
            del self._provider_connections[provider_id]
        
        if provider_id in self._tool_providers:
            self._tool_providers[provider_id].status = "disconnected"
        
        return True
    
    def export_registry(self, path: str) -> bool:
        return self._registry.export_registry(path)
    
    def import_registry(self, path: str, overwrite: bool = False) -> int:
        return self._registry.import_registry(path, overwrite)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        return self._registry.get_registry_summary()
    
    def cleanup_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.cleanup()
                del self._sessions[session_id]
    
    def invalidate_tool_cache(self, tool_name: Optional[str] = None):
        if tool_name:
            cache_key = f"tool_{tool_name}"
            with self._cache_lock:
                if cache_key in self._tool_cache:
                    del self._tool_cache[cache_key]
        else:
            self._invalidate_cache()
    
    def trigger_hot_reload(self, file_path: str):
        self._LOG.info(f"Hot reload triggered for: {file_path}")
        
        self._invalidate_cache()
        
        self._discover_and_register_tools()
    
    def shutdown(self):
        if self._shutdown:
            return
        
        self._shutdown = True
        
        self._LOG.info("Shutting down POPSSMCPPlaza...")
        
        self._monitor.shutdown()
        
        self._execution_manager.shutdown()
        
        with self._lock:
            for session_id, session in self._sessions.items():
                session.cleanup()
            self._sessions.clear()
        
        self._async_executor.shutdown(wait=True)
        
        for provider_id, conn in self._provider_connections.items():
            try:
                if hasattr(conn, 'close'):
                    conn.close()
            except Exception:
                pass
        
        self._server_stop_event.set()
        
        self._LOG.info("POPSSMCPPlaza shutdown complete")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


class POPSSMCPSession:
    def __init__(
        self,
        plaza: POPSSMCPPlaza,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.plaza = plaza
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or {}
        
        self._lock = threading.RLock()
        
        self.tool_cache: Dict[str, Any] = {}
        self.global_context: Dict[str, Any] = {}
        
        self.execution_history: List[Dict[str, Any]] = []
        
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        self.is_active = True
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        mode: POPSSMCPExecutionMode = POPSSMCPExecutionMode.AUTO,
        **kwargs
    ) -> Any:
        arguments = arguments or {}
        
        cache_key = f"{tool_name}_{str(sorted(arguments.items()))}"
        
        if self.config.get('use_cache', True) and cache_key in self.tool_cache:
            self.plaza.logger.debug(f"Using cached result for {tool_name}")
            return self.tool_cache[cache_key]
        
        start_time = time.time()
        
        try:
            result = self.plaza.execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                mode=mode,
                session_id=self.session_id,
                **kwargs
            )
            
            if self.config.get('use_cache', True):
                self.tool_cache[cache_key] = result
            
            execution_time = time.time() - start_time
            self.plaza._monitor.record_execution_time(execution_time)
            
            self.execution_history.append({
                'tool_name': tool_name,
                'arguments': arguments,
                'result': result,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            })
            
            self.last_activity = datetime.now()
            
            return result
            
        except Exception as e:
            self.plaza.logger.error(f"Tool execution failed in session: {tool_name}: {e}")
            raise
    
    def set_global_context(self, key: str, value: Any):
        with self._lock:
            self.global_context[key] = value
    
    def get_global_context(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.global_context.get(key, default)
    
    def clear_cache(self):
        with self._lock:
            self.tool_cache.clear()
    
    def cleanup(self):
        with self._lock:
            self.tool_cache.clear()
            self.global_context.clear()
            self.execution_history.clear()
            self.is_active = False
        
        self.plaza.cleanup_session(self.session_id)
    
    def get_session_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'session_id': self.session_id,
                'created_at': self.created_at.isoformat(),
                'last_activity': self.last_activity.isoformat(),
                'is_active': self.is_active,
                'cache_size': len(self.tool_cache),
                'context_keys': len(self.global_context),
                'execution_count': len(self.execution_history)
            }
