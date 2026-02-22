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

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar
from queue import Queue, Empty
from collections import deque

from utils.dc import PiscesLxLogger

from .types import (
    POPSSMCPConfiguration,
    POPSSMCPExecutionContext,
    POPSSMCPModuleStats,
    POPSSMCPModuleStatus
)

T = TypeVar('T')

class POPSSMCPUnifiedToolExecutor:
    def __init__(self):
        self._LOG = self._configure_logging()
        self.tools: Dict[str, Any] = {}
        self.tool_metadata: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._LOG.info("POPSSMCPUnifiedToolExecutor initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        _LOG = get_logger("PiscesLx.Core.MCP.UnifiedExecutor")
        return _LOG
    
    def register_tool(self, tool_metadata: Any):
        with self._lock:
            if hasattr(tool_metadata, 'name'):
                self.tools[tool_metadata.name] = tool_metadata
                self.tool_metadata[tool_metadata.name] = tool_metadata
                self._LOG.info(f"Tool registered in unified executor: {tool_metadata.name}")
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        with self._lock:
            if tool_name not in self.tools:
                raise ValueError(f"Tool not found: {tool_name}")
            
            tool = self.tools[tool_name]
            if hasattr(tool, 'function') and callable(tool.function):
                return tool.function(arguments)
            elif callable(tool):
                return tool(arguments)
            
            raise ValueError(f"Tool {tool_name} is not callable")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        with self._lock:
            return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        with self._lock:
            return list(self.tools.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_tools': len(self.tools),
                'tool_names': list(self.tools.keys())
            }

class POPSSMCPExecutionMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    THREADED = "threaded"
    REMOTE = "remote"
    AUTO = "auto"

@dataclass
class POPSSMCPExecutionResult:
    execution_id: str
    tool_name: str
    mode: POPSSMCPExecutionMode
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSMCPExecutionMetrics:
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    tool_execution_counts: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    last_execution_time: Optional[datetime] = None

class POPSSMCPExecutionManager:
    def __init__(self, config: Optional[POPSSMCPConfiguration] = None):
        self.config = config or POPSSMCPConfiguration()
        self._LOG = self._configure_logging()
        
        self._lock = threading.RLock()
        
        self.pending_executions: Queue = Queue(maxsize=self.config.max_queue_size)
        self.execution_results: Dict[str, POPSSMCPExecutionResult] = {}
        self.active_executions: Dict[str, POPSSMCPExecutionResult] = {}
        self.execution_history: deque = deque(maxlen=self.config.max_execution_history)
        
        self.metrics = POPSSMCPExecutionMetrics()
        
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self._setup_thread_pool()
        
        self._execution_loop: Optional[asyncio.AbstractEventLoop] = None
        self._execution_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._result_handlers: Dict[str, Callable] = {}
        self._error_handlers: Dict[str, Callable] = {}
        
        self.default_timeout = 60.0
        self.max_retries = 3
        self.retry_delay = 0.1
        
        self._LOG.info("POPSSMCPExecutionManager initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        _LOG = get_logger("PiscesLx.Core.MCP.Execution")
        return _LOG
    
    def _setup_thread_pool(self):
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="piscesl1_mcp_exec"
        )
        self._LOG.info(f"Thread pool initialized with {self.config.max_workers} workers")
    
    def set_result_handler(self, tool_name: str, handler: Callable):
        self._result_handlers[tool_name] = handler
    
    def set_error_handler(self, tool_name: str, handler: Callable):
        self._error_handlers[tool_name] = handler
    
    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        mode: POPSSMCPExecutionMode = POPSSMCPExecutionMode.AUTO,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        session_id: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> POPSSMCPExecutionResult:
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        effective_mode = mode
        if mode == POPSSMCPExecutionMode.AUTO:
            effective_mode = self._determine_execution_mode(tool_name, arguments)
        
        result = POPSSMCPExecutionResult(
            execution_id=execution_id,
            tool_name=tool_name,
            mode=effective_mode,
            success=False,
            metadata={
                'timeout': timeout or self.default_timeout,
                'retries': retries if retries is not None else self.max_retries,
                'session_id': session_id,
                'priority': priority,
                'attempt': 0,
                **kwargs
            }
        )
        
        if effective_mode == POPSSMCPExecutionMode.SYNC:
            return self._execute_sync(tool_name, arguments, result)
        elif effective_mode == POPSSMCPExecutionMode.ASYNC:
            return self._execute_async(tool_name, arguments, result)
        elif effective_mode == POPSSMCPExecutionMode.THREADED:
            return self._execute_threaded(tool_name, arguments, result)
        elif effective_mode == POPSSMCPExecutionMode.REMOTE:
            return self._execute_remote(tool_name, arguments, result)
        else:
            return self._execute_sync(tool_name, arguments, result)
    
    def _determine_execution_mode(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> POPSSMCPExecutionMode:
        if arguments is None:
            return POPSSMCPExecutionMode.SYNC
        
        complexity = self._estimate_complexity(arguments)
        if complexity == "low":
            return POPSSMCPExecutionMode.SYNC
        elif complexity == "medium":
            return POPSSMCPExecutionMode.THREADED
        else:
            return POPSSMCPExecutionMode.ASYNC
    
    def _estimate_complexity(self, arguments: Dict[str, Any]) -> str:
        if not arguments:
            return "low"
        
        total_size = 0
        for value in arguments.values():
            if isinstance(value, (str, bytes)):
                total_size += len(value)
            elif isinstance(value, (list, dict)):
                total_size += len(str(value))
        
        if total_size < 1000:
            return "low"
        elif total_size < 10000:
            return "medium"
        else:
            return "high"
    
    def _execute_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: POPSSMCPExecutionResult
    ) -> POPSSMCPExecutionResult:
        start_time = time.time()
        
        try:
            function = self._get_tool_function(tool_name)
            if function is None:
                raise ValueError(f"Tool function not found: {tool_name}")
            
            execution_context = POPSSMCPExecutionContext(
                session_id=result.metadata.get('session_id', ''),
                tool_name=tool_name,
                arguments=arguments,
                start_time=datetime.now()
            )
            
            exec_result = function(arguments)
            
            result.success = True
            result.result = exec_result
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            self._update_metrics(result, success=True)
            self._handle_result(result)
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            if result.metadata.get('attempt', 0) < result.metadata.get('retries', self.max_retries):
                return self._retry_execution(tool_name, arguments, result)
            
            self._update_metrics(result, success=False)
            self._handle_error(result)
        
        return result
    
    def _execute_threaded(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: POPSSMCPExecutionResult
    ) -> POPSSMCPExecutionResult:
        if self.thread_pool is None:
            return self._execute_sync(tool_name, arguments, result)
        
        start_time = time.time()
        result.start_time = datetime.now()
        
        future: Future = self.thread_pool.submit(
            self._execute_with_context,
            tool_name,
            arguments,
            result.metadata.get('session_id', '')
        )
        
        timeout = result.metadata.get('timeout', self.default_timeout)
        
        try:
            exec_result = future.result(timeout=timeout)
            
            result.success = True
            result.result = exec_result
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            self._update_metrics(result, success=True)
            self._handle_result(result)
            
        except Exception as e:
            future.cancel()
            
            result.success = False
            result.error = str(e)
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            self._update_metrics(result, success=False)
            self._handle_error(result)
        
        return result
    
    async def _execute_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: POPSSMCPExecutionResult
    ) -> POPSSMCPExecutionResult:
        start_time = time.time()
        result.start_time = datetime.now()
        
        try:
            function = self._get_async_tool_function(tool_name)
            if function is None:
                return self._execute_sync(tool_name, arguments, result)
            
            timeout = result.metadata.get('timeout', self.default_timeout)
            
            exec_result = await asyncio.wait_for(
                function(arguments),
                timeout=timeout
            )
            
            result.success = True
            result.result = exec_result
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            self._update_metrics(result, success=True)
            self._handle_result(result)
            
        except asyncio.TimeoutError:
            result.success = False
            result.error = "Execution timed out"
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            self._update_metrics(result, success=False)
            self._handle_error(result)
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.execution_time = time.time() - start_time
            result.end_time = datetime.now()
            
            self._update_metrics(result, success=False)
            self._handle_error(result)
        
        return result
    
    def _execute_remote(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: POPSSMCPExecutionResult
    ) -> POPSSMCPExecutionResult:
        start_time = time.time()
        result.start_time = datetime.now()
        
        self._LOG.warning(f"Remote execution mode requested for {tool_name}, falling back to sync")
        
        return self._execute_sync(tool_name, arguments, result)
    
    def _execute_with_context(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str
    ) -> Any:
        function = self._get_tool_function(tool_name)
        if function is None:
            raise ValueError(f"Tool function not found: {tool_name}")
        
        execution_context = POPSSMCPExecutionContext(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            start_time=datetime.now()
        )
        
        return function(arguments)
    
    def _retry_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: POPSSMCPExecutionResult
    ) -> POPSSMCPExecutionResult:
        attempt = result.metadata.get('attempt', 0) + 1
        result.metadata['attempt'] = attempt
        
        delay = self.retry_delay * (2 ** (attempt - 1))
        
        self._LOG.info(f"Retrying execution {result.execution_id} (attempt {attempt}) after {delay}s delay")
        
        time.sleep(delay)
        
        if result.mode == POPSSMCPExecutionMode.THREADED:
            return self._execute_threaded(tool_name, arguments, result)
        else:
            return self._execute_sync(tool_name, arguments, result)
    
    def _get_tool_function(self, tool_name: str) -> Optional[Callable]:
        from .registry import POPSSMCPToolRegistry
        registry = POPSSMCPToolRegistry()
        return registry.tool_functions.get(tool_name)
    
    def _get_async_tool_function(self, tool_name: str) -> Optional[Callable]:
        return None
    
    def _update_metrics(self, result: POPSSMCPExecutionResult, success: bool):
        with self._lock:
            self.metrics.total_executions += 1
            self.metrics.last_execution_time = datetime.now()
            
            if success:
                self.metrics.successful_executions += 1
            else:
                self.metrics.failed_executions += 1
            
            mode_key = result.mode.value
            self.metrics.mode_distribution[mode_key] = self.metrics.mode_distribution.get(mode_key, 0) + 1
            
            tool_key = result.tool_name
            self.metrics.tool_execution_counts[tool_key] = self.metrics.tool_execution_counts.get(tool_key, 0) + 1
            
            if not success and result.error:
                error_type = result.error.split(':')[0] if ':' in result.error else result.error
                self.metrics.error_counts[error_type] = self.metrics.error_counts.get(error_type, 0) + 1
            
            if self.metrics.total_executions > 0:
                self.metrics.average_execution_time = (
                    self.metrics.total_execution_time / self.metrics.total_executions
                )
    
    def _handle_result(self, result: POPSSMCPExecutionResult):
        handler = self._result_handlers.get(result.tool_name)
        if handler:
            try:
                handler(result)
            except Exception as e:
                self._LOG.error(f"Error in result handler for {result.tool_name}: {e}")
    
    def _handle_error(self, result: POPSSMCPExecutionResult):
        handler = self._error_handlers.get(result.tool_name)
        if handler:
            try:
                handler(result)
            except Exception as e:
                self._LOG.error(f"Error in error handler for {result.tool_name}: {e}")
    
    def batch_execute(
        self,
        executions: List[Dict[str, Any]],
        mode: POPSSMCPExecutionMode = POPSSMCPExecutionMode.THREADED,
        timeout: Optional[float] = None
    ) -> List[POPSSMCPExecutionResult]:
        results = []
        
        for exec_config in executions:
            tool_name = exec_config.get('tool_name')
            arguments = exec_config.get('arguments', {})
            
            result = self.execute(
                tool_name=tool_name,
                arguments=arguments,
                mode=mode,
                timeout=timeout,
                **exec_config
            )
            results.append(result)
        
        return results
    
    def execute_parallel(
        self,
        executions: List[Dict[str, Any]],
        max_concurrent: int = 10,
        timeout: Optional[float] = None
    ) -> List[POPSSMCPExecutionResult]:
        semaphore = threading.Semaphore(max_concurrent)
        results: List[POPSSMCPExecutionResult] = []
        lock = threading.Lock()
        
        def execute_one(exec_config: Dict[str, Any]):
            with semaphore:
                result = self.execute(
                    tool_name=exec_config.get('tool_name'),
                    arguments=exec_config.get('arguments', {}),
                    mode=POPSSMCPExecutionMode.THREADED,
                    timeout=timeout,
                    **exec_config
                )
                with lock:
                    results.append(result)
        
        threads = []
        for exec_config in executions:
            t = threading.Thread(target=execute_one, args=(exec_config,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        return sorted(results, key=lambda x: x.start_time)
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    'execution_id': exec_id,
                    'tool_name': result.tool_name,
                    'mode': result.mode.value,
                    'status': 'running' if result.end_time is None else 'completed',
                    'start_time': result.start_time.isoformat(),
                    'duration': (datetime.now() - result.start_time).total_seconds()
                }
                for exec_id, result in self.active_executions.items()
            ]
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            recent = list(self.execution_history)[-limit:]
            return [
                {
                    'execution_id': result.execution_id,
                    'tool_name': result.tool_name,
                    'mode': result.mode.value,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat() if result.end_time else None
                }
                for result in recent
            ]
    
    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_executions': self.metrics.total_executions,
                'successful_executions': self.metrics.successful_executions,
                'failed_executions': self.metrics.failed_executions,
                'success_rate': (
                    self.metrics.successful_executions / max(self.metrics.total_executions, 1)
                ),
                'total_execution_time': self.metrics.total_execution_time,
                'average_execution_time': self.metrics.average_execution_time,
                'mode_distribution': self.metrics.mode_distribution,
                'tool_execution_counts': self.metrics.tool_execution_counts,
                'error_counts': self.metrics.error_counts,
                'queue_size': self.pending_executions.qsize(),
                'active_count': len(self.active_executions)
            }
    
    def clear_history(self):
        with self._lock:
            self.execution_history.clear()
            self._LOG.info("Execution history cleared")
    
    def shutdown(self):
        self._LOG.info("Shutting down execution manager")
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        self._stop_event.set()
        
        self._LOG.info("Execution manager shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
