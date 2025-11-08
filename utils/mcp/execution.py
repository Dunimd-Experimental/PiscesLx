#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
import time
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum

try:
    from utils.log.core import PiscesLxCoreLog
    logger = PiscesLxCoreLog("Arctic.Utils.MCP.Execution")
except ImportError:
    # Fallback to simple logger if utils.log.core is not available
    import logging
    logger = logging.getLogger("Arctic.Utils.MCP.Execution")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class PiscesLxCoreMCPExecutionMode(Enum):
    """Execution mode for tools."""
    SYNC = "sync"
    ASYNC = "async"
    THREADED = "threaded"
    REMOTE = "remote"


class PiscesLxCoreMCPExecutionStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class PiscesLxCoreMCPExecutionResult:
    """Result of tool execution."""
    success: bool
    result: Any
    execution_time: float
    status: PiscesLxCoreMCPExecutionStatus
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    mode: PiscesLxCoreMCPExecutionMode = PiscesLxCoreMCPExecutionMode.SYNC


@dataclass
class PiscesLxCoreMCPExecutionConfig:
    """Configuration for tool execution."""
    timeout: float = 30.0
    max_workers: int = 4
    retry_count: int = 1
    retry_delay: float = 0.5
    enable_cancellation: bool = True
    enable_timeout: bool = True


class PiscesLxCoreMCPExecutionManager:
    """Unified execution manager for MCP tools."""
    
    def __init__(self, config: Optional[PiscesLxCoreMCPExecutionConfig] = None):
        """Initialize the execution manager."""
        self.config = config or PiscesLxCoreMCPExecutionConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.active_executions: Dict[str, asyncio.Task] = {}
        self._shutdown = False
        self._lock = threading.Lock()
        
        logger.info(f"MCPExecutionManager initialized with {self.config.max_workers} workers")
    
    async def execute_tool(
        self,
        tool_func: Callable,
        parameters: Dict[str, Any],
        execution_id: str,
        mode: PiscesLxCoreMCPExecutionMode = PiscesLxCoreMCPExecutionMode.SYNC,
        timeout: Optional[float] = None
    ) -> PiscesLxCoreMCPExecutionResult:
        """
        Execute a tool with unified interface.
        
        Args:
            tool_func: Tool function to execute
            parameters: Parameters for the tool
            execution_id: Unique execution identifier
            mode: Execution mode
            timeout: Optional timeout override
            
        Returns:
            Execution result
        """
        if self._shutdown:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.CANCELLED,
                error_message="Execution manager is shutdown",
                error_code="MANAGER_SHUTDOWN",
                mode=mode
            )
        
        timeout = timeout or self.config.timeout
        start_time = time.time()
        
        try:
            if mode == PiscesLxCoreMCPExecutionMode.ASYNC:
                result = await self._execute_async(tool_func, parameters, execution_id, timeout)
            elif mode == PiscesLxCoreMCPExecutionMode.THREADED:
                result = await self._execute_threaded(tool_func, parameters, execution_id, timeout)
            elif mode == PiscesLxCoreMCPExecutionMode.REMOTE:
                result = await self._execute_remote(tool_func, parameters, execution_id, timeout)
            else:  # SYNC
                result = await self._execute_sync(tool_func, parameters, execution_id, timeout)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Execution {execution_id} failed: {e}")
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                mode=mode
            )
    
    async def _execute_sync(
        self,
        tool_func: Callable,
        parameters: Dict[str, Any],
        execution_id: str,
        timeout: float
    ) -> PiscesLxCoreMCPExecutionResult:
        """Execute tool synchronously."""
        try:
            if asyncio.iscoroutinefunction(tool_func):
                # Function is async, run it in executor to make it sync
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: asyncio.run(tool_func(**parameters))
                )
            else:
                # Function is sync, run it directly
                result = tool_func(**parameters)
            
            return PiscesLxCoreMCPExecutionResult(
                success=True,
                result=result,
                execution_time=0.0,  # Will be set by caller
                status=PiscesLxCoreMCPExecutionStatus.COMPLETED,
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
            
        except Exception as e:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="SYNC_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.SYNC
            )
    
    async def _execute_async(
        self,
        tool_func: Callable,
        parameters: Dict[str, Any],
        execution_id: str,
        timeout: float
    ) -> PiscesLxCoreMCPExecutionResult:
        """Execute tool asynchronously."""
        try:
            if not asyncio.iscoroutinefunction(tool_func):
                return PiscesLxCoreMCPExecutionResult(
                    success=False,
                    result=None,
                    execution_time=0.0,
                    status=PiscesLxCoreMCPExecutionStatus.FAILED,
                    error_message="Tool function is not async",
                    error_code="ASYNC_MISMATCH",
                    mode=PiscesLxCoreMCPExecutionMode.ASYNC
                )
            
            # Create task for execution
            task = asyncio.create_task(tool_func(**parameters))
            
            with self._lock:
                self.active_executions[execution_id] = task
            
            try:
                if self.config.enable_timeout:
                    result = await asyncio.wait_for(task, timeout=timeout)
                else:
                    result = await task
                
                return PiscesLxCoreMCPExecutionResult(
                    success=True,
                    result=result,
                    execution_time=0.0,
                    status=PiscesLxCoreMCPExecutionStatus.COMPLETED,
                    mode=PiscesLxCoreMCPExecutionMode.ASYNC
                )
                
            finally:
                with self._lock:
                    self.active_executions.pop(execution_id, None)
                
        except asyncio.TimeoutError:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=timeout,
                status=PiscesLxCoreMCPExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {timeout}s",
                error_code="TIMEOUT",
                mode=PiscesLxCoreMCPExecutionMode.ASYNC
            )
        except Exception as e:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="ASYNC_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.ASYNC
            )
    
    async def _execute_threaded(
        self,
        tool_func: Callable,
        parameters: Dict[str, Any],
        execution_id: str,
        timeout: float
    ) -> PiscesLxCoreMCPExecutionResult:
        """Execute tool in a separate thread."""
        try:
            loop = asyncio.get_event_loop()
            
            def run_in_thread():
                return tool_func(**parameters)
            
            # Submit to thread pool
            future = self.executor.submit(run_in_thread)
            
            with self._lock:
                self.active_executions[execution_id] = future
            
            try:
                if self.config.enable_timeout:
                    result = await loop.run_in_executor(None, future.result, timeout)
                else:
                    result = await loop.run_in_executor(None, future.result)
                
                return PiscesLxCoreMCPExecutionResult(
                    success=True,
                    result=result,
                    execution_time=0.0,
                    status=PiscesLxCoreMCPExecutionStatus.COMPLETED,
                    mode=PiscesLxCoreMCPExecutionMode.THREADED
                )
                
            finally:
                with self._lock:
                    self.active_executions.pop(execution_id, None)
                
        except FutureTimeoutError:
            future.cancel()
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=timeout,
                status=PiscesLxCoreMCPExecutionStatus.TIMEOUT,
                error_message=f"Threaded execution timed out after {timeout}s",
                error_code="THREAD_TIMEOUT",
                mode=PiscesLxCoreMCPExecutionMode.THREADED
            )
        except Exception as e:
            return PiscesLxCoreMCPExecutionResult(
                success=False,
                result=None,
                execution_time=0.0,
                status=PiscesLxCoreMCPExecutionStatus.FAILED,
                error_message=str(e),
                error_code="THREAD_EXECUTION_ERROR",
                mode=PiscesLxCoreMCPExecutionMode.THREADED
            )
    
    async def _execute_remote(
        self,
        tool_func: Callable,
        parameters: Dict[str, Any],
        execution_id: str,
        timeout: float
    ) -> PiscesLxCoreMCPExecutionResult:
        """Execute tool remotely (placeholder for remote execution)."""
        # This will be implemented when we integrate remote functionality
        return PiscesLxCoreMCPExecutionResult(
            success=False,
            result=None,
            execution_time=0.0,
            status=PiscesLxCoreMCPExecutionStatus.FAILED,
            error_message="Remote execution not yet implemented",
            error_code="REMOTE_NOT_IMPLEMENTED",
            mode=PiscesLxCoreMCPExecutionMode.REMOTE
        )
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: Execution identifier to cancel
            
        Returns:
            True if cancellation was successful
        """
        with self._lock:
            task = self.active_executions.get(execution_id)
            if task:
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
                    return True
                elif hasattr(task, 'cancel'):
                    task.cancel()
                    return True
        
        return False
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        with self._lock:
            return list(self.active_executions.keys())
    
    async def shutdown(self):
        """Shutdown the execution manager."""
        self._shutdown = True
        
        # Cancel all active executions
        with self._lock:
            for execution_id, task in list(self.active_executions.items()):
                await self.cancel_execution(execution_id)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("MCPExecutionManager shutdown completed")


# Global execution manager instance
_execution_manager: Optional[PiscesLxCoreMCPExecutionManager] = None


def get_execution_manager(config: Optional[PiscesLxCoreMCPExecutionConfig] = None) -> PiscesLxCoreMCPExecutionManager:
    """Get the global execution manager instance."""
    global _execution_manager
    if _execution_manager is None:
        _execution_manager = PiscesLxCoreMCPExecutionManager(config)
    return _execution_manager


def execute_tool_sync(
    tool_func: Callable,
    parameters: Dict[str, Any],
    timeout: float = 30.0
) -> PiscesLxCoreMCPExecutionResult:
    """
    Convenience function for synchronous tool execution.
    
    Args:
        tool_func: Tool function to execute
        parameters: Parameters for the tool
        timeout: Execution timeout
        
    Returns:
        Execution result
    """
    manager = get_execution_manager()
    
    async def _execute():
        return await manager.execute_tool(
            tool_func=tool_func,
            parameters=parameters,
            execution_id=f"sync_{int(time.time() * 1000000)}",
            mode=PiscesLxCoreMCPExecutionMode.SYNC,
            timeout=timeout
        )
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_execute())


async def execute_tool_async(
    tool_func: Callable,
    parameters: Dict[str, Any],
    execution_id: str,
    timeout: float = 30.0
) -> PiscesLxCoreMCPExecutionResult:
    """
    Convenience function for asynchronous tool execution.
    
    Args:
        tool_func: Tool function to execute
        parameters: Parameters for the tool
        execution_id: Unique execution identifier
        timeout: Execution timeout
        
    Returns:
        Execution result
    """
    manager = get_execution_manager()
    return await manager.execute_tool(
        tool_func=tool_func,
        parameters=parameters,
        execution_id=execution_id,
        mode=PiscesLxCoreMCPExecutionMode.ASYNC,
        timeout=timeout
    )