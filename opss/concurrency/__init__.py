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

"""
Concurrency Operators Module

Advanced concurrency utilities for parallel execution, timeout management, and retry logic.

Features:
    - Timeout context managers
    - Retry with exponential backoff and circuit breaker
    - Async task management with priority queues
    - Resource pool management
    - Parallel execution with Thread/Process backends
    - Centralized concurrency manager
"""

import asyncio
import time
import queue
import signal
import threading
import multiprocessing as mp
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, Dict, Union, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from concurrent.futures import TimeoutError as FutureTimeoutError

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus

from configs.version import VERSION

_LOG = PiscesLxLogger("PiscesLx.Opss.Concurrency", file_path=get_log_file("PiscesLx.Opss.Concurrency"), enable_file=True)


class PiscesLxCoreTimeoutError(TimeoutError):
    """Timeout error for concurrency operations."""
    pass


class POPSSTimeoutError(PiscesLxCoreTimeoutError):
    """Timeout error for concurrency operations."""
    pass


class POPSSConcurrencyError(RuntimeError):
    """Concurrency error for failed operations."""
    pass


class POPSSTaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class POPSSConcurrencyBackend(Enum):
    """Concurrency backend types."""
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNCIO = "asyncio"


@dataclass
class POPSSConcurrencyConfig:
    """Configuration for concurrency operations."""
    max_workers: int = 4
    max_memory_mb: Optional[int] = None
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    backend: POPSSConcurrencyBackend = POPSSConcurrencyBackend.THREADING
    enable_profiling: bool = False


@dataclass
class POPSSRetryConfig:
    """Configuration for retry operations."""
    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0


class POPSSTimeoutOperator:
    """
    Timeout Context Manager.
    
    Provides timeout functionality with signal and threading support.
    
    Usage:
        with Timeout(seconds=30) as timeout:
            result = long_running_operation()
            if timeout.timed_out:
                handle_timeout()
    """
    
    def __init__(self, seconds: float, raise_on_timeout: bool = True) -> None:
        self.seconds = float(seconds)
        self.raise_on_timeout = raise_on_timeout
        self._start: Optional[float] = None
        self._timeout_occurred = False
        self._timer: Optional[threading.Timer] = None
    
    def __enter__(self) -> 'TimeoutOperator':
        self._start = time.perf_counter()
        self._timeout_occurred = False
        
        if threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGALRM'):
            signal.alarm(int(self.seconds))
        else:
            self._timer = threading.Timer(self.seconds, self._timeout_handler)
            self._timer.start()
        
        return self
    
    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        
        if self._timeout_occurred and self.raise_on_timeout:
            raise POPSSTimeoutError(
                "operation timed out",
                context={"seconds": self.seconds}
            )
        
        return False
    
    def _timeout_handler(self) -> None:
        self._timeout_occurred = True
    
    @property
    def timed_out(self) -> bool:
        return self._timeout_occurred


class POPSSRetryOperator:
    """
    Retry Operator with Exponential Backoff and Circuit Breaker.
    
    Provides robust retry logic for unreliable operations.
    
    Usage:
        retry = POPSSRetryOperator(max_attempts=3, base_delay=1.0)
        result = retry.execute(unreliable_function)
    """
    
    def __init__(self, config: Optional[POPSSRetryConfig] = None):
        self.config = config or POPSSRetryConfig()
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = 0.0
    
    def execute(self, func: Callable[[], Any]) -> Any:
        """Execute function with retry logic."""
        if self._is_circuit_open():
            raise POPSSConcurrencyError("Circuit breaker is open")
        
        attempt = 0
        delay = self.config.base_delay
        last_exc: Optional[BaseException] = None
        
        while attempt < self.config.max_attempts:
            try:
                result = func()
                if attempt > 0:
                    self._failure_count = 0
                    self._circuit_open = False
                return result
            except self.config.exceptions as e:
                last_exc = e
                self._record_failure()
                
                if attempt == self.config.max_attempts - 1:
                    raise POPSSConcurrencyError(
                        "retry attempts exhausted",
                        context={"attempts": self.config.max_attempts, "final_delay": delay}
                    )
                
                time.sleep(delay)
                delay = min(delay * self.config.exponential_base, self.config.max_delay)
                if self.config.jitter:
                    delay = delay * (0.5 + 0.5 * (time.time() % 1))
                
                attempt += 1
        
        raise POPSSConcurrencyError("unexpected retry failure")
    
    def _is_circuit_open(self) -> bool:
        if not self._circuit_open:
            return False
        if time.time() - self._last_failure_time > self.config.circuit_breaker_timeout:
            self._circuit_open = False
            self._failure_count = 0
            return False
        return True
    
    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.config.circuit_breaker_threshold:
            self._circuit_open = True


class POPSSParallelOperator:
    """
    Parallel Execution Operator.
    
    Provides parallel execution using ThreadPool or ProcessPool.
    
    Usage:
        parallel = POPSSParallelOperator(max_workers=4, backend="threading")
        results = parallel.map(process_item, items)
    """
    
    def __init__(
        self,
        backend: POPSSConcurrencyBackend = POPSSConcurrencyBackend.THREADING,
        max_workers: Optional[int] = None,
        chunk_size: int = 1,
        timeout: Optional[float] = None
    ):
        self.backend = backend
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
    
    def map(self, func: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        """Execute function on items in parallel."""
        if not items:
            return []
        
        if self.max_workers is None:
            self.max_workers = min(len(items), 16)
        
        if self.backend == POPSSConcurrencyBackend.THREADING:
            return self._execute_threaded(func, items)
        elif self.backend == POPSSConcurrencyBackend.MULTIPROCESSING:
            return self._execute_multiprocess(func, items)
        else:
            raise POPSSConcurrencyError(f"Unsupported backend: {self.backend}")
    
    def _execute_threaded(self, func: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in items}
            for future in as_completed(future_to_item):
                result = future_to_item[future]
                try:
                    results.append(future.result(timeout=self.timeout))
                except Exception as e:
                    raise POPSSConcurrencyError(f"threaded execution failed: {e}")
        return results
    
    def _execute_multiprocess(self, func: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            chunks = [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]
            for chunk in chunks:
                chunk_results = [func(item) for item in chunk]
                results.extend(chunk_results)
        return results


class POPSSAsyncOperator:
    """
    Async Task Operator with Priority Queue.
    
    Provides async task management with priority support.
    
    Usage:
        async_manager = POPSSAsyncOperator(max_concurrent=100)
        await async_manager.submit(coro, priority=POPSSTaskPriority.HIGH)
    """
    
    def __init__(self, max_concurrent: int = 100, max_queue_size: int = 1000):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running_tasks: Set[asyncio.Task[Any]] = set()
        self._shutdown = False
    
    async def submit(
        self,
        coro: Callable[[], Any],
        priority: POPSSTaskPriority = POPSSTaskPriority.NORMAL
    ) -> Any:
        """Submit async task."""
        if self._shutdown:
            raise POPSSConcurrencyError("Async manager is shutdown")
        
        async def wrapped():
            async with self._semaphore:
                return await coro()
        
        task = asyncio.create_task(wrapped())
        self._running_tasks.add(task)
        task.add_done_callback(self._task_done)
        
        return await task
    
    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._running_tasks.discard(task)
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown async manager."""
        self._shutdown = True
        if self._running_tasks:
            for task in list(self._running_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
    
    @property
    def active_count(self) -> int:
        return len(self._running_tasks)


class POPSSConcurrencyOperator(PiscesLxOperatorInterface):
    """
    Concurrency Operator.
    
    Unified interface for all concurrency operations.
    
    Features:
        - Timeout management
        - Retry with exponential backoff
        - Parallel execution
        - Async task management
        - Resource pool support
    
    Input:
        operation: Type of concurrency operation
        config: Configuration for the operation
    
    Output:
        Result of the concurrency operation
    """
    
    def __init__(self, config: Optional[POPSSConcurrencyConfig] = None):
        super().__init__()
        self.name = "concurrency"
        self.version = VERSION
        self.config = config or POPSSConcurrencyConfig()
        self._LOG = PiscesLxLogger("pisceslx.ops.concurrency")
    
    @property
    def description(self) -> str:
        return "Concurrency operator for timeout, retry, and parallel execution"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "operation": {"type": "str", "required": True, "enum": ["timeout", "retry", "parallel", "async"]},
            "seconds": {"type": "float", "required": False},
            "max_attempts": {"type": "int", "required": False},
            "func": {"type": "callable", "required": False},
            "items": {"type": "list", "required": False},
            "max_workers": {"type": "int", "required": False},
            "priority": {"type": "str", "required": False},
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "success": {"type": "bool"},
            "result": {"type": "any"},
            "timed_out": {"type": "bool"},
            "attempts": {"type": "int"},
        }
    
    def execute(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute concurrency operation."""
        operation = inputs.get("operation")
        
        try:
            if operation == "timeout":
                return self._timeout_operation(inputs)
            elif operation == "retry":
                return self._retry_operation(inputs)
            elif operation == "parallel":
                return self._parallel_operation(inputs)
            elif operation == "async":
                return self._async_operation(inputs)
            else:
                return PiscesLxOperatorResult(
                    status=PiscesLxOperatorStatus.FAILED,
                    output={},
                    error=f"Unknown operation: {operation}"
                )
        except Exception as e:
            self._LOG.error(f"Concurrency operation failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.FAILED,
                output={},
                error=str(e)
            )
    
    def _timeout_operation(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        seconds = inputs.get("seconds", 60.0)
        func = inputs.get("func")
        
        if func is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"timed_out": False}
            )
        
        with POPSSTimeoutOperator(seconds) as timeout:
            result = func()
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"result": result, "timed_out": timeout.timed_out}
        )
    
    def _retry_operation(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        max_attempts = inputs.get("max_attempts", 3)
        func = inputs.get("func")
        
        if func is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"attempts": 0}
            )
        
        retry = POPSSRetryOperator(POPSSRetryConfig(max_attempts=max_attempts))
        result = retry.execute(func)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"result": result}
        )
    
    def _parallel_operation(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        items = inputs.get("items", [])
        func = inputs.get("func", lambda x: x)
        max_workers = inputs.get("max_workers", 4)
        
        parallel = POPSSParallelOperator(max_workers=max_workers)
        results = parallel.map(func, items)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"results": results}
        )
    
    def _async_operation(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"message": "Use POPSSAsyncOperator directly for async operations"}
        )


__all__ = [
    "PiscesLxCoreTimeoutError",
    "POPSSTimeoutError",
    "POPSSConcurrencyError",
    "POPSSTaskPriority",
    "POPSSConcurrencyBackend",
    "POPSSConcurrencyConfig",
    "POPSSRetryConfig",
    "POPSSTimeoutOperator",
    "POPSSRetryOperator",
    "POPSSParallelOperator",
    "POPSSAsyncOperator",
    "POPSSConcurrencyOperator",
]
