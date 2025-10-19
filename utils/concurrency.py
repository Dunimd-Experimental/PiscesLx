#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

import time
import queue
import signal
import asyncio
import threading
from enum import Enum
import multiprocessing as mp
from dataclasses import dataclass
from contextlib import contextmanager
from utils.log.core import PiscesLxCoreLog
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, Dict, Union, Set
from utils.error import PiscesLxCoreTimeoutError, PiscesLxCoreConcurrencyError, PiscesLxCoreMemoryError

logger = PiscesLxCoreLog("PiscesLx.Utils.Concurrency")

class _TaskPriority(Enum):
    """Enumeration representing different task priority levels.
    
    Values are ordered from highest to lowest priority.
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class _ConcurrencyBackend(Enum):
    """Enumeration representing different concurrency backend types."""
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNCIO = "asyncio"

@dataclass
class TaskResult:
    """Data class representing the result of a task execution.

    Attributes:
        task_id (str): Unique identifier for the task.
        success (bool): Indicates whether the task execution was successful.
        result (Any, optional): The result of the task if successful. Defaults to None.
        error (Optional[Exception], optional): The exception raised if the task failed. Defaults to None.
        execution_time (float, optional): The time taken to execute the task in seconds. Defaults to 0.0.
        memory_usage_mb (float, optional): The memory usage during task execution in megabytes. Defaults to 0.0.
        start_time (float, optional): The timestamp when the task started. Defaults to 0.0.
        end_time (float, optional): The timestamp when the task ended. Defaults to 0.0.
    """
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

@dataclass
class ConcurrencyConfig:
    """Data class representing the configuration for concurrency operations.

    Attributes:
        max_workers (int, optional): The maximum number of workers to use. Defaults to 4.
        max_memory_mb (Optional[int], optional): The maximum memory usage in megabytes. Defaults to None.
        timeout_seconds (float, optional): The overall timeout in seconds. Defaults to 300.0.
        retry_attempts (int, optional): The number of retry attempts for failed tasks. Defaults to 3.
        retry_delay (float, optional): The delay between retry attempts in seconds. Defaults to 1.0.
        backend (_ConcurrencyBackend, optional): The concurrency backend to use. Defaults to _ConcurrencyBackend.THREADING.
        enable_profiling (bool, optional): Whether to enable profiling. Defaults to False.
        enable_memory_monitoring (bool, optional): Whether to enable memory monitoring. Defaults to False.
        task_timeout (float, optional): The timeout for individual tasks in seconds. Defaults to 60.0.
        queue_size (int, optional): The size of the task queue. Defaults to 1000.
        max_async_tasks (int, optional): The maximum number of asynchronous tasks. Defaults to 100.
        max_queue_size (int, optional): The maximum size of the task queue. Defaults to 1000.
        backend: _ConcurrencyBackend = _ConcurrencyBackend.THREADING
        default_priority: _TaskPriority = _TaskPriority.NORMAL
        default_max_workers (int, optional): The default maximum number of workers. Defaults to 4.
        default_timeout (float, optional): The default timeout for tasks in seconds. Defaults to 60.0.
        resource_pools (Dict[str, Dict[str, Any]], optional): The resource pools configuration. Defaults to None.
    """
    max_workers: int = 4
    max_memory_mb: Optional[int] = None
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    backend: _ConcurrencyBackend = _ConcurrencyBackend.THREADING
    enable_profiling: bool = False
    enable_memory_monitoring: bool = False
    task_timeout: float = 60.0
    queue_size: int = 1000
    max_async_tasks: int = 100
    max_queue_size: int = 1000
    default_priority: _TaskPriority = _TaskPriority.NORMAL
    default_max_workers: int = 4
    default_timeout: float = 60.0
    resource_pools: Dict[str, Dict[str, Any]] = None  # type: ignore

    def __post_init__(self):
        """Initialize additional attributes after the dataclass is created.
        
        Ensures that resource_pools is a dictionary and keeps max_queue_size consistent with queue_size 
        if max_queue_size is not explicitly set.
        """
        # Ensure resource_pools is a dict
        if self.resource_pools is None:
            self.resource_pools = {}
        # Keep max_queue_size consistent with queue_size if not explicitly set
        if not hasattr(self, "max_queue_size") or self.max_queue_size is None:
            self.max_queue_size = int(self.queue_size)
            
class PiscesLxCoreTimeout:
    """A context manager that provides timeout functionality with signal and threading support.
    
    This context manager can be used to limit the execution time of a block of code.
    It uses signals for the main thread and threading timers for other threads or platforms 
    that don't support signals.
    """

    def __init__(self, seconds: float, raise_on_timeout: bool = True) -> None:
        """Initialize the timeout context manager.

        Args:
            seconds (float): The timeout duration in seconds.
            raise_on_timeout (bool, optional): Whether to raise an exception when a timeout occurs. 
                                             Defaults to True.
        """
        self.seconds = float(seconds)
        self.raise_on_timeout = raise_on_timeout
        self._start: Optional[float] = None
        self._timeout_occurred = False
        self._original_handler = None
        self._timer = None

    def __enter__(self) -> "PiscesLxCoreTimeout":
        """Enter the context manager and start the timeout countdown.

        Uses signals for the main thread if available, otherwise uses a threading timer.

        Returns:
            PiscesLxCoreTimeout: The instance itself.
        """
        self._start = time.perf_counter()
        self._timeout_occurred = False
        
        # Use signals for main thread if SIGALRM is available
        if threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGALRM'):
            self._original_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(self.seconds))
        else:
            # Use threading timer for non-main threads or Windows
            self._timer = threading.Timer(self.seconds, self._timeout_handler_thread)
            self._timer.start()
        
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Exit the context manager and clean up resources.

        Args:
            exc_type: The type of exception raised, if any.
            exc: The exception instance, if any.
            tb: The traceback object, if any.

        Returns:
            bool: False to indicate not to suppress any exceptions.
        """
        # Clean up resources
        if self._original_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._original_handler)
        elif self._timer is not None:
            self._timer.cancel()
        
        if self._timeout_occurred and self.raise_on_timeout:
            raise PiscesLxCoreTimeoutError("operation timed out", context={"seconds": self.seconds})
        
        # Check elapsed time as a fallback mechanism
        if exc_type is None and self._start is not None:
            elapsed = time.perf_counter() - self._start
            if elapsed > self.seconds:
                if self.raise_on_timeout:
                    raise PiscesLxCoreTimeoutError(
                        "operation timed out", 
                        context={"seconds": self.seconds, "elapsed": elapsed}
                    )
                else:
                    self._timeout_occurred = True
        
        # Do not suppress exceptions
        return False
    
    def _timeout_handler(self, signum, frame):
        """Signal-based timeout handler.

        Args:
            signum: The signal number.
            frame: The current stack frame.
        """
        self._timeout_occurred = True
    
    def _timeout_handler_thread(self):
        """Threading-based timeout handler."""
        self._timeout_occurred = True
    
    @property
    def timed_out(self) -> bool:
        """Check if a timeout has occurred.

        Returns:
            bool: True if a timeout has occurred, False otherwise.
        """
        return self._timeout_occurred

class PiscesLxCoreRetry:
    """A retry helper that provides exponential backoff, jitter, and circuit breaker functionality.
    
    This class can be used to retry a function call multiple times when specific exceptions occur.
    It also includes a circuit breaker mechanism to prevent repeated failed attempts.
    """

    # use module-level logger (standardized)

    def __init__(
        self, 
        max_attempts: int = 3, 
        base_delay: float = 0.2, 
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 300.0
    ):
        """Initialize the retry helper.

        Args:
            max_attempts (int, optional): The maximum number of retry attempts. Defaults to 3.
            base_delay (float, optional): The base delay in seconds between retries. Defaults to 0.2.
            max_delay (float, optional): The maximum delay in seconds between retries. Defaults to 60.0.
            exponential_base (float, optional): The base for exponential backoff. Defaults to 2.0.
            jitter (bool, optional): Whether to add jitter to the delay. Defaults to True.
            exceptions (Tuple[Type[BaseException], ...], optional): The exceptions to catch and retry on. 
                                                                  Defaults to (Exception,).
            circuit_breaker_threshold (int, optional): The number of failures to trigger the circuit breaker. 
                                                    Defaults to 5.
            circuit_breaker_timeout (float, optional): The timeout in seconds for the circuit breaker. 
                                                     Defaults to 300.0.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._circuit_open = False

    def __call__(self, func: Callable[[], Any]) -> Any:
        """Execute the function with retry logic.

        Args:
            func (Callable[[], Any]): The function to execute.

        Returns:
            Any: The result of the function call.
        """
        return self.retry(func)

    def retry(self, func: Callable[[], Any]) -> Any:
        """Execute the function with retry logic.

        Args:
            func (Callable[[], Any]): The function to execute.

        Returns:
            Any: The result of the function call.

        Raises:
            PiscesLxCoreConcurrencyError: If the circuit breaker is open or all retry attempts are exhausted.
        """
        if self._is_circuit_open():
            raise PiscesLxCoreConcurrencyError(
                "Circuit breaker is open",
                context={
                    "failure_count": self._failure_count,
                    "timeout_remaining": self.circuit_breaker_timeout - (time.time() - self._last_failure_time)
                }
            )

        attempt = 0
        delay = self.base_delay
        last_exc: Optional[BaseException] = None
        
        while attempt < self.max_attempts:
            try:
                result = func()
                # If retry succeeded, reset failure count
                if attempt > 0:
                    self._failure_count = 0
                    self._circuit_open = False
                    try:
                        self._log.info(
                            "retry succeeded",
                            event="concurrency.retry.success",
                            attempts=attempt + 1,
                        )
                    except Exception as log_e:
                        logger.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                return result
                
            except self.exceptions as e:
                last_exc = e
                self._record_failure()
                
                if attempt == self.max_attempts - 1:
                    try:
                        self._log.error(
                            "retry attempts exhausted",
                            event="concurrency.retry.exhausted",
                            attempts=self.max_attempts,
                            final_delay=delay,
                            error=str(e),
                            error_class=type(e).__name__,
                            failure_count=self._failure_count,
                        )
                    except Exception as log_e:
                        _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                    raise PiscesLxCoreConcurrencyError(
                        "retry attempts exhausted",
                        context={
                            "attempts": self.max_attempts,
                            "final_delay": delay,
                            "failure_count": self._failure_count
                        },
                        cause=e,
                    )
                
                try:
                    self._log.warning(
                        "retrying after error",
                        event="concurrency.retry.attempt",
                        attempt=attempt + 1,
                        max_attempts=self.max_attempts,
                        delay_s=delay,
                        error=str(e),
                        error_class=type(e).__name__,
                        failure_count=self._failure_count,
                    )
                except Exception as log_e:
                    _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                
                time.sleep(delay)
                
                # Calculate next delay with exponential backoff and jitter
                delay = min(delay * self.exponential_base, self.max_delay)
                if self.jitter:
                    delay = delay * (0.5 + 0.5 * (time.time() % 1))  # Add jitter
                
                attempt += 1
        
        # This line should theoretically never be reached due to the raise above
        # But added to satisfy type checking
        raise PiscesLxCoreConcurrencyError(
            "retry attempts exhausted",
            context={
                "attempts": self.max_attempts,
                "final_delay": delay,
                "failure_count": self._failure_count
            },
            cause=last_exc
        )

    def _record_success(self) -> None:
        """
        Record a successful execution and update the circuit breaker state.
        
        Decrements the failure count and resets the circuit breaker if the failure count reaches zero.
        Logs the reset event if possible.
        """
        self._failure_count = max(0, self._failure_count - 1)
        if self._failure_count == 0:
            self._circuit_open = False
            try:
                self._log.info(
                    "circuit breaker reset",
                    event="concurrency.circuit_breaker.reset",
                    failure_count=self._failure_count,
                )
            except Exception as log_e:
                _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
    
    def _reset_circuit_breaker(self) -> None:
        """
        Manually reset the circuit breaker state.
        
        Resets the circuit breaker to its initial state by setting the failure count to zero,
        clearing the last failure time, and closing the circuit. Logs the manual reset event if possible.
        """
        self._circuit_open = False
        self._failure_count = 0
        self._last_failure_time = 0.0
        try:
            self._log.info(
                "circuit breaker manually reset",
                event="concurrency.circuit_breaker.manual_reset",
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

    def _record_failure(self) -> None:
        """
        Record a failure and update the circuit breaker state.
        
        Increments the failure count, records the current time as the last failure time,
        and opens the circuit breaker if the failure count reaches or exceeds the threshold.
        Logs the circuit opening event if possible.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.circuit_breaker_threshold:
            self._circuit_open = True
            try:
                self._log.error(
                    "circuit breaker opened",
                    event="concurrency.circuit_breaker.open",
                    failure_count=self._failure_count,
                    threshold=self.circuit_breaker_threshold,
                )
            except Exception as log_e:
                _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

    def _is_circuit_open(self) -> bool:
        """
        Check if the circuit breaker is open.
        
        If the circuit is open, checks if the circuit breaker timeout has expired.
        If the timeout has expired, resets the circuit breaker state.

        Returns:
            bool: True if the circuit breaker is open, False otherwise.
        """
        if not self._circuit_open:
            return False
            
        # Check if the circuit breaker timeout has expired
        if time.time() - self._last_failure_time > self.circuit_breaker_timeout:
            self._circuit_open = False
            self._failure_count = 0
            return False
            
        return True

    @staticmethod
    def retry_static(
        func: Callable[[], Any],
        retries: int = 3,
        backoff: float = 0.2,
        exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    ) -> Any:
        """
        Static retry method provided for backward compatibility.

        Args:
            func (Callable[[], Any]): The function to execute.
            retries (int, optional): The maximum number of retry attempts. Defaults to 3.
            backoff (float, optional): The base delay in seconds between retries. Defaults to 0.2.
            exceptions (Tuple[Type[BaseException], ...], optional): The exceptions to catch and retry on. 
                                                                  Defaults to (Exception,).

        Returns:
            Any: The result of the function call.
        """
        retry_instance = PiscesLxCoreRetry(
            max_attempts=retries,
            base_delay=backoff,
            exceptions=exceptions
        )
        return retry_instance.retry(func)

class PiscesLxCoreAsyncManager:
    """Advanced async task manager with priority queues and resource limits."""



    def __init__(self,
                 max_concurrent: int = 100,
                 max_queue_size: int = 1000,
                 default_priority: _TaskPriority = _TaskPriority.NORMAL):
        """
        Initialize the async task manager.

        Args:
            max_concurrent (int, optional): The maximum number of concurrent tasks. Defaults to 100.
            max_queue_size (int, optional): The maximum size of the task queue. Defaults to 1000.
            default_priority (_TaskPriority, optional): The default priority for tasks. Defaults to _TaskPriority.NORMAL.
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_priority = default_priority
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._task_queue: asyncio.PriorityQueue[Tuple[int, int, asyncio.Task[Any]]] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._running_tasks: Set[asyncio.Task[Any]] = set()
        self._task_counter = 0
        self._shutdown = False

    async def submit(self, 
                    coro: Callable[[], Any], 
                    priority: Optional[_TaskPriority] = None,
                    timeout: Optional[float] = None) -> asyncio.Task[Any]:
        """
        Submit an async task with a specified priority.

        Args:
            coro (Callable[[], Any]): The coroutine to execute.
            priority (Optional[_TaskPriority], optional): The priority of the task. Defaults to the default priority of the manager.
            timeout (Optional[float], optional): The timeout for the task in seconds. Defaults to None.

        Returns:
            asyncio.Task[Any]: The created async task.

        Raises:
            PiscesLxCoreConcurrencyError: If the async manager is shutdown.
        """
        if self._shutdown:
            raise PiscesLxCoreConcurrencyError("Async manager is shutdown")

        priority = priority or self.default_priority
        self._task_counter += 1

        try:
            logger.debug(
                "submitting async task",
                event="concurrency.async.submit",
                priority=priority.value,
                timeout=timeout,
                queue_size=self._task_queue.qsize(),
                running_tasks=len(self._running_tasks),
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        # Create task with semaphore protection
        async def _wrapped_coro():
            async with self._semaphore:
                if timeout:
                    return await asyncio.wait_for(coro(), timeout=timeout)
                else:
                    return await coro()

        task = asyncio.create_task(_wrapped_coro())
        
        # Add to priority queue (lower priority value = higher priority)
        await self._task_queue.put((priority.value, self._task_counter, task))
        self._running_tasks.add(task)

        # Setup cleanup
        task.add_done_callback(self._task_done_callback)

        return task

    def _task_done_callback(self, task: asyncio.Task[Any]) -> None:
        """
        Callback function executed when a task is completed.

        Removes the completed task from the running tasks set and logs the completion event if possible.

        Args:
            task (asyncio.Task[Any]): The completed task.
        """
        self._running_tasks.discard(task)
        try:
            logger.debug(
                "task completed",
                event="concurrency.async.completed",
                running_tasks=len(self._running_tasks),
                exception=task.exception() if task.exception() else None,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the async manager.

        Cancels all running tasks and waits for them to complete within the specified timeout.
        Logs the shutdown process and results if possible.

        Args:
            timeout (float, optional): The timeout for waiting tasks to complete in seconds. Defaults to 30.0.
        """
        self._shutdown = True
        
        try:
            logger.info(
                "shutting down async manager",
                event="concurrency.async.shutdown",
                running_tasks=len(self._running_tasks),
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        # Cancel remaining tasks
        if self._running_tasks:
            for task in list(self._running_tasks):
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete with timeout
            done, pending = await asyncio.wait(self._running_tasks, timeout=timeout)
            
            if pending:
                try:
                    logger.warning(
                        "some tasks did not complete during shutdown",
                        event="concurrency.async.shutdown_timeout",
                        pending_tasks=len(pending),
                    )
                except Exception as log_e:
                    _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                
                # Force cancel remaining tasks
                for task in pending:
                    task.cancel()
        
        try:
            logger.info(
                "async manager shutdown complete",
                event="concurrency.async.shutdown_complete",
            )
        except Exception as log_e:
            logger.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics of the async manager.

        Returns:
            Dict[str, Any]: A dictionary containing statistics such as the number of running tasks,
                           queue size, maximum concurrent tasks, maximum queue size, and shutdown status.
        """
        return {
            "running_tasks": len(self._running_tasks),
            "queue_size": self._task_queue.qsize(),
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "shutdown": self._shutdown,
        }

    @property
    def active_tasks(self) -> int:
        """
        Get the number of active tasks.

        Returns:
            int: The number of currently running tasks.
        """
        return len(self._running_tasks)

    @property
    def queue_size(self) -> int:
        """
        Get the current size of the task queue.

        Returns:
            int: The number of tasks in the queue.
        """
        return self._task_queue.qsize()

class PiscesLxCoreResourcePool:
    """Manages a pool of limited resources such as connections and file handles."""



    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 0,
        resource_timeout: float = 30.0,
        health_check: Optional[Callable[[Any], bool]] = None
    ):
        """Initialize the resource pool.

        Args:
            factory (Callable[[], Any]): A callable to create new resources.
            max_size (int, optional): The maximum number of resources the pool can hold. Defaults to 10.
            min_size (int, optional): The minimum number of resources to initialize the pool with. Defaults to 0.
            resource_timeout (float, optional): Timeout in seconds when acquiring a resource from the pool. Defaults to 30.0.
            health_check (Optional[Callable[[Any], bool]], optional): A callable to check resource health. Defaults to None.
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.resource_timeout = resource_timeout
        self.health_check = health_check
        self._pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_size)
        self._semaphore = asyncio.Semaphore(max_size)
        self._created_count = 0
        self._active_count = 0
        self._shutdown = False

    async def initialize(self) -> None:
        """Initialize the pool by creating the minimum number of resources."""
        try:
            logger.info(
                "Initializing resource pool",
                event="concurrency.pool.initialize",
                min_size=self.min_size,
                max_size=self.max_size,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        for _ in range(self.min_size):
            try:
                resource = await self._create_resource()
                await self._pool.put(resource)
            except Exception as e:
                try:
                    logger.error(
                        "Failed to create initial resource",
                        event="concurrency.pool.init_failed",
                        error=str(e),
                        error_class=type(e).__name__,
                    )
                except Exception as log_e:
                    _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

    async def _create_resource(self) -> Any:
        """Create a new resource.

        Raises:
            PiscesLxCoreConcurrencyError: If the maximum pool size is reached.

        Returns:
            Any: The newly created resource.
        """
        if self._created_count >= self.max_size:
            raise PiscesLxCoreConcurrencyError("Maximum pool size reached")

        resource = await asyncio.get_event_loop().run_in_executor(None, self.factory)
        self._created_count += 1

        try:
            logger.debug(
                "Created new resource",
                event="concurrency.pool.resource_created",
                total_created=self._created_count,
                active=self._active_count,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        return resource

    async def acquire(self) -> Any:
        """Acquire a resource from the pool.

        If the pool is empty, a new resource will be created. If a health check is provided,
        unhealthy resources will be destroyed and replaced.

        Raises:
            PiscesLxCoreConcurrencyError: If the resource pool is shutdown.

        Returns:
            Any: An available resource.
        """
        if self._shutdown:
            raise PiscesLxCoreConcurrencyError("Resource pool is shutdown")

        try:
            self._log.debug(
                "Acquiring resource",
                event="concurrency.pool.acquire",
                queue_size=self._pool.qsize(),
                active=self._active_count,
                created=self._created_count,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        async with self._semaphore:
            try:
                # Try to get a resource from the pool with a timeout
                resource = await asyncio.wait_for(self._pool.get(), timeout=self.resource_timeout)

                # Perform health check if the check function is provided
                if self.health_check and not self.health_check(resource):
                    try:
                        logger.warning(
                            "Resource failed health check, creating new one",
                            event="concurrency.pool.health_check_failed",
                        )
                    except Exception as log_e:
                        _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                    await self._destroy_resource(resource)
                    resource = await self._create_resource()

            except asyncio.TimeoutError:
                # Pool is empty, create a new resource
                resource = await self._create_resource()
            except asyncio.QueueEmpty:
                # Pool is empty, create a new resource
                resource = await self._create_resource()

            self._active_count += 1
            return resource

    async def release(self, resource: Any) -> None:
        """Release a resource back to the pool.

        If the pool is full or shutting down, the resource will be destroyed.

        Args:
            resource (Any): The resource to be released.
        """
        self._active_count -= 1

        try:
            self._log.debug(
                "Releasing resource",
                event="concurrency.pool.release",
                queue_size=self._pool.qsize(),
                active=self._active_count,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        if self._shutdown:
            await self._destroy_resource(resource)
            return

        try:
            # Try to put the resource back into the pool
            self._pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool is full, destroy the resource
            await self._destroy_resource(resource)

    async def _destroy_resource(self, resource: Any) -> None:
        """Destroy a resource by calling its close or disconnect method if available.

        Args:
            resource (Any): The resource to be destroyed.
        """
        try:
            if hasattr(resource, 'close'):
                await asyncio.get_event_loop().run_in_executor(None, resource.close)
            elif hasattr(resource, 'disconnect'):
                await asyncio.get_event_loop().run_in_executor(None, resource.disconnect)
        except Exception as e:
            try:
                self._log.warning(
                    "Error destroying resource",
                    event="concurrency.pool.destroy_error",
                    error=str(e),
                    error_class=type(e).__name__,
                )
            except Exception as log_e:
                _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
        finally:
            self._created_count -= 1

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the resource pool and destroy all resources in the pool.

        Args:
            timeout (float, optional): Not currently used in this implementation. Defaults to 30.0.
        """
        self._shutdown = True

        try:
            self._log.info(
                "Shutting down resource pool",
                event="concurrency.pool.shutdown",
                active=self._active_count,
                created=self._created_count,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        # Destroy all resources in the pool
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                await self._destroy_resource(resource)
            except asyncio.QueueEmpty:
                break

    @property
    def available(self) -> int:
        """Get the number of available resources in the pool.

        Returns:
            int: The number of available resources.
        """
        return self._pool.qsize()

    @property
    def active(self) -> int:
        """Get the number of active resources.

        Returns:
            int: The number of active resources.
        """
        return self._active_count


class PiscesLxCoreConcurrencyManager:
    """A centralized manager for all concurrency primitives."""



    def __init__(self, config: Optional['ConcurrencyConfig'] = None):
        """Initialize the concurrency manager.

        Args:
            config (Optional[ConcurrencyConfig], optional): Configuration for the concurrency manager.
                If None, a default configuration will be used. Defaults to None.
        """
        self.config = config or ConcurrencyConfig()
        self._async_manager: Optional[PiscesLxCoreAsyncManager] = None
        self._resource_pools: Dict[str, PiscesLxCoreResourcePool] = {}
        self._parallel_executors: Dict[str, 'PiscesLxCoreParallel'] = {}
        self._retry_handlers: Dict[str, PiscesLxCoreRetry] = {}
        self._timeout_handlers: Dict[str, PiscesLxCoreTimeout] = {}
        self._shutdown = False

    async def initialize(self) -> None:
        """Initialize the concurrency manager, including the async manager and resource pools."""
        try:
            self._log.info(
                "Initializing concurrency manager",
                event="concurrency.manager.initialize",
                config=self.config,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        # Initialize the async manager
        self._async_manager = PiscesLxCoreAsyncManager(
            max_concurrent=self.config.max_async_tasks,
            max_queue_size=self.config.max_queue_size,
            default_priority=self.config.default_priority
        )

        # Initialize resource pools based on the configuration
        for pool_name, pool_config in self.config.resource_pools.items():
            pool = PiscesLxCoreResourcePool(
                factory=pool_config['factory'],
                max_size=pool_config.get('max_size', 10),
                min_size=pool_config.get('min_size', 0),
                resource_timeout=pool_config.get('timeout', 30.0),
                health_check=pool_config.get('health_check')
            )
            await pool.initialize()
            self._resource_pools[pool_name] = pool

    def get_async_manager(self) -> PiscesLxCoreAsyncManager:
        """Get the async task manager.

        Raises:
            PiscesLxCoreConcurrencyError: If the async manager is not initialized.

        Returns:
            PiscesLxCoreAsyncManager: The async task manager instance.
        """
        if not self._async_manager:
            raise PiscesLxCoreConcurrencyError("Async manager not initialized")
        return self._async_manager

    def get_resource_pool(self, name: str) -> PiscesLxCoreResourcePool:
        """Get a resource pool by its name.

        Args:
            name (str): The name of the resource pool.

        Raises:
            PiscesLxCoreConcurrencyError: If the specified resource pool is not found.

        Returns:
            PiscesLxCoreResourcePool: The resource pool instance.
        """
        if name not in self._resource_pools:
            raise PiscesLxCoreConcurrencyError(f"Resource pool '{name}' not found")
        return self._resource_pools[name]

    def create_parallel_executor(
        self,
        name: str,
        backend: _ConcurrencyBackend = _ConcurrencyBackend.THREADING,
        max_workers: Optional[int] = None,
        **kwargs
    ) -> 'PiscesLxCoreParallel':
        """Create a parallel executor and store it in the manager.

        Args:
            name (str): The name to identify the parallel executor.
            backend (_ConcurrencyBackend, optional): The concurrency backend to use. Defaults to _ConcurrencyBackend.THREADING.
            max_workers (Optional[int], optional): The maximum number of workers. If None, use the default value from config. Defaults to None.
            **kwargs: Additional arguments to pass to the parallel executor.

        Returns:
            PiscesLxCoreParallel: The newly created parallel executor.
        """
        executor = PiscesLxCoreParallel(
            backend=backend,
            max_workers=max_workers or self.config.default_max_workers,
            timeout=kwargs.get('timeout', self.config.default_timeout),
            error_handler=kwargs.get('error_handler'),
            result_processor=kwargs.get('result_processor')
        )
        self._parallel_executors[name] = executor
        return executor

    def create_retry_handler(
        self,
        name: str,
        max_attempts: int = 3,
        base_delay: float = 0.2,
        **kwargs
    ) -> PiscesLxCoreRetry:
        """Create a retry handler and store it in the manager.

        Args:
            name (str): The name to identify the retry handler.
            max_attempts (int, optional): The maximum number of retry attempts. Defaults to 3.
            base_delay (float, optional): The base delay between retries in seconds. Defaults to 0.2.
            **kwargs: Additional arguments to pass to the retry handler.

        Returns:
            PiscesLxCoreRetry: The newly created retry handler.
        """
        retry_handler = PiscesLxCoreRetry(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=kwargs.get('max_delay', 60.0),
            exponential_base=kwargs.get('exponential_base', 2.0),
            jitter=kwargs.get('jitter', True),
            exceptions=kwargs.get('exceptions', (Exception,)),
            circuit_breaker_threshold=kwargs.get('circuit_breaker_threshold', 5),
            circuit_breaker_timeout=kwargs.get('circuit_breaker_timeout', 300.0)
        )
        self._retry_handlers[name] = retry_handler
        return retry_handler

    def create_timeout_handler(
        self,
        name: str,
        timeout: float,
        **kwargs
    ) -> PiscesLxCoreTimeout:
        """Create a timeout handler and store it in the manager.

        Args:
            name (str): The name to identify the timeout handler.
            timeout (float): The timeout duration in seconds.
            **kwargs: Additional arguments to pass to the timeout handler.

        Returns:
            PiscesLxCoreTimeout: The newly created timeout handler.
        """
        timeout_handler = PiscesLxCoreTimeout(
            timeout=timeout,
            raise_on_timeout=kwargs.get('raise_on_timeout', True)
        )
        self._timeout_handlers[name] = timeout_handler
        return timeout_handler

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the concurrency manager and all its components.

        Args:
            timeout (float, optional): Timeout for the shutdown process. Defaults to 30.0.
        """
        self._shutdown = True

        try:
            self._log.info(
                "Shutting down concurrency manager",
                event="concurrency.manager.shutdown",
                resource_pools=len(self._resource_pools),
                executors=len(self._parallel_executors),
                retry_handlers=len(self._retry_handlers),
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        # Shutdown the async manager
        if self._async_manager:
            await self._async_manager.shutdown(timeout)

        # Shutdown all resource pools
        shutdown_tasks = []
        for pool in self._resource_pools.values():
            shutdown_tasks.append(pool.shutdown(timeout))

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Clear all handlers
        self._resource_pools.clear()
        self._parallel_executors.clear()
        self._retry_handlers.clear()
        self._timeout_handlers.clear()

    def __enter__(self):
        """Enter the context manager.

        Returns:
            PiscesLxCoreConcurrencyManager: The instance itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and shutdown the concurrency manager.

        Args:
            exc_type: The type of exception raised, if any.
            exc_val: The exception instance, if any.
            exc_tb: The traceback object, if any.
        """
        if self._async_manager:
            try:
                # Try to run shutdown synchronously
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new event loop for shutdown
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(self.shutdown())
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(loop)
                else:
                    loop.run_until_complete(self.shutdown())
            except Exception as log_e:
                _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))


class PiscesLxCoreParallel:
    """Performs advanced parallel execution using ThreadPoolExecutor and ProcessPoolExecutor."""



    def __init__(
        self,
        backend: _ConcurrencyBackend = _ConcurrencyBackend.THREADING,
        max_workers: Optional[int] = None,
        chunk_size: int = 1,
        timeout: Optional[float] = None,
        error_handler: Optional[Callable[[Exception, Any], Any]] = None,
        result_processor: Optional[Callable[[List[Any]], List[Any]]] = None
    ):
        """Initialize the parallel executor.

        Args:
            backend (_ConcurrencyBackend, optional): The concurrency backend to use. Defaults to _ConcurrencyBackend.THREADING.
            max_workers (Optional[int], optional): The maximum number of workers. Defaults to None.
            chunk_size (int, optional): The number of items to process in each chunk. Defaults to 1.
            timeout (Optional[float], optional): The timeout for each task. Defaults to None.
            error_handler (Optional[Callable[[Exception, Any], Any]], optional): A callable to handle task errors. Defaults to None.
            result_processor (Optional[Callable[[List[Any]], List[Any]]], optional): A callable to process task results. Defaults to None.
        """
        self.backend = backend
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.error_handler = error_handler
        self.result_processor = result_processor

    def map(self, func: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        """Execute a function on a sequence of items in parallel.

        Args:
            func (Callable[[Any], Any]): The function to execute.
            items (Sequence[Any]): The sequence of items to process.

        Raises:
            PiscesLxCoreConcurrencyError: If the backend is unsupported or parallel execution fails.

        Returns:
            List[Any]: The results of the function execution.
        """
        if not items:
            return []

        if self.max_workers is None:
            self.max_workers = min(len(items), 16)

        try:
            self._log.info(
                "Starting parallel execution",
                event="concurrency.parallel.start",
                backend=self.backend.value,
                items=len(items),
                max_workers=self.max_workers,
                chunk_size=self.chunk_size,
                timeout=self.timeout,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        start_time = time.time()
        results: List[Any] = []
        errors: List[Tuple[Exception, Any]] = []

        try:
            if self.backend == _ConcurrencyBackend.THREADING:
                results = self._execute_threaded(func, items, errors)
            elif self.backend == _ConcurrencyBackend.MULTIPROCESSING:
                results = self._execute_multiprocess(func, items, errors)
            else:
                raise PiscesLxCoreConcurrencyError(
                    f"Unsupported backend: {self.backend}",
                    context={"backend": self.backend.value}
                )

            # Process results if a result processor is provided
            if self.result_processor:
                results = self.result_processor(results)

            # Handle errors if an error handler is provided
            if errors and self.error_handler:
                for error, item in errors:
                    try:
                        fallback_result = self.error_handler(error, item)
                        results.append(fallback_result)
                    except Exception as e:
                        try:
                            self._log.error(
                                "Error handler failed",
                                event="concurrency.parallel.error_handler_failed",
                                error=str(e),
                                error_class=type(e).__name__,
                            )
                        except Exception as log_e:
                            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        except Exception as e:
            try:
                self._log.error(
                    "Parallel execution failed",
                    event="concurrency.parallel.failed",
                    error=str(e),
                    error_class=type(e).__name__,
                    duration=time.time() - start_time,
                )
            except Exception as log_e:
                _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
            raise PiscesLxCoreConcurrencyError(
                "Parallel execution failed",
                context={"error": str(e), "error_class": type(e).__name__, "duration": time.time() - start_time},
                cause=e,
            )

        duration = time.time() - start_time
        try:
            self._log.info(
                "Parallel execution completed",
                event="concurrency.parallel.completed",
                backend=self.backend.value,
                items=len(items),
                results=len(results),
                errors=len(errors),
                duration=duration,
                throughput=len(items) / duration if duration > 0 else 0,
            )
        except Exception as log_e:
            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))

        return results

    def _execute_threaded(self, func: Callable[[Any], Any], items: Sequence[Any], errors: List[Tuple[Exception, Any]]) -> List[Any]:
        """Execute tasks using ThreadPoolExecutor.

        Args:
            func (Callable[[Any], Any]): The function to execute.
            items (Sequence[Any]): The sequence of items to process.
            errors (List[Tuple[Exception, Any]]): A list to store errors and corresponding items.

        Raises:
            PiscesLxCoreConcurrencyError: If a task fails and no error handler is provided.

        Returns:
            List[Any]: The results of the function execution.
        """
        results: List[Any] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(func, item): item for item in items}

            # Process completed tasks
            for future in as_completed(future_to_item, timeout=self.timeout):
                item = future_to_item[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except Exception as e:
                    if self.error_handler:
                        errors.append((e, item))
                    else:
                        try:
                            self._log.error(
                                "Threaded execution item failed",
                                event="concurrency.threaded.item_failed",
                                error=str(e),
                                error_class=type(e).__name__,
                            )
                        except Exception as log_e:
                            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                        raise PiscesLxCoreConcurrencyError(
                            "Threaded execution item failed",
                            context={"error": str(e), "error_class": type(e).__name__},
                            cause=e,
                        )

        return results

    def _execute_multiprocess(self, func: Callable[[Any], Any], items: Sequence[Any], errors: List[Tuple[Exception, Any]]) -> List[Any]:
        """Execute tasks using ProcessPoolExecutor in chunks.

        Args:
            func (Callable[[Any], Any]): The function to execute.
            items (Sequence[Any]): The sequence of items to process.
            errors (List[Tuple[Exception, Any]]): A list to store errors and corresponding items.

        Raises:
            PiscesLxCoreConcurrencyError: If a chunk fails and no error handler is provided.

        Returns:
            List[Any]: The results of the function execution.
        """
        results: List[Any] = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Split items into chunks for better performance
            chunks = [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]

            future_to_chunk = {executor.submit(self._process_chunk, func, chunk): chunk for chunk in chunks}

            for future in as_completed(future_to_chunk, timeout=self.timeout):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result(timeout=self.timeout)
                    results.extend(chunk_results)
                except Exception as e:
                    if self.error_handler:
                        for item in chunk:
                            errors.append((e, item))
                    else:
                        try:
                            self._log.error(
                                "Multiprocess execution chunk failed",
                                event="concurrency.multiprocess.chunk_failed",
                                chunk_size=len(chunk),
                                error=str(e),
                                error_class=type(e).__name__,
                            )
                        except Exception as log_e:
                            _module_log.debug("CONCURRENCY_LOG_ERROR", error=str(log_e))
                        raise PiscesLxCoreConcurrencyError(
                            "Multiprocess execution chunk failed",
                            context={"chunk_size": len(chunk), "error": str(e), "error_class": type(e).__name__},
                            cause=e,
                        )

        return results

    def _process_chunk(self, func: Callable[[Any], Any], chunk: Sequence[Any]) -> List[Any]:
        """Process a chunk of items by applying a function to each item.

        Args:
            func (Callable[[Any], Any]): The function to execute.
            chunk (Sequence[Any]): The sequence of items to process.

        Returns:
            List[Any]: The results of the function execution on the chunk.
        """
        return [func(item) for item in chunk]

    @staticmethod
    def map(fn: Callable[[Any], Any], items: Iterable[Any], max_workers: int = 4) -> List[Any]:
        """Static method for backward compatibility to perform parallel execution.

        Args:
            fn (Callable[[Any], Any]): The function to execute.
            items (Iterable[Any]): The iterable of items to process.
            max_workers (int, optional): The maximum number of workers. Defaults to 4.

        Returns:
            List[Any]: The results of the function execution.
        """
        parallel = PiscesLxCoreParallel(
            backend=_ConcurrencyBackend.THREADING,
            max_workers=max_workers
        )
        return parallel.map(fn, list(items))