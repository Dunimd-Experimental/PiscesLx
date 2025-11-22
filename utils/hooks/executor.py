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

import time
import asyncio
import concurrent.futures
from ..log.core import PiscesLxCoreLog
from typing import Any, Callable, List, Optional
from .types import PiscesLxCoreAlgorithmicListener, PiscesLxCoreExecutionResult

logger = PiscesLxCoreLog("HOOKS")

class PiscesLxCoreHookExecutor:
    """Object-oriented encapsulation for executing listeners."""
    
    def __init__(self, max_workers: int = 16, max_coroutines: int = 128) -> None:
        """Initialize the executor.

        Args:
            max_workers (int, optional): The maximum number of worker threads. Defaults to 16 (reduced from 64 to avoid resource exhaustion).
            max_coroutines (int, optional): The maximum number of concurrent coroutines. Defaults to 128 (reduced from 256).
        """
        self.max_workers = max_workers
        self.max_coroutines = max_coroutines
        self._thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = False
        logger.info("Executor initialization completed", event="HOOK_EXECUTOR_INIT", workers=max_workers, coroutines=max_coroutines)
    
    def initialize(self) -> None:
        """Perform lazy initialization of the executor."""
        if not self._initialized:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="piscesl1_hook"
            )
            self._semaphore = asyncio.Semaphore(self.max_coroutines)
            self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the executor gracefully."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        self._initialized = False
        logger.success("Executor has been shut down")
    
    @property
    def thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get the thread pool. Initialize it if not initialized yet.

        Returns:
            concurrent.futures.ThreadPoolExecutor: The thread pool instance.
        """
        if not self._initialized:
            self.initialize()
        return self._thread_pool
    
    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get the semaphore. Initialize it if not initialized yet.

        Returns:
            asyncio.Semaphore: The semaphore instance.
        """
        if not self._initialized:
            self.initialize()
        return self._semaphore
    
    def _run_sync_listener(self, listener: PiscesLxCoreAlgorithmicListener, event_type: str, payload: Any) -> PiscesLxCoreExecutionResult:
        """Execute a synchronous listener.

        Args:
            listener (PiscesLxCoreAlgorithmicListener): The listener to execute.
            event_type (str): The type of the event.
            payload (Any): The payload to pass to the listener.

        Returns:
            PiscesLxCoreExecutionResult: The execution result of the listener.
        """
        start = time.perf_counter()
        
        try:
            result = listener.callback(payload)
            execution_time = time.perf_counter() - start
            try:
                listener.executed()
            except Exception as e:
                logger.debug("Failed to mark the listener as executed", {
                    "event_type": event_type,
                    "callback": getattr(listener.callback, "__name__", str(listener.callback)),
                    "error": str(e),
                    "error_class": type(e).__name__,
                })
            
            return PiscesLxCoreExecutionResult(
                event_type=event_type,
                execution_time=execution_time,
                executed=1,
                errors=0,
                result=result
            )
        except Exception as exc:
            execution_time = time.perf_counter() - start
            logger.warning("Synchronous listener failed", {
                "event_type": event_type,
                "callback": getattr(listener.callback, "__name__", str(listener.callback)),
                "error": str(exc),
                "error_class": type(exc).__name__,
            })
            
            return PiscesLxCoreExecutionResult(
                event_type=event_type,
                execution_time=execution_time,
                executed=0,
                errors=1,
                exception=str(exc)
            )
    
    async def _run_async_listener(self, listener: PiscesLxCoreAlgorithmicListener, event_type: str, payload: Any) -> PiscesLxCoreExecutionResult:
        """Execute an asynchronous listener.

        Args:
            listener (PiscesLxCoreAlgorithmicListener): The listener to execute.
            event_type (str): The type of the event.
            payload (Any): The payload to pass to the listener.

        Returns:
            PiscesLxCoreExecutionResult: The execution result of the listener.
        """
        start = time.perf_counter()
        
        try:
            async with self.semaphore:
                result = await listener.callback(payload)
                execution_time = time.perf_counter() - start
                try:
                    listener.executed()
                except Exception as e:
                    logger.debug("Failed to mark the listener as executed", {
                        "event_type": event_type,
                        "callback": getattr(listener.callback, "__name__", str(listener.callback)),
                        "error": str(e),
                        "error_class": type(e).__name__,
                    })
                
                return PiscesLxCoreExecutionResult(
                    event_type=event_type,
                    execution_time=execution_time,
                    executed=1,
                    errors=0,
                    result=result
                )
        except Exception as exc:
            execution_time = time.perf_counter() - start
            logger.warning("Asynchronous listener failed", {
                "event_type": event_type,
                "callback": getattr(listener.callback, "__name__", str(listener.callback)),
                "error": str(exc),
                "error_class": type(exc).__name__,
            })
            
            return PiscesLxCoreExecutionResult(
                event_type=event_type,
                execution_time=execution_time,
                executed=0,
                errors=1,
                exception=str(exc)
            )
    
    def run_listener(self, listener: PiscesLxCoreAlgorithmicListener, event_type: str, payload: Any) -> PiscesLxCoreExecutionResult:
        """Execute a single listener.

        Args:
            listener (PiscesLxCoreAlgorithmicListener): The listener to execute.
            event_type (str): The type of the event.
            payload (Any): The payload to pass to the listener.

        Returns:
            PiscesLxCoreExecutionResult: The execution result of the listener.
        """
        if not listener.should_execute():
            return PiscesLxCoreExecutionResult(
                event_type=event_type,
                execution_time=0.0,
                executed=0,
                errors=0,
                result=None
            )
        
        callback = listener.callback
        
        if asyncio.iscoroutinefunction(callback):
            # Execute asynchronously
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.create_task(self._run_async_listener(listener, event_type, payload))
                return asyncio.run_coroutine_threadsafe(future, loop).result()
            except RuntimeError:
                return asyncio.run(self._run_async_listener(listener, event_type, payload))
        else:
            # Execute synchronously - add timeout to prevent deadlocks
            try:
                future = self.thread_pool.submit(self._run_sync_listener, listener, event_type, payload)
                # Add 30-second timeout to prevent indefinite blocking
                return future.result(timeout=30.0)
            except concurrent.futures.TimeoutError:
                logger.error("Listener execution timed out after 30 seconds", {
                    "event_type": event_type,
                    "callback": getattr(callback, "__name__", str(callback))
                })
                return PiscesLxCoreExecutionResult(
                    event_type=event_type,
                    execution_time=30.0,
                    executed=0,
                    errors=1,
                    exception="Listener execution timed out after 30 seconds"
                )
            except Exception as e:
                logger.error("Listener execution failed", {
                    "event_type": event_type,
                    "callback": getattr(callback, "__name__", str(callback)),
                    "error": str(e),
                    "error_class": type(e).__name__
                })
                return PiscesLxCoreExecutionResult(
                    event_type=event_type,
                    execution_time=0.0,
                    executed=0,
                    errors=1,
                    exception=str(e)
                )
    
    def run_listeners_parallel(
        self,
        listeners: List[PiscesLxCoreAlgorithmicListener],
        event_type: str,
        payload: Any,
        max_workers: Optional[int] = None
    ) -> List[PiscesLxCoreExecutionResult]:
        """Execute multiple listeners in parallel.

        Args:
            listeners (List[PiscesLxCoreAlgorithmicListener]): The list of listeners to execute.
            event_type (str): The type of the event.
            payload (Any): The payload to pass to each listener.
            max_workers (Optional[int], optional): The maximum number of worker threads. If provided and different from the current value, 
                the thread pool will be re-initialized. Defaults to None.

        Returns:
            List[PiscesLxCoreExecutionResult]: The list of execution results for each listener.
        """
        if not listeners:
            return []
        
        # Dynamically adjust the thread pool
        if max_workers and max_workers != self.max_workers:
            self.max_workers = max_workers
            self.shutdown()
            self.initialize()
        
        # Submit tasks
        futures = []
        for listener in listeners:
            future = self.thread_pool.submit(self.run_listener, listener, event_type, payload)
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                # Add timeout to prevent indefinite blocking on individual futures
                result = future.result(timeout=35.0)  # Slightly longer than individual listener timeout
                results.append(result)
            except concurrent.futures.TimeoutError:
                logger.error("Parallel listener execution timed out", {"event_type": event_type})
                results.append(PiscesLxCoreExecutionResult(
                    event_type=event_type,
                    execution_time=35.0,
                    executed=0,
                    errors=1,
                    exception="Parallel listener execution timed out after 35 seconds"
                ))
            except Exception as exc:
                logger.error("Listener execution failed", {"error": str(exc), "error_class": type(exc).__name__})
                results.append(PiscesLxCoreExecutionResult(
                    event_type=event_type,
                    execution_time=0.0,
                    executed=0,
                    errors=1,
                    exception=str(exc)
                ))
        
        return results

    def execute(self, event_type: str, listeners: List[PiscesLxCoreAlgorithmicListener], **kwargs: Any) -> dict:
        """Execute all listeners for a given event and return summarized information.

        Args:
            event_type (str): The type of the event.
            listeners (List[PiscesLxCoreAlgorithmicListener]): The list of listeners to execute.
            **kwargs (Any): The payload to pass to each listener.

        Returns:
            dict: A dictionary containing the following keys:
                'results': List[PiscesLxCoreExecutionResult],
                'total_time': float,   # Cumulative execution time (per listener)
                'errors': int,         # Number of errors (per listener)
                'count': int           # Number of listeners
        """
        payload = kwargs
        
        # Add overall timeout to prevent indefinite blocking
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Hook execution timed out after 60 seconds for event: {event_type}")
        
        # Set timeout for entire execution (Unix only, Windows will ignore)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second overall timeout
        
        try:
            results = self.run_listeners_parallel(listeners, event_type, payload)
            total_time = sum((r.execution_time for r in results), 0.0)
            errors = sum((r.errors for r in results), 0)
            return {
                'results': results,
                'total_time': float(total_time),
                'errors': int(errors),
                'count': len(results),
            }
        except TimeoutError as e:
            logger.error("Hook execution overall timeout", {"event_type": event_type, "error": str(e)})
            return {
                'results': [],
                'total_time': 60.0,
                'errors': len(listeners) if listeners else 1,
                'count': 0,
                'timeout_error': str(e)
            }
        finally:
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)