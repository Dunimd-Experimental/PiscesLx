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
Operator Execution Engine

This module implements the runtime execution system for operators, providing
synchronous and asynchronous execution, resource management, exception handling,
and performance monitoring.

Key Responsibilities:
    - Synchronous operator execution
    - Asynchronous operator execution via thread pool
    - Batch processing of multiple operators
    - Pipeline execution with data flow between operators
    - Execution context tracking and monitoring
    - Cancellation support for long-running operations

Architecture:
    The executor uses Python's ThreadPoolExecutor for concurrent execution,
    providing thread safety through proper synchronization. Execution contexts
    are tracked to enable monitoring and cancellation.

Thread Safety:
    - ThreadPoolExecutor handles concurrent task execution
    - Active executions are tracked with thread-safe dictionaries
    - Proper locking for shared state access

Usage Patterns:
    Sync Execution:
        result = executor.execute_sync("operator_name", inputs, config)

    Async Execution:
        result = await executor.execute_async("operator_name", inputs, config)

    Batch Execution:
        results = executor.execute_batch([
            {"operator": "op1", "inputs": {...}},
            {"operator": "op2", "inputs": {...}}
        ])

    Pipeline Execution:
        results = executor.execute_pipeline([
            {"operator": "transform", "inputs": {...}},
            {"operator": "filter", "inputs": {...}}
        ])
"""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union
)

from .interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from .registry import PiscesLxOperatorRegistry

from ..dc import (
    PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing,
    PiscesLxSystemMonitor
)

from utils.paths import get_log_file


@dataclass
class PiscesLxExecutionContext:
    """
    Container for execution context information.

    This dataclass tracks all relevant information about an active operator
    execution, including inputs, configuration, timing, and thread metadata.

    Attributes:
        operator_name: Name of the operator being executed
        inputs: Input data dictionary for the operator
        config: Configuration parameters for execution
        start_time: Unix timestamp when execution started
        thread_id: ID of the thread executing the operator

    Usage:
        context = PiscesLxExecutionContext(
            operator_name="transformer",
            inputs={"data": value},
            config={"timeout": 60},
            start_time=time.time(),
            thread_id=threading.get_ident()
        )
    """
    operator_name: str
    inputs: Dict[str, Any]
    config: Dict[str, Any]
    start_time: float
    thread_id: int


class PiscesLxOperatorExecutor:
    """
    Operator execution engine with thread pool management.

    This class handles the actual execution of operators, managing thread pool
    resources, tracking active executions, and providing comprehensive error
    handling and performance monitoring.

    Attributes:
        registry: Reference to the operator registry
        max_workers: Maximum concurrent worker threads
        thread_pool: ThreadPoolExecutor for concurrent execution
        active_executions: Dictionary of active execution contexts
        logger: Logger instance for execution events

    Thread Safety:
        - ThreadPoolExecutor handles concurrent task execution
        - Active executions dictionary protected by GIL for CPython
        - Proper error handling for thread-safe operation

    Usage:
        executor = PiscesLxOperatorExecutor(registry, max_workers=4)
        result = executor.execute_sync("my_operator", inputs, config)
        executor.shutdown()
    """

    def __init__(
        self,
        registry: PiscesLxOperatorRegistry,
        max_workers: int = 4
    ):
        """
        Initialize the operator executor with thread pool.

        Args:
            registry: Operator registry for operator lookup
            max_workers: Maximum concurrent workers (default 4)
        """
        self.registry = registry
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.active_executions: Dict[str, PiscesLxExecutionContext] = {}
        from utils.paths import get_log_file
        self.logger = PiscesLxLogger(f"PiscesLx.Core.OPSC.{self.__class__.__name__}", file_path=get_log_file(f"PiscesLx.Core.OPSC.{self.__class__.__name__}"), enable_file=True)
        self.metrics = PiscesLxMetrics()
        self.tracing = PiscesLxTracing()
        self.system_monitor = PiscesLxSystemMonitor()

    def _resolve_timeout(self, operator, config: Dict[str, Any]) -> Optional[float]:
        t = config.get("timeout")
        if t is None and hasattr(operator, "config"):
            t = getattr(operator.config, "timeout", None)
        try:
            return float(t) if t is not None else None
        except Exception:
            return None

    def _resolve_retries(self, operator, config: Dict[str, Any]) -> int:
        r = config.get("retries")
        if r is None and hasattr(operator, "config"):
            r = getattr(operator.config, "retries", 0)
        try:
            return max(0, int(r))
        except Exception:
            return 0

    def execute_sync(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> 'PiscesLxOperatorResult':
        """
        Execute an operator synchronously.

        This method runs the operator in the calling thread, blocking until
        completion. For non-blocking execution, use execute_async().

        Args:
            operator_name: Name of the operator to execute
            inputs: Input data dictionary
            config: Optional configuration overrides

        Returns:
            PiscesLxOperatorResult with status, output, and metadata

        Process:
            1. Create execution context with timing and thread info
            2. Generate unique execution ID
            3. Track execution in active_executions
            4. Look up or create operator instance
            5. Validate inputs against schema
            6. Execute operator
            7. Record execution time and metadata
            8. Clean up active execution

        Error Handling:
            - Operator not found: Returns FAILED status
            - Input validation failure: Returns FAILED status
            - Execution exception: Returns FAILED with error message

        Usage:
            result = executor.execute_sync(
                "data_transformer",
                {"data": [1, 2, 3]},
                {"timeout": 60}
            )

            if result.is_success():
                process(result.output)
        """
        config = config or {}
        start_time = time.time()
        thread_id = threading.get_ident()

        span = self.tracing.start_span(
            f"exec_{operator_name}",
            attributes={"operator": operator_name, "thread_id": thread_id}
        )

        execution_context = PiscesLxExecutionContext(
            operator_name=operator_name,
            inputs=inputs,
            config=config,
            start_time=start_time,
            thread_id=thread_id
        )

        execution_id = f"{operator_name}_{thread_id}_{int(start_time)}"
        self.active_executions[execution_id] = execution_context

        self.metrics.counter("executor_executions_started")
        self.metrics.gauge("executor_active_executions", len(self.active_executions))

        try:
            operator = self.registry.get_instance(operator_name)
            if not operator:
                operator = self.registry.create_instance(operator_name)
                if not operator:
                    return PiscesLxOperatorResult(
                        operator_name=operator_name,
                        status=PiscesLxOperatorStatus.FAILED,
                        error=f"Operator '{operator_name}' not found or failed to create"
                    )

            if not operator.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=operator_name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Input validation failed"
                )

            timeout = self._resolve_timeout(operator, config)
            retries = self._resolve_retries(operator, config)

            last_error: Optional[str] = None
            for attempt in range(retries + 1):
                try:
                    if timeout and timeout > 0:
                        future = self.thread_pool.submit(operator.execute, inputs, **config)
                        result = future.result(timeout=timeout)
                    else:
                        result = operator.execute(inputs, **config)
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt >= retries:
                        raise
                    self.logger.warning(
                        "operator_retry",
                        operator=operator_name,
                        attempt=attempt + 1,
                        retries=retries,
                        error=last_error
                    )
            else:
                raise RuntimeError(last_error or "Unknown execution failure")

            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.metadata.update({
                "thread_id": thread_id,
                "execution_id": execution_id,
                "start_time": execution_context.start_time
            })

            self.metrics.counter("executor_executions_completed")
            self.metrics.timer("executor_execution_time_ms", execution_time * 1000)
            self.tracing.end_span(span, status="ok")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.counter("executor_executions_failed")
            self.tracing.end_span(span, status="error", error_message=str(e))
            return PiscesLxOperatorResult(
                operator_name=operator_name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={"thread_id": thread_id, "execution_id": execution_id}
            )

        finally:
            self.active_executions.pop(execution_id, None)
            self.metrics.gauge("executor_active_executions", len(self.active_executions))

    async def execute_async(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> 'PiscesLxOperatorResult':
        """
        Execute an operator asynchronously.

        This method runs the operator in the thread pool without blocking
        the event loop, suitable for async applications.

        Args:
            operator_name: Name of the operator to execute
            inputs: Input data dictionary
            config: Optional configuration overrides

        Returns:
            PiscesLxOperatorResult from thread pool execution

        Implementation:
            Uses asyncio.run_in_executor() to bridge sync/async contexts.
            The actual execution happens in the thread pool.

        Usage:
            result = await executor.execute_async(
                "data_processor",
                {"data": large_dataset}
            )
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.execute_sync,
            operator_name,
            inputs,
            config
        )

    def execute_batch(
        self,
        executions: List[Dict[str, Any]]
    ) -> List['PiscesLxOperatorResult']:
        """
        Execute multiple operators concurrently.

        Submits all execution tasks to the thread pool and returns results
        as they complete. All tasks run in parallel up to max_workers limit.

        Args:
            executions: List of execution specifications, each containing:
                - operator: Operator name
                - inputs: Input data dictionary
                - config: Optional configuration

        Returns:
            List of PiscesLxOperatorResult in completion order

        Process:
            1. Submit all tasks to thread pool
            2. Collect futures as they complete
            3. Extract results or create error results for failures

        Timeout:
            Individual tasks have 300 second (5 minute) timeout.

        Usage:
            results = executor.execute_batch([
                {"operator": "transform1", "inputs": data1},
                {"operator": "transform2", "inputs": data2},
                {"operator": "filter", "inputs": data3}
            ])

            for result in results:
                if result.is_success():
                    process(result.output)
        """
        futures: List[Future] = []
        results: List[PiscesLxOperatorResult] = []

        for exec_info in executions:
            future = self.thread_pool.submit(
                self.execute_sync,
                exec_info['operator'],
                exec_info['inputs'],
                exec_info.get('config', {})
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
                results.append(result)
            except Exception as e:
                results.append(PiscesLxOperatorResult(
                    operator_name="unknown",
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Batch execution error: {e}"
                ))

        return results

    def execute_pipeline(
        self,
        pipeline: List[Dict[str, Any]]
    ) -> List['PiscesLxOperatorResult']:
        """
        Execute a sequence of operators with data flow.

        Operators in the pipeline execute sequentially, with each operator
        receiving the output of the previous operator as 'previous_output'.

        Args:
            pipeline: List of pipeline step specifications:
                - operator: Operator name
                - inputs: Input data (may reference previous_output)
                - config: Optional configuration

        Returns:
            List of PiscesLxOperatorResult for each step

        Data Flow:
            Step 1: Uses 'inputs' from configuration
            Step N: Receives 'previous_output' from Step N-1

        Early Termination:
            If a step fails and it's not the last step, the pipeline
            stops and returns results up to that point.

        Usage:
            results = executor.execute_pipeline([
                {
                    "operator": "load_data",
                    "inputs": {"path": "/data.csv"}
                },
                {
                    "operator": "transform",
                    "inputs": {"transform_params": {"normalize": True}}
                },
                {
                    "operator": "aggregate",
                    "inputs": {"method": "sum"}
                }
            ])

            for i, result in enumerate(results):
                print(f"Step {i+1}: {result.status}")
        """
        results: List[PiscesLxOperatorResult] = []
        previous_output = None

        for i, step in enumerate(pipeline):
            inputs = step.get('inputs', {}).copy()
            if previous_output is not None:
                inputs['previous_output'] = previous_output

            result = self.execute_sync(
                step['operator'],
                inputs,
                step.get('config', {})
            )

            results.append(result)

            if not result.is_success() and i < len(pipeline) - 1:
                break

            previous_output = result.output

        return results

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """
        Get information about currently active executions.

        Returns a list of dictionaries containing details about all
        executions that are currently running.

        Returns:
            List of execution information dictionaries:
            - execution_id: Unique identifier
            - operator_name: Name of executing operator
            - start_time: When execution started
            - duration: Elapsed time in seconds
            - thread_id: ID of executing thread

        Usage:
            for exec_info in executor.get_active_executions():
                print(f"{exec_info['operator_name']}: {exec_info['duration']:.1f}s")
        """
        return [
            {
                "execution_id": exec_id,
                "operator_name": context.operator_name,
                "start_time": context.start_time,
                "duration": time.time() - context.start_time,
                "thread_id": context.thread_id
            }
            for exec_id, context in self.active_executions.items()
        ]

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a specific active execution.

        This is a soft cancellation that removes the execution from
        the active tracking. For true cancellation, the operator must
        support interruption.

        Args:
            execution_id: The execution ID to cancel

        Returns:
            True if execution was found and cancelled, False otherwise

        Limitations:
            - Cannot interrupt already completed executions
            - Operators must cooperatively check for cancellation
            - Resources held by the operator may not be released

        Usage:
            if executor.cancel_execution("transform_12345_1234567890"):
                print("Execution cancelled")
        """
        context = self.active_executions.get(execution_id)
        if context:
            self.active_executions.pop(execution_id, None)
            return True
        return False

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor and release resources.

        This method shuts down the thread pool and clears all active
        executions. Call this when the executor is no longer needed.

        Args:
            wait: If True, wait for running tasks to complete

        Side Effects:
            - Thread pool shutdown
            - Active executions cleared
            - No further submissions accepted

        Usage:
            executor.shutdown(wait=True)
        """
        self.thread_pool.shutdown(wait=wait)
        self.active_executions.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Dictionary with executor statistics including metrics,
            device status, and execution history.
        """
        metrics_summary = {}
        system_metrics = self.system_monitor.collect()

        return {
            "active_executions": len(self.active_executions),
            "max_workers": self.max_workers,
            "metrics": metrics_summary,
            "system_metrics": system_metrics.to_dict()
        }

    def get_device_metrics(self) -> Dict[str, Any]:
        """
        Get current device metrics.

        Returns:
            Dictionary with CPU, GPU, and memory metrics.
        """
        cpu_usage = self.system_monitor.get_cpu_usage()
        mem_usage = self.system_monitor.get_memory_usage()
        mem_info = self.system_monitor.get_memory_info()

        return {
            "cpu_usage_percent": cpu_usage,
            "memory_percent": mem_usage,
            "memory_available_gb": mem_info.free_bytes / (1024 ** 3) if mem_info else 0,
            "memory_total_gb": mem_info.total_bytes / (1024 ** 3) if mem_info else 0
        }
