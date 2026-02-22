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

"""
Operator Core Scheduling System

This module implements the central controller for the OPSC framework, providing
unified management of operator registration, execution, monitoring, and lifecycle.
This is the central hub for all operator system operations.

Key Responsibilities:
    - System initialization and configuration
    - Operator registration and discovery
    - Single operator execution (sync/async)
    - Pipeline and batch processing coordination
    - Dependency management
    - Health monitoring and system status
    - Graceful shutdown and resource cleanup

Architecture:
    The core system integrates the registry and executor components,
    providing a unified API for all operator operations. It manages
    the system lifecycle and provides monitoring capabilities.

Thread Safety:
    - Registry operations are thread-safe
    - Executor operations are thread-safe
    - Core initialization is idempotent

Usage Patterns:
    Basic Usage:
        core = PiscesLxOperatorCore()
        core.register_operator(MyOperator)
        result = core.execute("my_operator", inputs)

    Context Manager:
        with PiscesLxOperatorCore() as core:
            core.register_operator(MyOperator)
            result = core.execute("my_operator", inputs)

    Pipeline:
        results = core.execute_pipeline([
            {"operator": "step1", "inputs": {...}},
            {"operator": "step2", "inputs": {...}}
        ])
"""

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional
)

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

if TYPE_CHECKING:
    from .interface import (
        PiscesLxOperatorInterface,
        PiscesLxOperatorResult,
        PiscesLxOperatorConfig
    )
    from .registry import PiscesLxOperatorRegistry
    from .executor import PiscesLxOperatorExecutor
    from .distributed import PiscesLxParallelConfig
    from .heterogeneous import PiscesLxHeterogeneousConfig


@dataclass
class PiscesLxOperatorSystemConfig:
    """
    Configuration for the OPSC system.

    This dataclass holds system-wide configuration parameters that control
    the behavior of the operator core, executor, and registry.

    Attributes:
        max_concurrent_executors: Maximum concurrent execution threads
        default_timeout: Default timeout for operator execution (seconds)
        enable_monitoring: Enable/disable monitoring features
        log_level: Logging verbosity level
        enable_distributed: Enable distributed execution support
        enable_heterogeneous: Enable heterogeneous (CPU/GPU) execution
        enable_gpu_scheduling: Enable GPU resource scheduling
        default_parallel_config: Default parallel execution configuration
        default_heterogeneous_config: Default heterogeneous execution config

    Default Values:
        - max_concurrent_executors: 8
        - default_timeout: 300.0 seconds (5 minutes)
        - enable_monitoring: True
        - log_level: "INFO"
        - enable_distributed: True
        - enable_heterogeneous: True

    Usage:
        config = PiscesLxOperatorSystemConfig(
            max_concurrent_executors=16,
            default_timeout=600.0,
            enable_monitoring=True,
            log_level="DEBUG"
        )
        core = PiscesLxOperatorCore(config=config)
    """
    max_concurrent_executors: int = 8
    default_timeout: float = 300.0
    enable_monitoring: bool = True
    log_level: str = "INFO"
    enable_distributed: bool = True
    enable_heterogeneous: bool = True
    enable_gpu_scheduling: bool = True


class PiscesLxOperatorCore:
    """
    Central controller for the OPSC operator system.

    This class serves as the main entry point for the operator framework,
    integrating the registry and executor components into a unified API.
    It manages the system lifecycle and provides comprehensive monitoring.

    Attributes:
        config: System configuration parameters
        registry: Operator registry instance
        executor: Operator executor instance
        logger: Logger instance for system events
        _initialized: Initialization flag

    Thread Safety:
        - Initialization is idempotent and thread-safe
        - Registry and executor handle their own thread safety

    Usage:
        # Synchronous usage
        core = PiscesLxOperatorCore()
        core.register_operator(MyOperator)
        result = core.execute("my_operator", inputs)
        core.shutdown()

        # Async usage
        async with PiscesLxOperatorCore() as core:
            await core.execute_async("my_operator", inputs)

        # Context manager
        with PiscesLxOperatorCore() as core:
            core.register_operator(MyOperator)
    """

    def __init__(
        self,
        config: Optional[PiscesLxOperatorSystemConfig] = None
    ):
        """
        Initialize the operator core system.

        Args:
            config: Optional system configuration
        """
        from .registry import PiscesLxOperatorRegistryHub, PiscesLxOperatorRegistry
        from .executor import PiscesLxOperatorExecutor
        from .heterogeneous import PiscesLxHeterogeneousExecutor, PiscesLxHeterogeneousConfig
        from .resources import PiscesLxGPUScheduler, PiscesLxResourceEstimator
        from ..dc import PiscesLxLogger

        self.config = config or PiscesLxOperatorSystemConfig()
        self.registry: PiscesLxOperatorRegistry = PiscesLxOperatorRegistryHub.get_registry()
        self.executor = PiscesLxOperatorExecutor(
            self.registry,
            max_workers=self.config.max_concurrent_executors
        )
        self.logger = PiscesLxLogger(f"PiscesLx.Core.OPSC.{self.__class__.__name__}", file_path=get_log_file(f"PiscesLx.Core.OPSC.{self.__class__.__name__}"), enable_file=True)
        self._initialized = False

        if self.config.enable_heterogeneous:
            hetero_config = PiscesLxHeterogeneousConfig()
            self.heterogeneous_executor = PiscesLxHeterogeneousExecutor(
                registry=self.registry,
                config=hetero_config
            )
        else:
            self.heterogeneous_executor = None

        if self.config.enable_gpu_scheduling:
            self.gpu_scheduler = PiscesLxGPUScheduler()
            self.resource_estimator = PiscesLxResourceEstimator()
        else:
            self.gpu_scheduler = None
            self.resource_estimator = None

    def initialize(self) -> None:
        """
        Initialize the core system if not already initialized.

        This method sets up logging configuration and logs system info.
        Safe to call multiple times (idempotent).

        Process:
            1. Check if already initialized
            2. Configure logging level
            3. Log system initialization message
            4. Log configuration summary
        """
        if self._initialized:
            return

        self._initialized = True
        self.logger.set_level(self.config.log_level)
        self.logger.info(
            "opsc_initialized",
            max_workers=self.config.max_concurrent_executors,
            default_timeout=self.config.default_timeout,
            monitoring=self.config.enable_monitoring
        )

    def register_operator(
        self,
        operator_class: type
    ) -> bool:
        """
        Register an operator class with the system.

        Args:
            operator_class: The operator class to register

        Returns:
            True if registration succeeded, False otherwise

        Process:
            1. Ensure system is initialized
            2. Attempt to register with registry
            3. Log result

        Usage:
            core.register_operator(DataTransformOperator)
            core.register_operator(FilterOperator)
        """
        if not self._initialized:
            self.initialize()

        try:
            self.registry.register(operator_class)
            meta = self.registry._class_metadata(operator_class)
            self.logger.info("opsc_register_ok", operator=meta.get("name") or operator_class.__name__, version=meta.get("version"))
            return True
        except Exception as e:
            self.logger.error("opsc_register_failed", operator=getattr(operator_class, "__name__", "unknown"), error=str(e))
            return False

    def unregister_operator(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Unregister an operator from the system.

        Args:
            name: Operator name to unregister
            version: Optional specific version

        Returns:
            True if operator was found and unregistered
        """
        return self.registry.unregister(name, version)

    def execute(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> 'PiscesLxOperatorResult':
        """
        Execute a single operator synchronously.

        Args:
            operator_name: Name of the operator to execute
            inputs: Input data dictionary
            config: Optional execution configuration

        Returns:
            PiscesLxOperatorResult with execution outcome

        Usage:
            result = core.execute(
                "data_transformer",
                {"data": [1, 2, 3]},
                {"timeout": 60}
            )

            if result.is_success():
                process(result.output)
        """
        if not self._initialized:
            self.initialize()

        return self.executor.execute_sync(operator_name, inputs, config)

    async def execute_async(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> 'PiscesLxOperatorResult':
        """
        Execute a single operator asynchronously.

        Args:
            operator_name: Name of the operator to execute
            inputs: Input data dictionary
            config: Optional execution configuration

        Returns:
            PiscesLxOperatorResult from async execution

        Usage:
            result = await core.execute_async(
                "data_processor",
                {"data": large_dataset}
            )
        """
        if not self._initialized:
            self.initialize()

        return await self.executor.execute_async(operator_name, inputs, config)

    def execute_pipeline(
        self,
        pipeline_config: List[Dict[str, Any]]
    ) -> List['PiscesLxOperatorResult']:
        """
        Execute a sequence of operators in pipeline mode.

        Operators execute sequentially with data flowing between steps.
        Each operator receives the output of the previous operator.

        Args:
            pipeline_config: List of pipeline step definitions

        Returns:
            List of PiscesLxOperatorResult for each step

        Early Termination:
            Pipeline stops if a step fails (unless it's the last step)

        Usage:
            results = core.execute_pipeline([
                {
                    "operator": "load_data",
                    "inputs": {"path": "/data.csv"}
                },
                {
                    "operator": "transform",
                    "inputs": {"mode": "normalize"}
                },
                {
                    "operator": "aggregate",
                    "inputs": {"method": "sum"}
                }
            ])

            for i, result in enumerate(results):
                print(f"Step {i+1}: {result.status}")
        """
        if not self._initialized:
            self.initialize()

        return self.executor.execute_pipeline(pipeline_config)

    def execute_batch(
        self,
        batch_config: List[Dict[str, Any]]
    ) -> List['PiscesLxOperatorResult']:
        """
        Execute multiple operators concurrently in batch mode.

        All operators in the batch execute concurrently up to the
        configured max workers limit.

        Args:
            batch_config: List of execution specifications

        Returns:
            List of PiscesLxOperatorResult in completion order

        Usage:
            results = core.execute_batch([
                {"operator": "op1", "inputs": {...}},
                {"operator": "op2", "inputs": {...}},
                {"operator": "op3", "inputs": {...}}
            ])

            for result in results:
                if result.is_success():
                    process(result.output)
        """
        if not self._initialized:
            self.initialize()

        return self.executor.execute_batch(batch_config)

    def list_available_operators(self) -> List[Dict[str, str]]:
        """
        List all operators registered with the system.

        Returns:
            List of operator information dictionaries sorted by name

        Information Returned:
            - name: Operator unique identifier
            - latest_version: Highest registered version
            - all_versions: List of all versions
            - description: Human-readable description

        Usage:
            for op in core.list_available_operators():
                print(f"{op['name']} v{op['latest_version']}")
        """
        if not self._initialized:
            self.initialize()

        return self.registry.list_operators()

    def get_operator_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific operator.

        Args:
            name: Operator name

        Returns:
            Operator info dictionary or None if not found

        Usage:
            info = core.get_operator_info("data_transformer")
            if info:
                print(f"Versions: {info['all_versions']}")
        """
        operators = self.list_available_operators()
        for op_info in operators:
            if op_info['name'] == name:
                return op_info
        return None

    def add_dependency(
        self,
        operator_name: str,
        dependency_name: str
    ) -> None:
        """
        Add a dependency relationship between operators.

        Records that one operator depends on another, affecting
        the execution order in pipelines.

        Args:
            operator_name: The dependent operator
            dependency_name: The operator being depended on

        Usage:
            # Filter depends on transformer
            core.add_dependency("filter_operator", "transform_operator")
        """
        self.registry.add_dependency(operator_name, dependency_name)

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """
        Get information about currently active operator executions.

        Returns:
            List of execution information dictionaries

        Usage:
            for exec_info in core.get_active_executions():
                print(f"{exec_info['operator_name']}: {exec_info['duration']:.1f}s")
        """
        return self.executor.get_active_executions()

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a specific active execution.

        Args:
            execution_id: The execution ID to cancel

        Returns:
            True if execution was found and cancelled

        Usage:
            if core.cancel_execution("transform_12345"):
                print("Execution cancelled")
        """
        return self.executor.cancel_execution(execution_id)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a system health check.

        Returns comprehensive information about system status
        including initialization state, registered operators, and
        executor status.

        Returns:
            Dictionary containing:
            - initialized: System initialization state
            - registered_operators: Count of registered operators
            - active_executions: Count of running executions
            - config: Current configuration values
            - gpu_status: GPU scheduler status (if enabled)
            - heterogeneous_status: Heterogeneous executor status (if enabled)

        Usage:
            health = core.health_check()
            print(f"Operators: {health['registered_operators']}")
            print(f"Active: {health['active_executions']}")
        """
        result = {
            "initialized": self._initialized,
            "registered_operators": len(self.registry._operators),
            "active_executions": len(self.executor.active_executions),
            "config": {
                "max_workers": self.config.max_concurrent_executors,
                "timeout": self.config.default_timeout,
                "monitoring": self.config.enable_monitoring,
                "distributed": self.config.enable_distributed,
                "heterogeneous": self.config.enable_heterogeneous,
                "gpu_scheduling": self.config.enable_gpu_scheduling
            }
        }

        if self.gpu_scheduler:
            result["gpu_status"] = self.gpu_scheduler.health_check()

        if self.heterogeneous_executor:
            result["heterogeneous_status"] = self.heterogeneous_executor.health_check()

        return result

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with executor stats, GPU stats, and device metrics.
        """
        stats = {
            "executor": self.executor.get_stats(),
            "registry": {
                "operators": self.registry.list_operators()
            }
        }

        if self.gpu_scheduler:
            stats["gpu_scheduler"] = self.gpu_scheduler.get_stats()

        if self.heterogeneous_executor:
            stats["heterogeneous"] = self.heterogeneous_executor.get_stats()

        return stats

    def estimate_resources(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate resource requirements for an operator.

        Args:
            operator_name: Name of the operator
            inputs: Input data
            config: Optional configuration

        Returns:
            Resource estimation dictionary
        """
        if not self.resource_estimator:
            return {"error": "Resource estimation not enabled"}

        operator = self.registry.get_instance(operator_name)
        if not operator:
            operator = self.registry.create_instance(operator_name)

        if not operator:
            return {"error": f"Operator '{operator_name}' not found"}

        input_shape = self._get_input_shape(inputs)
        request = self.resource_estimator.estimate_operator_resources(
            operator_name, input_shape, config
        )

        return {
            "operator_name": operator_name,
            "gpu_memory_gb": request.gpu_memory_gb,
            "compute_units": request.compute_units,
            "priority": request.priority.value,
            "preemptible": request.preemptible
        }

    def _get_input_shape(self, inputs: Dict[str, Any]) -> tuple:
        """Get shape of input data."""
        for key, value in inputs.items():
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    return tuple(value.shape)
            except ImportError:
                pass

            if hasattr(value, "shape"):
                return tuple(value.shape)

            if isinstance(value, (list, tuple)) and len(value) > 0:
                return (len(value),)

        return (1,)

    def execute_heterogeneous(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> 'PiscesLxOperatorResult':
        """
        Execute operator using heterogeneous executor.

        Uses CPU/GPU hybrid execution for optimal performance.

        Args:
            operator_name: Name of operator to execute
            inputs: Input data dictionary
            config: Optional execution configuration

        Returns:
            Execution result
        """
        if not self.heterogeneous_executor:
            return self.execute(operator_name, inputs, config)

        if not self._initialized:
            self.initialize()

        return self.heterogeneous_executor.execute_sync(operator_name, inputs, config)

    async def execute_heterogeneous_async(
        self,
        operator_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> 'PiscesLxOperatorResult':
        """
        Execute operator asynchronously using heterogeneous executor.

        Args:
            operator_name: Name of operator to execute
            inputs: Input data dictionary
            config: Optional execution configuration

        Returns:
            Execution result
        """
        if not self.heterogeneous_executor:
            return await self.execute_async(operator_name, inputs, config)

        if not self._initialized:
            self.initialize()

        return await self.heterogeneous_executor.execute_async(operator_name, inputs, config)

    def shutdown(self) -> None:
        """
        Gracefully shutdown the operator core system.

        This method:
        1. Shuts down the executor (with waiting)
        2. Shuts down heterogeneous executor (if enabled)
        3. Clears the registry
        4. Resets initialization state

        Safe to call multiple times.

        Usage:
            core.shutdown()
        """
        if not self._initialized:
            return

        self.executor.shutdown(wait=True)

        if self.heterogeneous_executor:
            self.heterogeneous_executor.shutdown()

        self.registry.clear()
        self._initialized = False

    def __enter__(self) -> 'PiscesLxOperatorCore':
        """
        Context manager entry point.

        Returns:
            Initialized core instance
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit point.

        Ensures proper shutdown when used in with statement.
        """
        self.shutdown()
