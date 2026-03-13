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
Operator Interface Definitions

This module defines the abstract base classes and data structures that establish
the contract for all operators in the PiscesL1 OPSC framework. All operators
must implement these interfaces to ensure interoperability and consistent behavior.

Key Components:
    - PiscesLxOperatorStatus: Enumeration of possible execution states
    - PiscesLxOperatorResult: Standardized result container with metadata
    - PiscesLxOperatorConfig: Configuration parameters for operator instances
    - PiscesLxOperatorInterface: Abstract base class for all operators

Design Principles:
    1. Interface Segregation: Small, focused interfaces for maximum flexibility
    2. Type Safety: Extensive type hints for IDE support and validation
    3. Extensibility: Easy to add new operators without modifying framework
    4. Observability: Built-in support for metrics and tracing
    5. Composability: Results can be chained and transformed

Result Flow:
    Input Validation -> Execute -> Result Creation -> Metadata Collection -> Return

The interface layer is intentionally minimal to avoid imposing implementation
constraints while ensuring semantic compatibility across all operators.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from configs.version import VERSION


class PiscesLxOperatorStatus(Enum):
    """
    Enumeration of possible execution states for operators.

    This enum defines the lifecycle states an operator can be in during and after
    execution. Status transitions follow a predictable pattern:
    PENDING -> RUNNING -> [SUCCESS | FAILED | CANCELLED]

    States:
        PENDING: Operator has been submitted but not yet started
        RUNNING: Operator is actively executing
        SUCCESS: Operator completed successfully without errors
        FAILED: Operator encountered an error during execution
        CANCELLED: Operator was explicitly cancelled before completion

    Usage:
        if result.status == PiscesLxOperatorStatus.SUCCESS:
            process_output(result.output)

    Integration:
        Status values are used for:
        - Conditional branching in pipelines
        - Error handling and retry logic
        - Monitoring and alerting thresholds
        - Audit trail and compliance reporting
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PiscesLxOperatorResult:
    """
    Standardized container for operator execution results.

    This dataclass encapsulates all information about an operator's execution,
    including the output data, execution status, timing information, and optional
    metadata. Results are immutable once created to ensure consistency.

    Attributes:
        operator_name: Unique identifier of the operator that produced this result
        status: Execution status from PiscesLxOperatorStatus enum
        output: The primary output data from the operator execution
        error: Error message if execution failed, None otherwise
        execution_time: Total execution time in seconds
        metadata: Additional contextual information about the execution
        timestamp: ISO format timestamp when execution completed

    Immutability:
        All fields are read-only after construction. To create a modified result,
        use the appropriate factory methods or construct a new instance.

    Thread Safety:
        Instances are immutable and safe to share across threads.

    Usage:
        result = PiscesLxOperatorResult(
            operator_name="data_transformer",
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"transformed": data},
            execution_time=0.125
        )

        if result.is_success():
            process_data(result.output)

    Serialization:
        Results can be serialized to JSON for logging or network transmission:
        {
            "operator_name": "...",
            "status": "success",
            "output": {...},
            "execution_time": 0.125,
            "timestamp": "2025-01-15 10:30:00"
        }
    """
    operator_name: str
    status: PiscesLxOperatorStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def is_success(self) -> bool:
        """
        Check if execution completed successfully.

        Returns:
            True if status is SUCCESS, False otherwise
        """
        return self.status == PiscesLxOperatorStatus.SUCCESS

    def is_failed(self) -> bool:
        """
        Check if execution failed.

        Returns:
            True if status is FAILED, False otherwise
        """
        return self.status == PiscesLxOperatorStatus.FAILED


@dataclass
class PiscesLxOperatorConfig:
    """
    Configuration parameters for operator instances.

    This dataclass holds all configurable parameters that control operator behavior,
    including execution constraints, retry policies, and custom parameters.

    Attributes:
        name: Unique identifier for the operator
        version: Semantic version string for version management
        timeout: Maximum execution time in seconds before timeout (default 300s)
        retries: Number of retry attempts on failure (default 3)
        parallel: Whether operator supports parallel execution
        dependencies: List of operator names this operator depends on
        parameters: Dictionary of custom configuration parameters
        gpu_memory_gb: Estimated GPU memory requirement in GB
        cpu_cores: Estimated CPU cores needed
        supports_gpu: Whether operator can run on GPU
        supports_cpu: Whether operator can run on CPU
        preferred_device: Preferred execution device ("gpu", "cpu", "auto")

    Default Values:
        - timeout: 300.0 seconds (5 minutes)
        - retries: 3 attempts
        - parallel: False (sequential execution)
        - dependencies: Empty list
        - parameters: Empty dictionary

    Usage:
        config = PiscesLxOperatorConfig(
            name="json_parser",
            version="1.2.0",
            timeout=60.0,
            retries=2,
            parameters={"strict_mode": True},
            gpu_memory_gb=4.0,
            supports_gpu=True
        )

    Validation:
        Configurations should be validated during operator setup to ensure
        all required parameters are present and within acceptable ranges.
    """
    name: str = "default_operator"
    version: str = VERSION
    timeout: float = 300.0
    retries: int = 3
    parallel: bool = False
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    supports_gpu: bool = True
    supports_cpu: bool = True
    preferred_device: str = "auto"


class PiscesLxOperatorInterface(ABC):
    """
    Abstract base class defining the interface for all operators.

    This ABC establishes the contract that all operators must fulfill, ensuring
    consistent behavior across the framework. Subclasses must implement all
    abstract properties and methods.

    Required Properties:
        name: Unique identifier for the operator
        version: Semantic version string
        description: Human-readable description of operator functionality
        input_schema: JSON Schema describing expected input format
        output_schema: JSON Schema describing output format

    Required Methods:
        validate_inputs: Verify input data matches expected schema
        execute: Perform the operator's core computation
        setup: Initialize resources before execution
        teardown: Release resources after execution

    Lifecycle:
        1. Instance creation with optional configuration
        2. setup() called to initialize resources
        3. execute() called one or more times with inputs
        4. teardown() called to release resources

    Usage:
        class MyOperator(PiscesLxOperatorInterface):
            @property
            def name(self) -> str:
                return "my_operator"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Does something useful"

            @property
            def input_schema(self) -> Dict[str, Any]:
                return {"type": "object", "required": ["data"]}

            @property
            def output_schema(self) -> Dict[str, Any]:
                return {"type": "object", "properties": {"result": {"type": "any"}}}

            def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
                return "data" in inputs

            def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
                result = do_computation(inputs["data"])
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={"result": result}
                )

    Thread Safety:
        Subclasses are responsible for implementing thread-safe execution
        if the operator will be used in concurrent contexts.
    """

    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None):
        """
        Initialize operator with optional configuration.

        Args:
            config: PiscesLxOperatorConfig instance or None for defaults
        """
        self.config = config or PiscesLxOperatorConfig(name=self.__class__.__name__, version=VERSION)
        self.operator_id = str(uuid.uuid4())
        from ..dc import PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing
        from utils.paths import get_log_file
        self.logger = PiscesLxLogger(f"PiscesLx.Core.OPSC.{self.__class__.__name__}", file_path=get_log_file(f"PiscesLx.Core.OPSC.{self.__class__.__name__}"), enable_file=True)
        self._metrics = PiscesLxMetrics()
        self._tracing = PiscesLxTracing()
        self._is_setup = False
        self._is_torn_down = False

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the unique name identifier for this operator.

        Returns:
            Unique string identifier for the operator
        """
        raise NotImplementedError("Operator must implement 'name'")

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Get the semantic version of this operator.

        Returns:
            Version string in format "major.minor.patch"
        """
        raise NotImplementedError("Operator must implement 'version'")

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get a human-readable description of this operator's functionality.

        Returns:
            Description string explaining what the operator does
        """
        raise NotImplementedError("Operator must implement 'description'")

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema describing expected input format.

        Returns:
            Dictionary representing the input validation schema
        """
        raise NotImplementedError("Operator must implement 'input_schema'")

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema describing output format.

        Returns:
            Dictionary representing the output structure schema
        """
        raise NotImplementedError("Operator must implement 'output_schema'")

    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate that input data matches the expected schema.

        This method should perform comprehensive validation of all input
        fields and return False if any required data is missing or invalid.

        Args:
            inputs: Dictionary of input data to validate

        Returns:
            True if inputs are valid, False otherwise

        Side Effects:
            May log validation errors using the framework logger
        """
        raise NotImplementedError("Operator must implement 'validate_inputs'")

    @abstractmethod
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute the operator's core computation.

        This is the main entry point for operator execution. Subclasses should
        implement their core business logic here and return a PiscesLxOperatorResult
        indicating success or failure.

        Args:
            inputs: Dictionary of input data from previous operators or callers
            **kwargs: Additional keyword arguments for flexibility

        Returns:
            PiscesLxOperatorResult with status, output, and metadata

        Best Practices:
            - Measure execution time for performance monitoring
            - Catch expected exceptions and return FAILED status
            - Include relevant metadata in the result
            - Handle partial failures gracefully
        """
        raise NotImplementedError("Operator must implement 'execute'")

    def setup(self) -> None:
        """
        Initialize resources before first execution.

        Override this method to allocate resources such as database connections,
        file handles, or external service clients. Called once after instance
        creation and before first execute() call.

        Default Implementation:
            Does nothing. Subclasses can override as needed.

        Usage:
            def setup(self) -> None:
                self.db_connection = connect_to_database()
                self.model = load_machine_learning_model()
        """
        if self._is_setup:
            return
        self._is_setup = True
        self._metrics.counter("operator_setups")
        self.logger.info("operator_setup", operator=self.name, version=self.version, operator_id=self.operator_id)

    def teardown(self) -> None:
        """
        Release resources after execution completes.

        Override this method to clean up resources allocated in setup().
        Called once when the operator is no longer needed.

        Default Implementation:
            Does nothing. Subclasses can override as needed.

        Usage:
            def teardown(self) -> None:
                self.db_connection.close()
                self.model.release_memory()
        """
        if self._is_torn_down:
            return
        self._is_torn_down = True
        self._metrics.counter("operator_teardowns")
        self.logger.info("operator_teardown", operator=self.name, version=self.version, operator_id=self.operator_id)

    def estimate_memory(self, inputs: Dict[str, Any]) -> float:
        """
        Estimate memory requirement for execution.

        Override this method to provide accurate memory estimates for
        better scheduling decisions.

        Args:
            inputs: Input data dictionary

        Returns:
            Estimated memory in GB

        Default Implementation:
            Returns the configured gpu_memory_gb value.
        """
        return getattr(self.config, "gpu_memory_gb", 0.0)

    def estimate_compute(self, inputs: Dict[str, Any]) -> float:
        """
        Estimate compute requirement for execution.

        Override this method to provide accurate compute estimates for
        better scheduling decisions.

        Args:
            inputs: Input data dictionary

        Returns:
            Estimated FLOPs or relative compute units

        Default Implementation:
            Returns 1.0 (one compute unit).
        """
        return 1.0

    def get_device_preference(self) -> str:
        """
        Get preferred execution device.

        Returns:
            "gpu", "cpu", or "auto"

        Default Implementation:
            Returns the configured preferred_device value.
        """
        return getattr(self.config, "preferred_device", "auto")

    def supports_device(self, device_type: str) -> bool:
        """
        Check if operator supports a specific device type.

        Args:
            device_type: "gpu" or "cpu"

        Returns:
            True if device is supported

        Default Implementation:
            Checks supports_gpu and supports_cpu config values.
        """
        if device_type == "gpu":
            return getattr(self.config, "supports_gpu", True)
        elif device_type == "cpu":
            return getattr(self.config, "supports_cpu", True)
        return True

    def __repr__(self) -> str:
        """
        Get string representation of the operator.

        Returns:
            Format: "ClassName(name='operator_name', version='1.0.0')"
        """
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
