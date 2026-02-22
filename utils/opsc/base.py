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
Operator Base Classes and Concrete Implementations

This module provides concrete operator implementations that serve as base classes
for custom operators. It includes common patterns for transform, filter, and
utility operators.

Key Components:
    - PiscesLxBaseOperator: Abstract base class with common functionality
    - PiscesLxTransformOperator: Base class for data transformation operators
    - PiscesLxFilterOperator: Base class for data filtering operators

Design Patterns:
    1. Template Method: Subclasses implement abstract execute method
    2. Strategy Pattern: Different operators use different strategies
    3. Chain of Responsibility: Filters can be chained

Transform Operators:
    Transform operators modify input data and produce new output.
    Common use cases:
    - Data normalization and scaling
    - Format conversion
    - Feature engineering
    - Data enrichment

Filter Operators:
    Filter operators evaluate conditions and return boolean results.
    Common use cases:
    - Validation checks
    - Condition evaluation
    - Data cleansing
    - Business rule enforcement

Usage:
    class MyTransform(PiscesLxTransformOperator):
        @property
        def name(self) -> str:
            return "my_transform"

        def transform(self, data: Any) -> Any:
            return processed_data

    class MyFilter(PiscesLxFilterOperator):
        @property
        def name(self) -> str:
            return "my_filter"

        def filter(self, data: Any) -> bool:
            return data.is_valid
"""

import time
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from .interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)
from ..dc import PiscesLxLogger, PiscesLxMetrics, PiscesLxTracing
from ..paths import get_log_file
from configs.version import VERSION


class PiscesLxBaseOperator(PiscesLxOperatorInterface, ABC):
    """
    Abstract base class providing common operator functionality.

    This class implements the PiscesLxOperatorInterface and provides
    default implementations for commonly needed functionality.
    Subclasses focus on implementing the business logic.

    Common Features:
        - Default input/output schema validation
        - Automatic execution time tracking
        - Error handling with context
        - Configuration management

    Usage:
        class MyOperator(PiscesLxBaseOperator):
            @property
            def name(self) -> str:
                return "my_operator"

            def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
                # Business logic here
                return PiscesLxOperatorResult(...)
    """

    @property
    def name(self) -> str:
        if getattr(self.config, "name", None):
            return str(self.config.name)
        return self.__class__.__name__

    @property
    def version(self) -> str:
        if getattr(self.config, "version", None):
            return str(self.config.version)
        return VERSION

    @property
    def description(self) -> str:
        doc = (self.__class__.__doc__ or "").strip()
        if doc:
            return doc.splitlines()[0].strip()
        return self.__class__.__name__

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Get default input schema accepting any dictionary.

        Returns:
            JSON Schema for object input
        """
        return {
            "type": "object",
            "additionalProperties": True
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Get default output schema allowing any output.

        Returns:
            JSON Schema for any output
        """
        return {
            "type": "any",
            "description": "Any valid output"
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate that inputs is a dictionary.

        Default implementation checks that inputs is a dict.
        Override for custom validation.

        Args:
            inputs: Input data to validate

        Returns:
            True if inputs is valid dict
        """
        return isinstance(inputs, dict)

    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute the operator's business logic.

        This method provides common error handling and timing.
        Subclasses should override transform() or filter() methods.

        Args:
            inputs: Input data dictionary
            **kwargs: Additional keyword arguments

        Returns:
            PiscesLxOperatorResult with execution outcome
        """
        start_time = time.time()
        span = self._tracing.start_span(
            f"base_exec_{self.__class__.__name__}",
            attributes={"operator": self.name}
        )
        try:
            if not self._is_setup:
                self.setup()

            if not self.validate_inputs(inputs):
                self._metrics.counter("operator_validation_failures")
                self._tracing.end_span(span, status="error", error_message="Input validation failed")
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Input validation failed",
                    execution_time=time.time() - start_time
                )

            out = self._execute_impl(inputs, **kwargs)
            if isinstance(out, PiscesLxOperatorResult):
                result = out
            else:
                result = PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output=out
                )

            result.execution_time = time.time() - start_time
            result.metadata.setdefault("operator_id", self.operator_id)
            result.metadata.setdefault("version", self.version)
            self._metrics.counter("operator_executions_success")
            self._metrics.timer("operator_execution_time_ms", result.execution_time * 1000)
            self._tracing.end_span(span, status="ok")
            return result

        except Exception as e:
            self.logger.error("operator_execute_failed", operator=self.name, error=str(e))
            self._metrics.counter("operator_executions_failed")
            self._tracing.end_span(span, status="error", error_message=str(e))
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                metadata={"operator_id": self.operator_id, "version": self.version}
            )

    def _execute_impl(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Internal execution implementation.

        Override this in subclasses to implement specific logic.

        Args:
            inputs: Validated input data
            **kwargs: Additional arguments

        Returns:
            Result of the operation
        """
        raise NotImplementedError("Subclasses must implement _execute_impl")


class PiscesLxTransformOperator(PiscesLxBaseOperator, ABC):
    """
    Abstract base class for data transformation operators.

    This class provides the template method pattern for transformation
    operations. Subclasses implement the transform() method.

    Attributes:
        transform_count: Number of transformations performed

    Usage:
        class NormalizeOperator(PiscesLxTransformOperator):
            @property
            def name(self) -> str:
                return "normalize"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Normalize data to 0-1 range"

            def transform(self, data: Any) -> Any:
                return (data - min) / (max - min)
    """

    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None):
        """
        Initialize transform operator.

        Args:
            config: Optional configuration
        """
        super().__init__(config)
        self.transform_count = 0

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Get input schema for numeric data.

        Returns:
            JSON Schema for numeric input
        """
        return {
            "type": "number",
            "description": "Numeric data to transform"
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Get output schema for numeric data.

        Returns:
            JSON Schema for numeric output
        """
        return {
            "type": "number",
            "description": "Transformed numeric data"
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate that required transform inputs are present.

        Args:
            inputs: Input data to validate

        Returns:
            True if validation passes
        """
        return "data" in inputs or isinstance(inputs, (int, float))

    def _execute_impl(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute transformation and return result.

        Args:
            inputs: Validated input data
            **kwargs: Additional arguments

        Returns:
            Result containing transformed data
        """
        data = inputs.get("data", inputs)

        try:
            transformed = self.transform(data)

            self.transform_count += 1

            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"transformed": transformed},
                metadata={
                    "transform_count": self.transform_count,
                    "input_type": type(data).__name__,
                    "output_type": type(transformed).__name__
                }
            )

        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Transform failed: {str(e)}"
            )

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Perform the actual transformation.

        Subclasses must implement this method.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        raise NotImplementedError("TransformOperator must implement transform()")


class PiscesLxFilterOperator(PiscesLxBaseOperator, ABC):
    """
    Abstract base class for data filtering operators.

    This class provides the template method pattern for filter
    operations. Subclasses implement the filter() method.

    Attributes:
        filter_count: Number of items filtered
        pass_count: Number of items that passed the filter

    Usage:
        class RangeFilter(PiscesLxFilterOperator):
            @property
            def name(self) -> str:
                return "range_filter"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Filter data within specified range"

            def filter(self, data: Any) -> bool:
                return self.min <= data <= self.max
    """

    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None):
        """
        Initialize filter operator.

        Args:
            config: Optional configuration
        """
        super().__init__(config)
        self.filter_count = 0
        self.pass_count = 0

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Get input schema for filterable data.

        Returns:
            JSON Schema for filterable input
        """
        return {
            "type": "any",
            "description": "Data to be filtered"
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Get output schema for filter result.

        Returns:
            JSON Schema with passed status
        """
        return {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "data": {"type": "any"}
            }
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate filter inputs.

        Args:
            inputs: Input data to validate

        Returns:
            True if validation passes
        """
        return "data" in inputs

    def _execute_impl(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute filter and return result.

        Args:
            inputs: Validated input data
            **kwargs: Additional arguments

        Returns:
            Result containing pass/fail status
        """
        data = inputs.get("data")

        try:
            passed = self.filter(data)

            self.filter_count += 1
            if passed:
                self.pass_count += 1

            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"passed": passed, "data": data},
                metadata={
                    "filter_count": self.filter_count,
                    "pass_count": self.pass_count,
                    "pass_rate": self.pass_count / self.filter_count if self.filter_count > 0 else 0
                }
            )

        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Filter failed: {str(e)}"
            )

    @abstractmethod
    def filter(self, data: Any) -> bool:
        """
        Evaluate the filter condition.

        Subclasses must implement this method.

        Args:
            data: Data to evaluate

        Returns:
            True if data passes the filter, False otherwise
        """
        raise NotImplementedError("FilterOperator must implement filter()")
