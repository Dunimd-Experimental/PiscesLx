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
Validation Operators Module

Comprehensive validation and assertion utilities for data and configuration validation.

Features:
    - Type checking with detailed error messages
    - Value range validation
    - Custom validation rules
    - Schema validation
    - Consistent error reporting
"""

from typing import Any, Iterable, Tuple, Type, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field

from utils.dc import PiscesLxLogger
from configs.version import VERSION
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from utils.error import PiscesLxCoreValidationError


from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Opss.Validation", file_path=get_log_file("PiscesLx.Opss.Validation"), enable_file=True)


@dataclass
class POPSSValidationRule:
    """A single validation rule."""
    name: str
    validate: Callable[[Any], Tuple[bool, str]]
    error_message: str = "Validation failed"


@dataclass
class POPSSValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    validated_value: Any = None


class POPSSValidator:
    """
    A lightweight helper class for validation and assertions.
    
    Raises PiscesLxCoreValidationError with consistent error messages.
    
    Usage:
        validator = PiscesLxValidator()
        validator.require(condition, "Custom error message")
        validator.expect_type(value, str, "value_name")
        validator.expect_in(value, [1, 2, 3], "choice")
    """
    
    @staticmethod
    def require(condition: bool, message: str, **ctx: Any) -> None:
        """
        Raise a validation error if the given condition is not met.
        
        Args:
            condition: The condition to check.
            message: The error message to raise if the condition is False.
            **ctx: Additional context information.
        
        Raises:
            PiscesLxCoreValidationError: If condition is False.
        """
        if not condition:
            raise PiscesLxCoreValidationError(message, context=ctx or None)
    
    @staticmethod
    def expect_type(obj: Any, types: Union[Type, Tuple[Type, ...]], name: str) -> None:
        """
        Verify that an object is of the expected type(s).
        
        Args:
            obj: The object to check.
            types: The expected type or tuple of types.
            name: The name of the object for error reporting.
        
        Raises:
            PiscesLxCoreValidationError: If obj is not instance of types.
        """
        if not isinstance(obj, types):
            if isinstance(types, type):
                tname = types.__name__
            else:
                tname = "|".join(t.__name__ for t in types)
            raise PiscesLxCoreValidationError(
                "invalid type",
                context={"name": name, "expected": tname, "got": type(obj).__name__},
            )
    
    @staticmethod
    def expect_in(value: Any, options: Iterable[Any], name: str) -> None:
        """
        Verify that a value is among the expected options.
        
        Args:
            value: The value to check.
            options: An iterable containing the expected options.
            name: The name of the value for error reporting.
        
        Raises:
            PiscesLxCoreValidationError: If value is not in options.
        """
        options_list = list(options)
        if value not in options_list:
            raise PiscesLxCoreValidationError(
                "invalid value",
                context={"name": name, "value": value, "options": options_list},
            )
    
    @staticmethod
    def expect_range(value: float, min_val: float, max_val: float, name: str) -> None:
        """
        Verify that a numeric value is within a range.
        
        Args:
            value: The numeric value to check.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            name: The name of the value for error reporting.
        
        Raises:
            PiscesLxCoreValidationError: If value is out of range.
        """
        if not (min_val <= value <= max_val):
            raise PiscesLxCoreValidationError(
                "value out of range",
                context={"name": name, "value": value, "min": min_val, "max": max_val},
            )
    
    @staticmethod
    def expect_non_empty(value: Any, name: str) -> None:
        """
        Verify that a collection or string is not empty.
        
        Args:
            value: The value to check.
            name: The name of the value for error reporting.
        
        Raises:
            PiscesLxCoreValidationError: If value is empty.
        """
        if hasattr(value, '__len__'):
            if len(value) == 0:
                raise PiscesLxCoreValidationError(
                    "empty value",
                    context={"name": name},
                )
        elif value is None or value == "":
            raise PiscesLxCoreValidationError(
                "empty value",
                context={"name": name},
            )
    
    @staticmethod
    def expect_length(value: Any, min_len: int, max_len: int, name: str) -> None:
        """
        Verify that a collection or string has expected length.
        
        Args:
            value: The value to check.
            min_len: Minimum allowed length.
            max_len: Maximum allowed length.
            name: The name of the value for error reporting.
        
        Raises:
            PiscesLxCoreValidationError: If value length is out of range.
        """
        if not hasattr(value, '__len__'):
            raise PiscesLxCoreValidationError(
                "value has no length",
                context={"name": name},
            )
        length = len(value)
        if not (min_len <= length <= max_len):
            raise PiscesLxCoreValidationError(
                "invalid length",
                context={"name": name, "length": length, "min": min_len, "max": max_len},
            )
    
    @staticmethod
    def expect_match(value: str, pattern: str, name: str) -> None:
        """
        Verify that a string matches a regex pattern.
        
        Args:
            value: The string to check.
            pattern: Regex pattern to match.
            name: The name of the value for error reporting.
        
        Raises:
            PiscesLxCoreValidationError: If value doesn't match pattern.
        """
        import re
        if not re.match(pattern, value):
            raise PiscesLxCoreValidationError(
                "pattern mismatch",
                context={"name": name, "value": value, "pattern": pattern},
            )


class POPSSValidatorOperator(PiscesLxOperatorInterface):
    """
    Validation Operator.
    
    Provides comprehensive validation capabilities for data, configs, and schemas.
    
    Features:
        - Type checking
        - Value range validation
        - Schema validation
        - Custom validation rules
        - Batch validation
    
    Input:
        operation: Type of validation
        value: Value to validate
        expected: Expected type/value/range
        rules: Custom validation rules
    
    Output:
        POPSSValidationResult with success status and details
    """
    
    def __init__(self):
        super().__init__()
        self.name = "validation"
        self.version = VERSION
        self._LOG = PiscesLxLogger("pisceslx.ops.validation")
        self._custom_rules: Dict[str, POPSSValidationRule] = {}
    
    @property
    def description(self) -> str:
        return "Validation operator for data and configuration validation"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "operation": {"type": "str", "required": True, "enum": ["type", "range", "in", "length", "pattern", "schema", "custom", "batch"]},
            "value": {"type": "any", "required": True},
            "expected": {"type": "any", "required": False},
            "name": {"type": "str", "required": False},
            "rules": {"type": "list", "required": False},
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "success": {"type": "bool"},
            "is_valid": {"type": "bool"},
            "errors": {"type": "list"},
            "validated_value": {"type": "any"},
        }
    
    def execute(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Execute validation operation."""
        operation = inputs.get("operation", "type")
        
        try:
            if operation == "type":
                return self._validate_type(inputs)
            elif operation == "range":
                return self._validate_range(inputs)
            elif operation == "in":
                return self._validate_in(inputs)
            elif operation == "length":
                return self._validate_length(inputs)
            elif operation == "pattern":
                return self._validate_pattern(inputs)
            elif operation == "schema":
                return self._validate_schema(inputs)
            elif operation == "custom":
                return self._validate_custom(inputs)
            elif operation == "batch":
                return self._validate_batch(inputs)
            else:
                return PiscesLxOperatorResult(
                    status=PiscesLxOperatorStatus.FAILED,
                    output={},
                    error=f"Unknown operation: {operation}"
                )
        except PiscesLxCoreValidationError as e:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.FAILED,
                output={"is_valid": False, "errors": [str(e)]}
            )
        except Exception as e:
            self._LOG.error(f"Validation operation failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.FAILED,
                output={},
                error=str(e)
            )
    
    def _validate_type(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate type."""
        value = inputs.get("value")
        expected = inputs.get("expected")
        name = inputs.get("name", "value")
        
        POPSSValidator.expect_type(value, expected, name)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_range(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate numeric range."""
        value = inputs.get("value")
        expected = inputs.get("expected")
        name = inputs.get("name", "value")
        
        if isinstance(expected, (list, tuple)) and len(expected) == 2:
            min_val, max_val = expected
        else:
            min_val = expected.get("min", 0)
            max_val = expected.get("max", float('inf'))
        
        POPSSValidator.expect_range(value, min_val, max_val, name)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_in(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate value in options."""
        value = inputs.get("value")
        expected = inputs.get("expected")
        name = inputs.get("name", "value")
        
        POPSSValidator.expect_in(value, expected, name)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_length(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate length."""
        value = inputs.get("value")
        expected = inputs.get("expected")
        name = inputs.get("name", "value")
        
        if isinstance(expected, (list, tuple)) and len(expected) == 2:
            min_len, max_len = expected
        else:
            min_len = expected.get("min", 0)
            max_len = expected.get("max", float('inf'))
        
        POPSSValidator.expect_length(value, min_len, max_len, name)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_pattern(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate pattern match."""
        value = inputs.get("value")
        expected = inputs.get("expected")
        name = inputs.get("name", "value")
        
        POPSSValidator.expect_match(value, expected, name)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_schema(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate against schema."""
        import json
        
        value = inputs.get("value")
        schema = inputs.get("expected")
        errors = []
        
        if isinstance(schema, dict) and isinstance(value, dict):
            for key, expected_type in schema.items():
                if key in value:
                    try:
                        POPSSValidator.expect_type(value[key], expected_type, key)
                    except PiscesLxCoreValidationError as e:
                        errors.append(str(e))
        
        if errors:
            raise PiscesLxCoreValidationError(
                "schema validation failed",
                context={"errors": errors}
            )
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_custom(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate using custom rules."""
        value = inputs.get("value")
        rule_name = inputs.get("expected")
        
        if rule_name not in self._custom_rules:
            raise PiscesLxCoreValidationError(
                "unknown rule",
                context={"rule_name": rule_name}
            )
        
        rule = self._custom_rules[rule_name]
        is_valid, error_msg = rule.validate(value)
        
        if not is_valid:
            raise PiscesLxCoreValidationError(
                rule.error_message,
                context={"value": value, "rule": rule_name}
            )
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={"is_valid": True, "validated_value": value}
        )
    
    def _validate_batch(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Validate multiple values."""
        values = inputs.get("value", [])
        results = []
        
        for item in values:
            try:
                self.execute(item)
                results.append({"is_valid": True, "value": item})
            except PiscesLxCoreValidationError as e:
                results.append({"is_valid": False, "value": item, "error": str(e)})
        
        all_valid = all(r["is_valid"] for r in results)
        
        return PiscesLxOperatorResult(
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "is_valid": all_valid,
                "results": results,
                "success_count": sum(1 for r in results if r["is_valid"]),
                "failure_count": sum(1 for r in results if not r["is_valid"]),
            }
        )
    
    def add_rule(self, name: str, validate: Callable[[Any], Tuple[bool, str]], 
                 error_message: str = "Validation failed") -> None:
        """Add a custom validation rule."""
        self._custom_rules[name] = POPSSValidationRule(
            name=name,
            validate=validate,
            error_message=error_message
        )


__all__ = [
    "POPSSValidationRule",
    "POPSSValidationResult",
    "POPSSValidator",
    "POPSSValidatorOperator",
]
