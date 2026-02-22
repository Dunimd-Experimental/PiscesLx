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
Core utilities module for PiscesLx.

This module provides basic utilities for error handling, concurrency,
and other core functionality.

For advanced operators (quantization, checkpoint, scaling, validation, concurrency),
please use the opss module instead:
    from opss.quantize import QuantizationEngine
    from opss.train.checkpoint import CheckpointOperator
    from opss.scaling import PiscesLxChinchillaScaler
    from opss.validation import ValidatorOperator
    from opss.concurrency import ConcurrencyOperator
"""

from utils.dc import (
    PiscesLxCoreError,
    PiscesLxCoreErrorCode,
    PiscesLxCoreErrorContext,
    PiscesLxCoreLogLevel,
)


class PiscesLxCoreValidationError(PiscesLxCoreError):
    """Validation-specific error."""

    def __init__(
        self,
        message: str,
        field: str = "unknown",
        value = None,
        context = None
    ):
        self.field = field
        self.value = value
        super().__init__(
            PiscesLxCoreErrorCode.VALIDATION_ERROR,
            message,
            context
        )


class PiscesLxCoreConcurrencyError(PiscesLxCoreError):
    """Concurrency-specific error."""

    def __init__(
        self,
        message: str,
        resource: str = "unknown",
        context = None
    ):
        self.resource = resource
        super().__init__(
            PiscesLxCoreErrorCode.INTERNAL_ERROR,
            message,
            context
        )


class PiscesLxCoreTimeoutError(PiscesLxCoreError):
    """Timeout-specific error."""

    def __init__(
        self,
        message: str,
        timeout_ms: float = 0.0,
        context = None
    ):
        self.timeout_ms = timeout_ms
        super().__init__(
            PiscesLxCoreErrorCode.TIMEOUT_ERROR,
            message,
            context
        )


class PiscesLxCoreNetworkError(PiscesLxCoreError):
    """Network-specific error."""

    def __init__(
        self,
        message: str,
        endpoint: str = "unknown",
        context = None
    ):
        self.endpoint = endpoint
        super().__init__(
            PiscesLxCoreErrorCode.SERVICE_UNAVAILABLE,
            message,
            context
        )


class PiscesLxCoreMemoryError(PiscesLxCoreError):
    """Memory-specific error."""

    def __init__(
        self,
        message: str,
        required_bytes: int = 0,
        available_bytes: int = 0,
        context = None
    ):
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        super().__init__(
            PiscesLxCoreErrorCode.INTERNAL_ERROR,
            message,
            context
        )


__all__ = [
    'PiscesLxCoreError',
    'PiscesLxCoreErrorCode',
    'PiscesLxCoreErrorContext',
    'PiscesLxCoreLogLevel',
    'PiscesLxCoreValidationError',
    'PiscesLxCoreConcurrencyError',
    'PiscesLxCoreTimeoutError',
    'PiscesLxCoreNetworkError',
    'PiscesLxCoreMemoryError',
]
