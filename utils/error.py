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

import traceback
from enum import Enum
from typing import Any, Dict, Optional

# Use dms_core logging instead of standard logging
import dms_core
logger = dms_core.log

class PiscesLxCoreErrorCode(Enum):
    """Enumeration of error codes for PiscesL1 utilities."""
    UNKNOWN = "unknown"  # Unknown error
    VALIDATION = "validation"  # Input/data/schema validation error
    CONCURRENCY = "concurrency"  # Thread/process/lock related error
    TIMEOUT = "timeout"  # Operation timeout error
    NETWORK = "network"  # Network/HTTP/Socket related error

class PiscesLxCoreError(Exception):
    """Base error class for all PiscesL1 utilities.
    
    Prefer raising a subclass with an appropriate error code.
    Automatically logs errors to the project's logging system.
    """
    
    # Class-level logger instance
    _logger = None
    
    @classmethod
    def _get_logger(cls):
        """Get module-level logger (standardized)."""
        return logger

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
        code: PiscesLxCoreErrorCode = PiscesLxCoreErrorCode.UNKNOWN,
    ) -> None:
        """Initialize a PiscesLxCoreError instance.

        Args:
            message (str): Error message describing the issue.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
            code (PiscesLxCoreErrorCode): Error code indicating the type of error. Defaults to UNKNOWN.
        """
        self.message = message
        self.context: Dict[str, Any] = dict(context or {})
        self.cause = cause
        self.code = code
        
        # Log the error automatically
        logger = self._get_logger()
        error_data = self.to_dict(include_cause=True)
        logger._Ferror("PiscesLx.Core.Error", f"{self.__class__.__name__} occurred: {self.message}")
        
        # Keep Exception args minimal to avoid overly long reprs
        super().__init__(self.__str__())

    def __str__(self) -> str:
        """Return a string representation of the error.

        Returns:
            str: String representation including error code, message, and context if available.
        """
        base = f"{self.code.value}: {self.message}" if self.code else self.message
        if self.context:
            return f"{base} | context={self.context}"
        return base

    def __repr__(self) -> str:
        """Return a detailed string representation of the error instance.

        Returns:
            str: Detailed representation including error code, message, and context.
        """
        return f"{self.__class__.__name__}(code={self.code.value!r}, message={self.message!r}, context={self.context!r})"

    def with_context(self, **more: Any) -> "PiscesLxCoreError":
        """Fluently attach additional context and return self.

        Args:
            **more (Any): Additional context key-value pairs.

        Returns:
            PiscesLxCoreError: Self with updated context.
        """
        self.context.update(more)
        return self

    def to_dict(self, include_cause: bool = False) -> Dict[str, Any]:
        """Serialize error for logs/reports/APIs.

        Args:
            include_cause (bool): Whether to include the cause information in the output. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing error information.
        """
        data: Dict[str, Any] = {
            "type": self.__class__.__name__,
            "code": self.code.value,
            "message": self.message,
            "context": self.context,
        }
        if include_cause and self.cause is not None:
            data["cause_type"] = self.cause.__class__.__name__
            data["cause_message"] = str(self.cause)
            try:
                data["cause_traceback"] = traceback.format_exception_only(type(self.cause), self.cause)[-1].strip()
            except Exception as log_e:
                logger._Fdebug("PiscesLx.Core.Error", f"ERROR_TRACEBACK_FAILED: {str(log_e)}")
        return data

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        message: Optional[str] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        code: PiscesLxCoreErrorCode = PiscesLxCoreErrorCode.UNKNOWN,
    ) -> "PiscesLxCoreError":
        """Wrap an arbitrary exception with standardized error.

        Args:
            exc (BaseException): The original exception to wrap.
            message (Optional[str]): Custom error message. If None, the original exception's message will be used. Defaults to None.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            code (PiscesLxCoreErrorCode): Error code indicating the type of error. Defaults to UNKNOWN.

        Returns:
            PiscesLxCoreError: A new PiscesLxCoreError instance wrapping the original exception.
        """
        return cls(message or str(exc), context=context, cause=exc, code=code)

class PiscesLxCoreValidationError(PiscesLxCoreError):
    """Error class for input/data/schema validation failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreValidationError instance.

        Args:
            message (str): Error message describing the validation failure.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.VALIDATION)

class PiscesLxCoreConcurrencyError(PiscesLxCoreError):
    """Error class for thread/process/lock/timeouts related failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreConcurrencyError instance.

        Args:
            message (str): Error message describing the concurrency error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.CONCURRENCY)

class PiscesLxCoreTimeoutError(PiscesLxCoreError):
    """Error class for operation timeouts."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreTimeoutError instance.

        Args:
            message (str): Error message describing the timeout error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.TIMEOUT)

class PiscesLxCoreNetworkError(PiscesLxCoreError):
    """Error class for network/HTTP/Socket related failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreNetworkError instance.

        Args:
            message (str): Error message describing the network error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.NETWORK)

class PiscesLxCoreMemoryError(PiscesLxCoreError):
    """Error class for memory allocation/management failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreMemoryError instance.

        Args:
            message (str): Error message describing the memory error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.CONCURRENCY)

__all__ = [
    # base
    "PiscesLxCoreError",
    "PiscesLxCoreErrorCode",
    # validation
    "PiscesLxCoreValidationError",
    # runtime
    "PiscesLxCoreConcurrencyError",
    "PiscesLxCoreTimeoutError",
    "PiscesLxCoreNetworkError",
    "PiscesLxCoreMemoryError",
]