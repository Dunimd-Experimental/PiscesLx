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

import traceback
from enum import Enum
from typing import Any, Dict, Optional
from utils.log.core import PiscesLxCoreLog

# Module-level fallback logger for debug paths where delayed logger may fail
logger = PiscesLxCoreLog("PiscesLx.Utils.Error")

class PiscesLxCoreErrorCode(Enum):
    """Enumeration of error codes for PiscesL1 utilities."""
    UNKNOWN = "unknown"  # Unknown error
    VALIDATION = "validation"  # Input/data/schema validation error
    CONFIG = "config"  # Configuration loading/merging error
    IO = "io"  # Generic I/O error
    FILESYSTEM = "filesystem"  # Filesystem operation error
    CONCURRENCY = "concurrency"  # Thread/process/lock related error
    TIMEOUT = "timeout"  # Operation timeout error
    NETWORK = "network"  # Network/HTTP/Socket related error
    CACHE = "cache"  # Cache operation error
    LOG = "log"  # Logging pipeline error
    HOOKS = "hooks"  # Events bus/registry/dispatch error
    OBSERVABILITY = "observability"  # Observability service error
    REPORTER = "reporter"  # Report rendering/writing error
    METRICS = "metrics"  # Metrics collection/flush/export error
    EXPORTER = "exporter"  # Exporter initialization/transport error
    DEVICE = "device"  # Device detection/setup/runtime error


class PiscesLxCoreError(Exception):
    """Base error class for all PiscesL1 utilities.
    
    Prefer raising a subclass with an appropriate error code.
    Automatically logs errors to the project's logging system.
    """
    
    # Class-level logger instance
    _logger = None
    
    @classmethod
    def _get_logger(cls) -> "PiscesLxCoreLog":
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
        logger.error(
            f"{self.__class__.__name__} occurred",
            data=error_data
        )
        
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
                logger.debug("ERROR_TRACEBACK_FAILED", error=str(log_e))
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


# ---------- Validation & Config ----------
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


class PiscesLxCoreConfigError(PiscesLxCoreError):
    """Error class for configuration loading/merging failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreConfigError instance.

        Args:
            message (str): Error message describing the configuration error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.CONFIG)


# ---------- Filesystem / IO ----------
class PiscesLxCoreIOError(PiscesLxCoreError):
    """Error class for generic I/O operations (read/write/serialize)."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreIOError instance.

        Args:
            message (str): Error message describing the I/O error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.IO)


class PiscesLxCoreFilesystemError(PiscesLxCoreError):
    """Error class for filesystem operations (mkdir/permissions/path)."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreFilesystemError instance.

        Args:
            message (str): Error message describing the filesystem error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.FILESYSTEM)


# ---------- Runtime / Concurrency / Network ----------
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


# ---------- Domain-specific (utils subpackages) ----------
class PiscesLxCoreCacheError(PiscesLxCoreError):
    """Error class for cache put/get/evict failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreCacheError instance.

        Args:
            message (str): Error message describing the cache error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.CACHE)


class PiscesLxCoreLogError(PiscesLxCoreError):
    """Error class for logging pipeline failures (handlers/formatters)."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreLogError instance.

        Args:
            message (str): Error message describing the logging error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.LOG)


class PiscesLxCoreHooksError(PiscesLxCoreError):
    """Error class for events bus/registry/dispatch failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreHooksError instance.

        Args:
            message (str): Error message describing the hooks error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.HOOKS)


class PiscesLxCoreObservabilityError(PiscesLxCoreError):
    """Error class for observability service-level failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreObservabilityError instance.

        Args:
            message (str): Error message describing the observability error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.OBSERVABILITY)


class PiscesLxCoreReporterError(PiscesLxCoreError):
    """Error class for report rendering/writing failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreReporterError instance.

        Args:
            message (str): Error message describing the reporter error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.REPORTER)


class PiscesLxCoreMetricsError(PiscesLxCoreError):
    """Error class for metrics collection/flush/export failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreMetricsError instance.

        Args:
            message (str): Error message describing the metrics error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.METRICS)


class PiscesLxCoreExporterError(PiscesLxCoreError):
    """Error class for exporter initialization/transport failures (Prom/OTLP/etc)."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreExporterError instance.

        Args:
            message (str): Error message describing the exporter error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.EXPORTER)


class PiscesLxCoreDeviceError(PiscesLxCoreError):
    """Error class for device detection/setup/runtime failures."""
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreDeviceError instance.

        Args:
            message (str): Error message describing the device error.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        super().__init__(message, context=context, cause=cause, code=PiscesLxCoreErrorCode.DEVICE)


# ---- Device-specific subclasses (migrated from utils/device/exceptions.py) ----
class PiscesLxCoreNoGPUError(PiscesLxCoreDeviceError):
    """Error raised when no GPU is detected but GPU computation is required."""
    def __init__(self, message: str = "No GPU detected", *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreNoGPUError instance.

        Args:
            message (str): Error message describing the issue. Defaults to "No GPU detected".
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        ctx = {"error_code": "NO_GPU_DETECTED"}
        if context:
            ctx.update(context)
        super().__init__(message, context=ctx, cause=cause)


class PiscesLxCoreGPUInsufficientError(PiscesLxCoreDeviceError):
    """Error raised when available GPU memory is insufficient for the requested operation."""
    def __init__(self, required_gb: int, available_gb: int, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreGPUInsufficientError instance.

        Args:
            required_gb (int): Required GPU memory in gigabytes.
            available_gb (int): Available GPU memory in gigabytes.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        message = f"Insufficient GPU memory: required {required_gb}GB, available {available_gb}GB"
        ctx = {"error_code": "GPU_MEMORY_INSUFFICIENT", "required_gb": required_gb, "available_gb": available_gb}
        if context:
            ctx.update(context)
        super().__init__(message, context=ctx, cause=cause)


class PiscesLxCorePlatformDetectionError(PiscesLxCoreDeviceError):
    """Error raised when platform-specific GPU detection fails."""
    def __init__(self, platform: str, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCorePlatformDetectionError instance.

        Args:
            platform (str): Platform name where GPU detection failed.
            message (str): Error message describing the failure.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        full_message = f"{platform} platform detection failed: {message}"
        ctx = {"error_code": "PLATFORM_DETECTION_FAILED", "platform": platform}
        if context:
            ctx.update(context)
        super().__init__(full_message, context=ctx, cause=cause)


class PiscesLxCoreDistributedSetupError(PiscesLxCoreDeviceError):
    """Error raised when distributed training/inference setup fails."""
    def __init__(self, message: str, *, world_size: Optional[int] = None, rank: Optional[int] = None, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreDistributedSetupError instance.

        Args:
            message (str): Error message describing the setup failure.
            world_size (Optional[int]): Total number of processes in the distributed setup. Defaults to None.
            rank (Optional[int]): Rank of the current process in the distributed setup. Defaults to None.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        ctx = {"error_code": "DISTRIBUTED_SETUP_FAILED", "world_size": world_size, "rank": rank}
        if context:
            ctx.update(context)
        super().__init__(message, context=ctx, cause=cause)


class PiscesLxCoreDeviceOrchestrationError(PiscesLxCoreDeviceError):
    """Error raised when device orchestration or strategy selection fails."""
    def __init__(self, strategy: str, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreDeviceOrchestrationError instance.

        Args:
            strategy (str): Device orchestration strategy that failed.
            message (str): Error message describing the failure.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        ctx = {"error_code": "ORCHESTRATION_FAILED", "strategy": strategy}
        if context:
            ctx.update(context)
        super().__init__(message, context=ctx, cause=cause)


class PiscesLxCoreConfigurationError(PiscesLxCoreDeviceError):
    """Error raised when device configuration is invalid or inconsistent."""
    def __init__(self, config_key: str, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        """Initialize a PiscesLxCoreConfigurationError instance.

        Args:
            config_key (str): Configuration key associated with the error.
            message (str): Error message describing the configuration issue.
            context (Optional[Dict[str, Any]]): Additional context information about the error. Defaults to None.
            cause (Optional[BaseException]): The original exception that caused this error. Defaults to None.
        """
        ctx = {"error_code": "INVALID_CONFIGURATION", "config_key": config_key}
        if context:
            ctx.update(context)
        super().__init__(message, context=ctx, cause=cause)

__all__ = [
    # base
    "PiscesLxCoreError",
    "PiscesLxCoreErrorCode",
    # validation & config
    "PiscesLxCoreValidationError",
    "PiscesLxCoreConfigError",
    # fs/io
    "PiscesLxCoreIOError",
    "PiscesLxCoreFilesystemError",
    # runtime
    "PiscesLxCoreConcurrencyError",
    "PiscesLxCoreTimeoutError",
    "PiscesLxCoreNetworkError",
    # subpackages
    "PiscesLxCoreCacheError",
    "PiscesLxCoreLogError",
    "PiscesLxCoreHooksError",
    "PiscesLxCoreObservabilityError",
    "PiscesLxCoreReporterError",
    "PiscesLxCoreMetricsError",
    "PiscesLxCoreExporterError",
    "PiscesLxCoreDeviceError",
    # device-specific
    "PiscesLxCoreNoGPUError",
    "PiscesLxCoreGPUInsufficientError",
    "PiscesLxCorePlatformDetectionError",
    "PiscesLxCoreDistributedSetupError",
    "PiscesLxCoreDeviceOrchestrationError",
    "PiscesLxCoreConfigurationError",
]