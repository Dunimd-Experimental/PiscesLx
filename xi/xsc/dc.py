#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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
Xi Core Library

Core utilities for Xi Studio and Xi Server.

Components:
    - XiLogger: Structured logging
    - XiLogLevel: Log level enumeration
    - XiErrorCode: Error code enumeration
    - XiErrorContext: Error context dataclass
    - XiError: Error exception class
"""

import datetime
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class XiLogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class XiErrorCode(Enum):
    INTERNAL_ERROR = ("X-001", "Internal server error")
    CONFIG_PARSE_ERROR = ("X-002", "Configuration parse error")
    FILESYSTEM_ERROR = ("X-003", "Filesystem operation error")
    VALIDATION_ERROR = ("X-004", "Validation error")
    SERVICE_UNAVAILABLE = ("X-005", "Service unavailable")
    TIMEOUT_ERROR = ("X-006", "Operation timeout")


@dataclass(frozen=True)
class XiErrorContext:
    component: str = "unknown"
    operation: str = "unknown"
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class XiError(Exception):
    def __init__(
        self,
        code: XiErrorCode,
        message: Optional[str] = None,
        context: Optional[XiErrorContext] = None
    ):
        self.code = code
        self.message = message or code.value[1]
        self.context = context or XiErrorContext()
        super().__init__(self.message)


class XiLogger:
    def __init__(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        enable_file: bool = False
    ):
        self.name = name
        self._context = context or {}
        self._file_path = file_path
        self._enable_file = enable_file
        self._file_handler = None
        self._lock = threading.Lock()

        if enable_file and file_path:
            self._init_file_handler(file_path)

    def _init_file_handler(self, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._file_handler = open(file_path, 'a', encoding='utf-8')
        except Exception:
            self._file_handler = None

    def _write_to_file(self, level: str, msg: str) -> None:
        if self._file_handler:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            level_fixed = level[:5].ljust(5)
            with self._lock:
                self._file_handler.write(f"{timestamp} | {level_fixed} | {self.name} | {msg}\n")
                self._file_handler.flush()

    def debug(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        print(f"[DEBUG] {self.name}: {formatted}")
        self._write_to_file("DEBUG", formatted)

    def info(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        print(f"[INFO] {self.name}: {formatted}")
        self._write_to_file("INFO", formatted)

    def warning(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        print(f"[WARNING] {self.name}: {formatted}")
        self._write_to_file("WARNING", formatted)

    def error(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        print(f"[ERROR] {self.name}: {formatted}")
        self._write_to_file("ERROR", formatted)

    def critical(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        print(f"[CRITICAL] {self.name}: {formatted}")
        self._write_to_file("CRITICAL", formatted)

    def _format(self, msg: str, **kwargs) -> str:
        if self._context:
            ctx_str = " ".join(f"{k}={v}" for k, v in self._context.items())
            msg = f"[{ctx_str}] {msg}"
        if kwargs:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = f"{msg} {extra}"
        return msg

    def with_context(self, **context) -> 'XiLogger':
        return XiLogger(self.name, {**self._context, **context})

    def __del__(self):
        if self._file_handler:
            try:
                self._file_handler.close()
            except Exception:
                pass


__all__ = [
    "XiLogLevel",
    "XiErrorCode",
    "XiErrorContext",
    "XiError",
    "XiLogger",
]
