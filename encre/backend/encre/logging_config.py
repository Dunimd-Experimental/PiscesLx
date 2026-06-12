#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

"""Structured logging configuration for encre.

Uses loguru when available, with a transparent fallback to the stdlib ``logging``
module so the library does not force a particular logging dependency on
downstream consumers.

Usage::

    from encre.logging_config import setup_logging, get_logger

    setup_logging(level="DEBUG", json_format=False)
    logger = get_logger(__name__)
    logger.info("agent started", extra={"turn": 1})
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_LOGURU_AVAILABLE: bool = False
try:
    from loguru import logger as _loguru_logger  # type: ignore[import-untyped]
    _LOGURU_AVAILABLE = True
except ImportError:
    _loguru_logger = None  # type: ignore[assignment]


class _StdlibLogger:
    """Thin wrapper that exposes a loguru-like API backed by stdlib ``logging``."""

    def __init__(self, name: str = "encre") -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, *args: Any, **kwargs: Any) -> None:
        extra: dict[str, Any] = kwargs.pop("extra", {}) or {}
        if extra:
            message = f"{message} | {extra!r}"
        self._logger.log(level, message, *args, **kwargs)

    def trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG - 5, message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, message, *args, **kwargs)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO + 5, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, *args, **kwargs)


def _get_loguru_serializer(json_format: bool = False):
    """Return a serialiser callable for loguru, or None for default formatting."""
    if not json_format:
        return None

    try:
        import json as _json
    except ImportError:
        return None

    def _serialize(record: Any) -> str:
        """Format a loguru record as a single JSON line."""
        subset: dict[str, Any] = {
            "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if record["time"] else "",
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        if record["extra"]:
            subset["extra"] = dict(record["extra"])
        if record["exception"]:
            subset["exception"] = str(record["exception"])
        return _json.dumps(subset, ensure_ascii=False, default=str) + "\n"

    return _serialize


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str = "",
    rotation: str = "10 MB",
    retention: str = "7 days",
    intercept_stdlib: bool = True,
) -> None:
    """Configure encre logging.

    When **loguru** is installed the function removes the default loguru
    handler, installs a new ``sys.stderr`` sink with the requested format, and
    optionally routes stdlib ``logging`` records through loguru (``intercept_stdlib``).

    When loguru is **not** installed a basic ``logging.StreamHandler`` is
    configured on the ``"encre"`` logger.  Downstream code that uses
    :func:`get_logger` will receive a ``_StdlibLogger`` wrapper so the API
    remains consistent.

    Args:
        level: One of ``TRACE``, ``DEBUG``, ``INFO``, ``SUCCESS``,
            ``WARNING``, ``ERROR``, ``CRITICAL``.  Case-insensitive.
        json_format: Emit JSON lines instead of human-readable text.
        log_file: Optional path for a persistent log file.  If empty, defaults
            to ``<data_dir>/logs/encre.log`` (``~/.dunimd/encre/logs/encre.log``).
        rotation: When to rotate the log file (loguru syntax).
        retention: How long to keep rotated logs.
        intercept_stdlib: Redirect stdlib ``logging`` to loguru (only
            effective when loguru is available).
    """
    level = level.upper()

    if not log_file:
        from encre.config import get_data_dir
        _log_dir = get_data_dir() / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(_log_dir / "encre.log")

    if _LOGURU_AVAILABLE:
        assert _loguru_logger is not None
        _loguru_logger.remove()  # Remove default stderr handler

        # Determine format string
        if json_format:
            fmt_kwargs = {"serialize": True}
        else:
            fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            fmt_kwargs = {"format": fmt}

        _loguru_logger.add(
            sys.stderr,
            level=level,
            colorize=True,
            **fmt_kwargs,
        )

        if log_file:
            _loguru_logger.add(
                log_file,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
                **fmt_kwargs,
            )

        if intercept_stdlib:
            # Re-route stdlib logging through loguru
            class _InterceptHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        lvl = _loguru_logger.level(record.levelname).name
                    except ValueError:
                        lvl = record.levelno
                    frame = logging.currentframe()
                    depth = 2
                    while frame and frame.f_code.co_filename == logging.__file__:
                        frame = frame.f_back
                        depth += 1
                    _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                        lvl, record.getMessage()
                    )

            logging.basicConfig(handlers=[_InterceptHandler()], level=logging.NOTSET, force=True)
    else:
        # Fallback: stdlib logging
        yim_logger = logging.getLogger("encre")
        yim_logger.setLevel(getattr(logging, level, logging.INFO))
        yim_logger.handlers.clear()

        handler: logging.Handler
        if json_format:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "function": "%(funcName)s", '
                '"line": %(lineno)d, "message": %(message)s}',
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
        else:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))

        yim_logger.addHandler(handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(handler.formatter)
            yim_logger.addHandler(file_handler)


def get_logger(name: str = "encre") -> Any:
    """Return a logger for *name*.

    Returns a **loguru** logger when loguru is installed, otherwise a
    :class:`_StdlibLogger` wrapper that presents a compatible API.
    """
    if _LOGURU_AVAILABLE:
        assert _loguru_logger is not None
        return _loguru_logger.bind(name=name)
    return _StdlibLogger(name)


# Module-level convenience logger for sub-modules that just want
# ``from encre.logging_config import logger``.
logger = get_logger("encre")
