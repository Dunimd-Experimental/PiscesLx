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

from __future__ import annotations

"""
The one place PiscesL1 is allowed to touch the standard ``logging`` module.

Project logs go through ``utils/dc.PiscesLxLogger`` (dmsc 0.1.8) — mandatory standard
§6.3. But third-party libraries (``transformers``, ``datasets``, ``modelscope``, the
ASGI server) emit *their own* stdlib loggers, and silencing them requires reaching into
stdlib: there is no dmsc handle onto a logger we do not own.

Rather than scatter ``import logging`` through a dozen modules, that single
responsibility lives here, and ``tools/check/checks_hygiene.py`` permits stdlib logging
only in this module and the two named bridge classes. Anything else fails the build.
"""

import logging
from typing import Dict, Iterable, Optional

__all__ = ["PiscesLxThirdPartyLogBridge"]


class PiscesLxThirdPartyLogBridge:
    """Quiet, capture and level control for *foreign* stdlib loggers only."""

    #: libraries whose chatter is suppressed during dataset download and loading
    DEFAULT_NOISY: Dict[str, str] = {
        "modelscope": "ERROR",
        "datasets": "WARNING",
        "transformers": "WARNING",
        "huggingface_hub": "WARNING",
        "urllib3": "WARNING",
    }

    @classmethod
    def quiet_default(cls) -> int:
        """Apply the standard suppression set. Returns how many loggers were tuned."""
        return cls.quiet(cls.DEFAULT_NOISY)

    @classmethod
    def quiet(cls, levels: Dict[str, str]) -> int:
        """Set a floor on named third-party loggers; unknown names are skipped."""
        tuned = 0
        for name, level in levels.items():
            resolved = getattr(logging, level.upper(), None)
            if resolved is None:
                continue
            logging.getLogger(name).setLevel(resolved)
            tuned += 1
        return tuned

    @classmethod
    def capture(cls, logger_name: Optional[str], sink_handler: logging.Handler) -> bool:
        """Attach a handler to a foreign logger so the UI can show its output."""
        target = logging.getLogger(logger_name or "")
        target.addHandler(sink_handler)
        return True

    @classmethod
    def release(cls, logger_name: Optional[str], sink_handler: logging.Handler) -> bool:
        """Detach a previously attached capture handler."""
        target = logging.getLogger(logger_name or "")
        target.removeHandler(sink_handler)
        return True

    @classmethod
    def configure_server_logging(cls, level: str, fmt: str) -> bool:
        """One-shot basic configuration for the ASGI/server process log stream."""
        resolved = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(level=resolved, format=fmt, force=False)
        return True

    @staticmethod
    def make_record(text: str, level_name: str) -> logging.LogRecord:
        """Build a record shaped like the stdlib's, for capture sinks."""
        level = getattr(logging, level_name.upper(), logging.INFO)
        return logging.LogRecord(
            name="PiscesLx.Dev", level=level, pathname=__file__,
            lineno=0, msg=text, args=(), exc_info=None,
        )
