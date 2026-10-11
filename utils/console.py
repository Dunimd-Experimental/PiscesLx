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

"""Console encoding guard for the tooling.

The acceptance suites print status glyphs (check marks, crosses) and metric names
that are outside the legacy Windows codepages. On a console whose default encoding
is cp936/cp1252, ``print`` raises ``UnicodeEncodeError`` *inside* the reporter, so a
run that is otherwise green aborts while writing its own summary — the tool fails to
report rather than the model failing to train.

`PiscesLxConsoleEncoding.ensure_utf8` reconfigures the standard streams to UTF-8 with
substitution, so reporting can never be the thing that breaks a run. It is idempotent
and a no-op on streams that cannot be reconfigured (captured pipes, notebooks).
"""

import sys

__all__ = ["PiscesLxConsoleEncoding"]


class PiscesLxConsoleEncoding:
    """Force UTF-8 output on the standard streams of a tooling process."""

    UTF8 = "utf-8"
    # Unencodable characters become '?' instead of raising inside a print call.
    ERROR_POLICY = "replace"
    STREAMS = ("stdout", "stderr")

    @classmethod
    def ensure_utf8(cls) -> bool:
        """Reconfigure stdout and stderr to UTF-8 with substitution.

        Returns:
            bool: True when at least one stream was reconfigured to UTF-8.
        """
        adjusted = False
        for name in cls.STREAMS:
            stream = getattr(sys, name, None)
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is None:
                continue
            try:
                reconfigure(encoding=cls.UTF8, errors=cls.ERROR_POLICY)
            except (ValueError, OSError):
                # A detached or closed stream simply keeps whatever it had.
                continue
            adjusted = True
        return adjusted

    @classmethod
    def child_environment(cls, environment: dict) -> dict:
        """Return a copy of `environment` that pins child-process output to UTF-8.

        Args:
            environment (dict): Parent environment mapping used to spawn a child.

        Returns:
            dict: Environment with ``PYTHONIOENCODING`` set to UTF-8/replace.
        """
        updated = dict(environment)
        updated["PYTHONIOENCODING"] = f"{cls.UTF8}:{cls.ERROR_POLICY}"
        return updated
