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

"""EntaSandbox — wrapper around the sandbox supervisor for tool execution.

Provides a clean interface for executing tools inside the EnTA sandbox
and checking for policy violations.
"""

from typing import Any


class EntaSandbox:
    """Wrapper around the EnTA sandbox supervisor.

    Delegates tool execution to an adapter and tracks policy violations.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        max_violations: int = 8,
        max_steps: int = 32,
    ) -> None:
        """Initialise the sandbox wrapper.

        Args:
            adapter: The tool adapter (e.g. ``_EncreToolAdapter``) used
                to execute individual tool calls.
            max_violations: Maximum allowed sandbox violations per trajectory.
            max_steps: Maximum allowed tool-call steps per trajectory.
        """
        self._adapter = adapter
        self._max_violations = int(max_violations)
        self._max_steps = int(max_steps)

    @property
    def max_violations(self) -> int:
        return self._max_violations

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return the OpenAI-format tool definitions from the adapter."""
        return self._adapter.openai_tools()

    async def execute_tool(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[str, bool, int]:
        """Execute a single tool and return ``(result, is_error, elapsed_ms)``."""
        return await self._adapter.execute(
            name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )

    def check_violations(self, steps: list[Any]) -> int:
        """Count sandbox policy violations across a list of steps.

        Args:
            steps: List of step records (must have ``tool``, ``is_error``,
                ``result`` attributes).

        Returns:
            Number of policy violations found.
        """
        import json

        violations = 0
        for step in steps:
            if step.tool != "bash":
                continue
            if not step.is_error:
                continue
            payload = step.result or ""
            try:
                parsed = json.loads(payload)
            except (ValueError, TypeError):
                parsed = {}
            if isinstance(parsed, dict):
                err = str(parsed.get("error", ""))
                lower = err.lower()
                if any(
                    token in lower
                    for token in (
                        "sandbox",
                        "violation",
                        "blocked",
                        "denied",
                        "permission",
                        "seccomp",
                    )
                ):
                    violations += 1
        return violations

    def is_exhausted(self, step_count: int, violation_count: int) -> bool:
        """True when the trajectory exceeded the safety/step budget."""
        return (
            step_count >= self._max_steps
            or violation_count >= self._max_violations
        )
