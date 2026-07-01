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

"""EntaTeacherClient — wrapper around the remote teacher / roundtable mechanism.

Provides a clean interface for generating training data from one or more
teacher models, either individually or via a roundtable (multi-teacher + judge).
"""

from typing import Any, Dict, List, Sequence, Tuple


class EntaTeacherClient:
    """Wrapper around the EnTA remote teacher / roundtable mechanism.

    Delegates to ``enta.build_roundtable_from_config`` and related
    utilities for teacher-driven data generation.
    """

    def __init__(
        self,
        enta_module: Any,
        cfg: Any,
    ) -> None:
        """Initialise the teacher client.

        Args:
            enta_module: The lazily-loaded EnTA module (``_bind_enta()``
                result).
            cfg: Configuration namespace with teacher specifications.
        """
        self._enta = enta_module
        self._cfg = cfg

        # Try to build a roundtable from config; None if not configured.
        try:
            self._roundtable = enta_module.build_roundtable_from_config(cfg)
        except Exception:
            self._roundtable = None

    @property
    def roundtable(self) -> Any | None:
        """The configured :class:`enta.TeacherRoundtable` or ``None``."""
        return self._roundtable

    def build_dataset(
        self,
        prompts: Sequence[Tuple[str, str]],
        *,
        system: str | None = None,
    ) -> List[Tuple[str, str, Any]]:
        """Generate training data via the roundtable.

        For each ``(prompt, _)`` pair the roundtable is invoked; the
        highest-scoring teacher response is selected as reference.

        Args:
            prompts: Sequence of ``(prompt, _)`` pairs.
            system: Optional system message for teachers.

        Returns:
            List of ``(prompt, reference, roundtable_result)`` tuples.
            Empty list when no roundtable is configured.
        """
        if self._roundtable is None:
            return []
        return self._run_roundtable(prompts, system=system)

    def _run_roundtable(
        self,
        items: Sequence[Tuple[str, str]],
        *,
        system: str | None = None,
    ) -> List[Tuple[str, str, Any]]:
        """Synchronously drive the async roundtable for every item."""
        import asyncio

        async def _drive() -> List[Tuple[str, str, Any]]:
            out: List[Tuple[str, str, Any]] = []
            for prompt, _ in items:
                messages: List[Dict[str, Any]] = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                result = await self._roundtable.run(prompt, messages=messages)
                if result.selected is not None and result.selected.text:
                    out.append((prompt, result.selected.text, result))
            return out

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(lambda: asyncio.run(_drive())).result()
            return loop.run_until_complete(_drive())
        except RuntimeError:
            return asyncio.run(_drive())
