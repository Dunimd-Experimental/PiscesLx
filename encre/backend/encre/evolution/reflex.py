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

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReflexResult:
    turn_number: int
    score: float  # 0.0 - 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    should_retry: bool = False
    timestamp: float = field(default_factory=time.time)


class EncreReflexLoop:
    """Lightweight self-reflection that runs after each turn.

    Does NOT call an LLM — uses heuristic rules on tool outcomes to produce
    actionable improvement hints. This keeps it fast and free.
    """

    MAX_HISTORY = 50

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._history: list[ReflexResult] = []
        self._consecutive_failures: dict[str, int] = {}
        self._turn_scores: list[float] = []

    def reflect(
        self,
        turn_number: int,
        tool_results: list[dict[str, Any]],
        turn_latency_ms: float,
    ) -> ReflexResult:
        if not self.enabled:
            return ReflexResult(turn_number=turn_number, score=1.0)

        total = len(tool_results)
        if total == 0:
            score = 0.5
            return ReflexResult(
                turn_number=turn_number,
                score=score,
                issues=["No tools were called this turn — agent may have given up too early"],
            )

        errors = [r for r in tool_results if r.get("is_error")]
        successes = [r for r in tool_results if not r.get("is_error")]
        error_rate = len(errors) / total if total > 0 else 0

        issues: list[str] = []
        suggestions: list[str] = []

        for err in errors:
            tool_name = err.get("tool_name", "unknown")
            self._consecutive_failures[tool_name] = self._consecutive_failures.get(tool_name, 0) + 1
            c = self._consecutive_failures[tool_name]
            if c >= 3:
                issues.append(f"Tool [{tool_name}] has failed {c} consecutive times")
                suggestions.append(f"Consider a completely different approach — [{tool_name}] is not working")

        for r in successes:
            tool_name = r.get("tool_name", "")
            self._consecutive_failures[tool_name] = 0

        # Latency check
        if turn_latency_ms > 60000:
            issues.append(f"Turn took {turn_latency_ms / 1000:.0f}s — very slow")
            suggestions.append("Consider batching operations or reducing scope per turn")

        # Repeated identical calls
        call_names = [r.get("tool_name", "") for r in tool_results]
        duplicates = {n: call_names.count(n) for n in set(call_names) if call_names.count(n) > 3}
        if duplicates:
            issues.append(f"Repeated tool calls detected: {duplicates}")
            suggestions.append("Consolidate repeated calls into fewer, broader invocations")

        # Score
        score = 1.0 - (error_rate * 0.7)
        if issues:
            score = max(0.1, score - 0.15 * len(issues))
        self._turn_scores.append(score)
        if len(self._turn_scores) > 20:
            self._turn_scores.pop(0)

        should_retry = error_rate > 0.5 and total > 1

        result = ReflexResult(
            turn_number=turn_number,
            score=round(score, 3),
            issues=issues,
            suggestions=suggestions,
            should_retry=should_retry,
        )
        self._history.append(result)
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)
        return result

    def get_improvement_context(self) -> str:
        if not self._history:
            return ""
        recent = self._history[-3:]
        hints: list[str] = []
        for r in recent:
            if r.suggestions:
                hints.extend(f"[Turn {r.turn_number}] {s}" for s in r.suggestions)
        if not hints:
            return ""
        return "**Self-reflection hints (recent turns):**\n" + "\n".join(f"  - {h}" for h in hints)

    def get_trend(self) -> str:
        if len(self._turn_scores) < 3:
            return "stable"
        recent = self._turn_scores[-3:]
        if recent[-1] > recent[0] + 0.15:
            return "improving"
        elif recent[-1] < recent[0] - 0.15:
            return "declining"
        return "stable"

    def get_average_score(self) -> float:
        if not self._turn_scores:
            return 1.0
        return sum(self._turn_scores) / len(self._turn_scores)

    def reset(self) -> None:
        self._history.clear()
        self._consecutive_failures.clear()
        self._turn_scores.clear()
