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
class ToolStrategy:
    tool_name: str
    param_signature: str  # hash of key param structure
    success_count: int = 0
    fail_count: int = 0
    total_latency_ms: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        total = self.success_count + self.fail_count
        return self.total_latency_ms / total if total > 0 else 0.0

    @property
    def score(self) -> float:
        return self.success_rate * (1.0 + min(self.success_count, 10) * 0.1)


class EncreStrategyOptimizer:
    MAX_STRATEGIES_PER_TOOL = 20
    MIN_SAMPLES_FOR_RECOMMENDATION = 3

    def __init__(self) -> None:
        self._strategies: dict[str, dict[str, ToolStrategy]] = {}

    def record_outcome(
        self,
        tool_name: str,
        params: dict[str, Any],
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        sig = _param_signature(params)
        strategies = self._strategies.setdefault(tool_name, {})
        if sig not in strategies:
            if len(strategies) >= self.MAX_STRATEGIES_PER_TOOL:
                self._evict_one(tool_name)
            strategies[sig] = ToolStrategy(
                tool_name=tool_name,
                param_signature=sig,
            )
        st = strategies[sig]
        if success:
            st.success_count += 1
        else:
            st.fail_count += 1
        st.total_latency_ms += latency_ms
        st.last_used = time.time()

    def suggest_strategy(self, tool_name: str, context: str) -> dict[str, Any] | None:
        strategies = self._strategies.get(tool_name, {})
        if not strategies:
            return None
        ranked = sorted(strategies.values(), key=lambda s: s.score, reverse=True)
        best = ranked[0]
        total = best.success_count + best.fail_count
        if total < self.MIN_SAMPLES_FOR_RECOMMENDATION:
            return None
        if best.success_rate < 0.5:
            return None
        return {"_strategy_hint": f"prefer pattern: {best.param_signature}", "_confidence": best.success_rate}

    def get_fallback(self, tool_name: str, current_params: dict[str, Any]) -> dict[str, Any] | None:
        strategies = self._strategies.get(tool_name, {})
        current_sig = _param_signature(current_params)
        ranked = sorted(strategies.values(), key=lambda s: s.score, reverse=True)
        for st in ranked:
            if st.param_signature != current_sig and st.success_rate > 0.6:
                total = st.success_count + st.fail_count
                if total >= self.MIN_SAMPLES_FOR_RECOMMENDATION:
                    return {"_fallback_hint": f"try alternative pattern: {st.param_signature}", "_confidence": st.success_rate}
        return None

    def get_statistics(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for tool_name, strategies in self._strategies.items():
            ranked = sorted(strategies.values(), key=lambda s: s.score, reverse=True)
            result[tool_name] = {
                "total_strategies": len(strategies),
                "best_success_rate": ranked[0].success_rate if ranked else 0.0,
                "total_samples": sum(s.success_count + s.fail_count for s in strategies.values()),
            }
        return result

    def _evict_one(self, tool_name: str) -> None:
        strategies = self._strategies.get(tool_name, {})
        if not strategies:
            return
        worst = min(strategies.values(), key=lambda s: (s.score, s.last_used))
        del strategies[worst.param_signature]

    def reset(self) -> None:
        self._strategies.clear()


def _param_signature(params: dict[str, Any]) -> str:
    keys = sorted(params.keys())
    types = []
    for k in keys:
        v = params[k]
        if isinstance(v, bool):
            types.append(f"{k}:bool")
        elif isinstance(v, str):
            if len(v) > 200:
                types.append(f"{k}:str*")
            elif re.search(r'^[\w\-./]+$', v):
                types.append(f"{k}:{v}")
            else:
                types.append(f"{k}:str")
        elif isinstance(v, (int, float)):
            types.append(f"{k}:num")
        elif isinstance(v, list):
            types.append(f"{k}:list({len(v)})")
        elif isinstance(v, dict):
            types.append(f"{k}:dict({','.join(sorted(v.keys())[:5])})")
        elif v is None:
            types.append(f"{k}:none")
        else:
            types.append(f"{k}:*")
    return ";".join(types)
