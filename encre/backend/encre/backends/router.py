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

from __future__ import annotations
import contextlib
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from encre.backends.base import BaseBackend
from encre.utils.types import BackendEvent, BackendFinish


class TaskCategory:
    """Categories for routing tasks to appropriate models."""
    CLASSIFICATION = "classification"     # Safety checks, evaluation, yes/no
    REASONING = "reasoning"               # Complex multi-step reasoning
    CODING = "coding"                     # Code generation, debugging, review
    RESEARCH = "research"                 # Web search, data gathering, analysis
    WRITING = "writing"                   # Documentation, content creation
    PLANNING = "planning"                 # Task decomposition, architecture
    EXECUTION = "execution"               # Tool-heavy execution (default)
    SUMMARIZATION = "summarization"       # Compaction, summarization


# Task category detection patterns
_CATEGORY_PATTERNS: dict[str, list[str]] = {
    TaskCategory.CLASSIFICATION: [
        r"\b(?:classify|categorize|is this safe|rate|rank|score|evaluate)\b",
        r"\b(?:true or false|yes or no|which (?:one|option)|select)\b",
    ],
    TaskCategory.REASONING: [
        r"\b(?:reason|deduce|infer|conclude|prove|derive|analyze|why|how does)\b",
        r"\b(?:step by step|think through|logic|logical|puzzle|paradox)\b",
    ],
    TaskCategory.CODING: [
        r"\b(?:code|implement|develop|program|debug|refactor|function|class|module)\b",
        r"\b(?:python|javascript|typescript|rust|go|java|c\+\+|sql|html|css)\b",
        r"\b(?:fix|bug|error|exception|compile|lint|type check|test)\b",
    ],
    TaskCategory.RESEARCH: [
        r"\b(?:research|search|find|look up|investigate|discover|learn about)\b",
        r"\b(?:documentation|docs|reference|spec|standard)\b",
    ],
    TaskCategory.WRITING: [
        r"\b(?:write|compose|draft|create (?:a |an )?(?:document|article|blog|post))\b",
        r"\b(?:readme|documentation|docstring|comment|summary|explain)\b",
        r"\b(?:describe|elaborate|clarify)\b",
    ],
    TaskCategory.PLANNING: [
        r"\b(?:plan|design|architecture|approach|strategy|roadmap|break down)\b",
        r"\b(?:decompose|organize|structure|outline|skeleton|scaffold)\b",
    ],
    TaskCategory.SUMMARIZATION: [
        r"\b(?:summarize|condense|compact|shorten|tl;dr|recap|gist)\b",
    ],
}


@dataclass
class Route:
    category: str
    backend: BaseBackend
    priority: int = 0  # Lower = preferred
    min_confidence: float = 0.3

    def matches(self, prompt: str) -> float:
        """Return match confidence 0.0-1.0 for this route."""
        patterns = _CATEGORY_PATTERNS.get(self.category, [])
        if not patterns:
            return 0.0
        prompt_lower = prompt.lower()
        hits = 0
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                hits += 1
        if not hits:
            return 0.0
        return min(hits / max(len(patterns) * 0.5, 1), 1.0)


@dataclass
class CostTracker:
    """Tracks token usage and cost across all routed backends."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    cache_hit_tokens: int = 0
    cache_savings_usd: float = 0.0
    cost_by_model: dict[str, float] = field(default_factory=dict)
    requests_by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cost_usd: float, cache_hit: int = 0, cache_savings: float = 0.0) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.cache_hit_tokens += cache_hit
        self.cache_savings_usd += cache_savings
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost_usd
        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_savings_usd": round(self.cache_savings_usd, 6),
            "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
            "requests_by_model": dict(self.requests_by_model),
        }


class RouterBackend(BaseBackend):
    """Routes requests to different backends based on task type.

    Classification/evaluation -> cheap model (haiku/gpt-4o-mini)
    Coding/reasoning -> strong model (sonnet/gpt-5/deepseek-v4)
    Summarization/execution -> balanced model
    Research/writing -> mid-tier model

    Usage:
        router = RouterBackend(
            routes={
                TaskCategory.CLASSIFICATION: OpenAIBackend(model="gpt-4o-mini"),
                TaskCategory.CODING: AnthropicBackend(model="claude-sonnet-4-6"),
                TaskCategory.REASONING: DeepSeekBackend(model="deepseek-v4-pro"),
            },
            default=OpenAIBackend(model="gpt-4o"),
        )
    """

    def __init__(
        self,
        routes: dict[str, BaseBackend],
        default: BaseBackend,
        track_costs: bool = True,
    ) -> None:
        self._routes: dict[str, Route] = {}
        for category, backend in routes.items():
            self._routes[category] = Route(category=category, backend=backend)
        self._default = default
        self._last_used: str = "default"
        self._route_counts: dict[str, int] = {}
        for category in self._routes:
            self._route_counts[category] = 0
        self._route_counts["default"] = 0
        self._cost_tracker = CostTracker() if track_costs else None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
        enable_caching: bool = False,
    ) -> AsyncGenerator[BackendEvent, None]:
        # Extract last user message for routing
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            prompt += block.get("text", "")
                break

        backend, route_name = self._select_backend(prompt)
        self._last_used = route_name
        self._route_counts[route_name] = self._route_counts.get(route_name, 0) + 1

        async for event in backend.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            enable_caching=enable_caching,
        ):
            if isinstance(event, BackendFinish) and self._cost_tracker is not None:
                self._record_usage(backend, event)
            yield event

    def _select_backend(self, prompt: str) -> tuple[BaseBackend, str]:
        if not prompt:
            return self._default, "default"

        best: tuple[Route, float] | None = None
        for route in self._routes.values():
            confidence = route.matches(prompt)
            if confidence > route.min_confidence and (
                best is None or confidence > best[1]
            ):
                best = (route, confidence)

        if best is not None:
            return best[0].backend, best[0].category

        return self._default, "default"

    def _record_usage(self, backend: BaseBackend, finish: BackendFinish) -> None:
        """Record token usage and cost from a BackendFinish event.

        Retrieves model pricing from the registry and computes cost
        based on input/output token counts in the usage dict.
        """
        if self._cost_tracker is None or finish.usage is None:
            return
        model_name = getattr(backend, "model", "unknown")
        input_tokens = finish.usage.get("input_tokens", 0)
        output_tokens = finish.usage.get("output_tokens", 0)
        cache_hit = finish.usage.get("cache_read_input_tokens", 0)

        try:
            from encre.backends.registry import resolve_model_info
            info = resolve_model_info(model_name)
            input_cost = info.pricing_input_per_1m * (input_tokens / 1_000_000)
            output_cost = info.pricing_output_per_1m * (output_tokens / 1_000_000)
            cache_discount = info.extra.get("cache_hit_discount", 0.0) if info.extra else 0.0
            cache_savings = input_cost * cache_discount * (cache_hit / max(input_tokens, 1))
            cost = input_cost + output_cost - cache_savings
        except Exception:
            cost = 0.0
            cache_savings = 0.0

        self._cost_tracker.record(
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cache_hit=cache_hit,
            cache_savings=cache_savings,
        )

    def supports_tool_calling(self) -> bool:
        return self._default.supports_tool_calling()

    def context_window_size(self) -> int:
        return self._default.context_window_size()

    def supports_thinking(self) -> bool:
        return self._default.supports_thinking()

    def supports_prompt_caching(self) -> bool:
        return self._default.supports_prompt_caching()

    def count_tokens(self, text: str) -> int:
        return self._default.count_tokens(text)

    @property
    def last_route(self) -> str:
        return self._last_used

    @property
    def cost_tracker(self) -> CostTracker | None:
        return self._cost_tracker

    @property
    def route_stats(self) -> dict[str, int]:
        """Return accumulated route usage counts across all requests.

        Unlike the previous implementation that reset counts on every
        call, this reflects the true cumulative usage since the router
        was created.
        """
        return dict(self._route_counts)

    async def aclose(self) -> None:
        for route in self._routes.values():
            with contextlib.suppress(Exception):
                await route.backend.aclose()
        with contextlib.suppress(Exception):
            await self._default.aclose()
