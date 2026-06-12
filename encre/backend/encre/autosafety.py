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

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from encre.backends.base import BaseBackend
from encre.backend import create_backend
from encre.prompts.loader import PromptLoader
from encre.utils.types import BackendText


class AutoDecision(Enum):
    SAFE = auto()         # Auto-approve — clearly safe
    LOW_RISK = auto()     # Auto-approve but log — probably safe
    ASK_USER = auto()     # Needs human judgment — ambiguous
    HIGH_RISK = auto()    # Auto-deny — likely dangerous
    BLOCK = auto()        # Auto-block — clearly malicious


@dataclass
class ClassificationResult:
    decision: AutoDecision
    confidence: float  # 0.0 - 1.0
    reasoning: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class UserDecisionRecord:
    """Records user permission decisions for learning."""
    tool_name: str
    tool_args_summary: str  # hashed/key features, not full args
    user_approved: bool
    timestamp: float = field(default_factory=time.time)


_loader = PromptLoader()


class EncreAutoSafetyClassifier:
    """Lightweight ML-based safety classifier for auto permission mode.

    Uses a fast/cheap model to evaluate whether a tool action is safe.
    Maintains a learning cache of user decisions to improve over time.

    Usage:
        classifier = EncreAutoSafetyClassifier(config)
        result = await classifier.classify("bash", {"command": "ls -la"})
        if result.decision in (AutoDecision.SAFE, AutoDecision.LOW_RISK):
            # auto-approve
    """

    SYSTEM_PROMPT = _loader.load("safety_classifier", category="autosafety")

    def __init__(
        self,
        backend_type: str = "openai",
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        cache_size: int = 1000,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._backend_type = backend_type
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._cache: dict[str, ClassificationResult] = {}
        self._cache_size = cache_size
        self._user_decisions: list[UserDecisionRecord] = []
        self._backend: BaseBackend | None = None
        self._total_classifications = 0
        self._cache_hits = 0

    def _get_backend(self) -> BaseBackend:
        if self._backend is None:
            # Use cheapest model by default
            if not self._model:
                if self._backend_type == "anthropic":
                    self._model = "claude-haiku-4-5-20251001"
                elif self._backend_type == "openai":
                    self._model = "gpt-4o-mini"
                else:
                    self._model = "default"
            self._backend = create_backend(
                self._backend_type,
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._model,
            )
        return self._backend

    def _make_cache_key(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Create a stable cache key from tool name + key arg patterns."""
        # Extract key features, not full args (which vary per call)
        features: list[str] = [tool_name]
        for key in sorted(tool_args.keys()):
            val = tool_args[key]
            if isinstance(val, str):
                # Extract command base (first word) or URL domain
                if key in ("command",):
                    base_cmd = val.split()[0] if val.split() else val[:50]
                    features.append(f"{key}={base_cmd}")
                elif key in ("url",):
                    from urllib.parse import urlparse
                    try:
                        parsed = urlparse(val)
                        features.append(f"url_domain={parsed.netloc}")
                    except Exception:
                        features.append(f"url={val[:60]}")
                elif key in ("path", "file_path"):
                    import os
                    features.append(f"ext={os.path.splitext(val)[1]}")
                else:
                    features.append(f"{key}={str(val)[:40]}")
            else:
                features.append(f"{key}={type(val).__name__}")
        return "|".join(features)

    def learn_from_user(self, tool_name: str, tool_args: dict[str, Any], approved: bool) -> None:
        """Record a user permission decision for future learning."""
        cache_key = self._make_cache_key(tool_name, tool_args)
        self._user_decisions.append(UserDecisionRecord(
            tool_name=tool_name,
            tool_args_summary=cache_key,
            user_approved=approved,
        ))
        # Keep only recent decisions
        if len(self._user_decisions) > self._cache_size:
            self._user_decisions = self._user_decisions[-self._cache_size:]

    def get_user_pattern(self, tool_name: str) -> dict[str, Any] | None:
        """Get user's historical approval pattern for a tool."""
        relevant = [d for d in self._user_decisions if d.tool_name == tool_name]
        if not relevant:
            return None
        approved = sum(1 for d in relevant if d.user_approved)
        return {
            "total": len(relevant),
            "approved": approved,
            "denied": len(relevant) - approved,
            "approval_rate": approved / len(relevant) if relevant else 0,
        }

    async def classify(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ClassificationResult:
        """Classify a tool action's safety level."""
        self._total_classifications += 1

        # Fast path: pattern-based pre-filtering
        pattern_result = self._pattern_classify(tool_name, tool_args)
        if pattern_result.confidence > 0.95:
            self._cache_hits += 1
            return pattern_result

        # Cache lookup
        cache_key = self._make_cache_key(tool_name, tool_args)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # Check user history
        user_pattern = self.get_user_pattern(tool_name)

        # For clearly safe tools, skip LLM call
        if tool_name in ("file_read", "grep", "glob", "web_search",
                         "task_list", "task_get", "cron_list", "task_output"):
            result = ClassificationResult(
                decision=AutoDecision.SAFE,
                confidence=0.99,
                reasoning="Read-only tool, always safe",
                tool_name=tool_name,
                tool_args=tool_args,
            )
            self._cache_result(cache_key, result)
            return result

        # For clearly dangerous patterns, skip LLM call
        if pattern_result.decision == AutoDecision.BLOCK:
            return pattern_result

        # LLM-based classification
        start = time.time()
        try:
            backend = self._get_backend()
            eval_prompt = self._build_eval_prompt(tool_name, tool_args, user_pattern)
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt},
            ]

            full_response = ""
            async for event in backend.chat(
                messages=messages, max_tokens=512, temperature=0.0, enable_caching=False,
            ):
                if isinstance(event, BackendText):
                    full_response += event.text

            llm_decision = self._parse_response(full_response)
            llm_decision.tool_name = tool_name
            llm_decision.tool_args = tool_args
            llm_decision.latency_ms = (time.time() - start) * 1000

            self._cache_result(cache_key, llm_decision)
            return llm_decision

        except Exception:
            # Fallback: ask user on classifier failure
            return ClassificationResult(
                decision=AutoDecision.ASK_USER,
                confidence=0.0,
                reasoning="Classifier unavailable — defaulting to ask user",
                tool_name=tool_name,
                tool_args=tool_args,
            )

    def _pattern_classify(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> ClassificationResult:
        """Fast regex-based pre-classification for clear cases."""
        from encre.safety import analyze_bash_command, DangerLevel

        if tool_name == "bash":
            command = tool_args.get("command", "")
            if not command:
                return ClassificationResult(
                    decision=AutoDecision.SAFE, confidence=1.0,
                    reasoning="Empty command", tool_name=tool_name, tool_args=tool_args,
                )

            analysis = analyze_bash_command(command)

            if analysis.danger_level == DangerLevel.CRITICAL:
                return ClassificationResult(
                    decision=AutoDecision.BLOCK, confidence=1.0,
                    reasoning=f"Critical danger: {'; '.join(analysis.injection_details)}",
                    tool_name=tool_name, tool_args=tool_args,
                )

            if analysis.danger_level == DangerLevel.HIGH:
                return ClassificationResult(
                    decision=AutoDecision.HIGH_RISK, confidence=0.95,
                    reasoning=f"High risk: {'; '.join(analysis.injection_details)}",
                    tool_name=tool_name, tool_args=tool_args,
                )

            if analysis.danger_level == DangerLevel.SAFE:
                return ClassificationResult(
                    decision=AutoDecision.SAFE, confidence=0.98,
                    reasoning="Static analysis: safe command",
                    tool_name=tool_name, tool_args=tool_args,
                )

            # MEDIUM/LOW → need LLM evaluation
            return ClassificationResult(
                decision=AutoDecision.ASK_USER, confidence=0.5,
                reasoning="Medium risk — needs LLM evaluation",
                tool_name=tool_name, tool_args=tool_args,
            )

        # File operations
        if tool_name in ("file_write", "file_edit"):
            path = tool_args.get("path", tool_args.get("file_path", ""))
            dangerous_paths = ["/etc/", "/boot/", "/sys/", "/proc/", "~/.ssh/",
                              "/root/", "C:\\Windows\\", "C:\\Windows\\System32\\"]
            for dp in dangerous_paths:
                if dp.lower() in path.lower():
                    return ClassificationResult(
                        decision=AutoDecision.BLOCK, confidence=1.0,
                        reasoning=f"Dangerous path: {path}", tool_name=tool_name,
                        tool_args=tool_args,
                    )

            # Check for .env / credential files
            sensitive_files = [".env", "credentials", "secret", ".pem", "id_rsa",
                             "password", "token", "api_key"]
            path_lower = path.lower()
            for sf in sensitive_files:
                if sf in path_lower:
                    return ClassificationResult(
                        decision=AutoDecision.ASK_USER, confidence=0.8,
                        reasoning=f"Potentially sensitive file: {path}",
                        tool_name=tool_name, tool_args=tool_args,
                    )

            return ClassificationResult(
                decision=AutoDecision.LOW_RISK, confidence=0.85,
                reasoning=f"File write to project path: {path}",
                tool_name=tool_name, tool_args=tool_args,
            )

        return ClassificationResult(
            decision=AutoDecision.ASK_USER, confidence=0.5,
            reasoning="Unknown tool — needs evaluation",
            tool_name=tool_name, tool_args=tool_args,
        )

    def _build_eval_prompt(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        user_pattern: dict[str, Any] | None,
    ) -> str:
        args_str = json.dumps(tool_args, indent=2, ensure_ascii=False)
        if len(args_str) > 2000:
            args_str = args_str[:2000] + "\n... truncated"
        user_pattern_str = ""
        if user_pattern:
            total = user_pattern["total"]
            rate = user_pattern["approval_rate"]
            user_pattern_str = f"User history: approved {rate:.0%} of {total} similar {tool_name} calls."
        return _loader.load_with_context(
            "safety_eval", category="autosafety",
            tool_name=tool_name,
            args_str=args_str,
            user_pattern=user_pattern_str,
        )

    def _parse_response(self, response: str) -> ClassificationResult:
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                return ClassificationResult(
                    decision=AutoDecision.ASK_USER, confidence=0.0,
                    reasoning=f"Could not parse classifier response"
                )

            safe = data.get("safe", False)
            risk = data.get("risk_level", "medium")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            if risk == "critical":
                decision = AutoDecision.BLOCK
            elif risk == "high":
                decision = AutoDecision.HIGH_RISK
            elif risk == "medium":
                decision = AutoDecision.ASK_USER if confidence < self._confidence_threshold else AutoDecision.LOW_RISK
            elif risk == "low":
                decision = AutoDecision.LOW_RISK
            else:
                decision = AutoDecision.SAFE

            return ClassificationResult(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
            )
        except Exception:
            return ClassificationResult(
                decision=AutoDecision.ASK_USER, confidence=0.0,
                reasoning="Parse error — defaulting to ask",
            )

    def _cache_result(self, key: str, result: ClassificationResult) -> None:
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_classifications": self._total_classifications,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / max(self._total_classifications, 1),
            "user_decisions_recorded": len(self._user_decisions),
        }
