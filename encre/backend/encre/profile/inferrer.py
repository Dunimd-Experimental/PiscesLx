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
import json
from typing import Any

from encre.logging_config import get_logger

logger = get_logger("encre.profile.inferrer")

_INFERENCE_PROMPT = """Analyze the conversation above and extract user traits. Return ONLY valid JSON with this exact structure:
{
  "inferred": {
    "expertise_level": "", "domain": "", "formality": "", "detail_preference": "",
    "tone": "", "response_style": "", "testing_preference": "", "learning_style": "",
    "error_tolerance": "", "typical_session_length": "", "os": "", "editor": ""
  },
  "confidences": {
    "expertise_level": 0.0, "domain": 0.0, "formality": 0.0, "detail_preference": 0.0,
    "tone": 0.0, "response_style": 0.0, "testing_preference": 0.0, "learning_style": 0.0,
    "error_tolerance": 0.0, "typical_session_length": 0.0, "os": 0.0, "editor": 0.0
  },
  "languages": [], "frameworks": [], "goals": []
}

Rules:
- expertise_level: one of "beginner", "intermediate", "advanced", "expert"
- formality: one of "casual", "semi-formal", "formal"
- detail_preference: one of "low", "medium", "high"
- tone: adjective describing communication tone (e.g. "friendly", "professional", "direct")
- response_style: e.g. "concise", "thorough", "example-driven"
- testing_preference: how they approach testing (e.g. "writes tests first", "tests after", "manual testing")
- learning_style: how they prefer to learn (e.g. "by example", "by reading docs", "hands-on")
- error_tolerance: one of "low", "medium", "high"
- typical_session_length: one of "short", "medium", "long"
- confidence: 0.0-1.0 how sure you are about each inference
- Leave fields empty if you cannot infer them with reasonable confidence (< 0.5)
- languages: programming languages mentioned
- frameworks: frameworks/tools mentioned
- goals: common task types observed

IMPORTANT: Return ONLY the JSON, no other text."""


class ProfileInferrer:
    MAX_USER_TOKENS = 800
    MAX_TOOL_TOKENS = 600
    MAX_TOTAL_MESSAGES = 20

    async def infer(self, messages: list[dict[str, Any]], backend: Any) -> tuple[dict[str, Any], dict[str, float]] | None:
        sample = self._sample_messages(messages)
        if not sample:
            return None

        infer_messages = list(sample)
        infer_messages.append({"role": "user", "content": _INFERENCE_PROMPT})

        try:
            full_text = ""
            async for event in backend.chat(messages=infer_messages, max_tokens=1024, stream=False):
                from encre.utils.types import BackendText, BackendFinish
                if isinstance(event, BackendText):
                    full_text += event.text
                elif isinstance(event, BackendFinish):
                    break

            parsed = self._parse_response(full_text)
            if parsed:
                inferred, confidences, languages, frameworks, goals = parsed
                if languages:
                    inferred["preferred_languages"] = languages
                if frameworks:
                    inferred["preferred_frameworks"] = frameworks
                if goals:
                    inferred["common_goals"] = goals
                return inferred, confidences
        except Exception as e:
            logger.warning("Profile inference call failed: %s", e)

        return None

    def _sample_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sampled: list[dict[str, Any]] = []
        char_count = 0
        for msg in reversed(messages):
            if len(sampled) >= self.MAX_TOTAL_MESSAGES:
                break
            role = msg.get("role", "")
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = " ".join(texts)
            if not isinstance(content, str):
                continue
            if role == "tool":
                if char_count + len(content) > self.MAX_TOOL_TOKENS * 4:
                    content = content[:self.MAX_TOOL_TOKENS * 4]
                sampled.insert(0, {"role": "user", "content": f"[tool result] {content[:200]}"})
                continue
            if role == "user":
                if char_count + len(content) > self.MAX_USER_TOKENS * 4:
                    content = content[:self.MAX_USER_TOKENS * 4]
                sampled.insert(0, msg)
                char_count += len(content)
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    names = [tc.get("function", {}).get("name", "") for tc in tool_calls if isinstance(tc, dict)]
                    tool_text = f"[used tools: {', '.join(names)}]"
                    sampled.insert(0, {"role": "assistant", "content": tool_text})
                else:
                    sampled.insert(0, {"role": "assistant", "content": content[:200]})
        return sampled

    def _parse_response(self, text: str) -> tuple[dict[str, Any], dict[str, float], list[str], list[str], list[str]] | None:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        text = text[start:end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        inferred: dict[str, Any] = data.get("inferred", {})
        confidences: dict[str, float] = data.get("confidences", {})
        languages: list[str] = data.get("languages", [])
        frameworks: list[str] = data.get("frameworks", [])
        goals: list[str] = data.get("goals", [])

        inferred = {k: v for k, v in inferred.items() if v is not None and v != ""}
        confidences = {k: v for k, v in confidences.items() if isinstance(v, (int, float))}

        if not inferred:
            return None

        return inferred, confidences, languages, frameworks, goals
