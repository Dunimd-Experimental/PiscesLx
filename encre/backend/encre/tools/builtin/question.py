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

"""Question tool — ask the user for clarification or additional information.

Use this when the user's request is ambiguous, incomplete, or when you need
more context to proceed confidently.  Instead of guessing, ask directly.

Returns structured JSON so the frontend can render an interactive Question
card with the question text, optional context, and optional choice buttons.
"""

from __future__ import annotations

import json
from typing import Any

from encre.tools.base import build_tool


async def _question_execute(**kwargs: Any) -> str:
    questions_raw = kwargs.get("questions")
    single_question = (kwargs.get("question") or "").strip()

    # Build a list of question items (either from `questions` or single `question`)
    items: list[dict[str, Any]] = []

    if questions_raw and isinstance(questions_raw, list):
        for q in questions_raw:
            if isinstance(q, dict):
                text = (q.get("question") or "").strip()
                if text:
                    item: dict[str, Any] = {"question": text}
                    if q.get("details"):
                        item["details"] = str(q["details"]).strip()
                    if q.get("options") and isinstance(q["options"], list):
                        item["options"] = [str(o) for o in q["options"]]
                    items.append(item)
    elif single_question:
        item = {"question": single_question}
        if kwargs.get("details"):
            item["details"] = str(kwargs["details"]).strip()
        if kwargs.get("options") and isinstance(kwargs["options"], list):
            item["options"] = [str(o) for o in kwargs["options"]]
        items.append(item)

    if not items:
        return 'Error: Provide at least one question via "question" or "questions".'

    return json.dumps({"_type": "question", "questions": items}, ensure_ascii=False)


EncreQuestionTool = build_tool(
    name="question",
    description=(
        "Ask the user questions when their request is ambiguous or you need "
        "more information. This pauses the conversation until the user answers.\n\n"
        "CRITICAL: If you have MULTIPLE things to ask, ask ALL of them at once "
        "using the 'questions' parameter (array of objects). Do NOT ask one at "
        "a time — the user can only respond once per question call.\n\n"
        "Usage examples:\n"
        '- Single: question(question="你要开发什么类型的网站？")\n'
        '- With options: question(question="你更偏好哪个方案？", options=["方案A: 性能优先", "方案B: 可维护性优先"])\n'
        '- Multiple at once: question(questions=[{question:"项目名称是什么？"},{question:"目标用户是谁？"},{question:"主要功能有哪些？"}])'
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A single question to ask the user. Use this for simple queries, or use 'questions' for multiple.",
            },
            "details": {
                "type": "string",
                "description": "Optional background context to help the user understand why you're asking.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of predefined options for the user to choose from.",
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question text."},
                        "details": {"type": "string", "description": "Optional context for this question."},
                        "options": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Optional predefined choices.",
                        },
                    },
                    "required": ["question"],
                },
                "description": "Multiple questions to ask at once. Each can have its own details and options. All are displayed together, and the user answers before the model continues.",
            },
        },
        "required": ["question"],
    },
    execute=_question_execute,
    intents=["general", "coding", "research", "data", "system"],
    category="communication",
    triggers=["ask user", "question", "clarify", "confirm", "ambiguity"],
    always_available=True,
    is_concurrency_safe=lambda _: True,
)


__all__ = ["EncreQuestionTool", "_set_backend"]
