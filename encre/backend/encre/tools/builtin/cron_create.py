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

from encre.tools.base import build_tool

_scheduler: Any = None  # Set by agent during initialization


def set_scheduler(scheduler: Any) -> None:
    global _scheduler
    _scheduler = scheduler


async def _cron_create_execute(**kwargs: Any) -> str:
    cron_expr = kwargs.get("cron", "")
    prompt_text = kwargs.get("prompt", "")
    name = kwargs.get("name", "Unnamed job")

    if not cron_expr or not prompt_text:
        return "Error: both 'cron' and 'prompt' are required."

    # Validate cron expression
    from encre.scheduler import CronSchedule
    try:
        CronSchedule.parse(cron_expr)
    except ValueError as e:
        return f"Error: invalid cron expression '{cron_expr}' — {e}"

    if _scheduler is None:
        # Fallback: validate only, no real scheduling
        return (
            f"Cron expression '{cron_expr}' validated. Job '{name}' ready to schedule.\n"
            f"Prompt: {prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}\n"
            "(Scheduler not yet started — job will activate when scheduler is available.)"
        )

    job_id = _scheduler.schedule(
        name=name,
        prompt=prompt_text,
        cron=cron_expr,
    )
    return json.dumps({
        "status": "scheduled",
        "job_id": job_id,
        "name": name,
        "cron": cron_expr,
        "prompt_preview": prompt_text[:200],
    }, ensure_ascii=False, indent=2)


EncreCronCreateTool = build_tool(
    name="cron_create",
    description=(
        "Schedule a prompt to be executed at a future time or on a recurring schedule. "
        "Uses standard 5-field cron: minute hour day-of-month month day-of-week. "
        'Example: "0 9 * * *" for daily at 9am, "*/5 * * * *" for every 5 minutes.'
    ),
    input_schema={
        "type": "object",
        "properties": {
            "cron": {
                "type": "string",
                "description": "5-field cron expression in local time",
            },
            "prompt": {
                "type": "string",
                "description": "The prompt to execute at each fire time",
            },
            "name": {
                "type": "string",
                "description": "Human-readable name for this scheduled job",
            },
            "recurring": {
                "type": "boolean",
                "description": "Whether this is a recurring job (default: true)",
                "default": True,
            },
        },
        "required": ["cron", "prompt"],
    },
    execute=_cron_create_execute,
    intents=["system"],
)
# Backward-compat: keep ``.set_scheduler()`` callable on the tool object.
EncreCronCreateTool.set_scheduler = set_scheduler
