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


async def _cron_delete_execute(**kwargs: Any) -> str:
    job_id = kwargs.get("job_id", "")
    if not job_id:
        return "Error: job_id is required."

    if _scheduler is None:
        return f"Job '{job_id}' cancellation noted. (Scheduler not available — no active jobs to cancel.)"

    cancelled = _scheduler.cancel(job_id)
    if cancelled:
        return json.dumps({"status": "cancelled", "job_id": job_id}, ensure_ascii=False)
    return json.dumps({"status": "not_found", "job_id": job_id,
                      "message": "No such job or already cancelled"}, ensure_ascii=False)


EncreCronDeleteTool = build_tool(
    name="cron_delete",
    description="Cancel/delete a previously scheduled cron job by its ID",
    input_schema={
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "The ID of the scheduled job to cancel",
            },
        },
        "required": ["job_id"],
    },
    execute=_cron_delete_execute,
    intents=["system"],
)
# Backward-compat: keep ``.set_scheduler()`` callable on the tool object.
EncreCronDeleteTool.set_scheduler = set_scheduler
