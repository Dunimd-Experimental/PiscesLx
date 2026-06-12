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
import time
from typing import Any

from encre.tools.base import build_tool

_scheduler: Any = None  # Set by agent during initialization


def set_scheduler(scheduler: Any) -> None:
    global _scheduler
    _scheduler = scheduler


async def _cron_list_execute(**kwargs: Any) -> str:
    if _scheduler is None:
        return json.dumps({"jobs": [], "message": "Scheduler not available."}, ensure_ascii=False)

    jobs = _scheduler.list_jobs()
    if not jobs:
        return json.dumps({"jobs": [], "message": "No scheduled jobs."}, ensure_ascii=False)

    now = time.time()
    result = []
    for job in jobs:
        entry = {
            "id": job.id,
            "name": job.name,
            "state": job.state.name,
            "schedule_type": job.schedule_type.name,
            "prompt_preview": job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt,
            "created_at": job.created_at,
        }
        if job.cron:
            entry["cron"] = job.cron.to_expression()
            next_fire = job.cron.next_fire(now) if job.state.name == "PENDING" else None
            if next_fire:
                entry["next_fire"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(next_fire))
        elif job.fire_at:
            entry["fire_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.fire_at))
        if job.last_fired:
            entry["last_fired"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.last_fired))
        if job.fail_count > 0:
            entry["fail_count"] = job.fail_count
        result.append(entry)

    return json.dumps({"jobs": result, "total": len(result)}, ensure_ascii=False, indent=2)


EncreCronListTool = build_tool(
    name="cron_list",
    description="List all scheduled cron jobs",
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    execute=_cron_list_execute,
    intents=["system"],
    is_concurrency_safe=lambda _: True,
)
# Backward-compat: keep ``.set_scheduler()`` callable on the tool object.
EncreCronListTool.set_scheduler = set_scheduler
