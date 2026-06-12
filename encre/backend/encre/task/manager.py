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
import time
import uuid
from typing import Any

from encre.task.types import EncreTask
from encre.utils.types import TaskStatus, TaskType


class EncreTaskManager:
    _tasks: dict[str, EncreTask] = {}

    @classmethod
    def create_task(
        cls,
        name: str,
        description: str,
        task_type: TaskType,
        prompt: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        now = time.time()
        task = EncreTask(
            id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            prompt=prompt,
            parent_id=parent_id,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        cls._tasks[task_id] = task
        return task_id

    @classmethod
    def get_task(cls, task_id: str) -> EncreTask | None:
        return cls._tasks.get(task_id)

    @classmethod
    def update_task(
        cls,
        task_id: str,
        status: TaskStatus | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> bool:
        task = cls._tasks.get(task_id)
        if task is None:
            return False
        if status is not None:
            task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        task.updated_at = time.time()
        return True

    @classmethod
    def list_tasks(cls, status: TaskStatus | None = None) -> list[EncreTask]:
        tasks = list(cls._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    @classmethod
    def delete_task(cls, task_id: str) -> bool:
        if task_id in cls._tasks:
            del cls._tasks[task_id]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        cls._tasks.clear()