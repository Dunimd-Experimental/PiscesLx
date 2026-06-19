#!/usr/bin/env python3

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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



"""In-process task store backing the EnCRE ``task_*`` tool family.

The previous implementation lived in ``enta.task.manager`` which has been
removed during the EnTA core slim-down.  This module is a complete,
in-process replacement that the adversarial-training tool palette relies
on.  Tasks are kept in a thread-safe in-memory dict keyed by task id; the
store is intentionally minimal so it can be used as-is from training
rollouts without any extra infrastructure.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class _TaskRecord:
    """Plain dataclass mirroring the legacy ``EncreTask`` schema."""

    id: str
    name: str
    description: str
    task_type: str
    prompt: str
    parent_id: str | None = None
    status: str = "pending"
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class EncreTaskStore:
    """Process-wide task ledger.

    All public methods are thread-safe.  The store is intentionally
    process-local: there is no network or persistence layer.  Training
    rollouts and live interactions are expected to live in a single
    process at a time, so an in-memory dict is sufficient and avoids
    pulling in any external service.
    """

    _instance: EncreTaskStore | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> EncreTaskStore:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks = {}
                    cls._instance._lock = threading.RLock()
        return cls._instance

    # ── creation ──────────────────────────────────────────────────────
    def create(
        self,
        *,
        name: str,
        description: str,
        task_type: str,
        prompt: str,
        parent_id: str | None = None,
    ) -> _TaskRecord:
        task_id = f"t_{uuid.uuid4().hex[:10]}"
        record = _TaskRecord(
            id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            prompt=prompt,
            parent_id=parent_id,
        )
        with self._lock:
            self._tasks[task_id] = record
        return record

    # ── read ──────────────────────────────────────────────────────────
    def get(self, task_id: str) -> _TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, status: str | None = None) -> list[_TaskRecord]:
        with self._lock:
            records = list(self._tasks.values())
        if status is None:
            return records
        return [r for r in records if r.status == status]

    # ── mutate ────────────────────────────────────────────────────────
    def update(
        self,
        task_id: str,
        *,
        status: str | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return False
            if status is not None:
                record.status = status
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            record.updated_at = time.time()
            return True

    def delete(self, task_id: str) -> bool:
        with self._lock:
            return self._tasks.pop(task_id, None) is not None

    # ── introspection ─────────────────────────────────────────────────
    def all(self) -> Iterable[_TaskRecord]:
        with self._lock:
            return list(self._tasks.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)


# Module-level singleton shortcut.
def get_store() -> EncreTaskStore:
    return EncreTaskStore()


__all__ = ["EncreTaskStore", "get_store"]
