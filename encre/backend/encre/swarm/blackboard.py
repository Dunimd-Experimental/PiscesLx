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

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlackboardEntry:
    key: str
    value: Any
    version: int
    namespace: str = "default"
    timestamp: float = field(default_factory=time.time)
    owner: str = ""


class EncreBlackboard:
    """Shared knowledge board for multi-agent collaboration.

    Features:
    - Namespace isolation (each agent team gets its own namespace)
    - Version tracking for conflict detection
    - Watch/subscribe mechanism for reactive agents
    - Automatic pruning of old entries
    """

    MAX_ENTRIES_PER_NAMESPACE = 1000
    MAX_WATCHERS = 200

    def __init__(self) -> None:
        self._store: dict[str, dict[str, BlackboardEntry]] = {}
        self._watchers: dict[str, dict[str, list[asyncio.Queue[BlackboardEntry]]]] = {}
        self._version_counter: dict[str, int] = {}

    def put(self, namespace: str, key: str, value: Any, owner: str = "") -> int:
        ns = self._store.setdefault(namespace, {})
        ver = self._version_counter.get(namespace, 0) + 1
        self._version_counter[namespace] = ver
        entry = BlackboardEntry(
            key=key,
            value=value,
            version=ver,
            namespace=namespace,
            owner=owner,
        )
        ns[key] = entry
        if len(ns) > self.MAX_ENTRIES_PER_NAMESPACE:
            oldest = min(ns.values(), key=lambda e: e.timestamp)
            del ns[oldest.key]
        self._notify(namespace, key, entry)
        return ver

    def get(self, namespace: str, key: str) -> tuple[Any, int] | None:
        ns = self._store.get(namespace, {})
        entry = ns.get(key)
        if entry is None:
            return None
        return (entry.value, entry.version)

    def get_all(self, namespace: str) -> dict[str, Any]:
        ns = self._store.get(namespace, {})
        return {k: e.value for k, e in ns.items()}

    def get_all_visible(self) -> str:
        if not self._store:
            return ""
        lines: list[str] = []
        for namespace, entries in self._store.items():
            if namespace.startswith("__"):
                continue
            for key, entry in entries.items():
                val_str = json.dumps(entry.value, ensure_ascii=False) if not isinstance(entry.value, str) else entry.value
                lines.append(f"[{namespace}/{key}] {val_str[:500]}")
        return "\n".join(lines)

    def delete(self, namespace: str, key: str) -> bool:
        ns = self._store.get(namespace, {})
        return ns.pop(key, None) is not None

    def compare_and_swap(self, namespace: str, key: str, expected_version: int, new_value: Any) -> bool:
        ns = self._store.get(namespace, {})
        entry = ns.get(key)
        if entry is None or entry.version != expected_version:
            return False
        self.put(namespace, key, new_value)
        return True

    async def watch(self, namespace: str, key: str, timeout: float = 300.0) -> BlackboardEntry | None:
        queue: asyncio.Queue[BlackboardEntry] = asyncio.Queue(maxsize=1)
        ns_watchers = self._watchers.setdefault(namespace, {})
        key_watchers = ns_watchers.setdefault(key, [])
        if len(key_watchers) >= self.MAX_WATCHERS:
            return None
        key_watchers.append(queue)
        try:
            entry = await asyncio.wait_for(queue.get(), timeout=timeout)
            return entry
        except asyncio.TimeoutError:
            return None
        finally:
            if queue in key_watchers:
                key_watchers.remove(queue)

    def _notify(self, namespace: str, key: str, entry: BlackboardEntry) -> None:
        ns_watchers = self._watchers.get(namespace, {})
        key_watchers = ns_watchers.get(key, [])
        for queue in list(key_watchers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass

    def reset(self) -> None:
        self._store.clear()
        self._watchers.clear()
        self._version_counter.clear()
