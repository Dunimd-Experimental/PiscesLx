#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

"""
Minimal publish-subscribe state store.

Inspired by Claude Code's ``createStore`` (35 lines) and Zustand's design.
Provides a single ``Store[T]`` interface: ``get_state`` / ``set_state`` /
``subscribe``.  No immutability helpers, no middleware -- just the bare
minimum needed for cross-component state sharing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Store(Generic[T]):
    """A minimal pub-sub state container.

    Usage::

        store: Store[AppState] = create_store(AppState())
        store.set_state(lambda prev: {**prev, "count": prev["count"] + 1})
        unsub = store.subscribe(lambda new, old: print(new, old))
        current = store.get_state()
    """

    def __init__(self, initial_state: T) -> None:
        self._state: T = initial_state
        self._listeners: list[Callable[[T, T], None]] = []

    def get_state(self) -> T:
        """Return the current state snapshot."""
        return self._state

    def set_state(self, updater: Callable[[T], T]) -> None:
        """Apply ``updater`` to the current state and notify listeners on change.

        Listeners are notified only when ``updater`` returns a **different**
        object (reference equality via ``is``).  Callers that mutate in place
        must return a new object to trigger notifications.
        """
        new_state = updater(self._state)
        if new_state is not self._state:
            old_state = self._state
            self._state = new_state
            for listener in self._listeners:
                listener(new_state, old_state)

    def subscribe(self, listener: Callable[[T, T], None]) -> Callable[[], None]:
        """Register a listener called as ``listener(new_state, old_state)``.

        Returns an ``unsubscribe`` callable.
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)


def create_store(initial_state: T) -> Store[T]:
    """Create a :class:`Store` with the given initial state."""
    return Store(initial_state)
