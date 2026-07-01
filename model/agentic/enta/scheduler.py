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

"""EntaScheduler — tracks curriculum progress and decides when to stop.

The scheduler maintains an internal curriculum level and uses evaluation
profiles to decide when the model has plateaued and training should stop.
"""

from typing import Any, Dict


class EntaScheduler:
    """Tracks curriculum progress for the EnTA outer training loop.

    The scheduler maintains:
    - ``_curriculum_level`` — current difficulty/complexity level (starts at 0)
    - ``_max_level`` — when this level is reached, ``should_stop()`` returns True
    - ``_patience`` — number of evaluations with no improvement before stopping
    - ``_best_score`` — best capability score seen so far
    - ``_no_improve_count`` — consecutive evaluations without improvement
    """

    def __init__(
        self,
        max_level: int = 5,
        patience: int = 3,
        improvement_threshold: float = 0.01,
    ) -> None:
        """Initialise the scheduler.

        Args:
            max_level: Maximum curriculum level.  When reached,
                ``should_stop()`` returns ``True``.
            patience: Number of evaluations with no improvement before
                the scheduler gives up.
            improvement_threshold: Minimum score improvement to count
                as meaningful progress.
        """
        self._max_level = int(max_level)
        self._patience = int(patience)
        self._improvement_threshold = float(improvement_threshold)

        self._curriculum_level: int = 0
        self._best_score: float = 0.0
        self._no_improve_count: int = 0

    @property
    def curriculum_level(self) -> int:
        """Current curriculum level."""
        return self._curriculum_level

    def should_stop(self) -> bool:
        """Return ``True`` when the curriculum is complete.

        Stopping conditions:
        1. Curriculum level reached ``max_level``.
        2. No improvement for ``patience`` consecutive evaluations.
        """
        if self._curriculum_level >= self._max_level:
            return True
        if self._no_improve_count >= self._patience:
            return True
        return False

    def update(self, profile: Dict[str, Any]) -> None:
        """Advance the curriculum based on an evaluation profile.

        Args:
            profile: Capability profile dict from
                :meth:`EntaEvaluator.evaluate`.  Must contain a
                ``capability_score`` key.
        """
        score = float(profile.get("capability_score", 0.0))

        if score > self._best_score + self._improvement_threshold:
            # Real improvement — advance curriculum.
            self._best_score = score
            self._no_improve_count = 0
            self._curriculum_level += 1
        else:
            # No meaningful improvement — count stagnation.
            self._no_improve_count += 1
