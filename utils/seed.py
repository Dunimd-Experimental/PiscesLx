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

"""Determinism control for training runs.

A metric measured on an unseeded run is a sample, not a result. On this project's host tier
two runs of the same configuration at the same step budget scored held-out next-token
cross-entropy 11.2703 and 11.6965 -- the same sign (both below the uniform baseline) but a
0.43 nat spread, which is larger than the margins several P1 acceptance numbers ask for.
This module is the single place that pins every generator a run touches, so a reported number
can be reproduced and a reported improvement can be attributed to the change that caused it.
"""

import random
from typing import Any, Dict, Optional

import torch

#: Seed values must be inside the 32-bit range so they are portable across generators.
SEED_MODULUS = 2 ** 31 - 1


class PiscesLxDeterminism:
    """Apply and report the random-state pinning for one process.

    Example:
        >>> PiscesLxDeterminism.apply(7)
        >>> PiscesLxDeterminism.describe()
        {'seed': 7, 'numpy': True, 'cuda': False}
    """

    _seed: Optional[int] = None

    @classmethod
    def apply(cls, seed: Optional[int]) -> Optional[int]:
        """Pin Python, NumPy and every Torch generator to ``seed``.

        Args:
            seed: The seed to install. ``None`` leaves the process unseeded (the default for
                accelerator hosts, where data order is decided by the loader and pinning
                would only hide non-determinism behind a misleading label).

        Returns:
            The seed actually installed, or None when nothing was pinned.
        """
        if seed is None:
            cls._seed = None
            return None
        value = int(seed) % SEED_MODULUS
        cls._seed = value
        random.seed(value)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
        try:
            import numpy
        except ImportError:
            numpy = None
        if numpy is not None:
            numpy.random.seed(value)
        return value

    @classmethod
    def current(cls) -> Optional[int]:
        """The installed seed, or None when the run is unseeded."""
        return cls._seed

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        """A flat, loggable report of what is pinned in this process."""
        try:
            import numpy
            numpy_present = True
        except ImportError:
            numpy_present = False
        return {
            "seed": cls._seed,
            "numpy": numpy_present,
            "cuda": bool(torch.cuda.is_available()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        }
