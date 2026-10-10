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
Model-layer device resolver.

The one implementation lives in ``utils.device.PiscesLxDeviceResolver``. This module
exists only because coding standard §6.1 requires every class inside ``model/`` to
carry the ``Yv`` prefix, so the model layer reaches the shared resolver through a name
that satisfies that rule. It deliberately holds no logic of its own: a second copy here
is what let ``model/core/cache.py`` default to the literal ``"cuda"`` and crash CPU-only
hosts while ``model/core/model.py`` behaved correctly.
"""

from utils.device import PiscesLxDeviceResolver

__all__ = ["YvDeviceResolver"]


class YvDeviceResolver(PiscesLxDeviceResolver):
    """``model/``-facing name for the project-wide device resolver."""
