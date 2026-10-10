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
Executable conformance checks for the mandatory PiscesL1 coding standards.

Every prohibition listed in ``docs/prompts/UPGRADE_06_CODING.md`` §6.8 and
``docs/prompts/UPGRADE_10_DELIVER.md`` §10.2 is mirrored by a checker here, so
acceptance is decided by script output rather than by a human declaration.
"""

from tools.check.base import (
    PiscesLxCheckReport,
    PiscesLxRepositoryScanner,
    PiscesLxRuleId,
    PiscesLxSeverity,
    PiscesLxSourceFile,
    PiscesLxStandardCheck,
    PiscesLxStandardHeader,
    PiscesLxViolation,
)
from tools.check.runner import PiscesLxCheckCatalog, PiscesLxCheckCli, PiscesLxCheckSuite

__all__ = [
    "PiscesLxCheckCatalog",
    "PiscesLxCheckCli",
    "PiscesLxCheckReport",
    "PiscesLxCheckSuite",
    "PiscesLxRepositoryScanner",
    "PiscesLxRuleId",
    "PiscesLxSeverity",
    "PiscesLxSourceFile",
    "PiscesLxStandardCheck",
    "PiscesLxStandardHeader",
    "PiscesLxViolation",
]
