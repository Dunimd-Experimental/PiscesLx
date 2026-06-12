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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from encre.memdir.system import MemoryHeader


def format_memory_manifest(memories: list[MemoryHeader]) -> str:
    if not memories:
        return ""

    lines: list[str] = []
    lines.append("# Memory Manifest")
    lines.append("")
    lines.append("| # | File | Age | Description | Type | Tags |")
    lines.append("|---|---|---|---|---|---|")

    for i, m in enumerate(memories, 1):
        desc = m.description or "-"
        if len(desc) > 60:
            desc = desc[:57] + "..."
        mtype = m.memory_type or "-"
        tags = ", ".join(m.tags) if m.tags else "-"
        if len(tags) > 40:
            tags = tags[:37] + "..."

        lines.append(
            f"| {i} | {m.filename} | {m.age_text} | {desc} | {mtype} | {tags} |"
        )

    lines.append("")
    return "\n".join(lines)
