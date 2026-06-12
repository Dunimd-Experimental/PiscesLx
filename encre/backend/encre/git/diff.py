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
import os
from dataclasses import dataclass

from encre.native import compute_diff as _native_compute_diff
from encre.native import apply_diff as _native_apply_diff


@dataclass
class GitDiffResult:
    files: int
    insertions: int
    deletions: int


class EncreGitDiff:
    @staticmethod
    def compute_diff(old: str, new: str) -> str:
        return _native_compute_diff(old, new)

    @staticmethod
    def apply_diff(content: str, diff: str) -> str:
        return _native_apply_diff(content, diff)

    @staticmethod
    def parse_diff_stats(diff_output: str) -> dict[str, int]:
        files = 0
        insertions = 0
        deletions = 0
        for line in diff_output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                files += 1
                try:
                    add = int(parts[0]) if parts[0] != "-" else 0
                    dlt = int(parts[1]) if parts[1] != "-" else 0
                    insertions += add
                    deletions += dlt
                except ValueError:
                    pass
        return {"total_files": files, "total_insertions": insertions, "total_deletions": deletions}

    @staticmethod
    def is_transient_git_state(workspace: str) -> bool:
        git_dir = os.path.join(workspace, ".git")
        if not os.path.exists(git_dir):
            return False
        transient_names = [
            "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD",
            "BISECT_START", "rebase-merge", "rebase-apply",
        ]
        for name in transient_names:
            if os.path.exists(os.path.join(git_dir, name)):
                return True
        return False
