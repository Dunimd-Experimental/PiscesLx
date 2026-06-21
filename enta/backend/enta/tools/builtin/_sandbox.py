#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


"""Workspace path remapping for the file tools.

The training loop sets a workspace path via
``enta.tools.builtin.bash.set_workspace`` (or directly here).  When the
model asks to read or write a path that is absolute or starts with
``/workspace`` (the conventional placeholder used by tool prompts), we
remap it to ``<workspace>/<rest>``.  Otherwise the path is returned
unchanged.  Path traversal sequences are normalised to prevent the
model from escaping the workspace root.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

_workspace: ContextVar[Optional[str]] = ContextVar(
    "enta_builtin_workspace", default=None
)


def set_workspace(path: Optional[str]) -> object:
    return _workspace.set(path)


def reset_workspace(token: object) -> None:
    _workspace.reset(token)


def get_workspace() -> Optional[str]:
    return _workspace.get()


def remap_tool_path(file_path: str) -> str:
    if file_path is None:
        return ""
    file_path = str(file_path)

    workspace = get_workspace()
    if not workspace:
        return os.path.normpath(file_path)

    if file_path.startswith("/workspace"):
        rel = file_path[len("/workspace"):].lstrip("/")
        candidate = os.path.join(workspace, rel) if rel else workspace
        return os.path.normpath(candidate)

    if os.path.isabs(file_path):
        return os.path.normpath(file_path)

    return os.path.normpath(os.path.join(workspace, file_path))
