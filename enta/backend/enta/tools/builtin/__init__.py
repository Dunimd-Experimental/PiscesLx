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



"""
EnCRE builtin tool set -- adversarial-training tool palette.

This is the curated, real tool surface that the trained model is exposed to
during adversarial training rollouts.  Every tool listed here is a fully
implemented class living in the same directory; the registry below is the
single source of truth for the model's tool palette.

The tool set is deliberately code- and shell-centric: it gives the model
the ability to read, edit, search, and execute code, while keeping external
side-effects (browsing, deployment, Docker, etc.) inside the sandbox so the
training loop can audit every action.
"""

from enta.tools.builtin.agent import EncreAgentTool
from enta.tools.builtin.apply_patch import EncreApplyPatchTool
from enta.tools.builtin.bash import EncreBashTool
from enta.tools.builtin.bash_io import (
    EncreBashKillTool,
    EncreBashListTool,
    EncreBashOutputTool,
)
from enta.tools.builtin.database import EncreDatabaseTool
from enta.tools.builtin.deploy import EncreDeployTool
from enta.tools.builtin.docker import EncreDockerTool
from enta.tools.builtin.file_edit import EncreFileEditTool
from enta.tools.builtin.file_read import EncreFileReadTool
from enta.tools.builtin.file_write import EncreFileWriteTool
from enta.tools.builtin.find_tool import EncreFindToolTool
from enta.tools.builtin.git_tool import EncreGitTool
from enta.tools.builtin.glob import EncreGlobTool
from enta.tools.builtin.grep import EncreGrepTool
from enta.tools.builtin.image import EncreImageTool
from enta.tools.builtin.lint_format import EncreLintFormatTool
from enta.tools.builtin.lsp import EncreLSPTool
from enta.tools.builtin.memory import (
    EncreMemoryCreateTool,
    EncreMemoryDeleteTool,
    EncreMemoryProfileTool,
    EncreMemoryReadTool,
    EncreMemorySearchTool,
    EncreMemoryUpdateTool,
)
from enta.tools.builtin.pdf import EncrePDFTool
from enta.tools.builtin.rest_client import EncreRESTTool
from enta.tools.builtin.spreadsheet import EncreSpreadsheetTool
from enta.tools.builtin.task_create import EncreTaskCreateTool
from enta.tools.builtin.task_get import EncreTaskGetTool
from enta.tools.builtin.task_list import EncreTaskListTool
from enta.tools.builtin.task_output import EncreTaskOutputTool
from enta.tools.builtin.task_stop import EncreTaskStopTool
from enta.tools.builtin.task_update import EncreTaskUpdateTool
from enta.tools.builtin.test_runner import EncreTestRunTool
from enta.tools.builtin.todo import EncreTodoTool
from enta.tools.builtin.web_fetch import EncreWebFetchTool
from enta.tools.builtin.web_search import EncreWebSearchTool
from enta.tools.builtin.workflow import EncreWorkflowTool


def build_default_tool_registry():
    """Instantiate every builtin tool and return them in a dict.

    The registry is built lazily and on demand by the training pipeline,
    so importing this module is cheap and side-effect free.
    """
    from enta.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tools = [
        EncreAgentTool(),
        EncreApplyPatchTool(),
        EncreBashTool(),
        EncreBashKillTool(),
        EncreBashListTool(),
        EncreBashOutputTool(),
        EncreDatabaseTool(),
        EncreDeployTool(),
        EncreDockerTool(),
        EncreFileEditTool(),
        EncreFileReadTool(),
        EncreFileWriteTool(),
        EncreFindToolTool(),
        EncreGitTool(),
        EncreGlobTool(),
        EncreGrepTool(),
        EncreImageTool(),
        EncreLintFormatTool(),
        EncreLSPTool(),
        EncreMemoryCreateTool(),
        EncreMemoryDeleteTool(),
        EncreMemoryProfileTool(),
        EncreMemoryReadTool(),
        EncreMemorySearchTool(),
        EncreMemoryUpdateTool(),
        EncrePDFTool(),
        EncreRESTTool(),
        EncreSpreadsheetTool(),
        EncreTaskCreateTool(),
        EncreTaskGetTool(),
        EncreTaskListTool(),
        EncreTaskOutputTool(),
        EncreTaskStopTool(),
        EncreTaskUpdateTool(),
        EncreTestRunTool(),
        EncreTodoTool(),
        EncreWebFetchTool(),
        EncreWebSearchTool(),
        EncreWorkflowTool(),
    ]
    registry.register_many(tools)
    return registry


DEFAULT_BUILTIN_TOOLS = build_default_tool_registry


__all__ = [
    "DEFAULT_BUILTIN_TOOLS",
    "EncreAgentTool",
    "EncreApplyPatchTool",
    "EncreBashKillTool",
    "EncreBashListTool",
    "EncreBashOutputTool",
    "EncreBashTool",
    "EncreDatabaseTool",
    "EncreDeployTool",
    "EncreDockerTool",
    "EncreFileEditTool",
    "EncreFileReadTool",
    "EncreFileWriteTool",
    "EncreFindToolTool",
    "EncreGitTool",
    "EncreGlobTool",
    "EncreGrepTool",
    "EncreImageTool",
    "EncreLSPTool",
    "EncreLintFormatTool",
    "EncreMemoryCreateTool",
    "EncreMemoryDeleteTool",
    "EncreMemoryProfileTool",
    "EncreMemoryReadTool",
    "EncreMemorySearchTool",
    "EncreMemoryUpdateTool",
    "EncrePDFTool",
    "EncreRESTTool",
    "EncreSpreadsheetTool",
    "EncreTaskCreateTool",
    "EncreTaskGetTool",
    "EncreTaskListTool",
    "EncreTaskOutputTool",
    "EncreTaskStopTool",
    "EncreTaskUpdateTool",
    "EncreTestRunTool",
    "EncreTodoTool",
    "EncreWebFetchTool",
    "EncreWebSearchTool",
    "EncreWorkflowTool",
    "build_default_tool_registry",
]
