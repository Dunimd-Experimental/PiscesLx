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
from encre.tools.builtin.file_read import EncreFileReadTool
from encre.tools.builtin.file_write import EncreFileWriteTool
from encre.tools.builtin.file_edit import EncreFileEditTool
from encre.tools.builtin.apply_patch import EncreApplyPatchTool
from encre.tools.builtin.bash import EncreBashTool
from encre.tools.builtin.bash_io import (
    EncreBashOutputTool,
    EncreBashKillTool,
    EncreBashListTool,
)
from encre.tools.builtin.grep import EncreGrepTool
from encre.tools.builtin.glob import EncreGlobTool
from encre.tools.builtin.web_fetch import EncreWebFetchTool
from encre.tools.builtin.web_search import EncreWebSearchTool
from encre.tools.builtin.todo import EncreTodoTool
from encre.tools.builtin.task_create import EncreTaskCreateTool
from encre.tools.builtin.task_get import EncreTaskGetTool
from encre.tools.builtin.task_list import EncreTaskListTool
from encre.tools.builtin.task_update import EncreTaskUpdateTool
from encre.tools.builtin.task_stop import EncreTaskStopTool
from encre.tools.builtin.task_output import EncreTaskOutputTool
from encre.tools.builtin.cron_create import EncreCronCreateTool
from encre.tools.builtin.cron_delete import EncreCronDeleteTool
from encre.tools.builtin.cron_list import EncreCronListTool
from encre.tools.builtin.agent import EncreAgentTool
from encre.tools.builtin.find_tool import EncreFindToolTool
from encre.tools.builtin.lsp import EncreLSPTool
from encre.tools.builtin.browser import EncreBrowserTool
from encre.tools.builtin.notebook import EncreNotebookTool
from encre.tools.builtin.database import EncreDatabaseTool
from encre.tools.builtin.docker import EncreDockerTool
from encre.tools.builtin.git_tool import EncreGitTool
from encre.tools.builtin.rest_client import EncreRESTTool
from encre.tools.builtin.pdf import EncrePDFTool
from encre.tools.builtin.spreadsheet import EncreSpreadsheetTool
from encre.tools.builtin.image import EncreImageTool
from encre.tools.builtin.deploy import EncreDeployTool
from encre.tools.builtin.desktop import EncreDesktopTool
from encre.tools.builtin.memory import (
    EncreMemoryCreateTool,
    EncreMemoryReadTool,
    EncreMemoryUpdateTool,
    EncreMemoryDeleteTool,
    EncreMemorySearchTool,
    EncreMemoryProfileTool,
)
from encre.tools.builtin.question import EncreQuestionTool

__all__ = [
    "EncreFileReadTool",
    "EncreFileWriteTool",
    "EncreFileEditTool",
    "EncreApplyPatchTool",
    "EncreBashTool",
    "EncreBashOutputTool",
    "EncreBashKillTool",
    "EncreBashListTool",
    "EncreGrepTool",
    "EncreGlobTool",
    "EncreWebFetchTool",
    "EncreWebSearchTool",
    "EncreTodoTool",
    "EncreTaskCreateTool",
    "EncreTaskGetTool",
    "EncreTaskListTool",
    "EncreTaskUpdateTool",
    "EncreTaskStopTool",
    "EncreTaskOutputTool",
    "EncreCronCreateTool",
    "EncreCronDeleteTool",
    "EncreCronListTool",
    "EncreAgentTool",
    "EncreFindToolTool",
    "EncreLSPTool",
    "EncreBrowserTool",
    "EncreNotebookTool",
    "EncreDatabaseTool",
    "EncreDockerTool",
    "EncreGitTool",
    "EncreRESTTool",
    "EncrePDFTool",
    "EncreSpreadsheetTool",
    "EncreImageTool",
    "EncreDeployTool",
    "EncreDesktopTool",
    "EncreMemoryCreateTool",
    "EncreMemoryReadTool",
    "EncreMemoryUpdateTool",
    "EncreMemoryDeleteTool",
    "EncreMemorySearchTool",
    "EncreMemoryProfileTool",
    "EncreQuestionTool",
]
