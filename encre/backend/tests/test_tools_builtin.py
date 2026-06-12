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

"""Tests for built-in tool implementations (surface-level, no network calls)."""

import asyncio

import pytest

from encre.tools.base import EncreTool


# ===========================================================================
# Tool base class
# ===========================================================================

class TestEncreTool:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EncreTool()

    def test_concrete_tool_instantiates(self):
        from encre.tools.builtin import EncreFileReadTool
        tool = EncreFileReadTool()
        assert isinstance(tool, EncreTool)

    def test_concrete_tool_has_name(self):
        from encre.tools.builtin import EncreFileReadTool
        tool = EncreFileReadTool()
        assert tool.name == "file_read"

    def test_concrete_tool_has_description(self):
        from encre.tools.builtin import EncreFileReadTool
        tool = EncreFileReadTool()
        assert len(tool.description) > 0


# ===========================================================================
# File tools format
# ===========================================================================

class TestFileToolsFormat:
    def test_file_read_openai_format(self):
        from encre.tools.builtin import EncreFileReadTool
        tool = EncreFileReadTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert "name" in fmt["function"]
        assert "parameters" in fmt["function"]

    def test_file_read_anthropic_format(self):
        from encre.tools.builtin import EncreFileReadTool
        tool = EncreFileReadTool()
        fmt = tool.to_anthropic_format()
        assert "name" in fmt
        assert "input_schema" in fmt

    def test_file_write_openai_format(self):
        from encre.tools.builtin import EncreFileWriteTool
        tool = EncreFileWriteTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_file_edit_openai_format(self):
        from encre.tools.builtin import EncreFileEditTool
        tool = EncreFileEditTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_bash_openai_format(self):
        from encre.tools.builtin import EncreBashTool
        tool = EncreBashTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_grep_openai_format(self):
        from encre.tools.builtin import EncreGrepTool
        tool = EncreGrepTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_glob_openai_format(self):
        from encre.tools.builtin import EncreGlobTool
        tool = EncreGlobTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# Web tools
# ===========================================================================

class TestWebTools:
    def test_web_fetch_format(self):
        from encre.tools.builtin import EncreWebFetchTool
        tool = EncreWebFetchTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert "url" in str(fmt["function"]["parameters"])

    def test_web_search_format(self):
        from encre.tools.builtin import EncreWebSearchTool
        tool = EncreWebSearchTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert "query" in str(fmt["function"]["parameters"])


# ===========================================================================
# Task management tools
# ===========================================================================

class TestTaskTools:
    def test_task_create_format(self):
        from encre.tools.builtin import EncreTaskCreateTool
        tool = EncreTaskCreateTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_task_list_format(self):
        from encre.tools.builtin import EncreTaskListTool
        tool = EncreTaskListTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_task_get_format(self):
        from encre.tools.builtin import EncreTaskGetTool
        tool = EncreTaskGetTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_task_update_format(self):
        from encre.tools.builtin import EncreTaskUpdateTool
        tool = EncreTaskUpdateTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_task_stop_format(self):
        from encre.tools.builtin import EncreTaskStopTool
        tool = EncreTaskStopTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_task_output_format(self):
        from encre.tools.builtin import EncreTaskOutputTool
        tool = EncreTaskOutputTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# Cron tools
# ===========================================================================

class TestCronTools:
    def test_cron_create_format(self):
        from encre.tools.builtin import EncreCronCreateTool
        tool = EncreCronCreateTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_cron_delete_format(self):
        from encre.tools.builtin import EncreCronDeleteTool
        tool = EncreCronDeleteTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"

    def test_cron_list_format(self):
        from encre.tools.builtin import EncreCronListTool
        tool = EncreCronListTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# Agent tool
# ===========================================================================

class TestAgentTool:
    def test_agent_tool_format(self):
        from encre.tools.builtin import EncreAgentTool
        tool = EncreAgentTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# LSP tool
# ===========================================================================

class TestLSPTool:
    def test_lsp_tool_format(self):
        from encre.tools.builtin import EncreLSPTool
        tool = EncreLSPTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# Browser tool
# ===========================================================================

class TestBrowserTool:
    def test_browser_tool_format(self):
        from encre.tools.builtin import EncreBrowserTool
        tool = EncreBrowserTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# Notebook tool
# ===========================================================================

class TestNotebookTool:
    def test_notebook_tool_format(self):
        from encre.tools.builtin.notebook import EncreNotebookTool
        tool = EncreNotebookTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# Todo tool
# ===========================================================================

class TestTodoTool:
    def test_todo_tool_format(self):
        from encre.tools.builtin import EncreTodoTool
        tool = EncreTodoTool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"


# ===========================================================================
# MCP tool
# ===========================================================================

class TestMCPTool:
    def test_mcp_tool_create(self):
        from encre.tools.mcp import EncreMCPTool
        tool = EncreMCPTool(command="echo hello")
        assert tool.name == "mcp"
        assert tool._command == "echo hello"

    def test_mcp_tool_format(self):
        from encre.tools.mcp import EncreMCPTool
        tool = EncreMCPTool(command="echo hello")
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
