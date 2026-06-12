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

"""Tests for builtin tools: file read/write/edit, grep, glob, task manager,
cron validation, tool registry, input schemas, and concurrency safety."""

import os
import tempfile
from pathlib import Path

import pytest

from encre.tools.base import EncreTool
from encre.tools.registry import ToolRegistry
from encre.tools.builtin import (
    EncreBashTool,
    EncreFileReadTool,
    EncreFileWriteTool,
    EncreFileEditTool,
    EncreGrepTool,
    EncreGlobTool,
    EncreTaskCreateTool,
    EncreTaskGetTool,
    EncreTaskListTool,
    EncreTaskUpdateTool,
    EncreCronCreateTool,
    EncreCronDeleteTool,
    EncreCronListTool,
)
from encre.task.manager import EncreTaskManager


# ===========================================================================
# File read tool
# ===========================================================================

class TestFileReadTool:
    """Test :class:`EncreFileReadTool`."""

    async def test_read_existing_file(self, temp_dir):
        tool = EncreFileReadTool()
        file_path = os.path.join(temp_dir, "README.md")
        result = await tool.execute(file_path=file_path)
        assert "# Test Project" in result

    async def test_read_nonexistent_file(self, temp_dir):
        tool = EncreFileReadTool()
        result = await tool.execute(file_path=os.path.join(temp_dir, "nonexistent.txt"))
        assert "Error" in result

    async def test_read_with_offset(self, temp_dir):
        tool = EncreFileReadTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(file_path=file_path, offset=3)
        # Should return content starting from line 3
        assert isinstance(result, str)

    async def test_read_with_limit(self, temp_dir):
        tool = EncreFileReadTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(file_path=file_path, limit=1)
        lines = result.strip().split("\n")
        assert len(lines) <= 2  # may include trailing newline

    async def test_read_empty_file(self, temp_dir):
        tool = EncreFileReadTool()
        file_path = os.path.join(temp_dir, "empty.txt")
        result = await tool.execute(file_path=file_path)
        assert result == ""

    def test_input_schema_required(self):
        assert "file_path" in EncreFileReadTool.input_schema.get("required", [])

    def test_is_concurrency_safe(self):
        tool = EncreFileReadTool()
        assert tool.is_concurrency_safe({"file_path": "/somewhere"}) is True


# ===========================================================================
# File write tool
# ===========================================================================

class TestFileWriteTool:
    """Test :class:`EncreFileWriteTool`."""

    async def test_write_new_file(self, temp_dir):
        tool = EncreFileWriteTool()
        file_path = os.path.join(temp_dir, "new_file.txt")
        result = await tool.execute(file_path=file_path, content="Hello, write!")
        assert "Successfully wrote" in result
        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "Hello, write!"

    async def test_write_overwrites_existing(self, temp_dir):
        tool = EncreFileWriteTool()
        file_path = os.path.join(temp_dir, "README.md")
        result = await tool.execute(file_path=file_path, content="Overwritten")
        assert "Successfully wrote" in result
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == "Overwritten"

    async def test_write_creates_parent_dirs(self, temp_dir):
        tool = EncreFileWriteTool()
        file_path = os.path.join(temp_dir, "nested", "deep", "file.txt")
        result = await tool.execute(file_path=file_path, content="Deep content")
        assert "Successfully wrote" in result
        assert os.path.exists(file_path)

    def test_input_schema_required(self):
        required = EncreFileWriteTool.input_schema.get("required", [])
        assert "file_path" in required
        assert "content" in required

    def test_is_concurrency_safe(self):
        tool = EncreFileWriteTool()
        assert tool.is_concurrency_safe({}) is False


# ===========================================================================
# File edit tool
# ===========================================================================

class TestFileEditTool:
    """Test :class:`EncreFileEditTool`."""

    async def test_edit_existing_file(self, temp_dir):
        tool = EncreFileEditTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(
            file_path=file_path,
            old_str="def hello():",
            new_str="def greeting():",
        )
        assert "Edit applied successfully" in result
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "def greeting():" in content
        assert "def hello():" not in content

    async def test_edit_non_unique_match(self, temp_dir):
        tool = EncreFileEditTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(
            file_path=file_path,
            old_str="\n",
            new_str="\n\n",
        )
        assert "Found" in result
        assert "occurrences" in result

    async def test_edit_no_match(self, temp_dir):
        tool = EncreFileEditTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(
            file_path=file_path,
            old_str="this string does not exist in file",
            new_str="nothing",
        )
        assert "Error" in result

    async def test_edit_nonexistent_file(self, temp_dir):
        tool = EncreFileEditTool()
        result = await tool.execute(
            file_path=os.path.join(temp_dir, "not_here.txt"),
            old_str="x",
            new_str="y",
        )
        assert "Error" in result

    def test_input_schema_required(self):
        required = EncreFileEditTool.input_schema.get("required", [])
        assert "file_path" in required
        assert "old_str" in required
        assert "new_str" in required

    def test_is_concurrency_safe(self):
        tool = EncreFileEditTool()
        assert tool.is_concurrency_safe({}) is False


# ===========================================================================
# Grep tool
# ===========================================================================

class TestGrepTool:
    """Test :class:`EncreGrepTool`."""

    async def test_grep_finds_string(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="def hello", path=temp_dir)
        assert "def hello" in result

    async def test_grep_no_match(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="FOOBARBAZQUX", path=temp_dir)
        assert "No matches found" in result

    async def test_grep_case_insensitive(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="DEF HELLO", path=temp_dir, **{"-i": True})
        assert "def hello" in result

    async def test_grep_files_with_matches_mode(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="def", path=temp_dir, output_mode="files_with_matches")
        assert "main.py" in result or "utils.py" in result

    async def test_grep_count_mode(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="def", path=temp_dir, output_mode="count")
        assert "match" in result.lower()

    async def test_grep_with_glob_filter(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="def", path=temp_dir, glob="*.py")
        assert "main.py" in result or "utils.py" in result
        assert "README.md" not in result

    async def test_grep_invalid_regex(self, temp_dir):
        tool = EncreGrepTool()
        result = await tool.execute(pattern="[invalid", path=temp_dir)
        assert "Error" in result

    async def test_grep_specific_file(self, temp_dir):
        tool = EncreGrepTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(pattern="hello", path=file_path)
        assert "def hello" in result

    def test_input_schema_required(self):
        required = EncreGrepTool.input_schema.get("required", [])
        assert "pattern" in required
        assert "path" in required

    def test_is_concurrency_safe(self):
        tool = EncreGrepTool()
        assert tool.is_concurrency_safe({}) is True


# ===========================================================================
# Glob tool
# ===========================================================================

class TestGlobTool:
    """Test :class:`EncreGlobTool`."""

    async def test_glob_py_files_root(self, temp_dir):
        tool = EncreGlobTool()
        result = await tool.execute(pattern="*.py", path=temp_dir)
        assert "main.py" in result

    async def test_glob_py_files_nested(self, temp_dir):
        tool = EncreGlobTool()
        result = await tool.execute(pattern="*/*.py", path=temp_dir)
        assert "utils.py" in result

    async def test_glob_md_files(self, temp_dir):
        tool = EncreGlobTool()
        result = await tool.execute(pattern="*.md", path=temp_dir)
        assert "README.md" in result

    async def test_glob_json_files(self, temp_dir):
        tool = EncreGlobTool()
        result = await tool.execute(pattern="**/*.json", path=temp_dir)
        assert "config.json" in result

    async def test_glob_no_match(self, temp_dir):
        tool = EncreGlobTool()
        result = await tool.execute(pattern="*.xyzzy", path=temp_dir)
        assert "No files match pattern" in result

    async def test_glob_default_path(self, temp_dir):
        tool = EncreGlobTool()
        result = await tool.execute(pattern="main.py")
        assert "main.py" in result

    def test_input_schema_required(self):
        required = EncreGlobTool.input_schema.get("required", [])
        assert "pattern" in required

    def test_is_concurrency_safe(self):
        tool = EncreGlobTool()
        assert tool.is_concurrency_safe({}) is True


# ===========================================================================
# Bash tool (safe commands only)
# ===========================================================================

class TestBashTool:
    """Test :class:`EncreBashTool` with safe commands."""

    async def test_bash_echo(self):
        tool = EncreBashTool()
        result = await tool.execute(command="echo hello world")
        assert "hello world" in result

    async def test_bash_pwd(self):
        tool = EncreBashTool()
        result = await tool.execute(command="pwd")
        assert result.strip() != ""

    async def test_bash_with_cwd(self, temp_dir):
        tool = EncreBashTool()
        result = await tool.execute(command="pwd", cwd=temp_dir)
        # On Windows, bash may translate paths (e.g. C:\Users\...\Temp\... -> /tmp/...).
        # We verify pwd ran successfully by checking the output is a non-empty path.
        result = result.strip()
        assert len(result) > 0
        assert "Error" not in result
        # The returned path should be absolute (start with / or drive letter)
        assert result.startswith("/") or ":" in result

    async def test_bash_command_not_found(self):
        tool = EncreBashTool()
        result = await tool.execute(command="nonexistentcommandxyz123")
        assert "Error" in result or "not found" in result.lower() or result.strip() != ""

    def test_input_schema_required(self):
        required = EncreBashTool.input_schema.get("required", [])
        assert "command" in required

    def test_is_concurrency_safe(self):
        tool = EncreBashTool()
        assert tool.is_concurrency_safe({}) is False


# ===========================================================================
# Task manager CRUD
# ===========================================================================

class TestTaskManagerCRUD:
    """Test :class:`EncreTaskManager` operations."""

    def setup_method(self):
        EncreTaskManager.clear()

    def teardown_method(self):
        EncreTaskManager.clear()

    def test_create_task(self):
        task_id = EncreTaskManager.create_task(
            name="Test task",
            description="A test",
            task_type="bash",
            prompt="echo hello",
        )
        assert task_id is not None
        assert len(task_id) > 0

    def test_get_task(self):
        task_id = EncreTaskManager.create_task(
            name="Get me",
            description="Test retrieval",
            task_type="agent",
            prompt="Do something",
        )
        task = EncreTaskManager.get_task(task_id)
        assert task is not None
        assert task.name == "Get me"
        assert task.task_type == "agent"

    def test_get_nonexistent_task(self):
        assert EncreTaskManager.get_task("nonexistent") is None

    def test_update_task_status(self):
        task_id = EncreTaskManager.create_task(
            name="Update me",
            description="Status change test",
            task_type="bash",
            prompt="run",
        )
        result = EncreTaskManager.update_task(task_id, status="running")
        assert result is True
        task = EncreTaskManager.get_task(task_id)
        assert task.status == "running"

    def test_update_task_with_result(self):
        task_id = EncreTaskManager.create_task(
            name="Result test",
            description="Set result",
            task_type="bash",
            prompt="run",
        )
        EncreTaskManager.update_task(task_id, status="completed", result="Success!")
        task = EncreTaskManager.get_task(task_id)
        assert task.status == "completed"
        assert task.result == "Success!"

    def test_update_nonexistent_task(self):
        assert EncreTaskManager.update_task("nonexistent", status="completed") is False

    def test_list_tasks_all(self):
        ids = []
        for i in range(3):
            tid = EncreTaskManager.create_task(
                name=f"Task {i}",
                description=f"Desc {i}",
                task_type="bash",
                prompt=f"cmd {i}",
            )
            ids.append(tid)
        tasks = EncreTaskManager.list_tasks()
        assert len(tasks) == 3

    def test_list_tasks_filter_by_status(self):
        tid1 = EncreTaskManager.create_task(name="Pending", description="...", task_type="bash", prompt="...")
        tid2 = EncreTaskManager.create_task(name="Running", description="...", task_type="bash", prompt="...")
        EncreTaskManager.update_task(tid2, status="running")

        pending = EncreTaskManager.list_tasks(status="pending")
        running = EncreTaskManager.list_tasks(status="running")
        assert len(pending) >= 1
        assert len(running) >= 1

    def test_delete_task(self):
        task_id = EncreTaskManager.create_task(name="Delete me", description="...", task_type="bash", prompt="...")
        assert EncreTaskManager.delete_task(task_id) is True
        assert EncreTaskManager.get_task(task_id) is None

    def test_delete_nonexistent_task(self):
        assert EncreTaskManager.delete_task("nonexistent") is False


# ===========================================================================
# Task tools (builtins)
# ===========================================================================

class TestTaskCreateTool:
    """Test :class:`EncreTaskCreateTool`."""

    def setup_method(self):
        EncreTaskManager.clear()

    def teardown_method(self):
        EncreTaskManager.clear()

    async def test_create_task_via_tool(self):
        tool = EncreTaskCreateTool()
        result = await tool.execute(
            name="My sub-task",
            description="Do the thing",
            task_type="bash",
            prompt="echo done",
        )
        assert "Task created:" in result
        task_id = result.split("Task created:")[1].strip()
        assert EncreTaskManager.get_task(task_id) is not None

    def test_input_schema_required(self):
        required = EncreTaskCreateTool.input_schema.get("required", [])
        assert "name" in required
        assert "task_type" in required
        assert "prompt" in required

    def test_is_concurrency_safe(self):
        tool = EncreTaskCreateTool()
        assert tool.is_concurrency_safe({}) is False


class TestTaskGetTool:
    """Test :class:`EncreTaskGetTool`."""

    def setup_method(self):
        EncreTaskManager.clear()

    def teardown_method(self):
        EncreTaskManager.clear()

    async def test_get_existing_task(self):
        task_id = EncreTaskManager.create_task(
            name="Detailed task",
            description="Check details",
            task_type="agent",
            prompt="Review code",
        )
        tool = EncreTaskGetTool()
        result = await tool.execute(task_id=task_id)
        assert "Detailed task" in result
        assert "agent" in result
        assert task_id in result

    async def test_get_nonexistent_task(self):
        tool = EncreTaskGetTool()
        result = await tool.execute(task_id="nonexistent-id")
        assert "Error" in result

    def test_input_schema_required(self):
        required = EncreTaskGetTool.input_schema.get("required", [])
        assert "task_id" in required

    def test_is_concurrency_safe(self):
        tool = EncreTaskGetTool()
        assert tool.is_concurrency_safe({}) is True


class TestTaskListTool:
    """Test :class:`EncreTaskListTool`."""

    def setup_method(self):
        EncreTaskManager.clear()

    def teardown_method(self):
        EncreTaskManager.clear()

    async def test_list_empty(self):
        tool = EncreTaskListTool()
        result = await tool.execute()
        assert "No tasks found" in result

    async def test_list_with_tasks(self):
        EncreTaskManager.create_task(name="T1", description="...", task_type="bash", prompt="...")
        EncreTaskManager.create_task(name="T2", description="...", task_type="bash", prompt="...")
        tool = EncreTaskListTool()
        result = await tool.execute()
        assert "T1" in result
        assert "T2" in result

    async def test_list_filtered(self):
        tid = EncreTaskManager.create_task(name="Running task", description="...", task_type="bash", prompt="...")
        EncreTaskManager.update_task(tid, status="running")
        tool = EncreTaskListTool()
        result = await tool.execute(status="running")
        assert "Running task" in result


class TestTaskUpdateTool:
    """Test :class:`EncreTaskUpdateTool`."""

    def setup_method(self):
        EncreTaskManager.clear()

    def teardown_method(self):
        EncreTaskManager.clear()

    async def test_update_status(self):
        task_id = EncreTaskManager.create_task(
            name="Status change",
            description="...",
            task_type="bash",
            prompt="...",
        )
        tool = EncreTaskUpdateTool()
        result = await tool.execute(task_id=task_id, status="completed", result="Done!")
        assert "updated successfully" in result.lower()
        task = EncreTaskManager.get_task(task_id)
        assert task.status == "completed"
        assert task.result == "Done!"

    async def test_update_nonexistent(self):
        tool = EncreTaskUpdateTool()
        result = await tool.execute(task_id="nonexistent", status="completed")
        assert "Error" in result

    def test_input_schema_required(self):
        required = EncreTaskUpdateTool.input_schema.get("required", [])
        assert "task_id" in required

    def test_is_concurrency_safe(self):
        tool = EncreTaskUpdateTool()
        assert tool.is_concurrency_safe({}) is False


# ===========================================================================
# Cron tools
# ===========================================================================

class TestCronCreateTool:
    """Test :class:`EncreCronCreateTool`."""

    async def test_validate_valid_cron(self):
        tool = EncreCronCreateTool()
        result = await tool.execute(
            cron="0 9 * * 1-5",
            prompt="Review PRs",
            name="Weekday review",
        )
        assert "validated" in result or "scheduled" in result or "ready" in result

    async def test_invalid_cron_rejected(self):
        tool = EncreCronCreateTool()
        result = await tool.execute(
            cron="invalid cron expr",
            prompt="Do something",
        )
        assert "Error" in result or "invalid" in result.lower()

    async def test_missing_cron(self):
        tool = EncreCronCreateTool()
        result = await tool.execute(cron="", prompt="Do something")
        assert "Error" in result

    async def test_missing_prompt(self):
        tool = EncreCronCreateTool()
        result = await tool.execute(cron="* * * * *", prompt="")
        assert "Error" in result

    async def test_with_scheduler_backend(self):
        from encre.scheduler import EncreScheduler
        sched = EncreScheduler()
        tool = EncreCronCreateTool()
        tool.set_scheduler(sched)
        result = await tool.execute(
            cron="0 12 * * *",
            prompt="Lunchtime check",
            name="Lunch check",
        )
        assert "job_id" in result.lower()
        tool.set_scheduler(None)  # Reset for other tests

    def test_input_schema_required(self):
        required = EncreCronCreateTool.input_schema.get("required", [])
        assert "cron" in required
        assert "prompt" in required

    def test_is_concurrency_safe(self):
        tool = EncreCronCreateTool()
        assert tool.is_concurrency_safe({}) is False


class TestCronDeleteTool:
    """Test :class:`EncreCronDeleteTool`."""

    def test_tool_exists_and_has_schema(self):
        tool = EncreCronDeleteTool()
        assert tool.name == "cron_delete"
        # The property in the schema is "job_id", not "id"
        props = tool.input_schema.get("properties", {})
        assert "job_id" in props
        assert props["job_id"]["type"] == "string"


class TestCronListTool:
    """Test :class:`EncreCronListTool`."""

    def test_tool_exists_and_has_schema(self):
        tool = EncreCronListTool()
        assert tool.name == "cron_list"
        assert hasattr(tool, "input_schema")


# ===========================================================================
# ToolRegistry
# ===========================================================================

class TestToolRegistry:
    """Test :class:`ToolRegistry`."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = EncreFileReadTool()
        registry.register(tool)
        assert registry.get("file_read") is tool

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_register_many(self):
        registry = ToolRegistry()
        tools = [EncreFileReadTool(), EncreFileWriteTool(), EncreGrepTool()]
        registry.register_many(tools)
        assert len(registry.all()) == 3
        assert registry.get("file_read") is not None
        assert registry.get("file_write") is not None
        assert registry.get("grep") is not None

    def test_register_overwrites(self):
        registry = ToolRegistry()
        t1 = EncreFileReadTool()
        t2 = EncreFileReadTool()
        registry.register(t1)
        registry.register(t2)
        assert len(registry.all()) == 1

    def test_get_openai_tools(self):
        registry = ToolRegistry()
        registry.register(EncreFileReadTool())
        openai_tools = registry.get_openai_tools()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert "function" in openai_tools[0]

    def test_get_anthropic_tools(self):
        registry = ToolRegistry()
        registry.register(EncreFileReadTool())
        anthropic_tools = registry.get_anthropic_tools()
        assert len(anthropic_tools) == 1
        assert "name" in anthropic_tools[0]
        assert "input_schema" in anthropic_tools[0]


# ===========================================================================
# Tool input schema validation
# ===========================================================================

class TestToolInputSchemas:
    """Verify every builtin tool has a well-formed input schema."""

    def test_all_tools_have_name(self):
        for tool_cls in [
            EncreFileReadTool, EncreFileWriteTool, EncreFileEditTool,
            EncreGrepTool, EncreGlobTool, EncreBashTool,
            EncreTaskCreateTool, EncreTaskGetTool, EncreTaskListTool, EncreTaskUpdateTool,
            EncreCronCreateTool, EncreCronDeleteTool, EncreCronListTool,
        ]:
            assert hasattr(tool_cls, "name"), f"{tool_cls.__name__} missing 'name'"
            assert isinstance(getattr(tool_cls, "name"), str)

    def test_all_tools_have_description(self):
        for tool_cls in [
            EncreFileReadTool, EncreFileWriteTool, EncreFileEditTool,
            EncreGrepTool, EncreGlobTool, EncreBashTool,
            EncreTaskCreateTool, EncreTaskGetTool, EncreTaskListTool, EncreTaskUpdateTool,
            EncreCronCreateTool, EncreCronDeleteTool, EncreCronListTool,
        ]:
            assert hasattr(tool_cls, "description"), f"{tool_cls.__name__} missing 'description'"

    def test_all_tools_have_input_schema(self):
        for tool_cls in [
            EncreFileReadTool, EncreFileWriteTool, EncreFileEditTool,
            EncreGrepTool, EncreGlobTool, EncreBashTool,
            EncreTaskCreateTool, EncreTaskGetTool, EncreTaskListTool, EncreTaskUpdateTool,
            EncreCronCreateTool, EncreCronDeleteTool, EncreCronListTool,
        ]:
            schema = getattr(tool_cls, "input_schema")
            assert isinstance(schema, dict), f"{tool_cls.__name__} input_schema not a dict"
            assert "type" in schema, f"{tool_cls.__name__} input_schema missing 'type'"
            assert schema["type"] == "object", f"{tool_cls.__name__} input_schema not 'object' type"

    def test_all_tools_have_to_openai_format(self):
        for tool_cls in [EncreFileReadTool, EncreBashTool, EncreGrepTool]:
            tool = tool_cls()
            fmt = tool.to_openai_format()
            assert "type" in fmt
            assert fmt["type"] == "function"

    def test_all_tools_have_to_anthropic_format(self):
        for tool_cls in [EncreFileReadTool, EncreBashTool, EncreGrepTool]:
            tool = tool_cls()
            fmt = tool.to_anthropic_format()
            assert "name" in fmt
            assert "input_schema" in fmt


# ===========================================================================
# Concurrency safety matrix
# ===========================================================================

class TestConcurrencySafety:
    """Verify is_concurrency_safe() returns expected values for each tool."""

    def test_read_only_tools_are_safe(self):
        assert EncreFileReadTool().is_concurrency_safe({}) is True
        assert EncreGrepTool().is_concurrency_safe({}) is True
        assert EncreGlobTool().is_concurrency_safe({}) is True
        assert EncreTaskGetTool().is_concurrency_safe({}) is True
        assert EncreTaskListTool().is_concurrency_safe({}) is True

    def test_write_tools_are_not_safe(self):
        assert EncreFileWriteTool().is_concurrency_safe({}) is False
        assert EncreFileEditTool().is_concurrency_safe({}) is False
        assert EncreBashTool().is_concurrency_safe({}) is False
        assert EncreTaskCreateTool().is_concurrency_safe({}) is False
        assert EncreTaskUpdateTool().is_concurrency_safe({}) is False
        assert EncreCronCreateTool().is_concurrency_safe({}) is False


# ===========================================================================
# EncreTool ABC compliance
# ===========================================================================

class TestEncreToolABC:
    """Test that :class:`EncreTool` ABC is properly defined."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EncreTool()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        tool = EncreFileReadTool()
        assert isinstance(tool, EncreTool)

    def test_execute_is_abstract(self):
        assert "execute" in EncreTool.__abstractmethods__

    def test_base_has_is_concurrency_safe(self):
        tool = EncreFileReadTool()
        assert hasattr(tool, "is_concurrency_safe")


# ===========================================================================
# Edge cases: file tool with special characters
# ===========================================================================

class TestFileToolsEdgeCases:
    """Edge-case tests for file tools."""

    async def test_read_with_offset_beyond_length(self, temp_dir):
        tool = EncreFileReadTool()
        result = await tool.execute(
            file_path=os.path.join(temp_dir, "main.py"),
            offset=999,
        )
        # Should return empty string when offset exceeds file length
        assert result == ""

    async def test_write_unicode_content(self, temp_dir):
        tool = EncreFileWriteTool()
        file_path = os.path.join(temp_dir, "unicode.txt")
        content = "中文测试\nEmoji: 🎉\nMixed: Café résumé"
        result = await tool.execute(file_path=file_path, content=content)
        assert "Successfully wrote" in result
        with open(file_path, "r", encoding="utf-8") as f:
            assert f.read() == content

    async def test_edit_multiline_match(self, temp_dir):
        tool = EncreFileEditTool()
        file_path = os.path.join(temp_dir, "main.py")
        result = await tool.execute(
            file_path=file_path,
            old_str="def hello():\n    return 'Hello, world!'",
            new_str="def hello():\n    return 'Hola, mundo!'",
        )
        assert "Edit applied successfully" in result
        with open(file_path, "r", encoding="utf-8") as f:
            assert "Hola, mundo!" in f.read()

    async def test_read_binary_file(self, temp_dir):
        tool = EncreFileReadTool()
        result = await tool.execute(file_path=os.path.join(temp_dir, "data.bin"))
        # Should read without crashing (may produce garbled text)
        assert isinstance(result, str)
