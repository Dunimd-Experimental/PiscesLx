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

"""Tests for encre.task — EncreTask, EncreTaskManager, and EncreTaskExecutor."""

import time
import uuid

import pytest


# ===========================================================================
# EncreTask dataclass
# ===========================================================================

class TestEncreTask:
    """Tests for the EncreTask dataclass."""

    def test_creation(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="task-001",
            name="Test Task",
            description="A test task",
            task_type="bash",
            prompt="echo hello",
        )
        assert task.id == "task-001"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.task_type == "bash"
        assert task.prompt == "echo hello"

    def test_default_status_pending(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="t1",
            name="Task",
            description="Desc",
            task_type="agent",
            prompt="do something",
        )
        assert task.status == "pending"

    def test_default_empty_result_and_error(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="t1",
            name="Task",
            description="Desc",
            task_type="agent",
            prompt="do something",
        )
        assert task.result == ""
        assert task.error == ""

    def test_default_parent_id_none(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="t1",
            name="Task",
            description="Desc",
            task_type="agent",
            prompt="do something",
        )
        assert task.parent_id is None

    def test_default_metadata_empty(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="t1",
            name="Task",
            description="Desc",
            task_type="agent",
            prompt="do something",
        )
        assert task.metadata == {}

    def test_timestamps_are_float(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="t1",
            name="Task",
            description="Desc",
            task_type="agent",
            prompt="do something",
        )
        assert isinstance(task.created_at, float)
        assert isinstance(task.updated_at, float)

    def test_with_metadata(self):
        from encre.task.types import EncreTask
        task = EncreTask(
            id="t1",
            name="Task",
            description="Desc",
            task_type="workflow",
            prompt="step1\nstep2",
            metadata={"priority": 1, "tags": ["urgent"]},
        )
        assert task.metadata["priority"] == 1
        assert "urgent" in task.metadata["tags"]

    def test_is_dataclass(self):
        from encre.task.types import EncreTask
        from dataclasses import is_dataclass
        assert is_dataclass(EncreTask)

    def test_task_type_valid_values(self):
        from encre.task.types import EncreTask
        for tt in ["bash", "agent", "workflow"]:
            task = EncreTask(
                id="t1", name=f"Task {tt}", description="", task_type=tt, prompt="test"
            )
            assert task.task_type == tt

    def test_status_valid_values(self):
        from encre.task.types import EncreTask
        for status in ["pending", "running", "completed", "failed", "killed"]:
            task = EncreTask(
                id="t1", name="Task", description="", task_type="bash", prompt="test",
                status=status,
            )
            assert task.status == status


# ===========================================================================
# EncreTaskManager CRUD
# ===========================================================================

class TestEncreTaskManager:
    """Tests for EncreTaskManager class-level CRUD operations."""

    @pytest.fixture(autouse=True)
    def _clear_before_test(self):
        """Clear tasks before each test to ensure isolation."""
        from encre.task.manager import EncreTaskManager
        EncreTaskManager.clear()

    def test_create_task_returns_id(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="Test",
            description="A test task",
            task_type="bash",
            prompt="echo hello",
        )
        assert isinstance(task_id, str)
        uuid.UUID(task_id)

    def test_create_task_stores_with_default_status(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="Test",
            description="Desc",
            task_type="agent",
            prompt="do work",
        )
        task = EncreTaskManager.get_task(task_id)
        assert task is not None
        assert task.status == "pending"
        assert task.result == ""
        assert task.error == ""

    def test_get_task_returns_none_for_missing(self):
        from encre.task.manager import EncreTaskManager
        task = EncreTaskManager.get_task("nonexistent-id")
        assert task is None

    def test_get_task_returns_created_task(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="My Task",
            description="Important",
            task_type="workflow",
            prompt="step1\nstep2",
        )
        task = EncreTaskManager.get_task(task_id)
        assert task is not None
        assert task.id == task_id
        assert task.name == "My Task"
        assert task.description == "Important"
        assert task.task_type == "workflow"

    def test_update_task_status(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="T", description="D", task_type="bash", prompt="echo hi"
        )
        result = EncreTaskManager.update_task(task_id, status="running")
        assert result is True
        task = EncreTaskManager.get_task(task_id)
        assert task.status == "running"

    def test_update_task_result(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="T", description="D", task_type="bash", prompt="echo hi"
        )
        EncreTaskManager.update_task(task_id, status="completed", result="success output")
        task = EncreTaskManager.get_task(task_id)
        assert task.status == "completed"
        assert task.result == "success output"

    def test_update_task_error(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="T", description="D", task_type="bash", prompt="invalid"
        )
        EncreTaskManager.update_task(task_id, status="failed", error="command not found")
        task = EncreTaskManager.get_task(task_id)
        assert task.status == "failed"
        assert task.error == "command not found"

    def test_update_nonexistent_task_returns_false(self):
        from encre.task.manager import EncreTaskManager
        result = EncreTaskManager.update_task("nonexistent", status="running")
        assert result is False

    def test_update_updates_timestamp(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="T", description="D", task_type="bash", prompt="echo"
        )
        original_updated_at = EncreTaskManager.get_task(task_id).updated_at
        time.sleep(0.01)
        EncreTaskManager.update_task(task_id, status="running")
        new_updated_at = EncreTaskManager.get_task(task_id).updated_at
        assert new_updated_at >= original_updated_at

    def test_list_tasks_returns_all(self):
        from encre.task.manager import EncreTaskManager
        EncreTaskManager.create_task(name="T1", description="D", task_type="bash", prompt="echo 1")
        EncreTaskManager.create_task(name="T2", description="D", task_type="bash", prompt="echo 2")
        EncreTaskManager.create_task(name="T3", description="D", task_type="bash", prompt="echo 3")
        tasks = EncreTaskManager.list_tasks()
        assert len(tasks) == 3

    def test_list_tasks_filter_by_status(self):
        from encre.task.manager import EncreTaskManager
        id1 = EncreTaskManager.create_task(name="T1", description="D", task_type="bash", prompt="echo 1")
        EncreTaskManager.create_task(name="T2", description="D", task_type="bash", prompt="echo 2")
        EncreTaskManager.update_task(id1, status="completed")
        pending = EncreTaskManager.list_tasks(status="pending")
        completed = EncreTaskManager.list_tasks(status="completed")
        assert len(pending) == 1
        assert len(completed) == 1

    def test_list_tasks_sorted_by_created_at_desc(self):
        from encre.task.manager import EncreTaskManager
        id1 = EncreTaskManager.create_task(name="First", description="D", task_type="bash", prompt="echo 1")
        time.sleep(0.01)
        id2 = EncreTaskManager.create_task(name="Second", description="D", task_type="bash", prompt="echo 2")
        tasks = EncreTaskManager.list_tasks()
        assert tasks[0].id == id2
        assert tasks[1].id == id1

    def test_delete_task_removes_it(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(name="T", description="D", task_type="bash", prompt="echo")
        assert EncreTaskManager.delete_task(task_id) is True
        assert EncreTaskManager.get_task(task_id) is None

    def test_delete_nonexistent_returns_false(self):
        from encre.task.manager import EncreTaskManager
        assert EncreTaskManager.delete_task("no-such-task") is False

    def test_create_task_with_parent(self):
        from encre.task.manager import EncreTaskManager
        parent_id = EncreTaskManager.create_task(
            name="Parent", description="P", task_type="agent", prompt="parent task"
        )
        child_id = EncreTaskManager.create_task(
            name="Child", description="C", task_type="bash", prompt="child task",
            parent_id=parent_id,
        )
        child = EncreTaskManager.get_task(child_id)
        assert child.parent_id == parent_id

    def test_create_task_with_metadata(self):
        from encre.task.manager import EncreTaskManager
        task_id = EncreTaskManager.create_task(
            name="T", description="D", task_type="bash", prompt="echo",
            metadata={"timeout": 30, "retries": 3},
        )
        task = EncreTaskManager.get_task(task_id)
        assert task.metadata == {"timeout": 30, "retries": 3}

    def test_clear_removes_all_tasks(self):
        from encre.task.manager import EncreTaskManager
        EncreTaskManager.create_task(name="T1", description="D", task_type="bash", prompt="echo 1")
        EncreTaskManager.create_task(name="T2", description="D", task_type="bash", prompt="echo 2")
        EncreTaskManager.clear()
        assert len(EncreTaskManager.list_tasks()) == 0


# ===========================================================================
# EncreTaskExecutor
# ===========================================================================

class TestEncreTaskExecutor:
    """Tests for EncreTaskExecutor."""

    @pytest.fixture(autouse=True)
    def _clear_before_test(self):
        from encre.task.manager import EncreTaskManager
        EncreTaskManager.clear()

    def test_construction(self):
        from encre.task.executor import EncreTaskExecutor
        executor = EncreTaskExecutor()
        assert executor is not None

    def test_execute_task_not_found(self):
        from encre.task.executor import EncreTaskExecutor

        async def _test():
            executor = EncreTaskExecutor()
            result = await executor.execute_task("nonexistent-id")
            assert "not found" in result.lower()

        import asyncio
        asyncio.run(_test())

    def test_execute_bash_task(self):
        from encre.task.manager import EncreTaskManager
        from encre.task.executor import EncreTaskExecutor

        async def _test():
            task_id = EncreTaskManager.create_task(
                name="Bash Test",
                description="Run echo",
                task_type="bash",
                prompt="echo hello world",
            )
            executor = EncreTaskExecutor()
            result = await executor.execute_task(task_id)
            assert "hello world" in result

            # Task status should be updated
            task = EncreTaskManager.get_task(task_id)
            assert task.status == "completed"
            assert task.result == result

        import asyncio
        asyncio.run(_test())

    def test_execute_bash_task_failure(self):
        from encre.task.manager import EncreTaskManager
        from encre.task.executor import EncreTaskExecutor

        async def _test():
            task_id = EncreTaskManager.create_task(
                name="Failing Bash",
                description="Run invalid command",
                task_type="bash",
                prompt="nonexistent_command_xyz 2>&1; exit 0",
            )
            executor = EncreTaskExecutor()
            result = await executor.execute_task(task_id)
            task = EncreTaskManager.get_task(task_id)
            # The task may complete or fail depending on shell behavior
            assert task.status in ("completed", "failed")

        import asyncio
        asyncio.run(_test())

    def test_execute_updates_status_to_running_then_completed(self):
        from encre.task.manager import EncreTaskManager
        from encre.task.executor import EncreTaskExecutor

        async def _test():
            task_id = EncreTaskManager.create_task(
                name="Status Test",
                description="Test status transitions",
                task_type="bash",
                prompt="echo done",
            )
            executor = EncreTaskExecutor()
            await executor.execute_task(task_id)
            task = EncreTaskManager.get_task(task_id)
            assert task.status == "completed"

        import asyncio
        asyncio.run(_test())

    def test_execute_unknown_task_type(self):
        from encre.task.manager import EncreTaskManager
        from encre.task.executor import EncreTaskExecutor

        async def _test():
            # Create a task with unsupported type directly
            from encre.task.types import EncreTask
            task_id = "fake-unknown-type-id"
            now = time.time()
            EncreTaskManager._tasks[task_id] = EncreTask(
                id=task_id,
                name="Unknown",
                description="Bad type",
                task_type="invalid_type",
                prompt="test",
                created_at=now,
                updated_at=now,
            )
            executor = EncreTaskExecutor()
            result = await executor.execute_task(task_id)
            assert "unknown" in result.lower()
            task = EncreTaskManager.get_task(task_id)
            # Unknown task types get status="completed" with error text in result
            assert task.status == "completed"
            assert "Unknown task type" in task.result

        import asyncio
        asyncio.run(_test())


# ===========================================================================
# Public API exports
# ===========================================================================

class TestTaskPublicAPI:
    """Verify the task module public exports."""

    def test_public_exports(self):
        from encre.task import EncreTask, EncreTaskManager, EncreTaskExecutor
        assert EncreTask is not None
        assert EncreTaskManager is not None
        assert EncreTaskExecutor is not None

    def test_task_type_literals(self):
        from encre.utils.types import TaskType, TaskStatus
        # Verify these are importable
        assert TaskType is not None
        assert TaskStatus is not None
