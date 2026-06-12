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

"""Tests for codebase indexer, LSP protocol, git, notebook, server types."""

import pytest


# ===========================================================================
# Codebase Indexer types
# ===========================================================================

class TestCodeIndex:
    def test_module_info(self):
        from encre.codebase.indexer import ModuleInfo
        mi = ModuleInfo(
            path="src/my_module.py",
            name="my_module",
            language="python",
            exports=["func_a", "ClassB"],
            imports=["os", "json"],
        )
        assert mi.name == "my_module"
        assert mi.language == "python"
        assert "func_a" in mi.exports
        assert "os" in mi.imports

    def test_module_info_defaults(self):
        from encre.codebase.indexer import ModuleInfo
        mi = ModuleInfo(path="test.py", name="test")
        assert mi.imports == []
        assert mi.imported_by == []
        assert mi.exports == []
        assert mi.language == ""
        assert mi.loc == 0

    def test_code_index_create(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        assert ci is not None


# ===========================================================================
# LSP types
# ===========================================================================

class TestLSPProtocol:
    def test_position(self):
        from encre.lsp.protocol import Position
        p = Position(line=10, character=5)
        assert p.line == 10
        assert p.character == 5

    def test_range(self):
        from encre.lsp.protocol import Position, Range
        start = Position(line=0, character=0)
        end = Position(line=10, character=20)
        r = Range(start=start, end=end)
        assert r.start.line == 0
        assert r.end.line == 10

    def test_location(self):
        from encre.lsp.protocol import Position, Range, Location
        r = Range(start=Position(line=1, character=0), end=Position(line=1, character=10))
        loc = Location(uri="file:///test.py", range=r)
        assert loc.uri == "file:///test.py"
        assert loc.range.start.line == 1

    def test_diagnostic(self):
        from encre.lsp.protocol import Position, Range, Diagnostic
        r = Range(start=Position(line=5, character=0), end=Position(line=5, character=10))
        diag = Diagnostic(
            range=r,
            message="Unused variable",
            severity=2,
            source="pyright",
        )
        assert diag.message == "Unused variable"
        assert diag.severity == 2
        assert diag.source == "pyright"

    def test_hover_result(self):
        from encre.lsp.protocol import HoverResult
        hr = HoverResult(contents="def foo(x: int) -> str", range=None)
        assert hr.contents == "def foo(x: int) -> str"
        assert hr.range is None

    def test_hover_result_with_range(self):
        from encre.lsp.protocol import HoverResult, Position, Range
        r = Range(start=Position(line=1, character=0), end=Position(line=1, character=10))
        hr = HoverResult(contents="def foo()", range=r)
        assert hr.range is not None

    def test_lsp_state(self):
        from encre.lsp.protocol import LSPState
        state = LSPState(status="running")
        assert state.status == "running"
        assert state.error is None

    def test_lsp_state_with_error(self):
        from encre.lsp.protocol import LSPState
        state = LSPState(status="stopped", error="connection refused")
        assert state.status == "stopped"
        assert state.error == "connection refused"


# ===========================================================================
# Git types
# ===========================================================================

class TestGitTypes:
    def test_git_state_default(self):
        from encre.git.repo import GitState
        gs = GitState(in_repo=False)
        assert gs.in_repo is False
        assert gs.is_clean is True
        assert gs.changed_files == []
        assert gs.untracked_files == []

    def test_git_state_in_repo(self):
        from encre.git.repo import GitState
        gs = GitState(
            in_repo=True,
            commit_hash="abc123",
            branch="main",
            remote_url="https://github.com/example/repo",
            is_clean=True,
            changed_files=[],
            untracked_files=[],
            has_unpushed=False,
            worktree_count=1,
        )
        assert gs.in_repo is True
        assert gs.branch == "main"
        assert gs.commit_hash == "abc123"
        assert gs.worktree_count == 1

    def test_git_diff_result(self):
        from encre.git.diff import GitDiffResult
        gdr = GitDiffResult(files=3, insertions=50, deletions=10)
        assert gdr.files == 3
        assert gdr.insertions == 50
        assert gdr.deletions == 10

    def test_git_repo_creation(self):
        from encre.git.repo import EncreGitRepo
        repo = EncreGitRepo(workspace=".")
        assert repo is not None

    def test_git_repo_is_in_repo(self):
        from encre.git.repo import EncreGitRepo
        repo = EncreGitRepo(workspace=".")
        assert isinstance(repo.is_in_repo(), bool)


# ===========================================================================
# Notebook types
# ===========================================================================

class TestNotebook:
    def test_session_create(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        assert sess is not None
        assert sess.kernel_name == "python3"

    def test_session_create_custom_kernel(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession(kernel_name="python3.12")
        assert sess.kernel_name == "python3.12"


# ===========================================================================
# Server protocol types
# ===========================================================================

class TestServerProtocol:
    def test_client_run(self):
        from encre.server.protocol import ClientRun
        msg = ClientRun(prompt="Hello", session_id="s1")
        assert msg.type == "run"
        assert msg.prompt == "Hello"
        assert msg.session_id == "s1"

    def test_client_run_from_dict(self):
        from encre.server.protocol import ClientRun
        msg = ClientRun.from_dict({"prompt": "Hello", "session_id": "s1"})
        assert msg.type == "run"
        assert msg.prompt == "Hello"

    def test_parse_client_message(self):
        from encre.server.protocol import parse_client_message
        import json
        raw = json.dumps({"type": "run", "prompt": "Hello", "session_id": "s1"})
        msg = parse_client_message(raw)
        assert msg is not None

    def test_parse_client_message_invalid(self):
        from encre.server.protocol import parse_client_message
        msg = parse_client_message("not json")
        assert msg is None

    def test_parse_client_message_ping(self):
        from encre.server.protocol import parse_client_message
        import json
        raw = json.dumps({"type": "ping"})
        msg = parse_client_message(raw)
        assert msg is not None

    def test_encode_server_message(self):
        from encre.server.protocol import encode_server_message
        encoded = encode_server_message("text_delta", text="Hello!")
        assert isinstance(encoded, str)
        assert "Hello!" in encoded


# ===========================================================================
# Server session manager
# ===========================================================================

class TestSessionManager:
    def test_session_info(self):
        from encre.server.session_manager import SessionInfo
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        agent = EncreAgent(config=EncreConfig(backend_type="openai", api_key="sk-fake"))
        si = SessionInfo(session_id="s1", agent=agent)
        assert si.session_id == "s1"
        assert si.is_running is False

    def test_session_manager_create(self):
        from encre.server.session_manager import SessionManager
        sm = SessionManager()
        assert sm is not None
        assert sm.active_count == 0

    def test_session_manager_create_session(self):
        from encre.server.session_manager import SessionManager
        from encre.config import EncreConfig
        sm = SessionManager()
        info = sm.create_session(EncreConfig(backend_type="openai", api_key="sk-fake"))
        assert info.session_id is not None
        assert sm.active_count == 1

    def test_session_manager_get_session(self):
        from encre.server.session_manager import SessionManager
        from encre.config import EncreConfig
        sm = SessionManager()
        info = sm.create_session(EncreConfig(backend_type="openai", api_key="sk-fake"))
        retrieved = sm.get_session(info.session_id)
        assert retrieved is not None
        assert retrieved.session_id == info.session_id

    def test_session_manager_list_sessions(self):
        from encre.server.session_manager import SessionManager
        from encre.config import EncreConfig
        sm = SessionManager()
        sm.create_session(EncreConfig(backend_type="openai", api_key="sk-fake"))
        sessions = sm.list_sessions()
        assert len(sessions) == 1

    def test_session_manager_remove(self):
        from encre.server.session_manager import SessionManager
        from encre.config import EncreConfig
        sm = SessionManager()
        info = sm.create_session(EncreConfig(backend_type="openai", api_key="sk-fake"))
        sm.remove_session(info.session_id)
        assert sm.active_count == 0
        assert sm.get_session(info.session_id) is None


# ===========================================================================
# Agent / Loop / Goal types
# ===========================================================================

class TestAgentTypes:
    def test_goal_definition(self):
        from encre.goal import GoalDefinition
        gd = GoalDefinition(description="Test feature", success_criteria="All tests pass", max_attempts=5)
        assert gd.description == "Test feature"
        assert gd.success_criteria == "All tests pass"
        assert gd.max_attempts == 5

    def test_goal_result(self):
        from encre.goal import GoalResult, GoalStatus
        gr = GoalResult(status=GoalStatus.SUCCESS, summary="Done", attempts=3)
        assert gr.status == GoalStatus.SUCCESS
        assert gr.attempts == 3

    def test_goal_status(self):
        from encre.goal import GoalStatus
        assert GoalStatus.PENDING is not None
        assert GoalStatus.IN_PROGRESS is not None
        assert GoalStatus.SUCCESS is not None
        assert GoalStatus.FAILED is not None
        assert GoalStatus.TIMEOUT is not None
        assert GoalStatus.MAX_ATTEMPTS is not None

    def test_goal_event(self):
        from encre.goal import GoalEvent, GoalStatus
        ge = GoalEvent(status=GoalStatus.IN_PROGRESS, attempt=1, message="Working...")
        assert ge.status == GoalStatus.IN_PROGRESS
        assert ge.attempt == 1
        assert ge.message == "Working..."

    def test_session_checkpoint(self):
        from encre.session import SessionCheckpoint
        sc = SessionCheckpoint(
            checkpoint_id="ckpt1",
            label="After turn 5",
            turn_count=5,
            tool_call_count=10,
        )
        assert sc.checkpoint_id == "ckpt1"
        assert sc.turn_count == 5
        assert sc.tool_call_count == 10

    def test_goal_loop_create(self):
        from encre.goal import EncreGoalLoop
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        agent = EncreAgent(config=EncreConfig(backend_type="openai", api_key="sk-fake"))
        loop = EncreGoalLoop(agent=agent, description="test", success_criteria="works")
        assert loop is not None
        assert loop._description == "test"

    def test_goal_runner_create(self):
        from encre.goal import EncreGoalRunner
        from encre.config import EncreConfig
        from encre.tools.registry import ToolRegistry
        from encre.hooks.system import EncreHookSystem
        from encre.safety import EncreSafetyEngine
        config = EncreConfig(backend_type="openai", api_key="sk-fake")
        registry = ToolRegistry()
        hooks = EncreHookSystem()
        safety = EncreSafetyEngine(config=config)
        runner = EncreGoalRunner(
            config=config,
            tool_registry=registry,
            hook_system=hooks,
            safety=safety,
        )
        assert runner is not None
