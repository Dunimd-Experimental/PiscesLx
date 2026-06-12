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

"""Tests for swarm subsystem: planner, consensus, blackboard, orchestrator, roles,
teammate, mailbox, swarm session."""

import asyncio

import pytest

from encre.swarm.mailbox import EncreMailbox, MailboxMessage
from encre.swarm.teammate import EncreTeammate, TeammateHandle
from encre.swarm.manager import EncreSwarmManager, SwarmProgress
from encre.swarm.planner import EncreTaskPlanner, TaskTree, TaskNode, _detect_pattern
from encre.swarm.consensus import EncreConsensus, Proposal, Vote, ConsensusResult
from encre.swarm.blackboard import EncreBlackboard, BlackboardEntry
from encre.swarm.orchestrator import EncreOrchestrator, OrchestrationEvent
from encre.swarm.roles import (
    AgentRole, RoleRegistry,
    ROLE_ARCHITECT, ROLE_CODER, ROLE_REVIEWER, ROLE_TESTER,
    ROLE_RESEARCHER, ROLE_DEBUGGER, ROLE_GENERAL,
)


# ===========================================================================
# Mailbox
# ===========================================================================

class TestMailbox:
    def test_create(self):
        mb = EncreMailbox(owner_id="agent1")
        assert mb.owner_id == "agent1"

    async def test_send_receive(self):
        mb_a = EncreMailbox(owner_id="a")
        mb_b = EncreMailbox(owner_id="b")
        await mb_a.send(mb_b, "hello")
        msg = await mb_b.receive(timeout=1.0)
        assert msg is not None
        assert msg.content == "hello"
        assert msg.sender == "a"

    async def test_receive_timeout(self):
        mb = EncreMailbox(owner_id="test", timeout=0.1)
        msg = await mb.receive(timeout=0.01)
        assert msg is None

    async def test_multiple_messages_fifo(self):
        mb_a = EncreMailbox(owner_id="a")
        mb_b = EncreMailbox(owner_id="b")
        await mb_a.send(mb_b, "first")
        await mb_a.send(mb_b, "second")
        msg1 = await mb_b.receive(timeout=1.0)
        msg2 = await mb_b.receive(timeout=1.0)
        assert msg1.content == "first"
        assert msg2.content == "second"

    def test_mailbox_message(self):
        msg = MailboxMessage(sender="a", content="test")
        assert msg.sender == "a"
        assert msg.content == "test"
        assert msg.metadata == {}

    def test_peek(self):
        async def _test():
            mb_a = EncreMailbox(owner_id="a")
            mb_b = EncreMailbox(owner_id="b")
            await mb_a.send(mb_b, "msg1")
            peeked = mb_b.peek()
            assert len(peeked) == 1
            # Message still available after peek
            msg = await mb_b.receive(timeout=1.0)
            assert msg is not None
        asyncio.run(_test())

    def test_clear(self):
        async def _test():
            mb_a = EncreMailbox(owner_id="a")
            mb_b = EncreMailbox(owner_id="b")
            await mb_a.send(mb_b, "msg1")
            mb_b.clear()
            msg = await mb_b.receive(timeout=0.1)
            assert msg is None
        asyncio.run(_test())


# ===========================================================================
# Teammate
# ===========================================================================

class TestTeammate:
    def test_create_teammate(self):
        tm = EncreTeammate(name="coder", task="write a function")
        assert tm.name == "coder"
        assert tm.task == "write a function"
        assert tm.mailbox is not None

    def test_teammate_handle(self):
        handle = TeammateHandle(teammate_id="tm1", name="reviewer", status="pending")
        assert handle.name == "reviewer"
        assert handle.status == "pending"


# ===========================================================================
# RoleRegistry & Roles
# ===========================================================================

class TestRoles:
    def test_role_constants(self):
        assert ROLE_ARCHITECT.name == "architect"
        assert ROLE_CODER.name == "coder"
        assert ROLE_REVIEWER.name == "reviewer"
        assert ROLE_TESTER.name == "tester"
        assert ROLE_RESEARCHER.name == "researcher"
        assert ROLE_DEBUGGER.name == "debugger"
        assert ROLE_GENERAL.name == "general"

    def test_agent_role_creation(self):
        role = AgentRole(name="custom", description="Custom role", allowed_tools=["bash"])
        assert role.name == "custom"
        assert "bash" in role.allowed_tools

    def test_role_registry_register(self):
        registry = RoleRegistry()
        custom = AgentRole(name="custom_role", description="Custom")
        registry.register(custom)
        assert registry.get("custom_role").name == "custom_role"

    def test_role_registry_get_defaults_to_general(self):
        registry = RoleRegistry()
        role = registry.get("nonexistent")
        assert role.name == "general"

    def test_role_registry_list_roles(self):
        registry = RoleRegistry()
        roles = registry.list_roles()
        assert "architect" in roles
        assert "coder" in roles
        assert "general" in roles

    def test_role_registry_get_for_task(self):
        registry = RoleRegistry()
        assert registry.get_for_task("design the system").name == "architect"
        assert registry.get_for_task("implement the feature").name == "coder"
        assert registry.get_for_task("audit the system for security issues").name == "reviewer"
        assert registry.get_for_task("test the application").name == "tester"
        assert registry.get_for_task("research best practices").name == "researcher"
        assert registry.get_for_task("debug the null pointer").name == "debugger"
        assert registry.get_for_task("something else").name == "general"

    def test_role_to_dict(self):
        d = ROLE_CODER.to_dict()
        assert d["name"] == "coder"
        assert "description" in d


# ===========================================================================
# TaskPlanner
# ===========================================================================

class TestTaskPlanner:
    def setup_method(self):
        self.planner = EncreTaskPlanner()

    def test_detect_pattern_build(self):
        assert _detect_pattern("build a web app") == "build"
        assert _detect_pattern("create an API") == "build"
        assert _detect_pattern("implement a cache layer") == "build"
        assert _detect_pattern("write a CLI tool") == "build"
        assert _detect_pattern("develop a mobile app") == "build"

    def test_detect_pattern_debug(self):
        assert _detect_pattern("debug the login flow") == "debug"
        assert _detect_pattern("fix a bug in auth") == "debug"

    def test_detect_pattern_research(self):
        assert _detect_pattern("research async patterns") == "research"
        assert _detect_pattern("investigate memory leak") == "research"

    def test_detect_pattern_refactor(self):
        assert _detect_pattern("refactor the database layer") == "refactor"
        assert _detect_pattern("clean up the utils module") == "refactor"

    def test_detect_pattern_none(self):
        assert _detect_pattern("hello world") is None

    def test_plan_build_pattern(self):
        tree = self.planner.plan("build a REST API")
        assert isinstance(tree, TaskTree)
        assert len(tree.nodes) == 5
        assert len(tree.entry_nodes) > 0
        assert len(tree.exit_nodes) > 0

    def test_plan_debug_pattern(self):
        tree = self.planner.plan("fix the authentication bug")
        assert len(tree.nodes) == 4

    def test_plan_research_pattern(self):
        tree = self.planner.plan("investigate database performance")
        assert len(tree.nodes) == 4

    def test_plan_refactor_pattern(self):
        tree = self.planner.plan("refactor the user service")
        assert len(tree.nodes) == 5

    def test_plan_unknown_falls_back_to_simple(self):
        tree = self.planner.plan("do something unusual and uncategorized")
        assert len(tree.nodes) == 1

    def test_task_tree_get_ready_nodes(self):
        tree = self.planner.plan("build a CLI")
        ready = tree.get_ready_nodes()
        assert len(ready) > 0
        for node in ready:
            assert node.status == "pending"
            assert node.dependencies == []

    def test_task_tree_all_done(self):
        tree = self.planner.plan("fix a bug")
        for node in tree.nodes.values():
            node.status = "completed"
        assert tree.all_done() is True

    def test_task_tree_has_failure(self):
        tree = self.planner.plan("fix a bug")
        first = list(tree.nodes.values())[0]
        first.status = "failed"
        assert tree.has_failure() is True

    def test_plan_with_llm_returns_prompt(self):
        prompt = self.planner.plan_with_llm("build a chat app", "using FastAPI")
        assert "build a chat app" in prompt
        assert "FastAPI" in prompt

    def test_plan_from_json(self):
        import json
        data = {
            "tasks": [
                {"id": "t1", "name": "Design", "description": "architect", "role": "architect", "dependencies": [], "priority": 10},
                {"id": "t2", "name": "Code", "description": "implement", "role": "coder", "dependencies": ["t1"], "priority": 5},
            ],
            "entry_tasks": ["t1"],
            "exit_tasks": ["t2"],
        }
        tree = EncreTaskPlanner.plan_from_json("test goal", json.dumps(data))
        assert len(tree.nodes) == 2
        assert tree.entry_nodes == ["t1"]
        assert tree.exit_nodes == ["t2"]

    def test_decompose_async(self):
        async def _test():
            tree = await self.planner.decompose("build a web app")
            assert isinstance(tree, TaskTree)
            assert len(tree.nodes) > 0
        asyncio.run(_test())


# ===========================================================================
# Consensus
# ===========================================================================

class TestConsensus:
    def setup_method(self):
        self.consensus = EncreConsensus()

    def test_create_proposal(self):
        p = self.consensus.create_proposal(
            title="Use FastAPI",
            description="Should we use FastAPI for the backend?",
            options=["yes", "no"],
            proposed_by="architect",
        )
        assert p.title == "Use FastAPI"
        assert len(p.options) == 2

    def test_cast_vote(self):
        p = self.consensus.create_proposal("Test", "desc", ["A", "B"])
        v = self.consensus.cast_vote(proposal_id=p.id, voter_id="coder1", choice="A", reasoning="Best option")
        assert v.choice == "A"

    def test_tally_unanimous(self):
        p = self.consensus.create_proposal("Test", "desc", ["A", "B"])
        self.consensus.cast_vote(p.id, "v1", "A")
        self.consensus.cast_vote(p.id, "v2", "A")
        self.consensus.cast_vote(p.id, "v3", "A")
        result = self.consensus.tally(p)
        assert result.winner == "A"
        assert result.is_consensus is True
        assert result.vote_counts["A"] == 3

    def test_tally_no_consensus(self):
        p = self.consensus.create_proposal("Test", "desc", ["A", "B"])
        self.consensus.cast_vote(p.id, "v1", "A")
        self.consensus.cast_vote(p.id, "v2", "B")
        result = self.consensus.tally(p)
        assert result.is_consensus is False

    def test_tally_empty(self):
        p = self.consensus.create_proposal("Test", "desc", ["A", "B"])
        result = self.consensus.tally(p)
        assert result.winner == "A"
        assert result.vote_counts["A"] == 0

    def test_proposal_to_dict(self):
        p = self.consensus.create_proposal("T", "D", ["X"], proposed_by="me")
        d = p.to_dict()
        assert d["title"] == "T"
        assert d["proposed_by"] == "me"


# ===========================================================================
# Blackboard
# ===========================================================================

class TestBlackboard:
    def setup_method(self):
        self.bb = EncreBlackboard()

    def test_put_get(self):
        self.bb.put("default", "key1", "value1", owner="agent1")
        result = self.bb.get("default", "key1")
        assert result is not None
        assert result[0] == "value1"

    def test_get_nonexistent(self):
        assert self.bb.get("default", "nonexistent") is None

    def test_get_all(self):
        self.bb.put("ns1", "k1", "v1", owner="a")
        self.bb.put("ns1", "k2", "v2", owner="a")
        all_data = self.bb.get_all("ns1")
        assert all_data["k1"] == "v1"
        assert all_data["k2"] == "v2"

    def test_get_all_visible(self):
        self.bb.put("public_ns", "key", "value")
        visible = self.bb.get_all_visible()
        assert "public_ns/key" in visible
        assert "value" in visible

    def test_delete(self):
        self.bb.put("default", "k1", "v1")
        assert self.bb.delete("default", "k1") is True
        assert self.bb.get("default", "k1") is None

    def test_delete_nonexistent(self):
        assert self.bb.delete("default", "nonexistent") is False

    def test_overwrite(self):
        self.bb.put("default", "k1", "v1")
        self.bb.put("default", "k1", "v2")
        result = self.bb.get("default", "k1")
        assert result[0] == "v2"

    def test_version_increment(self):
        v1 = self.bb.put("default", "k1", "v1")
        v2 = self.bb.put("default", "k1", "v2")
        assert v2 > v1

    def test_compare_and_swap(self):
        v = self.bb.put("default", "k1", "v1")
        assert self.bb.compare_and_swap("default", "k1", v, "v2") is True
        result = self.bb.get("default", "k1")
        assert result[0] == "v2"

    def test_compare_and_swap_wrong_version(self):
        self.bb.put("default", "k1", "v1")
        assert self.bb.compare_and_swap("default", "k1", 999, "v2") is False

    def test_blackboard_entry(self):
        entry = BlackboardEntry(key="test", value=42, version=1, namespace="ns1", owner="agent1")
        assert entry.key == "test"
        assert entry.value == 42
        assert entry.version == 1

    def test_reset(self):
        self.bb.put("default", "k1", "v1")
        self.bb.reset()
        assert self.bb.get("default", "k1") is None


# ===========================================================================
# Orchestrator
# ===========================================================================

class TestOrchestrator:
    def test_create(self):
        blackboard = EncreBlackboard()
        roles = RoleRegistry()
        roles.register(ROLE_GENERAL)
        orch = EncreOrchestrator(
            role_registry=roles,
            blackboard=blackboard,
            max_concurrent=3,
        )
        assert orch is not None

    def test_orchestration_event(self):
        event = OrchestrationEvent(type="task_completed", task_id="t1", task_name="Test", role="general")
        assert event.type == "task_completed"
        assert event.task_id == "t1"
