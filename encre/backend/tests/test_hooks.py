#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for hooks system and event types."""

import asyncio

import pytest

from encre.hooks.system import EncreHookSystem
from encre.hooks.types import (
    HookStartedEvent,
    HookProgressEvent,
    HookResponseEvent,
)


class TestHookEventTypes:
    def test_started_event(self):
        event = HookStartedEvent(
            hook_id="h1", hook_name="test_hook", event_type="pre_tool_exec"
        )
        assert event.hook_id == "h1"
        assert event.hook_name == "test_hook"
        assert event.event_type == "pre_tool_exec"

    def test_progress_event(self):
        event = HookProgressEvent(
            hook_id="h1",
            hook_name="test_hook",
            event_type="on_tool_progress",
            output="running",
            stdout="out",
            stderr="",
        )
        assert event.hook_id == "h1"
        assert event.output == "running"
        assert event.stdout == "out"

    def test_response_event(self):
        event = HookResponseEvent(
            hook_id="h1",
            hook_name="test_hook",
            event_type="post_tool_exec",
            output="success",
            exit_code=0,
            outcome="success",
        )
        assert event.hook_id == "h1"
        assert event.output == "success"
        assert event.outcome == "success"
        assert event.exit_code == 0

    def test_response_event_error(self):
        event = HookResponseEvent(
            hook_id="h1",
            hook_name="test_hook",
            event_type="post_tool_exec",
            output="something went wrong",
            exit_code=1,
            outcome="error",
        )
        assert event.exit_code == 1
        assert event.outcome == "error"


class TestHookSystem:
    def test_create(self):
        hooks = EncreHookSystem()
        assert hooks is not None
        assert hooks._handlers is not None
        assert hooks.enabled is True

    def test_register_handler(self):
        hooks = EncreHookSystem()

        async def handler(name, context, extra):
            return {"block": False}

        hid = hooks.register_handler("pre_tool_exec", handler, "test_handler")
        assert hid == "test_handler"
        assert len(hooks._handlers["pre_tool_exec"]) == 1

    def test_register_handler_auto_id(self):
        hooks = EncreHookSystem()

        async def handler(name, context, extra):
            return {}

        hid = hooks.register_handler("pre_tool_exec", handler)
        assert isinstance(hid, str)
        assert len(hid) > 0

    def test_unregister_handler(self):
        hooks = EncreHookSystem()

        async def handler(name, context, extra):
            return {}

        hid = hooks.register_handler("pre_tool_exec", handler, "test_handler")
        result = hooks.unregister_handler(hid)
        assert result is True
        assert len(hooks._handlers["pre_tool_exec"]) == 0

    def test_unregister_nonexistent(self):
        hooks = EncreHookSystem()
        assert hooks.unregister_handler("nonexistent_id") is False

    def test_emit_pre_tool(self):
        async def _test():
            hooks = EncreHookSystem()
            called = False

            async def handler(name, context, extra):
                nonlocal called
                called = True
                return {"block": False}

            hooks.register_handler("pre_tool_exec", handler, "test")
            result = await hooks.emit_pre_tool("bash", {"cmd": "ls"})
            assert called is True

        asyncio.run(_test())

    def test_emit_pre_tool_block(self):
        async def _test():
            hooks = EncreHookSystem()

            async def handler(name, context, extra):
                return {"block": True, "block_reason": "unsafe"}

            hooks.register_handler("pre_tool_exec", handler, "test")
            result = await hooks.emit_pre_tool("bash", {"cmd": "rm -rf /"})
            assert result is not None
            assert result.get("block") is True

        asyncio.run(_test())

    def test_emit_post_tool(self):
        async def _test():
            hooks = EncreHookSystem()

            async def handler(name, context, extra):
                return {"extra_context": "injected context"}

            hooks.register_handler("post_tool_exec", handler, "test")
            result = await hooks.emit_post_tool("bash", {"cmd": "ls"}, "file1.txt")
            assert isinstance(result, str)
            assert "injected context" in result

        asyncio.run(_test())

    def test_emit_session_start(self):
        async def _test():
            hooks = EncreHookSystem()
            called = False

            async def handler(name, context, extra):
                nonlocal called
                called = True
                return {}

            hooks.register_handler("on_session_start", handler, "test")
            await hooks.emit_session_start()
            assert called is True

        asyncio.run(_test())

    def test_emit_turn_start(self):
        async def _test():
            hooks = EncreHookSystem()
            called = False

            async def handler(name, context, extra):
                nonlocal called
                called = True
                return {}

            hooks.register_handler("on_turn_start", handler, "test")
            await hooks.emit_turn_start(1)
            assert called is True

        asyncio.run(_test())

    def test_emit_error(self):
        async def _test():
            hooks = EncreHookSystem()
            called = False

            async def handler(name, context, extra):
                nonlocal called
                called = True
                return {}

            hooks.register_handler("on_error", handler, "test")
            await hooks.emit_error(ValueError("test error"), "testing")
            assert called is True

        asyncio.run(_test())

    def test_on_event_observer(self):
        hooks = EncreHookSystem()
        events = []

        def observer(event):
            events.append(event)

        hooks.on_event(observer)

        async def _test():
            async def handler(name, context, extra):
                return {}

            hooks.register_handler("pre_tool_exec", handler, "test")
            await hooks.emit_pre_tool("bash", {"cmd": "ls"})
            # Should have received at least started + response events
            assert len(events) >= 2

        asyncio.run(_test())

    def test_register_invalid_event_type(self):
        hooks = EncreHookSystem()

        async def handler(name, context, extra):
            return {}

        with pytest.raises(ValueError):
            hooks.register_handler("invalid_event_type", handler)

    def test_disabled_hooks(self):
        async def _test():
            hooks = EncreHookSystem()
            hooks.enabled = False

            async def handler(name, context, extra):
                return {"block": True}

            hooks.register_handler("pre_tool_exec", handler, "test")
            result = await hooks.emit_pre_tool("bash", {"cmd": "ls"})
            assert result is None

        asyncio.run(_test())
