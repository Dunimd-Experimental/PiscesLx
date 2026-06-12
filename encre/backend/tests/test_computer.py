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

"""Tests for encre.computer.browser — EncreBrowserSession and BrowserState."""

import time

import pytest


# ===========================================================================
# BrowserState dataclass
# ===========================================================================

class TestBrowserState:
    """Tests for the BrowserState dataclass."""

    def test_default_creation(self):
        from encre.computer.browser import BrowserState
        state = BrowserState()
        assert state.url == ""
        assert state.title == ""
        assert state.html == ""
        assert state.text == ""

    def test_creation_with_values(self):
        from encre.computer.browser import BrowserState
        state = BrowserState(
            url="https://example.com",
            title="Example Domain",
            html="<html><body>Example</body></html>",
            text="Example",
        )
        assert state.url == "https://example.com"
        assert state.title == "Example Domain"
        assert state.html == "<html><body>Example</body></html>"
        assert state.text == "Example"

    def test_is_dataclass(self):
        from encre.computer.browser import BrowserState
        from dataclasses import is_dataclass
        assert is_dataclass(BrowserState)

    def test_all_fields_have_defaults(self):
        from encre.computer.browser import BrowserState
        state = BrowserState()
        for field_name in ["url", "title", "html", "text"]:
            assert getattr(state, field_name) == ""


# ===========================================================================
# EncreBrowserSession construction
# ===========================================================================

class TestEncreBrowserSessionConstruction:
    """Tests for EncreBrowserSession construction."""

    def test_default_construction(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        assert session is not None
        assert session.headless is True
        assert session.viewport_width == 1280
        assert session.viewport_height == 800
        assert session.timeout == 30000

    def test_custom_construction(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession(
            headless=False,
            viewport_width=1920,
            viewport_height=1080,
            timeout=60000,
        )
        assert session.headless is False
        assert session.viewport_width == 1920
        assert session.viewport_height == 1080
        assert session.timeout == 60000

    def test_initial_internal_state(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        assert session._pw is None
        assert session._browser is None
        assert session._context is None
        assert session._page is None

    def test_initial_browser_state_empty(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        assert session._state.url == ""
        assert session._state.title == ""
        assert session._state.html == ""
        assert session._state.text == ""

    def test_last_used_timestamp_set(self):
        from encre.computer.browser import EncreBrowserSession
        before = time.time()
        session = EncreBrowserSession()
        after = time.time()
        assert before <= session._last_used <= after


# ===========================================================================
# EncreBrowserSession state methods (no browser needed)
# ===========================================================================

class TestEncreBrowserSessionState:
    """Tests for state methods that don't require Playwright."""

    def test_get_state_before_navigate(self):
        from encre.computer.browser import EncreBrowserSession

        async def _test():
            session = EncreBrowserSession()
            state = await session.get_state()
            assert isinstance(state, object)
            from encre.computer.browser import BrowserState
            assert isinstance(state, BrowserState)

        import asyncio
        asyncio.run(_test())

    def test_is_idle_fresh_session(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        # A fresh session is not idle (last_used is now)
        assert session.is_idle(max_idle_seconds=600) is False

    def test_is_idle_with_custom_threshold(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        # With a zero-second threshold, it should be idle immediately
        assert session.is_idle(max_idle_seconds=0) is True

    def test_save_cookies_before_browser(self):
        from encre.computer.browser import EncreBrowserSession

        async def _test():
            session = EncreBrowserSession()
            cookies = await session.save_cookies()
            assert cookies == []

        import asyncio
        asyncio.run(_test())

    def test_close_before_browser(self):
        from encre.computer.browser import EncreBrowserSession

        async def _test():
            session = EncreBrowserSession()
            await session.close()
            assert session._browser is None
            assert session._pw is None
            assert session._page is None

        import asyncio
        asyncio.run(_test())

    def test_close_is_idempotent(self):
        from encre.computer.browser import EncreBrowserSession

        async def _test():
            session = EncreBrowserSession()
            await session.close()
            await session.close()
            # Should not raise

        import asyncio
        asyncio.run(_test())


# ===========================================================================
# EncreBrowserSession check_playwright
# ===========================================================================

class TestEncreBrowserSessionCheckPlaywright:
    """Tests for _check_playwright helper."""

    def test_check_playwright_returns_bool(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        result = session._check_playwright()
        assert isinstance(result, bool)


# ===========================================================================
# EncreBrowserSession public API exports
# ===========================================================================

class TestBrowserPublicAPI:
    """Verify the public API matches expectations."""

    def test_public_exports(self):
        from encre.computer import EncreBrowserSession, BrowserState
        assert EncreBrowserSession is not None
        assert BrowserState is not None

    def test_browser_methods_exist(self):
        from encre.computer.browser import EncreBrowserSession
        session = EncreBrowserSession()
        # All expected async methods should exist
        assert hasattr(session, "navigate")
        assert hasattr(session, "click")
        assert hasattr(session, "type_text")
        assert hasattr(session, "screenshot")
        assert hasattr(session, "get_html")
        assert hasattr(session, "get_text")
        assert hasattr(session, "execute_js")
        assert hasattr(session, "get_state")
        assert hasattr(session, "wait_for_selector")
        assert hasattr(session, "scroll_to")
        assert hasattr(session, "fill_form")
        assert hasattr(session, "press_key")
        assert hasattr(session, "save_cookies")
        assert hasattr(session, "load_cookies")
        assert hasattr(session, "close")
        assert hasattr(session, "is_idle")
