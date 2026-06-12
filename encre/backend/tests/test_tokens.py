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

"""Tests for token estimation utilities: estimate_tokens, count_message_tokens."""

import pytest

from encre.utils.tokens import (
    estimate_tokens,
    estimate_tokens_simple,
    count_message_tokens,
    is_tiktoken_available,
)


class TestEstimateTokens:
    """Verify estimate_tokens() returns sensible integer counts."""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        count = estimate_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_long_string(self):
        count = estimate_tokens("hello world " * 100)
        assert isinstance(count, int)
        assert count > 50

    def test_result_is_int(self):
        count = estimate_tokens("any text")
        assert isinstance(count, int)

    def test_non_negative(self):
        count = estimate_tokens("test")
        assert count >= 0

    def test_grows_with_length(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("hi " * 200)
        assert long > short

    def test_model_kwarg_accepted(self):
        count = estimate_tokens("hello", model="gpt-4o")
        assert isinstance(count, int)
        assert count > 0

    def test_different_model_kwarg(self):
        count = estimate_tokens("hello", model="gpt-4")
        assert isinstance(count, int)
        assert count > 0

    def test_unicode_text(self):
        count = estimate_tokens("你好世界")
        assert isinstance(count, int)
        assert count > 0

    def test_code_snippet(self):
        code = "def foo():\n    return 42\n"
        count = estimate_tokens(code)
        assert count > 0

    def test_special_characters(self):
        text = "!@#$%^&*()_+{}|:\"<>?[];',./"
        count = estimate_tokens(text)
        assert isinstance(count, int)

    def test_pure_whitespace(self):
        count = estimate_tokens("     ")
        assert isinstance(count, int)


class TestEstimateTokensSimple:
    """Verify estimate_tokens_simple() compatibility wrapper."""

    def test_returns_int(self):
        count = estimate_tokens_simple("hello")
        assert isinstance(count, int)

    def test_empty_string(self):
        count = estimate_tokens_simple("")
        assert count == 0

    def test_consistency_with_main_function(self):
        c1 = estimate_tokens_simple("hello world")
        c2 = estimate_tokens("hello world")
        assert c1 == c2


class TestCountMessageTokens:
    """Verify count_message_tokens() for message dicts."""

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        count = count_message_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_empty_messages_list(self):
        count = count_message_tokens([])
        assert count == 0

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
        ]
        count = count_message_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_message_with_empty_content(self):
        msgs = [{"role": "user", "content": ""}]
        count = count_message_tokens(msgs)
        assert isinstance(count, int)
        # Should have at least the per-message overhead (4 tokens)
        assert count >= 4

    def test_message_with_missing_content_key(self):
        msgs = [{"role": "user"}]
        count = count_message_tokens(msgs)
        assert isinstance(count, int)

    def test_list_content_blocks(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }]
        count = count_message_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_message_with_tool_calls(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "bash", "arguments": '{"cmd": "ls"}'},
                {"id": "call_2", "name": "read", "arguments": '{"path": "/tmp"}'},
            ],
        }]
        count = count_message_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_model_kwarg_accepted(self):
        msgs = [{"role": "user", "content": "hello"}]
        count = count_message_tokens(msgs, model="gpt-4")
        assert isinstance(count, int)

    def test_batch_grows_with_messages(self):
        single = count_message_tokens([{"role": "user", "content": "hello"}])
        double = count_message_tokens([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])
        assert double > single


class TestTiktokenAvailability:
    """Verify is_tiktoken_available() reports truthfully."""

    def test_import_works(self):
        """Just call it to ensure no import errors."""
        available = is_tiktoken_available()
        # It returns a bool regardless of whether tiktoken is installed
        assert isinstance(available, bool)
