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

"""
EncreSoulSystem — Agent identity, personality, and user relationship files.

Manages three markdown files that define the agent's core identity and
its relationship with the user:

- SOUL.md      — Agent's core personality, values, behavior principles
- IDENTITY.md  — Agent's background story, capabilities, self-description
- USER.md      — Agent's knowledge about the user (preferences, history)

These files are stored in the data directory alongside memory/ and are
encrypted on disk using AES-256-GCM (via encre.crypto). They are loaded
at startup and injected into the agent's system prompt.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from encre.logging_config import get_logger

logger = get_logger("encre.soul")

_SOUL_FILENAME = "SOUL.md"
_IDENTITY_FILENAME = "IDENTITY.md"
_USER_FILENAME = "USER.md"


def _read_file(filepath: str) -> str:
    """Read a file with decryption fallback. Returns empty string on failure."""
    if not os.path.isfile(filepath):
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return ""
        if raw.startswith("---") or raw.startswith("#"):
            return raw
        from encre.crypto import decrypt
        return decrypt(raw)
    except Exception:
        return ""


def _write_file(filepath: str, content: str) -> None:
    """Write a file with encryption. Creates parent directories."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        from encre.crypto import encrypt
        encrypted = encrypt(content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(encrypted)
    except Exception:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


@dataclass
class SoulFiles:
    """Container for the three soul file contents."""

    soul: str = ""
    identity: str = ""
    user: str = ""


class EncreSoulSystem:
    """Manages the agent's identity files (SOUL.md, IDENTITY.md, USER.md).

    These files define:
    - SOUL.md: The agent's core personality, values, and behavioral principles.
      This is the agent's "heart" — what it stands for, how it approaches problems.
    - IDENTITY.md: The agent's self-description, capabilities, background story.
      This is the agent's "resume" — what it can do, its expertise.
    - USER.md: The agent's evolving knowledge about the user — preferences,
      communication style, past context, learned habits.

    All files are encrypted at rest using AES-256-GCM.
    """

    def __init__(self, soul_dir: str | None = None) -> None:
        if soul_dir is None:
            from encre.config import get_data_dir
            soul_dir = str(get_data_dir() / "soul")
        self._soul_dir = soul_dir
        self._files = SoulFiles()
        self._loaded = False

    def ensure_defaults(self) -> None:
        """Create default soul files if they don't exist."""
        os.makedirs(self._soul_dir, exist_ok=True)

        soul_path = os.path.join(self._soul_dir, _SOUL_FILENAME)
        if not os.path.isfile(soul_path):
            default_soul = self._default_soul()
            _write_file(soul_path, default_soul)
            logger.info("Created default SOUL.md")

        identity_path = os.path.join(self._soul_dir, _IDENTITY_FILENAME)
        if not os.path.isfile(identity_path):
            default_identity = self._default_identity()
            _write_file(identity_path, default_identity)
            logger.info("Created default IDENTITY.md")

        user_path = os.path.join(self._soul_dir, _USER_FILENAME)
        if not os.path.isfile(user_path):
            default_user = self._default_user()
            _write_file(user_path, default_user)
            logger.info("Created default USER.md")

    def load(self) -> SoulFiles:
        """Load all three soul files from disk."""
        soul_path = os.path.join(self._soul_dir, _SOUL_FILENAME)
        identity_path = os.path.join(self._soul_dir, _IDENTITY_FILENAME)
        user_path = os.path.join(self._soul_dir, _USER_FILENAME)

        self._files = SoulFiles(
            soul=_read_file(soul_path),
            identity=_read_file(identity_path),
            user=_read_file(user_path),
        )
        self._loaded = True

        if not self._files.soul:
            self._files.soul = self._default_soul()
        if not self._files.identity:
            self._files.identity = self._default_identity()
        if not self._files.user:
            self._files.user = self._default_user()

        return self._files

    def get_soul_dir(self) -> str:
        return self._soul_dir

    def build_prompt(self) -> str:
        """Build the soul section to inject into the agent system prompt."""
        if not self._loaded:
            self.load()

        parts: list[str] = []
        if self._files.soul:
            parts.append("## Agent Identity (Soul)")
            parts.append(self._files.soul)
            parts.append("")

        if self._files.identity:
            parts.append("## Agent Capabilities")
            parts.append(self._files.identity)
            parts.append("")

        if self._files.user:
            parts.append("## About the User")
            parts.append(self._files.user)
            parts.append("")

        return "\n".join(parts)

    def update_soul(self, content: str) -> None:
        """Replace SOUL.md content."""
        self._files.soul = content
        path = os.path.join(self._soul_dir, _SOUL_FILENAME)
        _write_file(path, content)

    def update_identity(self, content: str) -> None:
        """Replace IDENTITY.md content."""
        self._files.identity = content
        path = os.path.join(self._soul_dir, _IDENTITY_FILENAME)
        _write_file(path, content)

    def update_user(self, content: str) -> None:
        """Replace USER.md content. This is updated automatically over time."""
        self._files.user = content
        path = os.path.join(self._soul_dir, _USER_FILENAME)
        _write_file(path, content)

    def append_user_note(self, note: str) -> None:
        """Append a note to USER.md (for learned preferences over time)."""
        if not self._loaded:
            self.load()
        existing = self._files.user
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"\n\n_Learned {timestamp}:_\n{note}"
        updated = existing + new_entry
        self.update_user(updated)

    def get_frontend_data(self) -> dict[str, Any]:
        """Return data for frontend display."""
        if not self._loaded:
            self.load()
        return {
            "soul": self._files.soul,
            "identity": self._files.identity,
            "user": self._files.user,
            "soul_dir": self._soul_dir,
        }

    @staticmethod
    def _default_soul() -> str:
        return """I am Encre, an autonomous AI agent built for thoughtful, thorough work.

My core principles:
- I take initiative and solve problems proactively
- I am honest about my limitations and uncertainties
- I prioritize the user's goals above my own curiosity
- I learn from every interaction to serve better over time
- I protect user privacy and handle all data with care

I approach every task with curiosity, precision, and a commitment to delivering real results — not just plausible answers."""

    @staticmethod
    def _default_identity() -> str:
        return """I am a general-purpose AI agent with expertise across software engineering, data analysis, research, and creative problem-solving.

My capabilities include:
- Full-stack software development (Python, TypeScript, Rust, Go, and more)
- Code analysis, review, refactoring, and debugging
- System architecture and design
- Data analysis and visualization
- Web research and information synthesis
- File and project management
- Git operations
- Running shell commands and scripts
- Multi-agent orchestration and delegation

I work best when given clear goals and the autonomy to figure out the best path to reach them."""

    @staticmethod
    def _default_user() -> str:
        return """I am learning about the user's preferences over time.

This file will be automatically updated as I observe:
- Communication style preferences (formal vs casual, detail level)
- Programming languages and frameworks the user works with
- Common task patterns and workflows
- Corrections and feedback the user provides
- Tools and technologies the user prefers

_Empty — will be populated through interactions._"""