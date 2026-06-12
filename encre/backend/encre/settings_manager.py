#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025-2026 Wenze Wei. All Rights Reserved.
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

import json
from pathlib import Path

from encre.crypto import encrypt, decrypt

_SETTINGS_PATH = Path("~/.dunimd/encre/settings.json").expanduser()

_GENERAL_SETTINGS_KEYS = frozenset({
    "shortcut_send_mode",
    "language",
    "default_link_behavior",
    "default_markdown_behavior",
    "startup_session_mode",
    "startup_session_behavior",
})


def load_settings() -> dict[str, str]:
    try:
        if not _SETTINGS_PATH.exists():
            return {}
        encrypted = _SETTINGS_PATH.read_text(encoding="utf-8").strip()
        if not encrypted:
            return {}
        decrypted = decrypt(encrypted)
        return json.loads(decrypted)
    except Exception:
        return {}


def save_settings(settings: dict[str, str]) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(settings, ensure_ascii=False, indent=2)
    encrypted = encrypt(raw)
    _SETTINGS_PATH.write_text(encrypted, encoding="utf-8")


def is_general_setting(key: str) -> bool:
    return key in _GENERAL_SETTINGS_KEYS


def load_custom_slash_commands() -> list[dict]:
    """Load custom slash commands from settings."""
    try:
        settings = load_settings()
        raw = settings.get("custom_slash_commands", "[]")
        if isinstance(raw, str):
            return json.loads(raw)
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def save_custom_slash_commands(commands: list[dict]) -> None:
    """Save custom slash commands into settings."""
    try:
        settings = load_settings()
        settings["custom_slash_commands"] = json.dumps(commands, ensure_ascii=False)
        save_settings(settings)
    except Exception:
        pass
