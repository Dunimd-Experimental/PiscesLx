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

import os
from dataclasses import dataclass, field
from typing import Any

from encre.config import get_data_dir


@dataclass
class RuleFile:
    source: str
    content: str
    priority: int = 0


_GLOBAL_RULES_DIR = "rules"

_PROJECT_RULE_PATTERNS: list[tuple[str, int, str]] = [
    (".encre/rules.md", 100, "encre"),
    (".cursorrules", 90, "cursor"),
    (".windsurfrules", 85, "windsurf"),
    (".clinerules", 80, "cline"),
    ("CLAUDE.md", 75, "claude"),
    ("AGENTS.md", 70, "agents"),
    (".github/copilot-instructions.md", 60, "copilot"),
]


class RulesLoader:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple[Any, ...], list[RuleFile]]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _project_signature(self, workspace_path: str) -> tuple[Any, ...]:
        if not workspace_path or not os.path.isdir(workspace_path):
            return ()
        sig: list[Any] = []
        for rel_path, priority, name in _PROJECT_RULE_PATTERNS:
            full_path = os.path.join(workspace_path, rel_path)
            try:
                st = os.stat(full_path)
                sig.append((name, rel_path, st.st_mtime_ns, st.st_size, priority))
            except OSError:
                sig.append((name, rel_path, None, None, priority))
        return tuple(sig)

    def _global_signature(self) -> tuple[Any, ...]:
        rules_dir = get_data_dir() / _GLOBAL_RULES_DIR
        if not rules_dir.is_dir():
            return ()
        sig: list[Any] = []
        try:
            for entry in sorted(rules_dir.iterdir(), key=lambda p: p.name):
                if entry.suffix.lower() != ".md" or not entry.is_file():
                    continue
                try:
                    st = entry.stat()
                    sig.append((entry.name, st.st_mtime_ns, st.st_size))
                except OSError:
                    sig.append((entry.name, None, None))
        except Exception:
            return ()
        return tuple(sig)

    def load_project_rules(self, workspace_path: str) -> list[RuleFile]:
        cache_key = f"project:{workspace_path}"
        if not workspace_path or not os.path.isdir(workspace_path):
            return []
        signature = self._project_signature(workspace_path)
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        rules: list[RuleFile] = []
        for rel_path, priority, name in _PROJECT_RULE_PATTERNS:
            full_path = os.path.join(workspace_path, rel_path)
            content = self._read_file(full_path)
            if content:
                rules.append(RuleFile(source=name, content=content, priority=priority))

        rules.sort(key=lambda r: -r.priority)
        self._cache[cache_key] = (signature, rules)
        return rules

    def load_global_rules(self) -> list[RuleFile]:
        cache_key = "global"
        rules_dir = get_data_dir() / _GLOBAL_RULES_DIR
        signature = self._global_signature()
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        if not rules_dir.is_dir():
            self._cache[cache_key] = (signature, [])
            return []

        rules: list[RuleFile] = []
        try:
            for entry in sorted(rules_dir.iterdir()):
                if entry.suffix.lower() == ".md" and entry.is_file():
                    content = self._read_file(str(entry))
                    if content:
                        name = entry.stem
                        rules.append(RuleFile(source=f"global:{name}", content=content, priority=50))
        except Exception:
            pass

        self._cache[cache_key] = (signature, rules)
        return rules

    def build_rules_prompt(self, workspace_path: str, enable_project: bool = True, enable_global: bool = True) -> str:
        blocks: list[str] = []

        if enable_global:
            global_rules = self.load_global_rules()
            for rule in global_rules:
                label = rule.source.replace("global:", "")
                blocks.append(f"[Global Rule: {label}]\n{rule.content}")

        if enable_project and workspace_path and os.path.isdir(workspace_path):
            project_rules = self.load_project_rules(workspace_path)
            for rule in project_rules:
                label = rule.source
                blocks.append(f"[Project Rule: {label}]\n{rule.content}")

        if not blocks:
            return ""

        joined = "\n\n---\n\n".join(blocks)
        return joined

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
            return ""
