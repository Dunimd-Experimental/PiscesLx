#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

"""
Failure-suppression conformance (§7): a swallowed exception is a hidden defect.

``except Exception: pass`` is how the multi-task uncertainty operator kept failing its
own input contract on every training step while the loop reported success: the result
was dropped, the operator logged to its own logger, and nothing surfaced. This check
makes that class of silence countable so it can be eliminated on evidence.
"""

import ast
import re

from tools.check.base import (
    PiscesLxRepositoryScanner,
    PiscesLxSourceFile,
    PiscesLxStandardCheck,
    PiscesLxViolation,
)

__all__ = ["PiscesLxSilentExceptCheck"]


class PiscesLxSilentExceptCheck(PiscesLxStandardCheck):
    """§7 — every handler must name what it recovered or justify staying silent."""

    rule = "NO-SILENT-EXCEPT"
    description = "no 'except ...: pass' / docstring-only handler without a deliberate marker"

    _DELIBERATE = re.compile(r"#\s*(deliberate|intentional|by design)\b", re.IGNORECASE)

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Scan every parseable module for discarding handlers."""
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> list:
        tree = source.tree
        if tree is None:
            return []
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            only = node.body[0] if len(node.body) == 1 else None
            if not self._is_silence(only):
                continue
            if self._DELIBERATE.search(self._window(source, node, only)):
                continue
            violations.append(
                PiscesLxViolation(
                    self.rule,
                    source.path,
                    node.lineno,
                    "handler discards the exception ("
                    + ("pass" if isinstance(only, ast.Pass) else "string only")
                    + ") — log or re-raise it, or mark the line '# deliberate: <why>'",
                    column=getattr(node, "col_offset", 0),
                    snippet=source.line_source(node.lineno),
                )
            )
        return violations

    @staticmethod
    def _is_silence(node) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Pass):
            return True
        return (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))

    @staticmethod
    def _window(source: PiscesLxSourceFile, node: ast.ExceptHandler, only) -> str:
        """The handler and the statement above it, where a justification comment sits."""
        lines = source.lines
        end = getattr(only, "end_lineno", node.lineno)
        return "\n".join(lines[max(0, node.lineno - 2):min(len(lines), end + 1)])
