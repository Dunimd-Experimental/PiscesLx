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
Naming, exposure, logging, header and project-spelling conformance checks.

Implements ``docs/prompts/UPGRADE_06_CODING.md`` §6.1-§6.6 as executable rules.
"""

import ast
import re
from pathlib import Path

from tools.check.base import (
    PiscesLxAstWalker,
    PiscesLxRegexLibrary,
    PiscesLxRepositoryScanner,
    PiscesLxSeverity,
    PiscesLxSourceFile,
    PiscesLxStandardCheck,
    PiscesLxStandardHeader,
    PiscesLxViolation,
)

__all__ = [
    "PiscesLxClassNamingCheck",
    "PiscesLxTopLevelFunctionCheck",
    "PiscesLxStdlibLoggingCheck",
    "PiscesLxCopyrightHeaderCheck",
    "PiscesLxProjectNameCheck",
]


class PiscesLxClassNamingCheck(PiscesLxStandardCheck):
    """§6.1 — ``model/`` classes are ``YvXxx…``; everything else is ``PiscesLxXxx…``."""

    rule = "NAMING-MODEL-YV"
    description = "class prefixes: Yv* inside model/, PiscesLx* outside model/"
    outside_rule = "NAMING-OUTSIDE-PISCESLX"

    MODEL_PREFIX = "Yv"
    OUTSIDE_PREFIX = "PiscesLx"
    _MODEL_SHAPE = re.compile(r"^_*(?:Yv[A-Z][A-Za-z0-9]*)$")
    _OUTSIDE_SHAPE = re.compile(r"^_*(?:PiscesLx[A-Z][A-Za-z0-9]*)$")
    _FORBIDDEN_IN_MODEL = re.compile(r"^(?:Pisces|PiscesL1|PISCES)")
    _DATA_LIKE = re.compile(r"^(?:Any|Dict|List|Set|Tuple|Options|Config|Args|Kwargs)$")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Collect class-prefix violations across the repository."""
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Check every class declaration in one module."""
        violations = []
        tree = source.tree
        if tree is None:
            return violations
        parents = PiscesLxAstWalker.enclosing_classes(tree)
        inside_model = source.in_model
        for node in PiscesLxAstWalker.iter_classes(tree):
            name = node.name
            if self._DATA_LIKE.match(name):
                continue
            expected = self.MODEL_PREFIX if inside_model else self.OUTSIDE_PREFIX
            shape = self._MODEL_SHAPE if inside_model else self._OUTSIDE_SHAPE
            nested = PiscesLxAstWalker.is_inside_class(node, parents)
            if shape.match(name):
                continue
            if inside_model and self._FORBIDDEN_IN_MODEL.match(name):
                violations.append(
                    PiscesLxViolation(
                        "NAMING-FORBIDDEN-PREFIX",
                        source.path,
                        node.lineno,
                        f"class '{name}' inside model/ uses a forbidden prefix; "
                        f"expected '{expected}Xxx…' (Pisces*/PiscesL1* names must not exist)",
                        column=getattr(node, "col_offset", 0),
                        snippet=source.line_source(node.lineno),
                    )
                )
                continue
            violations.append(
                PiscesLxViolation(
                    self.rule if inside_model else self.outside_rule,
                    source.path,
                    node.lineno,
                    f"class '{name}' must be named '{expected}XxxXxx…' "
                    f"({'inside' if inside_model else 'outside'} model/)"
                    + (" [nested class]" if nested else ""),
                    column=getattr(node, "col_offset", 0),
                    severity=PiscesLxSeverity.WARNING if nested else PiscesLxSeverity.ERROR,
                    snippet=source.line_source(node.lineno),
                )
            )
        return violations


class PiscesLxTopLevelFunctionCheck(PiscesLxStandardCheck):
    """§6.2 — no bare functions exposed at module level; functionality ships in classes."""

    rule = "EXPOSE-CLASSES-ONLY"
    description = "module-level def must move into the owning class"

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Collect module-level function definitions."""
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Report every ``def`` bound directly on the module body."""
        violations = []
        tree = source.tree
        if tree is None:
            return violations
        parents = PiscesLxAstWalker.enclosing_classes(tree)
        for node in self.module_level_defs(tree):
            if PiscesLxAstWalker.is_inside_main_guard(node, parents):
                continue
            violations.append(
                PiscesLxViolation(
                    self.rule,
                    source.path,
                    node.lineno,
                    f"top-level function '{node.name}()' is exposed; move it into a class "
                    "(method / @staticmethod / @classmethod)",
                    column=getattr(node, "col_offset", 0),
                    snippet=source.line_source(node.lineno),
                )
            )
        return violations


class PiscesLxStdlibLoggingCheck(PiscesLxStandardCheck):
    """§6.3 — logging goes through dmsc 0.1.8; the stdlib module is banned outright."""

    rule = "LOGGING-DMSC-ONLY"
    description = "replace stdlib logging with dmsc==0.1.8"

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Scan import and call sites for stdlib logging usage."""
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Flag stdlib logging imports (AST) and logger factory calls (text)."""
        violations = []
        tree = source.tree
        if tree is None:
            return violations
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "logging" or alias.name.startswith("logging."):
                        violations.append(
                            PiscesLxViolation(
                                self.rule,
                                source.path,
                                node.lineno,
                                f"`import {alias.name}` is forbidden; log through dmsc 0.1.8 "
                                "(see utils/dc.py)",
                                column=getattr(node, "col_offset", 0),
                                snippet=source.line_source(node.lineno),
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "logging" or node.module.startswith("logging.")):
                    violations.append(
                        PiscesLxViolation(
                            self.rule,
                            source.path,
                            node.lineno,
                            f"`from {node.module} import …` is forbidden; log through dmsc 0.1.8",
                            column=getattr(node, "col_offset", 0),
                            snippet=source.line_source(node.lineno),
                        )
                    )
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if PiscesLxRegexLibrary.LOGGING_CALL.search(line):
                violations.append(
                    PiscesLxViolation(
                        self.rule,
                        source.path,
                        index,
                        "stdlib logging call site; route through dmsc",
                        snippet=stripped[:160],
                    )
                )
        return self._dedupe(violations)

    @staticmethod
    def _dedupe(violations: list) -> list:
        seen = set()
        unique = []
        for violation in violations:
            key = (violation.rule, violation.path, violation.line)
            if key in seen:
                continue
            seen.add(key)
            unique.append(violation)
        return unique


class PiscesLxCopyrightHeaderCheck(PiscesLxStandardCheck):
    """§6.5 — the verbatim header, with ``from __future__ import annotations`` bound to it."""

    rule = "HEADER-VERBATIM"
    description = "copyright block is byte-exact and the future import follows it"

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Validate the header of every Python file."""
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Compare the leading region against the mandated header."""
        expected = PiscesLxStandardHeader.COPYRIGHT_BLOCK
        actual = PiscesLxStandardHeader.extract_leading_block(source.text)
        if actual == expected:
            return []
        if not actual:
            return [
                PiscesLxViolation(
                    self.rule,
                    source.path,
                    1,
                    "missing `#!/usr/bin/env python3` shebang — mandated header absent entirely",
                    snippet=source.line_source(1),
                )
            ]
        return [
            PiscesLxViolation(
                self.rule,
                source.path,
                self._first_divergent_line(expected.split("\n"), actual.split("\n")),
                "file header diverges from the mandated verbatim block "
                "(order: shebang → encoding → copyright → future import)",
                snippet=source.line_source(
                    self._first_divergent_line(expected.split("\n"), actual.split("\n"))
                ),
            )
        ]

    @staticmethod
    def _first_divergent_line(expected: list, actual: list) -> int:
        for index, (want, got) in enumerate(zip(expected, actual), start=1):
            if want != got:
                return index
        return max(1, min(len(expected), len(actual)))


class PiscesLxProjectNameCheck(PiscesLxStandardCheck):
    """§6.6 — only ``PiscesL1`` and ``PiscesLx`` may appear as the project name."""

    rule = "NAME-PISCESL1-PISCESLX"
    description = "project name spelled as PiscesL1 / PiscesLx only"

    _ANY = re.compile(r"[Pp][Ii][Ss][Cc][Ee][Ss][Ss]")
    _ALLOWED = ("PiscesL1", "PiscesLx")
    _SKIP_NAMES = frozenset({"paths.py"})

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Scan comments, docstrings and string literals for name variants."""
        violations = []
        for source in scanner.files:
            for index, line in enumerate(source.lines, start=1):
                for violation in self._scan_line(source, index, line):
                    violations.append(violation)
        return violations

    def _scan_line(self, source: PiscesLxSourceFile, index: int, line: str) -> list:
        found = []
        for match in self._ANY.finditer(line):
            start = match.start()
            long_form = line[start:start + 9]
            if long_form in self._ALLOWED:
                continue
            short = line[start:start + 6]
            found.append(
                PiscesLxViolation(
                    self.rule,
                    source.path,
                    index,
                    f"project name written as '{short}…'; only 'PiscesL1' (full name) or "
                    f"'PiscesLx' (codename prefix) are permitted",
                    snippet=line.strip()[:160],
                )
            )
        return found

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Text-level rule; :meth:`run` drives it directly."""
        del source
        return []


class PiscesLxIncompleteMarkerCheck(PiscesLxStandardCheck):
    """§10.2(7) — no stubs, placeholders or unfinished work shipped in the tree."""

    rule = "NO-STUB-NOR-TODO"
    description = "TODO/FIXME/placeholder/not-implemented markers are forbidden in shipped code"

    _PATTERNS = (
        re.compile(r"\bTODO\b[:\s]", re.IGNORECASE),
        re.compile(r"\bFIXME\b", re.IGNORECASE),
        re.compile(r"\bXXX\b(?!\w)"),
        re.compile(r"\bplaceholder\b", re.IGNORECASE),
        re.compile(r"\bstub\b", re.IGNORECASE),
        re.compile(r"raise\s+NotImplementedError"),
        re.compile(r"\bpass\s*#\s*(later|todo|fixme)\b", re.IGNORECASE),
    )
    _DOC_DIRS = ("docs/", "tests/", "tools/check/")

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Collect unfinished-work markers, excluding tests and documentation."""
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Report each marker hit with severity warning inside docs/tests."""
        violations = []
        relative = source.path.relative_to(source.root).as_posix()
        severity = (
            PiscesLxSeverity.WARNING
            if relative.startswith(self._DOC_DIRS) or "/test_" in relative
            else PiscesLxSeverity.ERROR
        )
        for index, line in enumerate(source.lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if PiscesLxRegexLibrary.LOGGING_IMPORT.match(stripped):
                continue
            for pattern in self._PATTERNS:
                if pattern.search(line):
                    violations.append(
                        PiscesLxViolation(
                            self.rule,
                            source.path,
                            index,
                            f"unfinished-work marker matched /{pattern.pattern}/",
                            severity=severity,
                            snippet=stripped[:160],
                        )
                    )
                    break
        return violations
