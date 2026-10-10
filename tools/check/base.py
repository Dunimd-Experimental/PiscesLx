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
Mandatory coding-standard check framework for PiscesLx.

Every prohibition in the upgrade directive is turned into a runnable check here
(see ``docs/prompts/UPGRADE_06_CODING.md`` §6.8). Acceptance is decided by the
output of these scripts, never by a human claim.
"""

import ast
import re
from pathlib import Path
from typing import Iterable

__all__ = [
    "PiscesLxRuleId",
    "PiscesLxSeverity",
    "PiscesLxViolation",
    "PiscesLxSourceFile",
    "PiscesLxCheckReport",
    "PiscesLxRepositoryScanner",
    "PiscesLxStandardCheck",
    "PiscesLxStandardHeader",
]


class PiscesLxRuleId:
    """Stable identifiers for each machine-checkable prohibition."""

    CLASS_NAMING_MODEL = "NAMING-MODEL-YV"
    CLASS_NAMING_OUTSIDE = "NAMING-OUTSIDE-PISCESLX"
    CLASS_NAMING_FORBIDDEN = "NAMING-FORBIDDEN-PREFIX"
    TOP_LEVEL_FUNCTION = "EXPOSE-CLASSES-ONLY"
    STDLIB_LOGGING = "LOGGING-DMSC-ONLY"
    COPYRIGHT_HEADER = "HEADER-VERBATIM"
    PROJECT_NAME = "NAME-PISCESL1-PISCESLX"
    HARDCODED_PATH = "NO-HARDCODED-PATH"
    HARDCODED_DEVICE = "NO-HARDCODED-DEVICE"
    HARDCODED_MODEL_ID = "NO-HARDCODED-MODEL-ID"
    MAGIC_NUMBER = "NO-MAGIC-NUMBER"
    RESIDUAL_ENTA = "CLEAN-ENTA-ENCRE"
    RESIDUAL_WATERMARK = "CLEAN-WATERMARK"
    RESIDUAL_MCP_SERVICE = "CLEAN-MCP-SERVICE-LAYER"
    RESIDUAL_DOC_MODALITY = "MODALITY-4-NO-DOCUMENT"
    RESIDUAL_AGENT_MODALITY = "MODALITY-4-NO-AGENT"
    ENGINE_EXTERNAL_MODEL = "ENGINE-NO-EXTERNAL-MODEL"
    DANGLING_IMPORT = "NO-DANGLING-IMPORT"
    INCOMPLETE_MARKER = "NO-STUB-NOR-TODO"

    @classmethod
    def all_rules(cls) -> tuple:
        """Return every declared rule identifier in declaration order."""
        return tuple(
            value
            for key, value in vars(cls).items()
            if not key.startswith("_") and isinstance(value, str) and key.isupper()
        )


class PiscesLxSeverity:
    """Exit-code semantics for a rule hit."""

    ERROR = "error"
    WARNING = "warning"


class PiscesLxStandardHeader:
    """The single permitted file header, compared verbatim modulo line endings."""

    COPYRIGHT_BLOCK = "\n".join(
        (
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            "",
            "# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.",
            "#",
            "# This file is part of PiscesL1.",
            "# The PiscesL1 project belongs to the Dunimd Team.",
            "#",
            "# Licensed under the Apache License, Version 2.0 (the \"License\");",
            "# You may not use this file except in compliance with the License.",
            "# You may obtain a copy of the License at",
            "#",
            "#     http://www.apache.org/licenses/LICENSE-2.0",
            "#",
            "# Unless required by applicable law or agreed to in writing, software",
            "# distributed under the License is distributed on an \"AS IS\" BASIS,",
            "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            "# See the License for the specific language governing permissions and",
            "# limitations under the License.",
            "#",
            "# DISCLAIMER: Users must comply with applicable AI regulations.",
            "# Non-compliance may result in service termination or legal liability.",
            "",
            "from __future__ import annotations",
        )
    )

    _ENCODING_LINE = "# -*- coding: utf-8 -*-"
    _FUTURE_LINE = "from __future__ import annotations"

    @classmethod
    def normalize(cls, text: str) -> str:
        """Fold CRLF/CR endings to LF so EOL style never masks a real violation."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def extract_leading_block(cls, text: str) -> str:
        """Return the shebang-through-future-import region of ``text``."""
        lines = cls.normalize(text).split("\n")
        if not lines or not lines[0].startswith("#!"):
            return ""
        collected = [lines[0]]
        for index, line in enumerate(lines[1:], start=1):
            collected.append(line)
            if line.strip() == cls._FUTURE_LINE.strip():
                return "\n".join(collected).rstrip("\n")
            if line.strip() and not line.startswith("#") and line.strip() != cls._ENCODING_LINE:
                return "\n".join(collected).rstrip("\n")
        return "\n".join(collected).rstrip("\n")


class PiscesLxViolation:
    """One rule hit pinned to a file and line."""

    __slots__ = ("rule", "path", "line", "column", "message", "severity", "snippet")

    def __init__(
        self,
        rule: str,
        path: Path,
        line: int,
        message: str,
        *,
        column: int = 0,
        severity: str = PiscesLxSeverity.ERROR,
        snippet: str = "",
    ) -> None:
        self.rule = rule
        self.path = path
        self.line = line
        self.column = column
        self.message = message
        self.severity = severity
        self.snippet = snippet

    def render(self, root: Path) -> str:
        """Render as ``rule: relative/path:line: message`` for CI log grepping."""
        try:
            location = self.path.relative_to(root)
        except ValueError:
            location = self.path
        suffix = f":{self.column}" if self.column else ""
        return f"{self.rule}: {location.as_posix()}:{self.line}{suffix}: {self.message}"


class PiscesLxSourceFile:
    """Lazily parsed view of one Python source file."""

    __slots__ = ("path", "root", "_text", "_lines", "_tree", "_parse_error")

    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root
        self._text = None
        self._lines = None
        self._tree = None
        self._parse_error = None

    @property
    def text(self) -> str:
        """File content with normalised line endings."""
        if self._text is None:
            self._load()
        return self._text

    @property
    def lines(self) -> list:
        """Physical lines of :attr:`text` without newline characters."""
        if self._lines is None:
            self._lines = self.text.split("\n")
        return self._lines

    @property
    def tree(self):
        """Module AST, or ``None`` when the file does not parse."""
        if self._tree is None and self._parse_error is None:
            self._parse()
        return self._tree

    @property
    def parse_error(self) -> str:
        """Syntax error description captured during :meth:`_parse`."""
        if self._tree is None and self._parse_error is None:
            self._parse()
        return self._parse_error or ""

    @property
    def is_init(self) -> bool:
        """Whether the file is a package re-export module."""
        return self.path.name == "__init__.py"

    @property
    def in_model(self) -> bool:
        """Whether the file lives under the repository ``model/`` package."""
        parts = self.path.relative_to(self.root).parts
        return bool(parts) and parts[0] == "model"

    def line_source(self, line: int) -> str:
        """Best-effort source text for 1-based ``line``."""
        lines = self.lines
        if 1 <= line <= len(lines):
            return lines[line - 1].strip()[:160]
        return ""

    def _load(self) -> None:
        try:
            raw = self.path.read_bytes()
        except OSError as error:
            self._text = ""
            self._parse_error = f"unreadable: {error}"
            return
        self._text = PiscesLxStandardHeader.normalize(
            raw.decode("utf-8", errors="replace")
        )

    def _parse(self) -> None:
        try:
            self._tree = ast.parse(self.text, filename=str(self.path))
        except SyntaxError as error:
            self._parse_error = f"{error.msg} (line {error.lineno})"
            self._tree = None


class PiscesLxRepositoryScanner:
    """Enumerates first-party Python sources, skipping caches and vendored trees."""

    SKIP_DIRS = frozenset(
        {
            ".git",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "env",
            "build",
            "dist",
            ".eggs",
            "site-packages",
            "third_party",
            "vendor",
        }
    )

    __slots__ = ("root", "explicit_paths", "_files", "_paths")

    def __init__(self, root: Path, explicit_paths: Iterable[Path] = ()) -> None:
        self.root = root
        self.explicit_paths = tuple(explicit_paths)
        self._files = None

    @property
    def files(self) -> tuple:
        """All discovered source files as :class:`PiscesLxSourceFile`."""
        if self._files is None:
            self._files = tuple(PiscesLxSourceFile(p, self.root) for p in self.paths)
        return self._files

    @property
    def paths(self) -> tuple:
        """Sorted absolute paths of every first-party ``*.py``."""
        self._collect()
        return self._paths

    def _collect(self) -> None:
        if hasattr(self, "_paths"):
            return
        found = set()
        if self.explicit_paths:
            for candidate in self.explicit_paths:
                self._absorb(candidate, found)
        else:
            self._walk(self.root, found)
        self._paths = tuple(sorted(found, key=lambda p: p.as_posix()))

    def _walk(self, directory: Path, sink: set) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name in self.SKIP_DIRS or name.startswith("."):
                    continue
                self._walk(entry, sink)
            elif name.endswith(".py"):
                if not self._is_skip_leaf(name):
                    sink.add(entry.resolve())

    @staticmethod
    def _is_skip_leaf(name: str) -> bool:
        return name in {"conftmp.py"}

    def _absorb(self, candidate: Path, sink: set) -> None:
        target = candidate if candidate.is_absolute() else (self.root / candidate)
        target = target.resolve()
        if target.is_dir():
            self._walk(target, sink)
        elif target.is_file() and target.name.endswith(".py"):
            sink.add(target)


class PiscesLxStandardCheck:
    """Base class for one prohibition turned into one executable check."""

    rule: str = ""
    description: str = ""

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Return :class:`PiscesLxViolation` list for the scanned repository."""
        raise NotImplementedError

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Per-file hook; overridden by AST or regex based checks."""
        return []

    def run_per_file(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Drive :meth:`inspect` across the corpus, recording parse failures."""
        violations = []
        for source in scanner.files:
            if source.parse_error:
                violations.append(
                    PiscesLxViolation(
                        self.rule,
                        source.path,
                        1,
                        f"syntax error blocks compliance check: {source.parse_error}",
                    )
                )
                continue
            violations.extend(self.inspect(source))
        return violations

    @staticmethod
    def module_level_defs(tree: ast.Module):
        """Yield ``ast.FunctionDef`` nodes bound at module body level."""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                yield node

    @staticmethod
    def matches_any(text: str, patterns: tuple) -> bool:
        """Whether any compiled pattern matches ``text``."""
        return any(pattern.search(text) for pattern in patterns)

    def summarize(self, violations: list) -> str:
        """Human-readable one-line summary used by the runner."""
        return f"{self.rule}: {len(violations)} violation(s) — {self.description}"


class PiscesLxCheckReport:
    """Aggregated violations with deterministic rendering and exit semantics."""

    __slots__ = ("root", "violations", "checks_run", "files_scanned", "duration_s")

    def __init__(self, root: Path, violations: list, checks_run: int, files_scanned: int, duration_s: float) -> None:
        self.root = root
        self.violations = violations
        self.checks_run = checks_run
        self.files_scanned = files_scanned
        self.duration_s = duration_s

    @property
    def errors(self) -> list:
        """Violations that must fail the build."""
        return [v for v in self.violations if v.severity == PiscesLxSeverity.ERROR]

    @property
    def warnings(self) -> list:
        """Advisory violations that are reported but do not fail the build."""
        return [v for v in self.violations if v.severity == PiscesLxSeverity.WARNING]

    def by_rule(self) -> dict:
        """Group violations by rule identifier, sorted."""
        grouped = {}
        for violation in self.violations:
            grouped.setdefault(violation.rule, []).append(violation)
        return {rule: grouped[rule] for rule in sorted(grouped)}

    @property
    def exit_code(self) -> int:
        """Non-zero when any error-severity violation exists."""
        return 1 if self.errors else 0

    def render(self, *, verbose: bool = True, max_rows: int = 0) -> str:
        """Render the report; ``max_rows=0`` prints every violation line."""
        out = [
            "PiscesL1 mandatory-standard conformance report",
            f"repository      : {self.root.as_posix()}",
            f"python files    : {self.files_scanned}",
            f"checks executed : {self.checks_run}",
            f"duration        : {self.duration_s:.2f}s",
            f"errors          : {len(self.errors)}",
            f"warnings        : {len(self.warnings)}",
            "",
        ]
        grouped = self.by_rule()
        if grouped:
            out.append("violations by rule")
            for rule, items in grouped.items():
                out.append(f"  {rule:<28} {len(items):>6}")
            out.append("")
        else:
            out.append("all mandatory standards satisfied")
            return "\n".join(out)
        if verbose:
            out.append("detail (rule: path:line: message)")
            total = 0
            for rule, items in grouped.items():
                limit = max_rows if max_rows else len(items)
                shown = 0
                for violation in items:
                    if shown >= limit:
                        out.append(f"  ... {len(items) - shown} more [{rule}]")
                        break
                    out.append("  " + violation.render(self.root))
                    shown += 1
                    total += 1
                del total
        return "\n".join(out)


class PiscesLxAstWalker:
    """Shared AST helpers used by several checks."""

    CLASS_NODES = (ast.ClassDef,)

    @classmethod
    def iter_classes(cls, tree: ast.Module):
        """Yield every class defined anywhere in the module."""
        for node in ast.walk(tree):
            if isinstance(node, cls.CLASS_NODES):
                yield node

    @classmethod
    def iter_functions(cls, tree: ast.Module):
        """Yield every function defined anywhere in the module."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node

    @classmethod
    def enclosing_classes(cls, tree: ast.Module) -> dict:
        """Map every nested node to the chain of class names enclosing it."""
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        return parents

    @classmethod
    def is_inside_class(cls, node: ast.AST, parents: dict) -> bool:
        """Whether ``node`` sits within a class body."""
        current = parents.get(node)
        while current is not None:
            if isinstance(current, ast.ClassDef):
                return True
            current = parents.get(current)
        return False

    @classmethod
    def is_inside_main_guard(cls, node: ast.AST, parents: dict) -> bool:
        """Whether ``node`` is guarded by ``if __name__ == "__main__"``."""
        current = parents.get(node)
        while current is not None:
            if isinstance(current, ast.If) and cls._is_main_test(current.test):
                return True
            current = parents.get(current)
        return False

    @staticmethod
    def _is_main_test(test: ast.expr) -> bool:
        if not isinstance(test, ast.Compare):
            return False
        left = test.left
        if isinstance(left, ast.Name) and left.id == "__name__":
            return True
        rendering = getattr(left, "value", None)
        return rendering == "__name__"


class PiscesLxRegexLibrary:
    """Compiled regexes shared by the text-level checks."""

    LOGGING_IMPORT = re.compile(r"^\s*(?:import\s+logging\b|from\s+logging\b)")
    LOGGING_CALL = re.compile(r"\blogs?ging\.(?:getLogger|basicConfig|config|Formatter|Handler|StreamHandler)\b")
    WINDOWS_ABS_PATH = re.compile(r"[\"'][A-Za-z]:[\\/]{1,2}")
    POSIX_ABS_PATH = re.compile(r"[\"']/(?:home|usr|mnt|opt|data|workspace|root|tmp|var)(?:/|\"|')")
    HOME_PATH = re.compile(r"[\"']~[\\/]")
    CUDA_DEVICE = re.compile(r"[\"']cuda:\d+[\"']|\bcuda:\d+\b")
    MULTI_GPU_DEVICE = re.compile(r"[\"'](?:gpu|npu|tpu|cpu):\d+[\"']")
    STRUCTURAL_NUMBERS = frozenset({"0", "1", "-1", "2", "0.0", "1.0"})

    @classmethod
    def absolute_path_patterns(cls) -> tuple:
        """Hardcoded-filesystem-location patterns."""
        return (cls.WINDOWS_ABS_PATH, cls.POSIX_ABS_PATH, cls.HOME_PATH)
