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
Package export conformance (§6.3): a package may advertise only what it binds.

``__all__`` is a promise — ``from pkg import *`` and every documented entry point
rely on it.  Renames and the §6.2 function-to-class conversions delete the symbol
while the advertisement survives, and nothing in the import graph notices until a
caller breaks.  Both binding styles used in this repository are checked: literal
``__all__`` lists and the module tables behind a PEP 562 ``__getattr__``.
"""

import ast

from tools.check.base import (
    PiscesLxRepositoryScanner,
    PiscesLxSourceFile,
    PiscesLxStandardCheck,
    PiscesLxViolation,
)

__all__ = ["PiscesLxDanglingExportCheck"]


class PiscesLxDanglingExportCheck(PiscesLxStandardCheck):
    """§6.3 — every advertised export resolves to a definition in the tree."""

    rule = "NO-DANGLING-EXPORT"
    description = ("__all__ entries are bound by the package and lazy-export tables "
                   "name a module plus a symbol that both exist")

    _FIRST_PARTY = ("model", "opss", "tools", "utils", "configs", "enta")

    def __init__(self) -> None:
        self._symbol_cache: dict = {}

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Check every package re-export module in the repository."""
        violations = []
        for source in scanner.files:
            if not source.is_init or source.tree is None:
                continue
            violations.extend(self._inspect_package(source))
        return violations

    def _inspect_package(self, source: PiscesLxSourceFile) -> list:
        tree = source.tree
        bound = self._bound_names(tree)
        violations = []
        for name, lineno in self._all_entries(tree):
            if name in bound:
                continue
            violations.append(
                PiscesLxViolation(
                    self.rule,
                    source.path,
                    lineno,
                    f"'{name}' is advertised in __all__ but nothing in "
                    f"{self._relative(source)} binds it — 'from "
                    f"{self._package_dotted(source)} import *' raises AttributeError",
                    snippet=source.line_source(lineno),
                )
            )
        for spec, attr, lineno in self._lazy_targets(tree):
            violations.extend(self._inspect_lazy(source, spec, attr, lineno))
        return violations

    def _inspect_lazy(self, source: PiscesLxSourceFile, spec: str,
                      attr: str, lineno: int) -> list:
        dotted = self._resolve_spec(source, spec)
        if dotted.split(".")[0] not in self._FIRST_PARTY:
            return []
        module_file = self._module_file(source.root, dotted)
        if module_file is None:
            return [
                PiscesLxViolation(
                    self.rule, source.path, lineno,
                    f"lazy export '{attr}' points at '{dotted}', which is not a "
                    "module in this tree",
                    snippet=source.line_source(lineno),
                )
            ]
        symbols = self._module_symbols(module_file)
        if symbols is None or attr in symbols:
            return []
        return [
            PiscesLxViolation(
                self.rule, source.path, lineno,
                f"lazy export '{attr}' is not defined by {dotted}",
                snippet=source.line_source(lineno),
            )
        ]

    # ── static name collection ────────────────────────────────────────────
    def _bound_names(self, tree: ast.Module) -> set:
        """Names a module binds at top level, lazy-export table keys included."""
        bound = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Import):
                bound.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                bound.update(a.asname or a.name for a in node.names if a.name != "*")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    bound.update(self._target_names(target))
                bound.update(self._dict_keys(node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
                bound.update(self._dict_keys(node.value))
        return bound

    @classmethod
    def _target_names(cls, target: ast.AST) -> set:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names = set()
            for element in target.elts:
                names.update(cls._target_names(element))
            return names
        return set()

    @staticmethod
    def _dict_keys(value: ast.AST) -> set:
        """Keys of a module-level dict literal — the PEP 562 lookup tables."""
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if not isinstance(value, ast.Dict):
            return set()
        return {
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

    @staticmethod
    def _all_entries(tree: ast.Module) -> list:
        """(name, line) pairs of a literal module-level ``__all__``."""
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            return [
                (element.value, element.lineno)
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
        return []

    @staticmethod
    def _lazy_targets(tree: ast.Module) -> list:
        """(module spec, attribute, line) triples taken from a dict literal."""
        found = []
        for node in tree.body:
            value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            if not isinstance(value, ast.Dict):
                continue
            for key, item in zip(value.keys, value.values):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    found.append((item.value, key.value, key.lineno))
                elif (
                    isinstance(item, ast.Tuple)
                    and len(item.elts) == 2
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                            for e in item.elts)
                ):
                    found.append((item.elts[0].value, item.elts[1].value, key.lineno))
        return found

    def _module_symbols(self, module_file) -> set:
        """Top-level names of *module_file*; ``None`` when it resolves dynamically."""
        key = str(module_file)
        if key in self._symbol_cache:
            return self._symbol_cache[key]
        try:
            tree = ast.parse(module_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            self._symbol_cache[key] = set()
            return self._symbol_cache[key]
        if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body):
            self._symbol_cache[key] = None
            return None
        symbols = self._bound_names(tree)
        self._symbol_cache[key] = symbols
        return symbols

    @staticmethod
    def _resolve_spec(source: PiscesLxSourceFile, spec: str) -> str:
        """Turn ``'.builder'`` or ``'opss.train.distill'`` into a dotted path."""
        if not spec.startswith("."):
            return spec
        package = PiscesLxDanglingExportCheck._package_dotted(source)
        levels = len(spec) - len(spec.lstrip("."))
        parts = package.split(".") if package else []
        base = parts[: len(parts) - levels + 1] if levels > 1 else parts
        tail = spec.lstrip(".").split(".") if spec.lstrip(".") else []
        return ".".join(part for part in base + tail if part)

    @staticmethod
    def _package_dotted(source: PiscesLxSourceFile) -> str:
        return ".".join(source.path.relative_to(source.root).parts[:-1])

    @staticmethod
    def _relative(source: PiscesLxSourceFile) -> str:
        return source.path.relative_to(source.root).as_posix()

    @staticmethod
    def _module_file(root, dotted: str):
        parts = dotted.split(".")
        module = root.joinpath(*parts).with_suffix(".py")
        if module.exists():
            return module
        package = root.joinpath(*parts, "__init__.py")
        return package if package.exists() else None

    def inspect(self, source: PiscesLxSourceFile) -> list:
        """Unused — :meth:`run` drives the package pass."""
        del source
        return []
