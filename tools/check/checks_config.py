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
Configuration-schema conformance checks (AGENTS.md §7, docs/audit/03 A36).

``getattr(cfg, "some_key", default)`` is *silent* when ``some_key`` was never declared:
the module keeps running with ``default`` and nobody ever learns that the declared knob
does nothing. This was measured across ``model/core`` — e.g. the code reads
``residual_dropout`` while :class:`YvConfig` declares ``residual_dropout_p`` (so dropout is
pinned at its literal default), and ``use_parallel`` while the declaration is
``use_parallel_attention_mlp`` (so the whole parallel-block branch is unreachable).

Two rules close that class of defect from both directions:

  * ``NO-UNDECLARED-CONFIG-READ`` — a defaulted ``getattr`` on a configuration receiver must
    name a field that some configuration schema actually declares.
  * ``NO-DEAD-CONFIG-FIELD`` — a field declared by ``YvConfig`` must be read somewhere, or the
    configuration file describing it is a fiction.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set

from tools.check.base import (
    PiscesLxRepositoryScanner,
    PiscesLxSeverity,
    PiscesLxSourceFile,
    PiscesLxStandardCheck,
    PiscesLxViolation,
)

__all__ = [
    "PiscesLxConfigSchemaCollector",
    "PiscesLxUndeclaredConfigReadCheck",
    "PiscesLxDeadConfigFieldCheck",
]


class PiscesLxConfigSchemaCollector:
    """The declared configuration schema, harvested from the repository itself.

    A *declaration* here means a class-level annotated assignment inside a dataclass-like
    class (``@dataclass`` decorated, or named ``*Config``/``*Settings``/``*Spec``), which is
    how every configuration object in this project is written. Instance attributes assigned
    in ``__init__`` are deliberately **not** treated as declarations: a defaulted read of an
    undeclared key creates such an attribute itself, so counting them would hide the very
    defect this schema exists to detect.
    """

    SCHEMA_SUFFIXES = ("Config", "Settings", "Spec", "Args", "Options")

    def __init__(self) -> None:
        self.fields_by_class: Dict[str, Set[str]] = {}
        self.schema_file_of: Dict[str, str] = {}

    @classmethod
    def is_schema_class(cls, node: ast.ClassDef) -> bool:
        """Whether a class declares configuration fields."""
        decorators = set()
        for decorator in node.decorator_list:
            name = decorator.func.__class__.__name__ if isinstance(decorator, ast.Call) else ""
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorators.add(decorator.func.id)
            elif isinstance(decorator, ast.Name):
                decorators.add(decorator.id)
        if {"dataclass", "dataclass_transform", "pydantic_dataclass"} & decorators:
            return True
        return node.name.endswith(cls.SCHEMA_SUFFIXES)

    def collect(self, scanner: PiscesLxRepositoryScanner) -> Dict[str, Set[str]]:
        """Build ``class name -> declared field names`` over the whole repository."""
        for source in scanner.files:
            tree = source.tree
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or not self.is_schema_class(node):
                    continue
                declared = self.fields_by_class.setdefault(node.name, set())
                for statement in node.body:
                    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                        declared.add(statement.target.id)
                    elif isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Name):
                                declared.add(target.id)
                self.schema_file_of.setdefault(node.name, str(source.path))
        return self.fields_by_class

    def all_fields(self) -> Set[str]:
        """Union of every declared configuration field."""
        union: Set[str] = set()
        for fields in self.fields_by_class.values():
            union.update(fields)
        return union


class PiscesLxUndeclaredConfigReadCheck(PiscesLxStandardCheck):
    """``NO-UNDECLARED-CONFIG-READ`` — a defaulted config read of an undeclared key."""

    CONFIG_RECEIVERS = (
        "cfg", "config", "self.cfg", "self.config", "self._cfg", "self._config",
        "model_config", "self.model_config", "train_config", "self.train_config",
        "self.config_obj", "conf", "self.conf",
    )
    MIN_DEFAULTS_REQUIRED = True

    rule = "NO-UNDECLARED-CONFIG-READ"
    description = "getattr(<config>, 'key', default) where 'key' is declared by no schema"

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Collect the schema first, then test every defaulted config read against it."""
        collector = PiscesLxConfigSchemaCollector()
        collector.collect(scanner)
        self._schema = collector.all_fields()
        return self.run_per_file(scanner)

    def inspect(self, source: PiscesLxSourceFile) -> List[PiscesLxViolation]:
        """Report each defaulted ``getattr`` on a configuration receiver for an unknown key."""
        tree = source.tree
        if tree is None:
            return []
        violations: List[PiscesLxViolation] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "getattr" or len(node.args) < 3:  # getattr(obj, key, default)
                continue
            if self._receiver_name(node.args[0]) not in self.CONFIG_RECEIVERS:
                continue
            key = node.args[1]
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            name = key.value
            if name in self._schema or name.startswith("_"):
                continue
            violations.append(
                PiscesLxViolation(
                    self.rule,
                    source.path,
                    node.lineno,
                    f"reads undeclared config key '{name}' with a silent default — the "
                    "declaration lives elsewhere (typo or renamed field) and this knob "
                    "cannot be set from any configuration file",
                    severity=PiscesLxSeverity.ERROR,
                )
            )
        return violations

    @staticmethod
    def _receiver_name(node: ast.AST) -> str:
        """Render an ``a.b`` chain as dotted text ('' for anything else)."""
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        parts.append(node.id)
        return ".".join(reversed(parts))


class PiscesLxDeadConfigFieldCheck(PiscesLxStandardCheck):
    """``NO-DEAD-CONFIG-FIELD`` — a ``YvConfig`` field nobody ever reads.

    Declared-but-unread fields are the mirror image of the same defect: the configuration
    surface advertises a control that no code path consults, which is how "the config says
    bf16 while the run uses fp32" style contradictions stay alive.
    """

    ROOT_SCHEMA = "YvConfig"

    rule = "NO-DEAD-CONFIG-FIELD"
    description = "YvConfig declares fields that no consumer reads"

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Compare declared root-schema fields against every textual use outside it."""
        collector = PiscesLxConfigSchemaCollector()
        fields_by_class = collector.collect(scanner)
        declared = fields_by_class.get(self.ROOT_SCHEMA, set())
        if not declared:
            return []
        root_path = collector.schema_file_of.get(self.ROOT_SCHEMA, "")
        used: Set[str] = set()
        anchor = 1
        for source in scanner.files:
            tree = source.tree
            if tree is None:
                continue
            if str(source.path) == root_path:
                anchor = self._declaration_line(tree)
                used.update(self._names_outside_schema(tree, self.ROOT_SCHEMA))
                continue
            used.update(self._all_names(tree))
        dead = sorted(declared - used)
        if not dead:
            return []
        return [
            PiscesLxViolation(
                self.rule,
                Path(root_path),
                anchor,
                f"{len(dead)} field(s) of {self.ROOT_SCHEMA} are declared but never read by "
                f"any Python module: {', '.join(dead[: 12])}"  # first 12 keep the line readable
                + (" …" if len(dead) > 12 else ""),  # 12-prefix truncation, see above
                severity=PiscesLxSeverity.WARNING,
            )
        ]

    @staticmethod
    def _all_names(tree: ast.AST) -> Set[str]:
        """Attribute, identifier and string-literal names appearing in a module."""
        names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
        return names

    @classmethod
    def _names_outside_schema(cls, tree: ast.Module, schema: str) -> Set[str]:
        """Uses inside the declaring file that are not the field declarations themselves."""
        names: Set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == schema:
                for statement in node.body:
                    if isinstance(statement, ast.AnnAssign):
                        if statement.value is not None:
                            names.update(cls._all_names(statement.value))
                    else:
                        names.update(cls._all_names(statement))
                continue
            names.update(cls._all_names(node))
        return names

    @staticmethod
    def _declaration_line(tree: ast.Module) -> int:
        """Line of the ``YvConfig`` class statement, for a locatable report."""
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "YvConfig":
                return node.lineno
        return 1
