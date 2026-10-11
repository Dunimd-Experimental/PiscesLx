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

__all__ = ["PiscesLxSilentExceptCheck", "PiscesLxDetachedScalarCheck"]


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


class PiscesLxDetachedScalarCheck(PiscesLxStandardCheck):
    """§7 — a trainable scalar must never be read out of the autograd graph.

    ``float(self.p)`` and ``self.p.item()`` on an ``nn.Parameter`` return the *value* and
    discard the node, so the parameter silently stops training while the module keeps
    working. That is invisible at runtime and was measured on the host tier: 64 of 75
    parameters with four elements or fewer had no gradient path at all, one source being
    ``F.scaled_dot_product_attention(..., scale=float(self.scale))``.
    """

    rule = "NO-DETACHED-SCALAR-PARAM"
    description = "no float(self.<Parameter>) / self.<Parameter>.item() reads"

    def run(self, scanner: PiscesLxRepositoryScanner) -> list:
        """Scan every module for parameters read as Python scalars."""
        return self.run_per_file(scanner)

    @staticmethod
    def _dotted_name(node) -> str:
        """Render an ``a.b.c`` attribute chain as the dotted text it spells ('' otherwise)."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        parts.append(node.id)
        return ".".join(reversed(parts))

    @classmethod
    def _is_parameter_call(cls, node) -> bool:
        """True for ``nn.Parameter(...)`` / ``torch.nn.Parameter(...)`` / ``Parameter(...)``."""
        if not isinstance(node, ast.Call):
            return False
        dotted = cls._dotted_name(node.func)
        if not dotted:
            return False
        last = dotted.split(".")[-1]
        prefix = dotted.rsplit(".", 1)[0] if "." in dotted else ""
        return last == "Parameter" and prefix in ("", "nn", "torch.nn")


    #: A scalar read that lands in a metrics/stats/report container is a deliberate
    #: observation, not a computation input, so it does not hide a gradient. The rule only
    #: forbids reading a trainable scalar out of the graph where the value is *used*.
    METRIC_TARGETS = ("metrics", "stats", "stat", "report", "reports", "log", "logs",
                      "history", "summary")

    @staticmethod
    def _statement_owners(tree):
        """Map every node to the statement that encloses it."""
        owners = {}

        def visit(node, statement):
            owners[id(node)] = statement
            for child in ast.iter_child_nodes(node):
                visit(child, child if isinstance(child, ast.stmt) else statement)

        visit(tree, tree)
        return owners

    @classmethod
    def _is_metric_read(cls, call, owners) -> bool:
        """True when the read is stored into a metrics-like container or detached block."""
        statement = owners.get(id(call))
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return False
        target = statement.targets[0]
        if not isinstance(target, ast.Subscript):
            return False
        base = target.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if not isinstance(base, ast.Name):
            return False
        return base.id.lower() in cls.METRIC_TARGETS

    def inspect(self, source: PiscesLxSourceFile) -> list:
        tree = source.tree
        if tree is None:
            return []
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            params = set()
            for assign in ast.walk(node):
                if not isinstance(assign, ast.Assign):
                    continue
                value = assign.value
                if isinstance(value, ast.Call) and self._is_parameter_call(value):
                    for target in assign.targets:
                        if isinstance(target, ast.Attribute) and isinstance(
                                target.value, ast.Name) and target.value.id == "self":
                            params.add(target.attr)
            if not params:
                continue
            owners = self._statement_owners(node)
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                name = self._detached_attr(call)
                if name in params and not self._is_metric_read(call, owners):
                    violations.append(
                        PiscesLxViolation(
                            self.rule, source.path, call.lineno,
                            f"'{self._render(call)}' reads trainable parameter self.{name} "
                            "out of the autograd graph — it can never receive a gradient; "
                            "fold the tensor into the computation instead"))
        return violations

    @staticmethod
    def _detached_attr(call) -> str:
        """Return the parameter name when ``call`` is float(self.X)/self.X.item(), else ''."""
        func, args = call.func, call.args
        if isinstance(func, ast.Name) and func.id == "float" and len(args) == 1:
            inner = args[0]
            if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name)                     and inner.value.id == "self":
                return inner.attr
        if isinstance(func, ast.Attribute) and func.attr == "item":
            inner = func.value
            if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name)                     and inner.value.id == "self":
                return inner.attr
        return ""

    @staticmethod
    def _render(call) -> str:
        """A short, faithful text for the report line."""
        try:
            import ast as _ast
            return _ast.unparse(call)[:60]
        except Exception:
            return "float(self.<param>) / self.<param>.item()"
