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
Move module-level functions into an owning class and update callers (§6.2).

Safety properties, each of which was violated by an earlier, discarded version of this
tool that silently rewrote ``main()`` calls in unrelated modules:

  1. **Import-scoped propagation.** A caller is rewritten only if it actually imports
     this module. Any other file that merely mentions the same identifier is reported
     untouched — a generic name like ``main`` or ``setup`` belongs to *that* file.
  2. **Definition sites are never rewritten** (``(?<!def )`` guard).
  3. **Call rewriting never appends a parenthesis**: the name is matched with a
     ``(?=\\s*\\()`` lookahead, so the replacement must be the dotted name only.
  4. **Every written file must re-parse**, otherwise it is skipped and reported.
  5. **Each source module is one reviewable change** — pass one file at a time.

Usage: ``python tools/dev/expose_classes.py <module.py> [--owner Class] [--apply]``
"""

import argparse
import ast
import pathlib
import re
import sys

__all__ = ["PiscesLxClassExposureTool"]


class PiscesLxClassExposureTool:
    """Per-module function-to-method migration with import-scoped rewriting."""

    SKIP_PARTS = ("__pycache__", ".git", "third_party", ".mypy_cache", ".ruff_cache")
    ALLOWED_DECORATORS = frozenset({"staticmethod", "classmethod"})
    GUARD = r"(?<!def )(?<![\w.])"

    def __init__(self, module: pathlib.Path, owner: str = "", apply: bool = False) -> None:
        self.module = pathlib.Path(module)
        self.owner = owner
        self.apply = apply
        self.notes = []
        self.reported = []

    # ── discovery ─────────────────────────────────────────────────────────────
    def source(self) -> str:
        return self.module.read_text(encoding="utf-8")

    def module_path_string(self) -> str:
        parts = list(self.module.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def collect(self):
        """Return movable top-level functions, or None when there is nothing to do."""
        tree = ast.parse(self.source())
        funcs = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        movable, skipped = [], []
        for fn in funcs:
            deco = {ast.unparse(d).split("(")[0] for d in fn.decorator_list}
            if fn.name.startswith("__") and fn.name.endswith("__"):
                skipped.append((fn.name, "dunder"))
            elif deco - self.ALLOWED_DECORATORS:
                skipped.append((fn.name, f"decorator {sorted(deco - self.ALLOWED_DECORATORS)}"))
            elif isinstance(fn, ast.AsyncFunctionDef):
                skipped.append((fn.name, "async"))
            else:
                movable.append(fn)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        return movable, classes, skipped

    # ── construction ──────────────────────────────────────────────────────────
    def class_name(self, classes) -> str:
        if self.owner:
            return self.owner
        inside_model = self.module.parts[0] == "model"
        prefix = "Yv" if inside_model else "PiscesLx"
        words = [w for w in re.split(r"[_\-]", self.module.stem) if w and w != "__init__"]
        return prefix + "".join(w[:1].upper() + w[1:] for w in words)

    def method_block(self, funcs, cls) -> str:
        names = [f.name for f in funcs]
        lines = self.source().split("\n")
        chunks = []
        for fn in sorted(funcs, key=lambda f: f.lineno):
            seg = lines[fn.lineno - 1:fn.end_lineno]
            indented = [("    " + l) if l.strip() else ""
                        for l in seg
                        if not l.strip().startswith(("@staticmethod", "@classmethod"))]
            body = "\n".join(indented)
            for other in names:
                body = re.sub(self.GUARD + re.escape(other) + r"(?=\s*\()",
                              f"{cls}.{other}", body)
            chunks.append("    @staticmethod\n" + body)
        return "\n\n".join(chunks) + "\n"

    def rebuilt(self, funcs, classes, cls):
        """The module source with functions moved into ``cls`` (new or existing)."""
        lines = self.source().split("\n")
        head = lines[:]
        for fn in sorted(funcs, key=lambda f: -f.lineno):
            start = fn.lineno - 1
            if fn.decorator_list:
                start = min(start, min(d.lineno for d in fn.decorator_list) - 1)
            end = fn.end_lineno
            while start > 0 and not head[start - 1].strip():
                start -= 1
            while end < len(head) and not head[end].strip():
                end += 1
            del head[start:end]

        methods = self.method_block(funcs, cls)
        existing = next((i for i, l in enumerate(head) if l.startswith(f"class {cls}")), None)
        if existing is None:
            doc = (
                f"class {cls}:\n"
                f'    """Operations formerly exposed as module-level functions of '
                f'{self.module.stem}.\n\n'
                f"    Standard §6.2 requires functionality be exposed through a class, so "
                f'each became a static method here.\n"""\n\n'
            )
            tail = (doc + methods).split("\n")
            body = "\n".join(head).rstrip("\n")
            return body + "\n\n" + "\n".join(tail)
        end = len(head)
        for j in range(existing + 1, len(head)):
            if head[j].startswith(("class ", "def ", "@", "if ", "for ", "while ", "try:")):
                end = j
                break
        while end > existing + 1 and not head[end - 1].strip():
            end -= 1
        merged = head[:end] + [""] + methods.rstrip("\n").split("\n") + head[end:]
        return "\n".join(merged)

    # ── callers ───────────────────────────────────────────────────────────────
    def importers(self, names):
        """Files that import this module, plus any file merely mentioning the names.

        Detection is AST-based and covers both call sites and *import statements*:
        ``from .rope import (\\n    precompute_freqs_cis,\\n)`` contains no ``name(``,
        so a call-only scan classified the file as unrelated and the rewrite left it
        importing a symbol that had just moved. That is exactly how an earlier run broke
        six modules.
        """
        dotted = self.module_path_string()
        call = re.compile(self.GUARD + r"(" + "|".join(map(re.escape, names)) + r")(?=\s*\()")
        callers, strangers = [], []
        for path in pathlib.Path(".").rglob("*.py"):
            if any(s in str(path) for s in self.SKIP_PARTS) or path == self.module:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            mentions_call = bool(call.search(text))
            imports_it, nodes = self._import_nodes(path, text, dotted, names)
            if not mentions_call and not imports_it:
                continue
            if imports_it:
                callers.append((path, text, nodes))
            else:
                strangers.append((path, text))
        return callers, strangers

    def _import_nodes(self, path: pathlib.Path, text: str, dotted: str, names=()):
        """Import nodes in *path* that resolve to *dotted* and bind one of our names."""
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return False, []
        parts = list(path.with_suffix("").parts)
        package_parts = parts[:-1]
        target = dotted.split(".")
        matched = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package_parts[:len(package_parts) - node.level + 1]
                resolved = base + (node.module.split(".") if node.module else [])
            else:
                resolved = (node.module or "").split(".")
            if resolved != target:
                continue
            # NOTE: do NOT write `a.name in {"*", ""} is False` — Python parses that as a
            # chained comparison, which is constantly False and silently hid every importer.
            bound = {a.name for a in node.names}
            if not names or (bound & set(names)):
                matched.append(node)
        return bool(matched), matched

    def rewrite_caller(self, path, text, cls, names, import_nodes=()):
        for name in names:
            text = re.sub(self.GUARD + re.escape(name) + r"(?=\s*\()", f"{cls}.{name}", text)
        if not import_nodes:
            return text
        lines = text.split("\n")
        # Rewrite from the bottom up so earlier line numbers stay valid.
        for node in sorted(import_nodes, key=lambda n: -n.lineno):
            kept = [a.asname or a.name for a in node.names
                    if a.name not in names and a.name != "*"]
            payload = ", ".join([cls] + kept)
            spec = "." * node.level + (node.module or "")
            indent = re.match(r"[ \t]*", lines[node.lineno - 1]).group(0)
            statement = f"{indent}from {spec} import {payload}"
            if len(statement) > 96 and kept:
                body = "".join(f"\n{indent}    {name}," for name in [cls] + kept)
                statement = f"{indent}from {spec} import ({body}\n{indent})"
            lines[node.lineno - 1:node.end_lineno] = statement.split("\n")
        return "\n".join(lines)

    # ── driver ────────────────────────────────────────────────────────────────
    def run(self) -> int:
        collected = self.collect()
        if collected is None:
            print(f"{self.module}: no movable top-level functions")
            return 0
        funcs, classes, skipped = collected
        for name, why in skipped:
            self.notes.append(f"{self.module}: kept in place: {name} ({why})")
        cls = self.class_name(classes)
        names = [f.name for f in funcs]
        print(f"{self.module}: {len(names)} function(s) -> {cls}"
              f"{' (new class)' if not any(l.startswith(f'class {cls}') for l in self.source().split(chr(10))) else ' (existing class)'}")
        try:
            new_source = self.rebuilt(funcs, classes, cls)
            ast.parse(new_source)
        except SyntaxError as error:
            print(f"  ABORT: rebuilt module invalid: {error}")
            return 1
        callers, strangers = self.importers(names)
        print(f"  importers to rewrite: {len(callers)}  "
              f"files mentioning the names without importing: {len(strangers)}")
        for path, _ in strangers:
            print(f"    REVIEW BY HAND (does not import this module): {path.as_posix()}")
        if not self.apply:
            print("  [dry-run] nothing written; pass --apply to commit the change")
            return 0
        self.module.write_text(new_source, encoding="utf-8", newline="")
        written = 0
        for path, text, nodes in callers:
            updated = self.rewrite_caller(path, text, cls, names, nodes)
            try:
                ast.parse(updated)
            except SyntaxError as error:
                print(f"    REFUSE {path}: {error}")
                continue
            path.write_text(updated, encoding="utf-8", newline="")
            written += 1
        print(f"  wrote module + {written}/{len(callers)} caller file(s)")
        for note in self.notes:
            print("  note:", note)
        return 0


    @classmethod
    def cli(cls, argv) -> int:
        """Argument-parsing entry point (kept a method: §6.2 allows no module-level def)."""
        parser = argparse.ArgumentParser(prog="tools/dev/expose_classes.py")
        parser.add_argument("module")
        parser.add_argument("--owner", default="", help="existing class to absorb into")
        parser.add_argument("--apply", action="store_true")
        args = parser.parse_args(argv)
        return cls(pathlib.Path(args.module), args.owner, args.apply).run()


if __name__ == "__main__":
    raise SystemExit(PiscesLxClassExposureTool.cli(sys.argv[1:]))