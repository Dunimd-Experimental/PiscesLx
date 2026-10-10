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
Import every first-party module and separate real defects from missing dependencies.

A rename or a moved symbol can break an import in a file nothing else touches, so a
green static check is not enough: the tree has to be imported. Failures are split into
two buckets on purpose — a locally absent optional dependency is not a code defect and
must not be reported as one, while an ``ImportError`` naming one of *our* symbols is
exactly the damage a bad refactor leaves behind.
"""

import argparse
import importlib
import pathlib
import sys
from typing import Dict, List, Tuple

__all__ = ["PiscesLxImportCensus"]


class PiscesLxImportCensus:
    """Walks the repository, imports every module, and classifies each failure."""

    SKIP_PARTS = ("__pycache__", ".git", "third_party", ".mypy_cache",
                  ".ruff_cache", "docs", "build", "dist")
    #: substrings that mark an environment gap rather than a source defect
    DEPENDENCY_MARKERS = (
        "No module named 'datasets'",
        "load_from_disk",
        "has no attribute '__version__'",
        "No module named 'evalscope'",
        "No module named 'vllm'",
        "No module named 'triton'",
        "No module named 'modelscope'",
        "No module named 'transformers'",
        "No module named 'psutil'",
        "No module named 'yaml'",
        "Torch not compiled with CUDA",
        "unknown location",
    )
    FIRST_PARTY = ("model", "opss", "tools", "utils", "configs")

    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root)

    def module_names(self) -> List[str]:
        names = set()
        for path in self.root.rglob("*.py"):
            rel = path.relative_to(self.root)
            if any(part in self.SKIP_PARTS for part in rel.parts):
                continue
            if path.name == "manage.py":
                continue
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
                if not parts:
                    continue
            names.add(".".join(parts))
        return sorted(names)

    def classify(self, error: BaseException) -> str:
        text = f"{type(error).__name__}: {error}"
        if any(marker in text for marker in self.DEPENDENCY_MARKERS):
            return "dependency"
        if isinstance(error, ModuleNotFoundError) and not any(
                f"'{pkg}'" in text for pkg in self.FIRST_PARTY):
            return "dependency"
        return "defect"

    def run(self, verbose: bool = False) -> Dict[str, List[Tuple[str, str]]]:
        result: Dict[str, List[Tuple[str, str]]] = {"defect": [], "dependency": [], "ok": []}
        for name in self.module_names():
            try:
                importlib.import_module(name)
            except SystemExit:
                result["defect"].append((name, "module executed sys.exit at import"))
            except BaseException as error:      # noqa: BLE001 - a crashing import is a defect
                bucket = self.classify(error)
                result[bucket].append((name, f"{type(error).__name__}: {error}"))
            else:
                result["ok"].append((name, ""))
        if verbose:
            for name, detail in result["defect"]:
                print(f"   DEFECT {name}: {detail[:120]}")
        return result

    def report(self, verbose: bool = False) -> int:
        """Print the census; exit non-zero when a source defect was found."""
        result = self.run(verbose)
        total = sum(len(v) for v in result.values())
        print(f"modules attempted : {total}")
        print(f"imported cleanly  : {len(result['ok'])}")
        print(f"blocked by missing dependency (not a code defect): {len(result['dependency'])}")
        print(f"SOURCE DEFECTS    : {len(result['defect'])}")
        for name, detail in result["defect"][:25]:
            print(f"   DEFECT {name}: {detail[:120]}")
        if len(result["defect"]) > 25:
            print(f"   ... {len(result['defect']) - 25} more")
        print("\nmissing-dependency list (for the environment owner):")
        seen = set()
        for name, detail in result["dependency"]:
            key = detail.split(":")[0]
            if key not in seen:
                seen.add(key)
                print(f"   {key}: {detail[:100]}")
        return 1 if result["defect"] else 0


class PiscesLxImportCensusCli:
    """Argument front end."""

    USAGE = "usage: python -m tools.check.import_census [--root PATH] [--verbose]"

    @classmethod
    def main(cls, argv) -> int:
        parser = argparse.ArgumentParser(prog="tools.check.import_census")
        parser.add_argument("--root", default=".")
        parser.add_argument("--verbose", action="store_true")
        args = parser.parse_args(argv)
        return PiscesLxImportCensus(pathlib.Path(args.root)).report(args.verbose)


if __name__ == "__main__":
    raise SystemExit(PiscesLxImportCensusCli.main(sys.argv[1:]))
