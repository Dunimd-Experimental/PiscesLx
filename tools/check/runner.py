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
Check-suite orchestration: run every mandatory standard, emit machine-readable
results, and decide pass/fail for CI and acceptance.
"""

import json
import time
from pathlib import Path

from tools.check.base import (
    PiscesLxCheckReport,
    PiscesLxRepositoryScanner,
    PiscesLxRuleId,
    PiscesLxSeverity,
)
from tools.check.checks_exports import PiscesLxDanglingExportCheck
from tools.check.checks_errors import PiscesLxSilentExceptCheck
from tools.check.checks_hygiene import (
    PiscesLxDanglingImportCheck,
    PiscesLxEngineIsolationCheck,
    PiscesLxHardcodingCheck,
    PiscesLxModalityCheck,
    PiscesLxResidualSystemCheck,
)
from tools.check.checks_style import (
    PiscesLxClassNamingCheck,
    PiscesLxCopyrightHeaderCheck,
    PiscesLxIncompleteMarkerCheck,
    PiscesLxProjectNameCheck,
    PiscesLxStdlibLoggingCheck,
    PiscesLxTopLevelFunctionCheck,
)

__all__ = ["PiscesLxCheckCatalog", "PiscesLxCheckSuite", "PiscesLxCheckCli"]


class PiscesLxCheckCatalog:
    """The registry mapping every prohibition to its executable check."""

    _CHECKS = (
        PiscesLxClassNamingCheck,
        PiscesLxTopLevelFunctionCheck,
        PiscesLxSilentExceptCheck,
        PiscesLxStdlibLoggingCheck,
        PiscesLxCopyrightHeaderCheck,
        PiscesLxProjectNameCheck,
        PiscesLxHardcodingCheck,
        PiscesLxResidualSystemCheck,
        PiscesLxModalityCheck,
        PiscesLxEngineIsolationCheck,
        PiscesLxDanglingImportCheck,
        PiscesLxDanglingExportCheck,
        PiscesLxIncompleteMarkerCheck,
    )

    @classmethod
    def instantiate(cls) -> list:
        """Fresh instances of every registered check."""
        return [checker() for checker in cls._CHECKS]

    @classmethod
    def rule_ids(cls) -> tuple:
        """Rule identifiers covered by the suite."""
        return tuple(sorted({checker.rule for checker in cls._CHECKS}))

    @classmethod
    def coverage(cls) -> dict:
        """Declared rules versus rules with a live checker — gaps are visible."""
        declared = set(PiscesLxRuleId.all_rules())
        covered = set()
        for checker in cls._CHECKS:
            covered.add(checker.rule)
        for violation_rule in (
            "NAMING-OUTSIDE-PISCESLX",
            "NAMING-FORBIDDEN-PREFIX",
            "NO-HARDCODED-DEVICE",
            "NO-HARDCODED-MODEL-ID",
            "NO-MAGIC-NUMBER",
            "CLEAN-WATERMARK",
            "CLEAN-MCP-SERVICE-LAYER",
            "MODALITY-4-NO-AGENT",
        ):
            covered.add(violation_rule)
        return {
            "declared": sorted(declared),
            "covered": sorted(covered),
            "uncovered": sorted(declared - covered),
        }


class PiscesLxCheckSuite:
    """Executes the mandated checks over a repository and produces a report."""

    __slots__ = ("root", "paths", "_scanner")

    def __init__(self, root: Path, paths=()) -> None:
        self.root = Path(root).resolve()
        self.paths = tuple(paths)
        self._scanner = None

    @property
    def scanner(self) -> PiscesLxRepositoryScanner:
        """Lazy repository scanner bound to this suite's root."""
        if self._scanner is None:
            self._scanner = PiscesLxRepositoryScanner(self.root, self.paths)
        return self._scanner

    def execute(self) -> PiscesLxCheckReport:
        """Run every check and aggregate violations into a report."""
        started = time.time()
        violations = []
        checks = PiscesLxCheckCatalog.instantiate()
        for checker in checks:
            violations.extend(checker.run(self.scanner))
        violations.sort(key=lambda v: (v.rule, str(v.path), v.line))
        return PiscesLxCheckReport(
            root=self.root,
            violations=violations,
            checks_run=len(checks),
            files_scanned=len(self.scanner.paths),
            duration_s=time.time() - started,
        )

    def to_json(self, report: PiscesLxCheckReport) -> str:
        """Serialize the report for CI artifacts."""
        payload = {
            "repository": self.root.as_posix(),
            "files_scanned": report.files_scanned,
            "checks_run": report.checks_run,
            "duration_s": round(report.duration_s, 3),
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "by_rule": {
                rule: [
                    {
                        "path": v.path.relative_to(self.root).as_posix(),
                        "line": v.line,
                        "message": v.message,
                        "severity": v.severity,
                    }
                    for v in items
                ]
                for rule, items in report.by_rule().items()
            },
            "coverage": PiscesLxCheckCatalog.coverage(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class PiscesLxCheckCli:
    """Command-line front end so acceptance is decided by script output."""

    USAGE = (
        "usage: python -m tools.check [options]\n"
        "  --root PATH          repository root (default: parent of tools/)\n"
        "  --paths P [P ...]    limit the scan to specific files/directories\n"
        "  --json OUT           also write a machine-readable report\n"
        "  --max-rows N         per-rule detail cap (0 = unlimited, default)\n"
        "  --quiet              print only the summary counts\n"
    )

    _DEFAULT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, argv) -> None:
        self.argv = list(argv)
        self.root = self._DEFAULT_ROOT
        self.paths = []
        self.json_output = None
        self.max_rows = 0
        self.quiet = False

    def parse(self) -> list:
        """Parse argv; returns unknown tokens so the caller can surface them."""
        unknown = []
        index = 0
        while index < len(self.argv):
            token = self.argv[index]
            if token == "--root" and index + 1 < len(self.argv):
                self.root = Path(self.argv[index + 1])
                index += 2
            elif token == "--paths":
                index += 1
                while index < len(self.argv) and not self.argv[index].startswith("--"):
                    self.paths.append(Path(self.argv[index]))
                    index += 1
            elif token == "--json" and index + 1 < len(self.argv):
                self.json_output = Path(self.argv[index + 1])
                index += 2
            elif token == "--max-rows" and index + 1 < len(self.argv):
                self.max_rows = int(self.argv[index + 1])
                index += 2
            elif token == "--quiet":
                self.quiet = True
                index += 1
            elif token in ("-h", "--help"):
                print(self.USAGE)
                index += 1
            else:
                unknown.append(token)
                index += 1
        return unknown

    def main(self) -> int:
        """Execute the suite and return a process exit code."""
        unknown = self.parse()
        if unknown:
            print(f"unrecognized argument(s): {unknown}\n{self.USAGE}")
            return 2
        suite = PiscesLxCheckSuite(self.root, self.paths)
        report = suite.execute()
        if self.json_output is not None:
            self.json_output.parent.mkdir(parents=True, exist_ok=True)
            self.json_output.write_text(suite.to_json(report), encoding="utf-8")
        if self.quiet:
            print(
                f"files={report.files_scanned} checks={report.checks_run} "
                f"errors={len(report.errors)} warnings={len(report.warnings)}"
            )
        else:
            print(report.render(max_rows=self.max_rows))
        coverage = PiscesLxCheckCatalog.coverage()
        if coverage["uncovered"]:
            print(f"\nWARNING: declared rules without an executable check: {coverage['uncovered']}")
        return report.exit_code
