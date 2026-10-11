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

"""Host memory budget enforcement.

A training run on a workstation must not silently consume the machine: on this project's
CPU-only host the failure mode of an over-committed build was a native allocation failure
(SIGSEGV), which kills the process without a Python exception and without flushing any
buffered log. That is undiagnosable after the fact and unacceptable while the machine is
being used for other work.

This module measures the two quantities that decide whether a step is safe -- the commit
charge still available from the operating system, and this process's own committed
footprint -- and turns a declared budget into an enforced ceiling: :meth:`guard` raises
before the allocator can be pushed past it, so an over-budget configuration fails as a
reported error at the stage that caused it.
"""

import ctypes
import platform
from typing import Any, Dict, Optional

BYTES_PER_GB = 1024 ** 3
#: procfs reports memory in kibibytes; the counters above are byte-denominated.
BYTES_PER_KIB = 1024.0
#: A run is refused when less than this much commit charge is left, even if the declared
#: budget is larger: below it the operating system itself starts failing to allocate.
MINIMUM_USABLE_COMMIT_GB = 0.5
#: Commit charge a model build costs per stored parameter, measured rather than assumed:
#: constructing the 224.7M-parameter host tier committed 6.81 GB (docs/audit/00 §13), i.e.
#: 30.3 B/param. The multiple over the stored weights is transient fp32 and block machinery,
#: which is why a build gate must scale with the parameter count and not with the ceiling.
BYTES_PER_PARAMETER_AT_BUILD = 30.3

class PiscesLxHostMemoryBudgetError(RuntimeError):
    """Raised when a measured footprint or remaining commit charge breaches the budget."""


class _PiscesLxWindowsCommitProbe:
    """Windows counters: free commit charge and this process's private bytes."""

    class _PiscesLxMemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    class _PiscesLxProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    @classmethod
    def available_commit_gb(cls) -> float:
        """Commit charge the operating system can still hand out, in GB."""
        status = cls._PiscesLxMemoryStatusEx()
        status.dwLength = ctypes.sizeof(cls._PiscesLxMemoryStatusEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return float(status.ullAvailPageFile) / BYTES_PER_GB

    @classmethod
    def process_footprint_gb(cls) -> float:
        """This process's committed private bytes in GB (peak, so a spike is not missed)."""
        counters = cls._PiscesLxProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(cls._PiscesLxProcessMemoryCountersEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return float(max(counters.PeakPagefileUsage, counters.PagefileUsage)) / BYTES_PER_GB


class _PiscesLxProcfsCommitProbe:
    """Linux probe: MemAvailable (+SwapFree) and VmHWM from procfs."""

    @staticmethod
    def _read(keywords: Dict[str, str]) -> Optional[int]:
        try:
            with open(keywords["path"], "r", encoding="ascii") as handle:
                text = handle.read()
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith(keywords["field"]):
                return int(line.split()[1])
        return None

    @classmethod
    def available_commit_gb(cls) -> float:
        """Usable memory the kernel reports, counting free swap, in GB."""
        available = cls._read({"path": "/proc/meminfo", "field": "MemAvailable:"})
        if available is None:
            return float("inf")
        swap = cls._read({"path": "/proc/meminfo", "field": "SwapFree:"}) or 0
        return (available + swap) * BYTES_PER_KIB / BYTES_PER_GB

    @classmethod
    def process_footprint_gb(cls) -> float:
        """Peak resident set size of this process in GB."""
        peak = cls._read({"path": "/proc/self/status", "field": "VmHWM:"})
        if peak is None:
            return 0.0
        return peak * BYTES_PER_KIB / BYTES_PER_GB


class PiscesLxHostMemoryBudget:
    """Enforced ceiling on this process's host memory, measured against the OS.

    Example:
        >>> budget = PiscesLxHostMemoryBudget(budget_gb=5.0)
        >>> budget.require_headroom(required_gb=1.0, stage="model build")
        >>> for step in range(100):
        ...     budget.guard(stage=f"step {step}")
    """

    def __init__(self, budget_gb: Optional[float] = None) -> None:
        """
        Args:
            budget_gb: Maximum GB this process may commit. ``None`` disables enforcement
                (measurements stay available), which is the default on accelerator hosts
                where the device, not the host, is the binding constraint.
        """
        if budget_gb is not None and budget_gb <= 0:
            raise ValueError(f"memory_budget_gb must be positive, got {budget_gb}")
        self.budget_gb = float(budget_gb) if budget_gb is not None else None
        self._probe = (_PiscesLxWindowsCommitProbe if platform.system() == "Windows"
                       else _PiscesLxProcfsCommitProbe)

    def enabled(self) -> bool:
        """Whether a ceiling was declared."""
        return self.budget_gb is not None

    def available_commit_gb(self) -> float:
        """Commit charge the operating system can still provide, in GB."""
        return self._probe.available_commit_gb()

    def process_footprint_gb(self) -> float:
        """This process's committed footprint in GB (peak, where the platform reports one)."""
        return self._probe.process_footprint_gb()

    def summary(self) -> Dict[str, float]:
        """Measured quantities as a flat dict, for logs and checkpoint metadata."""
        return {
            "budget_gb": self.budget_gb if self.budget_gb is not None else float("inf"),
            "available_commit_gb": self.available_commit_gb(),
            "process_footprint_gb": self.process_footprint_gb(),
        }

    def require_headroom(self, required_gb: float, stage: str) -> None:
        """Refuse to start a stage whose measured cost cannot fit in what is left.

        Args:
            required_gb: Expected additional commit for the stage.
            stage: Human-readable stage name used in the reported error.

        Raises:
            PiscesLxHostMemoryBudgetError: when the remaining commit charge cannot cover
                ``required_gb`` within the declared budget.
        """
        available = self.available_commit_gb()
        limit = available if self.budget_gb is None else min(available, self.budget_gb)
        if limit < max(required_gb, MINIMUM_USABLE_COMMIT_GB):
            # Three decimals on purpose: two quantities that disagree by a few megabytes both
            # print as "2.60" at two decimals, which reads as a self-contradiction
            # ("it needs 2.60 but only 2.60 is usable") instead of a shortfall.
            raise PiscesLxHostMemoryBudgetError(
                f"refusing {stage}: it needs {required_gb:.3f} GB but only "
                f"{limit:.3f} GB is usable (host free commit {available:.3f} GB, "
                f"declared budget {self.budget_gb})")

    def guard(self, stage: str) -> float:
        """Check this process against the ceiling; raise on breach.

        Args:
            stage: Name reported with the measurement when the ceiling is breached.

        Returns:
            The measured process footprint in GB.
        """
        footprint = self.process_footprint_gb()
        if self.budget_gb is not None and footprint > self.budget_gb:
            raise PiscesLxHostMemoryBudgetError(
                f"{stage}: process footprint {footprint:.2f} GB exceeds the declared budget "
                f"{self.budget_gb:.2f} GB; aborting instead of over-committing the host")
        return footprint
