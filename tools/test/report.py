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

"""
Test Report Module for PiscesL1.

This module provides the PiscesLxTestReport class for collecting,
aggregating, and displaying test results with Cargo-style dynamic
progress visualization.

Features:
    - Real-time progress display with animated progress bar
    - Cargo-style compilation output
    - Status tracking (PASS, FAIL, WARN, SKIP)
    - Duration measurement for each test
    - Summary statistics generation
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import sys
import time


class PiscesLxTestStatus(Enum):
    """Test result status enumeration."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class PiscesLxTestResult:
    """Single test result data container."""
    stage: str
    name: str
    status: str
    message: str
    duration: float = 0.0


class PiscesLxTestReport:
    """
    Test report collector with Cargo-style dynamic display.
    
    This class collects test results and provides real-time
    progress visualization similar to Cargo's compilation output.
    
    Attributes:
        results: List of all test results
        start_time: Report creation timestamp
        total_tests: Expected total number of tests
        current_stage: Current stage being executed
    
    Example:
        >>> report = PiscesLxTestReport(total_tests=40)
        >>> report.start()
        >>> report.add_result("Stage 1", "Python", "PASS", "3.11.5", 0.01)
        >>> report.finish()
    """
    
    STAGE_NAMES = {
        "Stage 1": "Environment Check",
        "Stage 2": "Project Structure Check",
        "Stage 3": "Module Import Check",
        "Stage 4": "Configuration Check",
        "Stage 5": "Model Instantiation Check",
        "Stage 6": "Forward Pass Check",
        "Stage 7": "Generation Check",
        "Stage 8": "Optimization Check",
    }
    
    STATUS_SYMBOLS = {
        "PASS": "✓",
        "FAIL": "✗",
        "WARN": "⚠",
        "SKIP": "⊘",
    }
    
    STATUS_COLORS = {
        "PASS": "\033[92m",
        "FAIL": "\033[91m",
        "WARN": "\033[93m",
        "SKIP": "\033[90m",
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "CYAN": "\033[96m",
        "DIM": "\033[2m",
    }
    
    PROGRESS_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, total_tests: int = 40):
        """Initialize the test report."""
        self.results: List[PiscesLxTestResult] = []
        self.start_time: float = 0.0
        self.total_tests: int = total_tests
        self.current_stage: str = ""
        self._spinner_idx: int = 0
        self._progress_line_printed: bool = False
    
    def _color(self, status: str) -> str:
        """Get color code for status."""
        return self.STATUS_COLORS.get(status, "")
    
    def _reset(self) -> str:
        """Get reset color code."""
        return self.STATUS_COLORS["RESET"]
    
    def _spinner(self) -> str:
        """Get next spinner character."""
        char = self.PROGRESS_CHARS[self._spinner_idx]
        self._spinner_idx = (self._spinner_idx + 1) % len(self.PROGRESS_CHARS)
        return char
    
    def _clear_progress_line(self) -> None:
        """Clear the progress line."""
        if self._progress_line_printed:
            sys.stdout.write("\r\033[K")
            self._progress_line_printed = False
    
    def _print_progress(self, stage: str, name: str) -> None:
        """Print animated progress line."""
        completed = len(self.results)
        total = self.total_tests
        percent = int((completed / total) * 100) if total > 0 else 0
        
        bar_width = 20
        filled = int((completed / total) * bar_width) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        spinner = self._spinner()
        stage_name = self.STAGE_NAMES.get(stage, stage)
        
        progress_str = (
            f"\r{self._color('CYAN')}{spinner}{self._reset()} "
            f"{self._color('BOLD')}Checking{self._reset()} "
            f"{self._color('DIM')}{stage_name}{self._reset()} "
            f"[{bar}] {completed}/{total} ({percent}%)"
        )
        
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        self._progress_line_printed = True
    
    def start(self) -> None:
        """Start the test run with header."""
        self.start_time = time.time()
        
        print()
        print(f"{self._color('BOLD')}{'═' * 55}{self._reset()}")
        print(f"{self._color('BOLD')}        PiscesL1 Project Health Check{self._reset()}")
        print(f"{self._color('BOLD')}{'═' * 55}{self._reset()}")
        print()
    
    def add_result(
        self, 
        stage: str, 
        name: str, 
        status: str, 
        message: str = "", 
        duration: float = 0.0
    ) -> None:
        """
        Add a test result with real-time display.
        
        Args:
            stage: Stage identifier (e.g., "Stage 1")
            name: Test name (e.g., "Python Version")
            status: Result status (PASS, FAIL, WARN, SKIP)
            message: Additional message or details
            duration: Test execution time in seconds
        """
        result = PiscesLxTestResult(
            stage=stage,
            name=name,
            status=status,
            message=message,
            duration=duration
        )
        self.results.append(result)
        
        self._clear_progress_line()
        
        if stage != self.current_stage:
            self.current_stage = stage
            stage_name = self.STAGE_NAMES.get(stage, stage)
            print(f"\n{self._color('BOLD')}[{stage}]{self._reset()} {stage_name}")
        
        symbol = self.STATUS_SYMBOLS.get(status, "?")
        color = self._color(status)
        reset = self._reset()
        
        status_str = f"  {color}{symbol}{reset} {name}"
        if message:
            status_str += f" {self._color('DIM')}({message}){reset}"
        if duration > 0.01:
            status_str += f" {self._color('DIM')}{duration:.2f}s{reset}"
        
        print(status_str)
        
        self._print_progress(stage, name)
    
    def finish(self) -> None:
        """Finish the test run with summary."""
        self._clear_progress_line()
        
        summary = self.get_summary()
        
        print()
        print(f"{self._color('BOLD')}{'═' * 55}{self._reset()}")
        
        passed_str = f"{self._color('PASS')}{summary['passed']}/{summary['total']}{self._reset()} passed"
        if summary['failed'] > 0:
            passed_str += f", {self._color('FAIL')}{summary['failed']}{self._reset()} failed"
        if summary['warned'] > 0:
            passed_str += f", {self._color('WARN')}{summary['warned']}{self._reset()} warnings"
        
        print(f"Summary: {passed_str}")
        
        if summary['is_healthy']:
            print(f"Status: {self._color('PASS')}✓ Project Healthy{self._reset()}")
        else:
            print(f"Status: {self._color('FAIL')}✗ Issues Found{self._reset()}")
        
        print(f"Time: {self._color('CYAN')}{summary['duration']:.1f}s{self._reset()}")
        print(f"{self._color('BOLD')}{'═' * 55}{self._reset()}")
        print()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary containing test statistics
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        duration = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "skipped": skipped,
            "duration": duration,
            "is_healthy": failed == 0,
        }
    
    def is_passed(self) -> bool:
        """Check if all tests passed (no failures)."""
        return all(r.status != "FAIL" for r in self.results)
    
    def print_report(self) -> None:
        """Print formatted report to console (for backward compatibility)."""
        self.finish()
