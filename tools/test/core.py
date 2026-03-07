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
Test Runner Core Module for PiscesL1.

This module provides the PiscesLxTestRunner class that orchestrates
all 8 stages of the project health check system with Cargo-style
dynamic progress visualization.

The runner coordinates:
    - Stage execution in order
    - Real-time progress display
    - Result collection and aggregation
    - Report generation
    - Quick check vs full check modes
"""

import os
import time
from typing import List, Optional, Set
from pathlib import Path

from .report import PiscesLxTestReport
from .stages import (
    PiscesLxEnvironmentChecker,
    PiscesLxStructureChecker,
    PiscesLxImportChecker,
    PiscesLxConfigChecker,
    PiscesLxModelChecker,
    PiscesLxForwardChecker,
    PiscesLxGenerationChecker,
    PiscesLxOptimizationChecker,
)


class PiscesLxTestRunner:
    """
    Main test runner for PiscesL1 project health checks.
    
    This class orchestrates all 8 stages of the health check system,
    collecting results and generating reports with real-time progress.
    
    Attributes:
        root_path: Project root directory
        config_name: Model configuration name
        verbose: Enable verbose output
        stages: List of stage numbers to run
    
    Example:
        >>> runner = PiscesLxTestRunner(config_name="7B")
        >>> report = runner.run_full_check()
        >>> report.finish()
    """
    
    STAGE_NAMES = {
        1: "Stage 1",
        2: "Stage 2",
        3: "Stage 3",
        4: "Stage 4",
        5: "Stage 5",
        6: "Stage 6",
        7: "Stage 7",
        8: "Stage 8",
    }
    
    QUICK_STAGES = {1, 2, 3, 4, 5}
    
    ESTIMATED_TESTS_PER_STAGE = {
        1: 6,
        2: 3,
        3: 7,
        4: 2,
        5: 4,
        6: 3,
        7: 3,
        8: 4,
    }
    
    def __init__(
        self, 
        root_path: str = None, 
        config_name: str = "7B", 
        verbose: bool = False,
        stages: Set[int] = None
    ):
        """
        Initialize the test runner.
        
        Args:
            root_path: Project root directory (defaults to cwd)
            config_name: Model configuration name (e.g., "7B")
            verbose: Enable verbose output
            stages: Specific stages to run (None = all stages)
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.config_name = config_name
        self.verbose = verbose
        self.stages = stages
    
    def run_full_check(self) -> PiscesLxTestReport:
        """
        Run all 8 stages of the health check.
        
        Returns:
            PiscesLxTestReport containing all results
        """
        return self._run_stages(stages={1, 2, 3, 4, 5, 6, 7, 8})
    
    def run_quick_check(self) -> PiscesLxTestReport:
        """
        Run quick check (stages 1-5 only).
        
        Returns:
            PiscesLxTestReport containing results from stages 1-5
        """
        return self._run_stages(stages=self.QUICK_STAGES)
    
    def _estimate_total_tests(self, stages: Set[int]) -> int:
        """Estimate total number of tests for progress bar."""
        return sum(self.ESTIMATED_TESTS_PER_STAGE.get(s, 3) for s in stages)
    
    def _run_stages(self, stages: Set[int]) -> PiscesLxTestReport:
        """
        Run specified stages with real-time progress display.
        
        Args:
            stages: Set of stage numbers to run
        
        Returns:
            PiscesLxTestReport containing all results
        """
        total_tests = self._estimate_total_tests(stages)
        report = PiscesLxTestReport(total_tests=total_tests)
        report.start()
        
        if 1 in stages:
            self._run_stage_1(report)
        
        if 2 in stages:
            self._run_stage_2(report)
        
        if 3 in stages:
            self._run_stage_3(report)
        
        if 4 in stages:
            self._run_stage_4(report)
        
        if 5 in stages:
            self._run_stage_5(report)
        
        if 6 in stages:
            self._run_stage_6(report)
        
        if 7 in stages:
            self._run_stage_7(report)
        
        if 8 in stages:
            self._run_stage_8(report)
        
        report.finish()
        return report
    
    def _run_stage_1(self, report: PiscesLxTestReport) -> None:
        """Run Stage 1: Environment Check."""
        checker = PiscesLxEnvironmentChecker(verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 1", name, status, message, duration)
    
    def _run_stage_2(self, report: PiscesLxTestReport) -> None:
        """Run Stage 2: Project Structure Check."""
        checker = PiscesLxStructureChecker(root_path=str(self.root_path), verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 2", name, status, message, duration)
    
    def _run_stage_3(self, report: PiscesLxTestReport) -> None:
        """Run Stage 3: Module Import Check."""
        checker = PiscesLxImportChecker(verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 3", name, status, message, duration)
    
    def _run_stage_4(self, report: PiscesLxTestReport) -> None:
        """Run Stage 4: Configuration Check."""
        checker = PiscesLxConfigChecker(
            root_path=str(self.root_path), 
            config_name=self.config_name,
            verbose=self.verbose
        )
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 4", name, status, message, duration)
    
    def _run_stage_5(self, report: PiscesLxTestReport) -> None:
        """Run Stage 5: Model Instantiation Check."""
        checker = PiscesLxModelChecker(config_name=self.config_name, verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 5", name, status, message, duration)
    
    def _run_stage_6(self, report: PiscesLxTestReport) -> None:
        """Run Stage 6: Forward Pass Check."""
        checker = PiscesLxForwardChecker(verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 6", name, status, message, duration)
    
    def _run_stage_7(self, report: PiscesLxTestReport) -> None:
        """Run Stage 7: Generation Check."""
        checker = PiscesLxGenerationChecker(verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 7", name, status, message, duration)
    
    def _run_stage_8(self, report: PiscesLxTestReport) -> None:
        """Run Stage 8: Optimization Check."""
        checker = PiscesLxOptimizationChecker(verbose=self.verbose)
        results = checker.run()
        for name, status, message, duration in results:
            report.add_result("Stage 8", name, status, message, duration)
