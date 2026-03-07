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
Test Command Entry Point for PiscesL1.

This module provides the test() function that serves as the entry point
for the `python manage.py test` command.

Usage:
    python manage.py test                    # Full check (all 8 stages)
    python manage.py test --quick            # Quick check (stages 1-5)
    python manage.py test --config 7B        # Specify model config
    python manage.py test --verbose          # Verbose output
    python manage.py test --stage 1,2,3      # Run specific stages
"""

import os
import sys
from pathlib import Path
from typing import Optional, Set

from .core import PiscesLxTestRunner
from .report import PiscesLxTestReport


def test(args=None, extra=None) -> bool:
    """
    Entry point for the test command.
    
    This function is called by manage.py when the user runs
    `python manage.py test`.
    
    Args:
        args: argparse namespace containing:
            - quick: bool - Run quick check (stages 1-5)
            - config: str - Model configuration name
            - verbose: bool - Enable verbose output
            - stage: str - Comma-separated stage numbers
        extra: Additional arguments (unused)
    
    Returns:
        bool: True if all tests passed, False otherwise
    
    Example:
        >>> from tools.test import test
        >>> success = test(args)
    """
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    quick = getattr(args, 'quick', False) if args else False
    config_arg = getattr(args, 'config', '7B') if args else '7B'
    verbose = getattr(args, 'verbose', False) if args else False
    stage_str = getattr(args, 'stage', None) if args else None
    
    # Extract config name from path if full path is provided
    if config_arg and ('/' in config_arg or '\\' in config_arg):
        config_name = Path(config_arg).stem
    else:
        config_name = config_arg
    
    stages = None
    if stage_str:
        try:
            stages = set(int(s.strip()) for s in stage_str.split(','))
        except ValueError:
            print(f"Invalid stage specification: {stage_str}")
            stages = None
    
    runner = PiscesLxTestRunner(
        root_path=root_path,
        config_name=config_name,
        verbose=verbose,
        stages=stages
    )
    
    if stages:
        report = runner._run_stages(stages)
    elif quick:
        report = runner.run_quick_check()
    else:
        report = runner.run_full_check()
    
    return report.is_passed()


def validate_test_args(args=None, extra=None) -> None:
    """
    Validate arguments for the test command.
    
    Args:
        args: argparse namespace
        extra: Additional arguments
    
    Raises:
        ValueError: If arguments are invalid
    """
    if args is not None and not isinstance(args, (dict, object)):
        raise ValueError("args must be None or an object-like container")
    if extra is not None and not isinstance(extra, (dict, str)):
        raise ValueError("extra must be None, dict or str")
