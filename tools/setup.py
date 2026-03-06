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
Environment Setup Module for PiscesL1.

This module provides automated environment setup functionality for the
PiscesL1 framework, including virtual environment creation, dependency
installation, and system configuration. It ensures a consistent and
reproducible development environment across different platforms.

Key Responsibilities:
    1. Virtual Environment Management:
       - Automatic detection of existing virtual environments
       - Creation of new virtual environments when needed
       - Cross-platform support (Windows, Linux, macOS)
       - Automatic re-execution within the virtual environment
    
    2. Dependency Installation:
       - Pip upgrade to latest version
       - Batch installation from requirements.txt
       - Fallback to individual package installation on failures
       - Detailed reporting of installation results
    
    3. Logging and Diagnostics:
       - Timestamped log messages with visual indicators
       - Success/failure tracking for each package
       - Summary of installation results

Directory Structure:
    The setup process creates the following structure:
    
    project_root/
    ├── .pisceslx/
    │   └── env/              # Virtual environment directory
    │       ├── bin/          # Unix executables
    │       │   ├── python
    │       │   └── pip
    │       ├── Scripts/      # Windows executables
    │       │   ├── python.exe
    │       │   └── pip.exe
    │       └── lib/          # Installed packages
    └── requirements.txt      # Dependency specification

Logging Format:
    All log messages follow a consistent format:
    
    MMDD HH:MM:SS | YYYY-MM-DDTHH:MM:SS.ffffffZ | EMOJI | [PiscesLx Core] | Message
    
    Log levels and their indicators:
    - info: 🟢 (green circle)
    - success: ✅ (check mark)
    - error: 🔴 (red circle)
    - warning: 🟡 (yellow circle)
    - debug: 🔵 (blue circle)

Platform Support:
    - Windows: Uses Scripts/ directory for executables, .exe extension
    - Linux: Uses bin/ directory for executables, no extension
    - macOS: Uses bin/ directory for executables, no extension

Usage Examples:
    Command-line usage:
        $ python manage.py setup
        
    Programmatic usage:
        from tools.setup import setup
        args = None  # or argparse.Namespace object
        setup(args)
    
    With custom arguments:
        from tools.setup import setup, validate_setup_args
        args = validate_setup_args(custom_args)
        setup(args)

Installation Flow:
    1. Validate input arguments
    2. Check if running in virtual environment
    3. If not in venv:
       a. Create virtual environment at .pisceslx/env
       b. Re-execute script within the new venv
    4. If in venv:
       a. Upgrade pip to latest version
       b. Attempt batch installation from requirements.txt
       c. On batch failure, install packages individually
       d. Report success/failure summary

Error Handling:
    The module handles various error conditions:
    - Virtual environment creation failures
    - Pip upgrade failures
    - Individual package installation failures
    - Missing requirements.txt file
    - Permission errors

Thread Safety:
    The setup function is not thread-safe and should only be called
    once during environment initialization. The os.execv call will
    replace the current process, terminating any concurrent operations.

Integration Points:
    - manage.py: Called via "python manage.py setup" command
    - tools/check.py: Validates environment after setup
    - requirements.txt: Source of dependency specifications

Note:
    The logging functions in this module are intentionally independent
    of the main logging infrastructure (utils/dc.py) to avoid circular
    dependencies during the setup process. Once setup is complete,
    the main logging system should be used for all other operations.
"""

import os
import sys
import platform
import subprocess
import datetime


def _log(level, message):
    """
    Internal logging function with timestamp and visual indicators.
    
    This function provides a self-contained logging mechanism that does not
    depend on the main logging infrastructure (utils/dc.py). This is essential
    during the setup phase when dependencies may not yet be installed.
    
    The function formats log messages with both local and ISO timestamps,
    a visual emoji indicator for the log level, and a consistent prefix.
    Messages are printed directly to stdout.
    
    When the script is invoked through manage.py, logging is suppressed to
    avoid duplicate output since manage.py handles its own logging.
    
    Args:
        level (str): The logging level indicator. Valid values are:
            - 'info': Informational message (🟢)
            - 'success': Successful operation (✅)
            - 'error': Error condition (🔴)
            - 'warning': Warning condition (🟡)
            - 'debug': Debug information (🔵)
            Any unrecognized level defaults to info (🟢).
        message (str): The log message content to display.
    
    Returns:
        None: This function prints to stdout and returns nothing.
    
    Side Effects:
        - Prints formatted log message to stdout
        - May suppress output when called from manage.py
    
    Example:
        >>> _log('info', 'Starting setup process')
        0125 14:30:00 | 2025-01-25T14:30:00.123456Z | 🟢 | [PiscesLx Core] | Starting setup process
        
        >>> _log('error', 'Installation failed')
        0125 14:30:01 | 2025-01-25T14:30:01.234567Z | 🔴 | [PiscesLx Core] | Installation failed
    """
    if 'manage.py' in sys.argv[0]:
        return

    now = datetime.datetime.now()
    local_ts = now.strftime("%m%d %H:%M:%S")
    iso_ts = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    level_emojis = {
        'info': '🟢',
        'success': '✅',
        'error': '🔴',
        'warning': '🟡',
        'debug': '🔵'
    }

    emoji = level_emojis.get(level, '🟢')
    print(f"{local_ts} | {iso_ts} | {emoji} | [PiscesLx Core] | {message}")


def logger_info(message):
    """
    Log an informational message.
    
    This function logs a message at the 'info' level, indicating normal
    operational progress. Use for routine status updates that don't
    indicate success, failure, or warning conditions.
    
    Args:
        message (str): The informational message to log.
    
    Returns:
        None
    
    Example:
        >>> logger_info("Checking virtual environment status")
        0125 14:30:00 | 2025-01-25T14:30:00.123456Z | 🟢 | [PiscesLx Core] | Checking virtual environment status
    """
    _log('info', message)


def logger_success(message):
    """
    Log a success message.
    
    This function logs a message at the 'success' level, indicating
    successful completion of an operation. Use when an operation
    completes without errors.
    
    Args:
        message (str): The success message to log.
    
    Returns:
        None
    
    Example:
        >>> logger_success("Virtual environment created successfully")
        0125 14:30:00 | 2025-01-25T14:30:00.123456Z | ✅ | [PiscesLx Core] | Virtual environment created successfully
    """
    _log('success', message)


def logger_error(message):
    """
    Log an error message.
    
    This function logs a message at the 'error' level, indicating
    a failure or error condition. Use when an operation fails or
    encounters an unrecoverable error.
    
    Args:
        message (str): The error message to log.
    
    Returns:
        None
    
    Example:
        >>> logger_error("Failed to create virtual environment")
        0125 14:30:00 | 2025-01-25T14:30:00.123456Z | 🔴 | [PiscesLx Core] | Failed to create virtual environment
    """
    _log('error', message)


def logger_warning(message):
    """
    Log a warning message.
    
    This function logs a message at the 'warning' level, indicating
    a non-critical issue or potential problem. Use when an operation
    completes but with caveats or potential issues.
    
    Args:
        message (str): The warning message to log.
    
    Returns:
        None
    
    Example:
        >>> logger_warning("Some packages failed to install")
        0125 14:30:00 | 2025-01-25T14:30:00.123456Z | 🟡 | [PiscesLx Core] | Some packages failed to install
    """
    _log('warning', message)


def setup(args):
    """
    Set up the PiscesL1 development environment automatically.
    
    This function orchestrates the complete environment setup process for
    the PiscesL1 framework. It handles virtual environment creation, pip
    upgrades, and dependency installation with robust error handling and
    fallback mechanisms.
    
    The setup process follows this flow:
    
    1. **Argument Validation**: Validates input arguments using
       validate_setup_args(). Continues even if validation fails.
    
    2. **Virtual Environment Check**: Determines if the script is already
       running within a virtual environment by comparing sys.prefix with
       sys.base_prefix.
    
    3. **Virtual Environment Creation** (if not in venv):
       - Creates directory structure at .pisceslx/env
       - Uses Python's venv module for creation
       - Re-executes the script within the new environment using os.execv
    
    4. **Pip Upgrade** (in venv):
       - Upgrades pip to the latest version
       - Reports success or failure
    
    5. **Dependency Installation**:
       - Attempts batch installation from requirements.txt
       - Falls back to individual package installation on batch failure
       - Tracks and reports success/failure for each package
    
    Virtual Environment Location:
        The virtual environment is created at:
        <project_root>/.pisceslx/env/
        
        This location keeps the environment separate from the source code
        while remaining within the project structure for easy management.
    
    Args:
        args: Command-line arguments or configuration object. This can be:
            - None: Default setup with no customization
            - argparse.Namespace: Parsed command-line arguments
            - dict: Dictionary of configuration options
            
            Currently, no specific arguments are used, but the parameter
            is reserved for future extensions such as:
            - python_version: Specify Python version for venv
            - requirements_path: Custom requirements file path
            - no_upgrade: Skip pip upgrade
    
    Returns:
        None: This function performs side effects only. The environment
        is configured in-place.
    
    Side Effects:
        - Creates .pisceslx/env/ directory structure
        - Creates or modifies virtual environment
        - Installs packages via pip
        - May replace current process via os.execv
    
    Raises:
        No exceptions are raised directly. All errors are logged and
        handled gracefully. The function returns early on critical errors.
    
    Example:
        Command-line usage:
            $ python manage.py setup
            
        Programmatic usage:
            >>> from tools.setup import setup
            >>> setup(None)
            🟢 Pisces auto environment setup...
            🟢 Not in virtual environment. Creating venv...
            ✅ Virtual environment created at /path/to/.pisceslx/env
            🟢 Re-running setup in venv...
            [Process restarts in venv]
            🟢 Already in virtual environment.
            🟢 Upgrading pip...
            ✅ Pip upgraded successfully
            🟢 Installing requirements.txt...
            ✅ Requirements installed successfully
    
    Note:
        When the function calls os.execv(), the current process is
        replaced. Any code after that call will not execute in the
        original process. The new process starts from the beginning
        of the script.
    """
    try:
        args = validate_setup_args(args)
    except Exception as e:
        logger_error(f"Invalid setup arguments: {e}")
        return

    logger_info("Pisces auto environment setup...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_dir = os.path.join(project_root, ".pisceslx", "env")
    os.makedirs(venv_dir, exist_ok=True)

    is_windows = platform.system().lower().startswith("win")

    if sys.prefix == sys.base_prefix:
        logger_info("Not in virtual environment. Creating venv...")
        python_executable = sys.executable
        subprocess.check_call([python_executable, "-m", "venv", venv_dir])
        logger_success(f"Virtual environment created at {venv_dir}")

        python_bin = os.path.join(venv_dir, "Scripts" if is_windows else "bin", "python" + (".exe" if is_windows else ""))
        logger_info("Re-running setup in venv...")
        os.execv(python_bin, [python_bin] + sys.argv)
        return
    else:
        logger_info("Already in virtual environment.")

    logger_info("Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        logger_success("Pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        logger_error(f"Failed to upgrade pip: {e}")
        return

    logger_info("Installing requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger_success("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger_warning("Failed to install all requirements, trying fallback approach...")

        requirements_path = os.path.join(project_root, "requirements.txt")

        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements = f.readlines()

            failed_packages = []
            installed_packages = []

            for line in requirements:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", line])
                    installed_packages.append(line)
                    logger_success(f"Installed: {line}")
                except subprocess.CalledProcessError:
                    failed_packages.append(line)
                    logger_warning(f"Failed to install: {line}")
            
            if installed_packages:
                logger_success(f"Successfully installed {len(installed_packages)} packages")
            if failed_packages:
                logger_warning(f"Failed to install {len(failed_packages)} packages: {failed_packages}")
                
        except FileNotFoundError:
            logger_error("requirements.txt not found")
            return
        except Exception as e:
            logger_error(f"Error reading requirements.txt: {e}")
            return


def validate_setup_args(args):
    """
    Validate and normalize setup arguments.
    
    This function ensures that the arguments passed to setup() are valid
    and in the expected format. Currently, it performs minimal validation
    as the setup function doesn't use specific arguments, but it provides
    a hook for future argument processing.
    
    The function is designed to be extensible for future argument options:
    - Custom requirements file path
    - Python version specification
    - Skip pip upgrade flag
    - Verbose output flag
    
    Args:
        args: Command-line arguments or configuration object. Can be:
            - None: No arguments provided
            - argparse.Namespace: Parsed command-line arguments
            - dict: Dictionary of configuration options
            - Any object: Passed through unchanged
    
    Returns:
        The validated arguments, currently returned unchanged. Future
        implementations may normalize or transform the arguments.
    
    Raises:
        Currently no exceptions are raised. Future implementations may
        raise ValueError for invalid argument combinations.
    
    Example:
        >>> validate_setup_args(None)
        None
        
        >>> args = argparse.Namespace(verbose=True)
        >>> validate_setup_args(args)
        Namespace(verbose=True)
    
    Note:
        This function is called automatically by setup() and typically
        does not need to be called directly by users.
    """
    return args
