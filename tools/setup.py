#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import os
import sys
import platform
import subprocess
import datetime

def _log(level, message):
    """
    A simple logging function that doesn't depend on utility modules.

    Args:
        level (str): Logging level, e.g., 'info', 'success', 'error', 'warning', 'debug'.
        message (str): The message to be logged.

    Returns:
        None
    """
    # Skip self-logging when invoked through manage.py to avoid duplicate logging
    if 'manage.py' in sys.argv[0]:
        return  

    # Get the current timestamp
    now = datetime.datetime.now()
    local_ts = now.strftime("%m%d %H:%M:%S")
    iso_ts = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Mapping of logging levels to emojis
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
    Log an info-level message.

    Args:
        message (str): The info message to be logged.

    Returns:
        None
    """
    _log('info', message)

def logger_success(message):
    """
    Log a success-level message.

    Args:
        message (str): The success message to be logged.

    Returns:
        None
    """
    _log('success', message)

def logger_error(message):
    """
    Log an error-level message.

    Args:
        message (str): The error message to be logged.

    Returns:
        None
    """
    _log('error', message)

def logger_warning(message):
    """
    Log a warning-level message.

    Args:
        message (str): The warning message to be logged.

    Returns:
        None
    """
    _log('warning', message)

def setup(args):
    """
    Automatically set up a virtual environment and install required packages if necessary,
    then automatically enter the virtual environment shell.

    Args:
        args: Command line arguments passed to the function.

    Returns:
        None
    """
    # Validate input arguments. Continue setup even if validation fails.
    try:
        args = validate_setup_args(args)
    except Exception as e:
        logger_error(f"Invalid setup arguments: {e}")
        return

    logger_info("Pisces auto environment setup...")

    # Determine the project root directory and virtual environment directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_dir = os.path.join(project_root, ".pisceslx", "env")
    os.makedirs(venv_dir, exist_ok=True)

    # Determine if the current operating system is Windows
    is_windows = platform.system().lower().startswith("win")

    # Check if the script is running outside a virtual environment
    if sys.prefix == sys.base_prefix:
        logger_info("Not in virtual environment. Creating venv...")
        # Create a new virtual environment
        python_executable = sys.executable
        subprocess.check_call([python_executable, "-m", "venv", venv_dir])
        logger_success(f"Virtual environment created at {venv_dir}")

        # Get the Python interpreter path within the virtual environment
        python_bin = os.path.join(venv_dir, "Scripts" if is_windows else "bin", "python" + (".exe" if is_windows else ""))
        logger_info("Re-running setup in venv...")
        # Re-run the script using the Python interpreter in the virtual environment
        os.execv(python_bin, [python_bin] + sys.argv)
        return
    else:
        logger_info("Already in virtual environment.")

    # Upgrade the pip package manager
    logger_info("Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        logger_success("Pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        logger_error(f"Failed to upgrade pip: {e}")
        return

    # Install packages from requirements.txt
    logger_info("Installing requirements.txt...")
    try:
        # Try to install all dependencies at once
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger_success("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger_warning("Failed to install all requirements, trying fallback approach...")

        # Read requirements.txt and install packages one by one, skipping failed ones
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
    Validate setup arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Validated arguments
    """
    return args