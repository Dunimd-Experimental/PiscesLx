#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import os
import sys
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.data.download")

def source():
    """
    Automatically enter the Pisces virtual environment shell.

    This function determines the current operating system and then executes the appropriate
    command to activate the virtual environment in the corresponding shell.
    On Windows, it supports both PowerShell and cmd. On other systems, it uses the default shell.
    """
    # Determine if the current operating system is Windows
    is_windows = os.name == 'nt'
    # Get the path to the virtual environment directory
    venv_dir = os.path.join(os.path.dirname(__file__), '..', 'venv')
    
    if is_windows:
        # Get the Windows command shell path, default to cmd.exe
        shell = os.environ.get("COMSPEC", "cmd.exe")
        # Check if the current shell is PowerShell
        if "powershell.exe" in shell.lower() or "pwsh.exe" in shell.lower():
            # Get the PowerShell activation script path
            activate = os.path.join(venv_dir, "Scripts", "Activate.ps1")
            logger.info("Auto-entering Pisces venv shell (PowerShell)...")
            os.execv(shell, [shell, "-NoExit", "-Command", f". '{activate}'"])
        else:
            # Get the cmd activation script path
            activate = os.path.join(venv_dir, "Scripts", "activate.bat")
            logger.info("Auto-entering Pisces venv shell (Windows cmd)...")
            os.execv(shell, [shell, "/K", activate])
    else:
        # Get the Unix-like system shell path, default to /bin/bash
        shell = os.environ.get("SHELL", "/bin/bash")
        # Get the Unix-like system activation script path
        activate = os.path.join(venv_dir, "bin", "activate")
        logger.info("Auto-entering Pisces venv shell (Linux/Mac)...")
        os.execv(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell}"])