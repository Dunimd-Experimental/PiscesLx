#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
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
from utils.log import RIGHT

def source():
    """
    Auto-enter Pisces virtual environment shell.
    """
    is_windows = os.name == 'nt'
    venv_dir = os.path.join(os.path.dirname(__file__), '..', 'venv')
    
    if is_windows:
        shell = os.environ.get("COMSPEC", "cmd.exe")
        # Check if using PowerShell
        if "powershell.exe" in shell.lower() or "pwsh.exe" in shell.lower():
            activate = os.path.join(venv_dir, "Scripts", "Activate.ps1")
            RIGHT("Auto-entering Pisces venv shell (PowerShell)...")
            os.execv(shell, [shell, "-NoExit", "-Command", f". '{activate}'"])
        else:
            activate = os.path.join(venv_dir, "Scripts", "activate.bat")
            RIGHT("Auto-entering Pisces venv shell (Windows cmd)...")
            os.execv(shell, [shell, "/K", activate])
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        activate = os.path.join(venv_dir, "bin", "activate")
        RIGHT("Auto-entering Pisces venv shell (Linux/Mac)...")
        os.execv(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell}"])