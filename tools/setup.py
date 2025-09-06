#!/usr/bin/env/python3

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
import platform
import subprocess
from utils.log import RIGHT

def setup(args):
    """
    Automatically set up a virtual environment and install required packages if necessary,
    then automatically enter the virtual environment shell.

    Args:
        args: Command line arguments passed to the function.

    Returns:
        None
    """
    RIGHT("Pisces auto environment setup...")
    
    # Create modelscope cache directory if not exists
    modelscope_dir = os.path.join(os.getcwd(), "modelscope")
    if not os.path.exists(modelscope_dir):
        os.makedirs(modelscope_dir, exist_ok=True)
        RIGHT(f"Created ModelScope cache directory: {modelscope_dir}")
    else:
        RIGHT(f"ModelScope cache directory already exists: {modelscope_dir}")
    
    # Get the current Python interpreter path
    py_exec = sys.executable
    # Define the virtual environment directory path
    venv_dir = os.path.join(os.getcwd(), "pisces_env")
    # Check if the current operating system is Windows
    is_windows = platform.system().lower().startswith("win")

    # Check if the script is running outside a virtual environment
    if sys.prefix == sys.base_prefix:
        RIGHT("Not in virtual environment. Creating venv...")
        # Create a new virtual environment
        subprocess.check_call([py_exec, "-m", "venv", venv_dir])
        RIGHT(f"Virtual environment created at {venv_dir}")
        # Get the Python interpreter path in the virtual environment
        python_bin = os.path.join(venv_dir, "Scripts" if is_windows else "bin", "python" + (".exe" if is_windows else ""))
        RIGHT("Re-running setup in venv...")
        # Re-run the script using the Python interpreter in the virtual environment
        os.execv(python_bin, [python_bin] + sys.argv)
        return
    else:
        RIGHT("Already in virtual environment.")

    # Upgrade the pip package manager
    RIGHT("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install packages from requirements.txt
    RIGHT("Installing requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    RIGHT("Pisces environment setup complete!")

    # Automatically enter the virtual environment shell
    if is_windows:
        # Get the Windows command shell path
        shell = os.environ.get("COMSPEC", "cmd.exe")
        # Check if the current shell is PowerShell
        if "powershell.exe" in shell.lower() or "pwsh.exe" in shell.lower():
            activate = os.path.join(venv_dir, "Scripts", "Activate.ps1")
            RIGHT("Auto-entering Pisces venv shell (PowerShell)...")
            # Enter the virtual environment using PowerShell
            os.execv(shell, [shell, "-NoExit", "-Command", f". '{activate}'"])
        else:
            activate = os.path.join(venv_dir, "Scripts", "activate.bat")
            RIGHT("Auto-entering Pisces venv shell (Windows cmd)...")
            # Enter the virtual environment using Windows cmd
            os.execv(shell, [shell, "/K", activate])
    else:
        # Get the Unix-like shell path
        shell = os.environ.get("SHELL", "/bin/bash")
        activate = os.path.join(venv_dir, "bin", "activate")
        RIGHT("Auto-entering Pisces venv shell (Linux/Mac)...")
        # Enter the virtual environment using Unix-like shell
        os.execv(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell}"])