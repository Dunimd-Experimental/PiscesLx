#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import platform
import subprocess

def setup(args):
    """Auto setup venv and install requirements if needed, then auto-enter venv shell"""
    print("✅\tPisces auto environment setup...")
    py_exec = sys.executable
    venv_dir = os.path.join(os.getcwd(), "pisces_env")
    is_windows = platform.system().lower().startswith("win")
    # Check if in venv
    if sys.prefix == sys.base_prefix:
        print("✅\tNot in virtual environment. Creating venv...")
        subprocess.check_call([py_exec, "-m", "venv", venv_dir])
        print(f"✅\tVirtual environment created at {venv_dir}")
        # Re-run in venv python
        python_bin = os.path.join(venv_dir, "Scripts" if is_windows else "bin", "python" + (".exe" if is_windows else ""))
        print("✅\tRe-running setup in venv...")
        os.execv(python_bin, [python_bin] + sys.argv)
        return
    else:
        print("✅\tAlready in virtual environment.")
    # Upgrade pip
    print("✅\tUpgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    # Install requirements
    print("✅\tInstalling requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅\tPisces environment setup complete!")
    # Auto enter venv shell
    if is_windows:
        shell = os.environ.get("COMSPEC", "cmd.exe")
        # Check if using PowerShell
        if "powershell.exe" in shell.lower() or "pwsh.exe" in shell.lower():
            activate = os.path.join(venv_dir, "Scripts", "Activate.ps1")
            print("✅\tAuto-entering Pisces venv shell (PowerShell)...")
            os.execv(shell, [shell, "-NoExit", "-Command", f". '{activate}'"])
        else:
            activate = os.path.join(venv_dir, "Scripts", "activate.bat")
            print("✅\tAuto-entering Pisces venv shell (Windows cmd)...")
            os.execv(shell, [shell, "/K", activate])
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        activate = os.path.join(venv_dir, "bin", "activate")
        print("✅\tAuto-entering Pisces venv shell (Linux/Mac)...")
        os.execv(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell}"])