#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Environment Setup for Welcome Screen

Handles environment setup operations including:
- Virtual environment creation
- Dependencies installation
- Directory structure creation
- Setup verification
"""

import os
import sys
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator

from ....core.dc import XiLogger
from ....config import get_xi_config, XiConfig


@dataclass
class SetupResult:
    """Result of a single setup step."""
    step: str
    success: bool
    message: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


class PiscesL1WelcomeSetup:
    """
    Environment setup handler for Xi Studio Welcome screen.
    
    Performs setup operations:
    1. Virtual environment creation
    2. Dependencies installation
    3. Directory structure creation
    4. Setup verification
    """
    
    STEP_LABELS: Dict[str, str] = {
        "venv_create": "Virtual Environment",
        "install_deps": "Installing Dependencies",
        "create_dirs": "Creating Directories",
        "verify_setup": "Verifying Setup",
    }
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = XiLogger("Xi.WelcomeSetup", enable_file=True)
        self._config: Optional[XiConfig] = None
    
    def get_step_label(self, step: str) -> str:
        """Get human-readable label for a setup step."""
        return self.STEP_LABELS.get(step, step)
    
    async def setup_all(self) -> AsyncGenerator[SetupResult, None]:
        """
        Run all setup steps and yield results.
        
        Yields:
            SetupResult for each setup step
        """
        all_success = True
        
        result = await self.setup_venv()
        yield result
        if not result.success:
            all_success = False
        
        result = await self.install_dependencies()
        yield result
        if not result.success:
            all_success = False
        
        result = await self.create_directories()
        yield result
        if not result.success:
            all_success = False
        
        result = await self.verify_setup()
        yield result
        if not result.success:
            all_success = False
    
    async def setup_venv(self) -> SetupResult:
        """
        Create virtual environment if configured.
        
        Checks:
        - Virtual environment configuration
        - Existing virtual environment
        - Creation if needed
        """
        step = "venv_create"
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            env_config = self._config.environment
            
            if not env_config.virtualenv or not env_config.virtualenv.enabled:
                return SetupResult(
                    step=step,
                    success=True,
                    message="Virtual environment not configured",
                    data={"enabled": False}
                )
            
            venv_path = self.project_root / env_config.virtualenv.path
            venv_data = {
                "enabled": True,
                "path": str(venv_path),
                "exists": venv_path.exists(),
            }
            
            if venv_path.exists():
                python_exe = self._get_venv_python(venv_path)
                if python_exe and python_exe.exists():
                    venv_data["python"] = str(python_exe)
                    return SetupResult(
                        step=step,
                        success=True,
                        message=f"Virtual environment exists at {env_config.virtualenv.path}",
                        data=venv_data
                    )
            
            if not env_config.virtualenv.create_if_missing:
                return SetupResult(
                    step=step,
                    success=False,
                    error="Virtual environment not found and auto-creation disabled",
                    message=f"Virtual environment not found at {env_config.virtualenv.path}",
                    data=venv_data
                )
            
            self.logger.info(f"Creating virtual environment at {venv_path}", event="xi.setup.venv.create")
            
            create_result = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if create_result.returncode != 0:
                error_msg = create_result.stderr or "Unknown error"
                return SetupResult(
                    step=step,
                    success=False,
                    error=f"Failed to create venv: {error_msg}",
                    message=f"Virtual environment creation failed",
                    data=venv_data
                )
            
            python_exe = self._get_venv_python(venv_path)
            if python_exe:
                venv_data["python"] = str(python_exe)
            
            return SetupResult(
                step=step,
                success=True,
                message=f"Virtual environment created at {env_config.virtualenv.path}",
                data=venv_data
            )
            
        except subprocess.TimeoutExpired:
            return SetupResult(
                step=step,
                success=False,
                error="Virtual environment creation timed out",
                message="Operation timed out after 300 seconds"
            )
        except Exception as e:
            return SetupResult(
                step=step,
                success=False,
                error=str(e),
                message=f"Failed to setup virtual environment: {e}"
            )
    
    async def install_dependencies(self) -> SetupResult:
        """
        Install dependencies from requirements files.
        
        Checks:
        - Requirements files existence
        - pip availability
        - Installation process
        """
        step = "install_deps"
        warnings = []
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            env_config = self._config.environment
            requirements = env_config.requirements
            
            if not requirements:
                return SetupResult(
                    step=step,
                    success=True,
                    message="No requirements files configured",
                    data={"installed": [], "total": 0}
                )
            
            python_exe = sys.executable
            if env_config.virtualenv and env_config.virtualenv.enabled:
                venv_path = self.project_root / env_config.virtualenv.path
                venv_python = self._get_venv_python(venv_path)
                if venv_python and venv_python.exists():
                    python_exe = str(venv_python)
            
            installed = []
            failed = []
            
            for req in requirements:
                req_path = self.project_root / req.path
                
                if not req_path.exists():
                    if req.required:
                        warnings.append(f"Required file not found: {req.path}")
                    continue
                
                self.logger.info(f"Installing dependencies from {req.path}", event="xi.setup.deps.install")
                
                install_result = subprocess.run(
                    [python_exe, "-m", "pip", "install", "-r", str(req_path)],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if install_result.returncode != 0:
                    error_msg = install_result.stderr or "Unknown error"
                    if req.required:
                        failed.append({"name": req.name, "path": req.path, "error": error_msg})
                    warnings.append(f"Failed to install {req.name}: {error_msg}")
                else:
                    installed.append({"name": req.name, "path": req.path})
            
            if failed:
                return SetupResult(
                    step=step,
                    success=False,
                    error=f"Failed to install: {', '.join(f['name'] for f in failed)}",
                    message=f"{len(failed)} required dependency file(s) failed",
                    data={"installed": installed, "failed": failed, "total": len(requirements)},
                    warnings=warnings
                )
            
            return SetupResult(
                step=step,
                success=True,
                message=f"Installed dependencies from {len(installed)} file(s)",
                data={"installed": installed, "total": len(requirements)},
                warnings=warnings
            )
            
        except subprocess.TimeoutExpired:
            return SetupResult(
                step=step,
                success=False,
                error="Dependency installation timed out",
                message="Operation timed out after 600 seconds"
            )
        except Exception as e:
            return SetupResult(
                step=step,
                success=False,
                error=str(e),
                message=f"Failed to install dependencies: {e}"
            )
    
    async def create_directories(self) -> SetupResult:
        """
        Create required directory structure.
        
        Creates:
        - Models directory
        - Checkpoints directory
        - Data directory
        - Outputs directory
        - Logs directory
        - Cache directory
        - Temp directory
        """
        step = "create_dirs"
        warnings = []
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            paths_config = self._config.paths
            created = []
            existing = []
            failed = []
            
            path_attrs = [
                ("models", paths_config.models),
                ("checkpoints", paths_config.checkpoints),
                ("data", paths_config.data),
                ("outputs", paths_config.outputs),
                ("logs", paths_config.logs),
                ("cache", paths_config.cache),
                ("temp", paths_config.temp),
                ("configs", paths_config.configs),
            ]
            
            for name, rel_path in path_attrs:
                full_path = self.project_root / rel_path
                
                if full_path.exists():
                    existing.append({"name": name, "path": rel_path})
                    continue
                
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    created.append({"name": name, "path": rel_path})
                    self.logger.debug(f"Created directory: {full_path}", event="xi.setup.dir.create")
                except PermissionError:
                    failed.append({"name": name, "path": rel_path, "error": "Permission denied"})
                    warnings.append(f"Cannot create {name}: permission denied")
                except Exception as e:
                    failed.append({"name": name, "path": rel_path, "error": str(e)})
                    warnings.append(f"Cannot create {name}: {e}")
            
            if failed:
                return SetupResult(
                    step=step,
                    success=False,
                    error=f"Failed to create: {', '.join(f['name'] for f in failed)}",
                    message=f"{len(failed)} directory(ies) could not be created",
                    data={"created": created, "existing": existing, "failed": failed},
                    warnings=warnings
                )
            
            message = f"Created {len(created)} directory(ies)"
            if existing:
                message += f", {len(existing)} already exist"
            
            return SetupResult(
                step=step,
                success=True,
                message=message,
                data={"created": created, "existing": existing, "total": len(path_attrs)},
                warnings=warnings
            )
            
        except Exception as e:
            return SetupResult(
                step=step,
                success=False,
                error=str(e),
                message=f"Failed to create directories: {e}"
            )
    
    async def verify_setup(self) -> SetupResult:
        """
        Verify the setup is complete and working.
        
        Checks:
        - Configuration loaded
        - Directories accessible
        - Python environment working
        """
        step = "verify_setup"
        warnings = []
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            checks = []
            
            config_check = {
                "name": "configuration",
                "passed": True,
                "message": "Configuration loaded successfully"
            }
            checks.append(config_check)
            
            paths_config = self._config.paths
            paths_ok = True
            missing_paths = []
            
            for name, rel_path in [
                ("models", paths_config.models),
                ("data", paths_config.data),
                ("outputs", paths_config.outputs),
                ("logs", paths_config.logs),
            ]:
                full_path = self.project_root / rel_path
                if not full_path.exists():
                    paths_ok = False
                    missing_paths.append(name)
            
            paths_check = {
                "name": "paths",
                "passed": paths_ok,
                "message": "All paths accessible" if paths_ok else f"Missing: {', '.join(missing_paths)}"
            }
            checks.append(paths_check)
            
            python_check = {
                "name": "python",
                "passed": True,
                "message": f"Python {sys.version_info.major}.{sys.version_info.minor} ready"
            }
            checks.append(python_check)
            
            env_config = self._config.environment
            if env_config.virtualenv and env_config.virtualenv.enabled:
                venv_path = self.project_root / env_config.virtualenv.path
                venv_python = self._get_venv_python(venv_path)
                
                venv_check = {
                    "name": "virtualenv",
                    "passed": venv_python is not None and venv_python.exists(),
                    "message": f"Virtual environment at {env_config.virtualenv.path}" if venv_python and venv_python.exists() else "Virtual environment not found"
                }
                checks.append(venv_check)
            
            all_passed = all(c["passed"] for c in checks)
            failed_checks = [c for c in checks if not c["passed"]]
            
            if failed_checks:
                return SetupResult(
                    step=step,
                    success=False,
                    error=f"Failed: {', '.join(c['name'] for c in failed_checks)}",
                    message="Setup verification failed",
                    data={"checks": checks},
                    warnings=warnings
                )
            
            return SetupResult(
                step=step,
                success=True,
                message="Setup verification complete",
                data={"checks": checks},
                warnings=warnings
            )
            
        except Exception as e:
            return SetupResult(
                step=step,
                success=False,
                error=str(e),
                message=f"Setup verification failed: {e}"
            )
    
    def _get_venv_python(self, venv_path: Path) -> Optional[Path]:
        """Get the Python executable path in a virtual environment."""
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        if python_exe.exists():
            return python_exe
        return None


_setup_instance: Optional[PiscesL1WelcomeSetup] = None


def get_welcome_setup(project_root: Optional[Path] = None) -> PiscesL1WelcomeSetup:
    """Get or create the global welcome setup instance."""
    global _setup_instance
    if _setup_instance is None:
        _setup_instance = PiscesL1WelcomeSetup(project_root)
    return _setup_instance
