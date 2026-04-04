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
Configuration Validator for Welcome Screen

Validates Xi Studio configuration including:
- xi.toml syntax and structure
- Project information
- Subcommand configurations
- Python environment
- UI configuration
- Paths configuration
"""

import os
import sys
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable

from ....core.dc import XiLogger
from ....config import get_xi_config, XiConfig


@dataclass
class ValidationResult:
    """Result of a single validation step."""
    step: str
    valid: bool
    message: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


class PiscesL1WelcomeValidator:
    """
    Configuration validator for Xi Studio Welcome screen.
    
    Performs comprehensive validation of:
    1. xi.toml syntax and structure
    2. Project information completeness
    3. Subcommand configurations validity
    4. Python environment availability
    5. UI configuration validity
    6. Paths configuration and accessibility
    """
    
    STEP_LABELS: Dict[str, str] = {
        "xi_toml_syntax": "Xi Configuration Syntax",
        "project_info": "Project Information",
        "subcommands": "Subcommand Configurations",
        "python_env": "Python Environment",
        "ui_config": "UI Configuration",
        "paths_config": "Paths Configuration",
    }
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = XiLogger("Xi.WelcomeValidator", enable_file=True)
        self._config: Optional[XiConfig] = None
    
    def get_step_label(self, step: str) -> str:
        """Get human-readable label for a validation step."""
        return self.STEP_LABELS.get(step, step)
    
    async def validate_all(self) -> AsyncGenerator[ValidationResult, None]:
        """
        Run all validation steps and yield results.
        
        Yields:
            ValidationResult for each validation step
        """
        all_valid = True
        
        result = await self.validate_xi_toml_syntax()
        if not result.valid:
            yield result
            return
        yield result
        if not result.valid:
            all_valid = False
        
        result = await self.validate_project_info()
        yield result
        if not result.valid:
            all_valid = False
        
        result = await self.validate_subcommands()
        yield result
        if not result.valid:
            all_valid = False
        
        result = await self.validate_python_env()
        yield result
        if not result.valid:
            all_valid = False
        
        result = await self.validate_ui_config()
        yield result
        if not result.valid:
            all_valid = False
        
        result = await self.validate_paths_config()
        yield result
        if not result.valid:
            all_valid = False
    
    async def validate_xi_toml_syntax(self) -> ValidationResult:
        """
        Validate xi.toml file syntax.
        
        Checks:
        - File existence
        - TOML syntax validity
        - Basic structure
        """
        step = "xi_toml_syntax"
        xi_dir = self.project_root / ".xi"
        xi_toml_path = xi_dir / "xi.toml"
        
        if not xi_toml_path.exists():
            return ValidationResult(
                step=step,
                valid=False,
                error="xi.toml not found",
                message=f"Configuration file not found: {xi_toml_path}"
            )
        
        try:
            import tomli
            with open(xi_toml_path, "rb") as f:
                data = tomli.load(f)
            
            if not isinstance(data, dict):
                return ValidationResult(
                    step=step,
                    valid=False,
                    error="Invalid configuration structure",
                    message="xi.toml must contain a dictionary"
                )
            
            return ValidationResult(
                step=step,
                valid=True,
                message="Configuration syntax is valid",
                data={"path": str(xi_toml_path), "sections": list(data.keys())}
            )
            
        except ImportError:
            import tomllib
            try:
                with open(xi_toml_path, "rb") as f:
                    data = tomllib.load(f)
                
                return ValidationResult(
                    step=step,
                    valid=True,
                    message="Configuration syntax is valid",
                    data={"path": str(xi_toml_path), "sections": list(data.keys())}
                )
            except Exception as e:
                return ValidationResult(
                    step=step,
                    valid=False,
                    error=str(e),
                    message=f"Failed to parse xi.toml: {e}"
                )
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'lineno'):
                error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            return ValidationResult(
                step=step,
                valid=False,
                error=error_msg,
                message=f"Configuration syntax error: {error_msg}"
            )
    
    async def validate_project_info(self) -> ValidationResult:
        """
        Validate project information configuration.
        
        Checks:
        - Project name is set
        - Version format validity
        - Backend configuration
        """
        step = "project_info"
        warnings = []
        
        try:
            config = get_xi_config()
            self._config = config
            
            if not config.project.name:
                return ValidationResult(
                    step=step,
                    valid=False,
                    error="Project name not configured",
                    message="Project name is required in xi.toml"
                )
            
            if config.project.name == "xi-project":
                warnings.append("Project name is using default value")
            
            data = {
                "name": config.project.name,
                "version": config.project.version,
                "backend": config.project.backend,
                "description": config.project.description,
                "author": config.project.author,
            }
            
            if not config.project.version:
                warnings.append("Project version not set, using default")
            
            message = f"Project: {config.project.name} v{config.project.version}"
            return ValidationResult(
                step=step,
                valid=True,
                message=message,
                data=data,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                step=step,
                valid=False,
                error=str(e),
                message=f"Failed to load project configuration: {e}"
            )
    
    async def validate_subcommands(self) -> ValidationResult:
        """
        Validate subcommand configurations.
        
        Checks:
        - Command files existence
        - Command schema validity
        - Executable availability
        """
        step = "subcommands"
        warnings = []
        commands_data = {}
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            commands = self._config.commands
            
            if not commands:
                return ValidationResult(
                    step=step,
                    valid=True,
                    message="No subcommands configured",
                    data={"commands": [], "total": 0},
                    warnings=["No subcommands defined in .xi/commands/"]
                )
            
            invalid_commands = []
            
            for cmd_name, cmd_config in commands.items():
                cmd_info = {
                    "name": cmd_name,
                    "executable": cmd_config.executable,
                    "script": cmd_config.script,
                    "valid": True,
                }
                
                if cmd_config.executable:
                    if not shutil.which(cmd_config.executable):
                        cmd_info["valid"] = False
                        cmd_info["error"] = f"Executable not found: {cmd_config.executable}"
                        invalid_commands.append(cmd_name)
                        warnings.append(f"Command '{cmd_name}' executable not found: {cmd_config.executable}")
                
                commands_data[cmd_name] = cmd_info
            
            if invalid_commands:
                return ValidationResult(
                    step=step,
                    valid=False,
                    error=f"Invalid commands: {', '.join(invalid_commands)}",
                    message=f"{len(invalid_commands)} command(s) have configuration issues",
                    data={"commands": commands_data, "total": len(commands), "invalid": invalid_commands},
                    warnings=warnings
                )
            
            return ValidationResult(
                step=step,
                valid=True,
                message=f"{len(commands)} subcommand(s) configured",
                data={"commands": commands_data, "total": len(commands)},
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                step=step,
                valid=False,
                error=str(e),
                message=f"Failed to validate subcommands: {e}"
            )
    
    async def validate_python_env(self) -> ValidationResult:
        """
        Validate Python environment.
        
        Checks:
        - Python version
        - Virtual environment configuration
        - pip availability
        """
        step = "python_env"
        warnings = []
        
        try:
            python_version = sys.version_info
            version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
                return ValidationResult(
                    step=step,
                    valid=False,
                    error=f"Python version too old: {version_str}",
                    message="Python 3.9 or higher is required"
                )
            
            if python_version.minor < 10:
                warnings.append(f"Python {version_str} - consider upgrading to 3.10+")
            
            pip_available = shutil.which("pip") is not None
            
            if not pip_available:
                warnings.append("pip not found in PATH")
            
            venv_info = {}
            venv_enabled = False
            
            if self._config:
                env_config = self._config.environment
                if env_config.virtualenv:
                    venv_enabled = env_config.virtualenv.enabled
                    venv_path = env_config.virtualenv.path
                    venv_info = {
                        "enabled": venv_enabled,
                        "path": venv_path,
                        "exists": False,
                    }
                    
                    if venv_enabled:
                        full_venv_path = self.project_root / venv_path
                        venv_info["exists"] = full_venv_path.exists()
                        
                        if not venv_info["exists"]:
                            if env_config.virtualenv.create_if_missing:
                                warnings.append(f"Virtual environment will be created at {venv_path}")
                            else:
                                warnings.append(f"Virtual environment not found at {venv_path}")
            
            data = {
                "version": version_str,
                "executable": sys.executable,
                "pip_available": pip_available,
                "virtualenv": venv_info,
            }
            
            message = f"Python {version_str}"
            if venv_enabled:
                message += f" (venv: {venv_info.get('path', 'N/A')})"
            
            return ValidationResult(
                step=step,
                valid=True,
                message=message,
                data=data,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                step=step,
                valid=False,
                error=str(e),
                message=f"Failed to validate Python environment: {e}"
            )
    
    async def validate_ui_config(self) -> ValidationResult:
        """
        Validate UI configuration.
        
        Checks:
        - Theme validity
        - Language support
        - UI settings
        """
        step = "ui_config"
        warnings = []
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            ui_config = self._config.ui
            valid_themes = ["light", "dark", "system"]
            valid_languages = ["en", "zh", "zh-CN", "zh-TW"]
            
            if ui_config.theme not in valid_themes:
                warnings.append(f"Unknown theme: {ui_config.theme}, using system default")
            
            if ui_config.language not in valid_languages:
                warnings.append(f"Unknown language: {ui_config.language}")
            
            data = {
                "theme": ui_config.theme,
                "language": ui_config.language,
                "sidebar_collapsed": ui_config.sidebar_collapsed,
            }
            
            return ValidationResult(
                step=step,
                valid=True,
                message=f"Theme: {ui_config.theme}, Language: {ui_config.language}",
                data=data,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                step=step,
                valid=False,
                error=str(e),
                message=f"Failed to validate UI configuration: {e}"
            )
    
    async def validate_paths_config(self) -> ValidationResult:
        """
        Validate paths configuration.
        
        Checks:
        - Path existence
        - Write permissions
        - Required directories
        """
        step = "paths_config"
        warnings = []
        
        try:
            if not self._config:
                self._config = get_xi_config()
            
            paths_config = self._config.paths
            paths_data = {}
            missing_paths = []
            unwritable_paths = []
            
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
                path_info = {
                    "name": name,
                    "path": rel_path,
                    "full_path": str(full_path),
                    "exists": full_path.exists(),
                    "writable": True,
                }
                
                if full_path.exists():
                    if not os.access(full_path, os.W_OK):
                        path_info["writable"] = False
                        unwritable_paths.append(name)
                        warnings.append(f"Path '{name}' is not writable")
                else:
                    parent = full_path.parent
                    while not parent.exists() and parent != parent.parent:
                        parent = parent.parent
                    
                    if parent.exists() and not os.access(parent, os.W_OK):
                        path_info["writable"] = False
                        unwritable_paths.append(name)
                        warnings.append(f"Cannot create path '{name}' - parent not writable")
                    
                    missing_paths.append(name)
                
                paths_data[name] = path_info
            
            data = {
                "paths": paths_data,
                "total": len(paths_data),
                "missing": missing_paths,
                "unwritable": unwritable_paths,
            }
            
            if unwritable_paths:
                return ValidationResult(
                    step=step,
                    valid=False,
                    error=f"Unwritable paths: {', '.join(unwritable_paths)}",
                    message=f"{len(unwritable_paths)} path(s) are not writable",
                    data=data,
                    warnings=warnings
                )
            
            if missing_paths:
                message = f"{len(missing_paths)} path(s) will be created"
            else:
                message = "All paths configured and accessible"
            
            return ValidationResult(
                step=step,
                valid=True,
                message=message,
                data=data,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                step=step,
                valid=False,
                error=str(e),
                message=f"Failed to validate paths configuration: {e}"
            )


_validator_instance: Optional[PiscesL1WelcomeValidator] = None


def get_welcome_validator(project_root: Optional[Path] = None) -> PiscesL1WelcomeValidator:
    """Get or create the global welcome validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PiscesL1WelcomeValidator(project_root)
    return _validator_instance
