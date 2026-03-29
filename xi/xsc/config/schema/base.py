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
Main configuration container.

This is the root configuration object that contains all sub-configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from .project import XiProjectConfig
from .paths import XiPathsConfig
from .api import XiApiConfig
from .ui import XiUiConfig
from .notification import XiNotificationConfig
from .environment import XiEnvironmentConfig
from .command import XiCommandConfig
from .parameter import XiParameterSchema


@dataclass
class XiConfig:
    """
    Main configuration container.
    
    This is the root configuration object that contains all
    sub-configurations for the Xi system.
    
    Attributes:
        project: Project metadata
        paths: Path configurations
        api: API server configuration
        ui: UI configuration
        notifications: Notification configuration
        environment: Environment configuration (Python, CUDA, requirements)
        commands: Command definitions (loaded separately)
        config_dir: Path to .xi/ directory
        project_root: Path to project root
    """
    project: XiProjectConfig = field(default_factory=XiProjectConfig)
    paths: XiPathsConfig = field(default_factory=XiPathsConfig)
    api: XiApiConfig = field(default_factory=XiApiConfig)
    ui: XiUiConfig = field(default_factory=XiUiConfig)
    notifications: XiNotificationConfig = field(default_factory=XiNotificationConfig)
    environment: XiEnvironmentConfig = field(default_factory=XiEnvironmentConfig)
    commands: Dict[str, XiCommandConfig] = field(default_factory=dict)
    config_dir: Optional[Path] = None
    project_root: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for API response.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "project": {
                "name": self.project.name,
                "version": self.project.version,
                "backend": self.project.backend,
                "description": self.project.description,
                "author": self.project.author,
                "first_launch": self.project.first_launch,
                "commands": {
                    "enabled": self.project.commands.enabled if self.project.commands else [],
                },
            },
            "paths": {
                "root": self.paths.root,
                "models": self.paths.models,
                "checkpoints": self.paths.checkpoints,
                "data": self.paths.data,
                "outputs": self.paths.outputs,
                "logs": self.paths.logs,
                "cache": self.paths.cache,
                "temp": self.paths.temp,
                "configs": self.paths.configs,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "cors_origins": self.api.cors_origins,
                "timeout": self.api.timeout,
                "max_workers": self.api.max_workers,
                "handshake": {
                    "enabled": self.api.handshake.enabled if self.api.handshake else True,
                    "timeout": self.api.handshake.timeout if self.api.handshake else 5,
                },
            },
            "ui": {
                "theme": self.ui.theme,
                "language": self.ui.language,
                "sidebar_collapsed": self.ui.sidebar_collapsed,
            },
            "notifications": {
                "enabled": self.notifications.enabled,
                "retention_days": self.notifications.retention_days,
                "max_count": self.notifications.max_count,
                "sound": self.notifications.sound,
            },
            "environment": {
                "python_path": self.environment.python_path,
                "cuda_required": self.environment.cuda_required,
                "cuda_version": self.environment.cuda_version,
                "requirements": [
                    {
                        "name": r.name,
                        "path": r.path,
                        "required": r.required,
                    }
                    for r in self.environment.requirements
                ],
                "virtualenv": {
                    "enabled": self.environment.virtualenv.enabled if self.environment.virtualenv else False,
                    "path": self.environment.virtualenv.path if self.environment.virtualenv else ".venv",
                    "create_if_missing": self.environment.virtualenv.create_if_missing if self.environment.virtualenv else False,
                },
            },
            "commands": {
                name: self._command_to_dict(cmd)
                for name, cmd in self.commands.items()
            },
        }
    
    def _command_to_dict(self, cmd: XiCommandConfig) -> Dict[str, Any]:
        """Convert command config to dictionary including schema."""
        result = {
            "executable": cmd.executable,
            "script": cmd.script,
            "args": cmd.args,
            "env": cmd.env,
            "cwd": cmd.cwd,
            "timeout": cmd.timeout,
            "background": cmd.background,
            "defaults": cmd.defaults,
        }
        if cmd.schema:
            result["schema"] = {
                "description": cmd.schema.description,
                "available": cmd.schema.available,
                "unavailable_reason": cmd.schema.unavailable_reason,
                "tabs": [
                    {
                        "name": t.name,
                        "label": t.label,
                        "available": t.available,
                        "unavailable_reason": t.unavailable_reason,
                    }
                    for t in cmd.schema.tabs
                ],
                "parameters": [
                    self._parameter_to_dict(p)
                    for p in cmd.schema.parameters
                ],
            }
        return result
    
    def _parameter_to_dict(self, p: XiParameterSchema) -> Dict[str, Any]:
        """Convert parameter schema to dictionary."""
        param_dict = {
            "name": p.name,
            "type": p.type,
            "description": p.description,
            "required": p.required,
            "default": p.default,
            "options": p.options,
            "min": p.min,
            "max": p.max,
            "source": p.source,
            "source_type": p.source_type,
            "filter": p.filter,
            "available": p.available,
            "unavailable_reason": p.unavailable_reason,
            "tab": p.tab,
        }
        
        if p.widget:
            param_dict["widget"] = {
                "type": p.widget.type,
                "style": {
                    "width": p.widget.style.width,
                    "height": p.widget.style.height,
                    "placeholder": p.widget.style.placeholder,
                    "prefix": p.widget.style.prefix,
                    "suffix": p.widget.style.suffix,
                    "class_name": p.widget.style.class_name,
                },
                "validation": {
                    "pattern": p.widget.validation.pattern,
                    "message": p.widget.validation.message,
                    "min_length": p.widget.validation.min_length,
                    "max_length": p.widget.validation.max_length,
                },
                "props": p.widget.props,
                "depends_on": p.widget.depends_on,
                "show_if": p.widget.show_if,
                "disabled_if": p.widget.disabled_if,
                "custom_component": p.widget.custom_component,
                "custom_props": p.widget.custom_props,
            }
        
        if p.value_mapping:
            param_dict["value_mapping"] = {
                "arg_format": p.value_mapping.arg_format,
                "arg_prefix": p.value_mapping.arg_prefix,
                "arg_separator": p.value_mapping.arg_separator,
                "skip_if": p.value_mapping.skip_if,
                "transform": p.value_mapping.transform,
                "default_if_empty": p.value_mapping.default_if_empty,
                "join_with": p.value_mapping.join_with,
                "wrap_value": p.value_mapping.wrap_value,
                "template": p.value_mapping.template,
            }
        
        return param_dict
    
    def get_resolved_paths(self) -> Dict[str, Path]:
        """
        Get all paths resolved to absolute paths.
        
        Returns:
            Dictionary of path name to resolved Path object
        """
        if not self.project_root:
            return {}
        
        resolved = {}
        for attr_name in ["models", "checkpoints", "data", "outputs", "logs", "cache", "temp", "configs"]:
            attr_value = getattr(self.paths, attr_name, None)
            if attr_value:
                path = Path(attr_value)
                if not path.is_absolute():
                    path = self.project_root / path
                resolved[attr_name] = path
        
        return resolved
