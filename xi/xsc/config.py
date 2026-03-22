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
Configuration Data Types for Xi Studio.

This module defines all configuration dataclasses used by the Xi
configuration system. These types represent the structure of the
xi.toml configuration file and command definitions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path


@dataclass
class XiProjectConfig:
    """
    Project metadata configuration.
    
    Attributes:
        name: Project name
        version: Project version
        backend: Backend type (e.g., "piscesl1", "custom")
        description: Project description
        author: Project author
        first_launch: Whether this is the first launch
        commands: Command configuration
    """
    name: str = "xi-project"
    version: str = "1.0.0"
    backend: str = "piscesl1"
    description: str = ""
    author: str = ""
    first_launch: bool = True
    commands: Optional['XiProjectCommandsConfig'] = None


@dataclass
class XiProjectCommandsConfig:
    """
    Project commands configuration.
    
    Attributes:
        enabled: List of enabled command names
    """
    enabled: List[str] = field(default_factory=list)


@dataclass
class XiPathsConfig:
    """
    Path configuration for project directories.
    
    All paths can be relative to project root or absolute.
    Variable substitution is supported: ${paths.xxx}, ${project.name}
    
    Attributes:
        root: Project root directory (usually ".")
        models: Directory for model files
        checkpoints: Directory for training checkpoints
        data: Directory for datasets
        outputs: Directory for output files
        logs: Directory for log files
        cache: Directory for cache files
        temp: Directory for temporary files
        configs: Directory for configuration files
    """
    root: str = "."
    models: str = ".pisceslx/models"
    checkpoints: str = ".pisceslx/checkpoints"
    data: str = ".pisceslx/data"
    outputs: str = ".pisceslx/outputs"
    logs: str = ".pisceslx/logs"
    cache: str = ".pisceslx/cache"
    temp: str = ".pisceslx/temp"
    configs: str = "configs"


@dataclass
class XiApiConfig:
    """
    API server configuration.
    
    Attributes:
        host: API server host
        port: API server port
        cors_origins: Allowed CORS origins
        timeout: Request timeout in seconds
        max_workers: Maximum worker threads
        handshake: Handshake configuration
    """
    host: str = "127.0.0.1"
    port: int = 3140
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    timeout: int = 120
    max_workers: int = 4
    handshake: Optional['XiApiHandshakeConfig'] = None


@dataclass
class XiApiHandshakeConfig:
    """
    API handshake configuration.
    
    Attributes:
        enabled: Whether handshake is enabled
        timeout: Handshake timeout in seconds
    """
    enabled: bool = True
    timeout: int = 5


@dataclass
class XiUiConfig:
    """
    UI configuration for frontend.
    
    Attributes:
        theme: Theme setting ("light", "dark", "system")
        language: UI language
        sidebar_collapsed: Default sidebar state
    """
    theme: str = "system"
    language: str = "en"
    sidebar_collapsed: bool = False


@dataclass
class XiNotificationConfig:
    """
    Notification system configuration.
    
    Attributes:
        enabled: Whether notifications are enabled
        retention_days: Days to keep notifications
        max_count: Maximum notification count
        sound: Whether to play sound on notification
    """
    enabled: bool = True
    retention_days: int = 30
    max_count: int = 1000
    sound: bool = False


@dataclass
class XiRequirementConfig:
    """
    Python requirements file configuration.
    
    Attributes:
        name: Name identifier for this requirements file
        path: Path to requirements.txt file (relative to project root)
        required: Whether this file is required for the project to work
    """
    name: str = "main"
    path: str = "requirements.txt"
    required: bool = True


@dataclass
class XiVirtualenvConfig:
    """
    Python virtual environment configuration.
    
    Attributes:
        enabled: Whether virtual environment is enabled
        path: Path to virtual environment directory
        create_if_missing: Whether to create venv if it doesn't exist
    """
    enabled: bool = False
    path: str = ".venv"
    create_if_missing: bool = False


@dataclass
class XiEnvironmentConfig:
    """
    Environment configuration for Python and CUDA.
    
    Attributes:
        python_path: Path to Python executable
        cuda_required: Whether CUDA is required
        cuda_version: Required CUDA version (e.g., "11.8", "12.1")
        requirements: List of requirements file configurations
        virtualenv: Virtual environment configuration
    """
    python_path: str = "python"
    cuda_required: bool = False
    cuda_version: str = ""
    requirements: List[XiRequirementConfig] = field(default_factory=list)
    virtualenv: Optional[XiVirtualenvConfig] = None


@dataclass
class XiWidgetValidation:
    """
    Widget validation configuration.
    
    Attributes:
        pattern: Regex pattern for validation
        message: Error message on validation failure
        min_length: Minimum length for string types
        max_length: Maximum length for string types
    """
    pattern: str = ""
    message: str = ""
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass
class XiWidgetStyle:
    """
    Widget style configuration.
    
    Attributes:
        width: Widget width (e.g., "full", "half", "auto")
        height: Widget height in pixels (for textarea, etc.)
        placeholder: Placeholder text
        prefix: Prefix text or icon
        suffix: Suffix text or icon
        class_name: Additional CSS classes
    """
    width: str = "full"
    height: Optional[int] = None
    placeholder: str = ""
    prefix: str = ""
    suffix: str = ""
    class_name: str = ""


@dataclass
class XiWidgetConfig:
    """
    Custom widget configuration for dynamic UI rendering.
    
    This allows XSC to define custom controls that XIS will render,
    enabling extensibility without modifying XIS code.
    
    Attributes:
        type: Widget type (text, textarea, number, slider, toggle, 
              select, multiselect, file, directory, color, date, 
              time, datetime, code, markdown, keyvalue, list, custom)
        style: Widget styling configuration
        validation: Validation rules
        props: Additional widget-specific properties
        depends_on: Other parameters this widget depends on
        show_if: Condition to show this widget
        disabled_if: Condition to disable this widget
        custom_component: Custom component name (for type="custom")
        custom_props: Custom component properties
    """
    type: str = "text"
    style: XiWidgetStyle = field(default_factory=XiWidgetStyle)
    validation: XiWidgetValidation = field(default_factory=XiWidgetValidation)
    props: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    show_if: str = ""
    disabled_if: str = ""
    custom_component: str = ""
    custom_props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XiValueMapping:
    """
    Value mapping configuration for command argument assembly.
    
    Defines how the frontend value should be transformed into
    command-line arguments.
    
    Attributes:
        arg_format: Format string for argument (e.g., "--{name}={value}", "-{name} {value}")
        arg_prefix: Prefix for argument name (e.g., "--", "-")
        arg_separator: Separator between name and value (e.g., "=", " ")
        skip_if: Skip this argument if value matches condition
        transform: Value transformation (lowercase, uppercase, str, int, float, json, path)
        default_if_empty: Default value if empty
        join_with: Join multiple values with this separator
        wrap_value: Wrap value in quotes
        template: Full template for complex argument generation
    """
    arg_format: str = ""
    arg_prefix: str = "--"
    arg_separator: str = " "
    skip_if: str = ""
    transform: str = ""
    default_if_empty: Any = None
    join_with: str = ","
    wrap_value: bool = False
    template: str = ""


@dataclass
class XiParameterSchema:
    """
    Parameter schema definition for command arguments.
    
    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, float, boolean, select, path)
        description: Human-readable description
        required: Whether the parameter is required
        default: Default value
        options: Available options for select type
        min: Minimum value for numeric types
        max: Maximum value for numeric types
        source: Source for dynamic options (directory path or variable)
        source_type: Type of source (directory, file, api)
        filter: Filter pattern for source (e.g., "*.yaml")
        available: Whether this parameter is available (has valid options/source)
        unavailable_reason: Reason why parameter is unavailable
        tab: Tab group this parameter belongs to
        widget: Custom widget configuration for UI rendering
        value_mapping: Value mapping for command assembly
    """
    name: str = ""
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    options: List[str] = field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None
    source: str = ""
    source_type: str = ""
    filter: str = ""
    available: bool = True
    unavailable_reason: str = ""
    tab: str = "basic"
    widget: Optional[XiWidgetConfig] = None
    value_mapping: Optional[XiValueMapping] = None


@dataclass
class XiTabSchema:
    """
    Tab schema definition for grouping parameters.
    
    Attributes:
        name: Tab name (used as identifier)
        label: Display label for the tab
        available: Whether this tab is available
        unavailable_reason: Reason why tab is unavailable
    """
    name: str = ""
    label: str = ""
    available: bool = True
    unavailable_reason: str = ""


@dataclass
class XiCommandSchema:
    """
    Command schema definition.
    
    Attributes:
        description: Command description
        parameters: List of parameter schemas
        tabs: List of tab schemas for grouping parameters
        available: Whether the entire command is available
        unavailable_reason: Reason why command is unavailable
    """
    description: str = ""
    parameters: List[XiParameterSchema] = field(default_factory=list)
    tabs: List[XiTabSchema] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str = ""


@dataclass
class XiCommandConfig:
    """
    Command definition configuration.
    
    Attributes:
        executable: Executable to run (e.g., "python")
        script: Script to execute (e.g., "manage.py")
        args: Default command arguments
        env: Environment variables
        cwd: Working directory (supports variable substitution)
        timeout: Command timeout in seconds
        background: Whether to run in background
        defaults: Default parameter values
        schema: Parameter schema for UI generation
    """
    executable: str = "python"
    script: str = "manage.py"
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = "${paths.root}"
    timeout: int = 3600
    background: bool = True
    defaults: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[XiCommandSchema] = None


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
