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
Configuration Loader for Xi Studio.

This module provides the configuration loading system for Xi Studio.
It handles TOML parsing, variable substitution, environment variable overrides,
and default value fallbacks.
"""

import os
import re
import tomllib
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.dc import XiLogger
from .schema import (
    XiConfig,
    XiProjectConfig,
    XiProjectCommandsConfig,
    XiPathsConfig,
    XiApiConfig,
    XiApiHandshakeConfig,
    XiUiConfig,
    XiNotificationConfig,
    XiEnvironmentConfig,
    XiRequirementConfig,
    XiVirtualenvConfig,
    XiCommandConfig,
    XiCommandSchema,
    XiParameterSchema,
    XiTabSchema,
    XiWidgetConfig,
    XiWidgetStyle,
    XiWidgetValidation,
    XiValueMapping,
)


class XiConfigLoader:
    """
    Configuration loader for Xi Studio.
    
    This class handles loading configuration from:
    1. .xi/xi.toml - Main configuration file
    2. .xi/commands/*.toml - Command definition files
    3. Environment variables (XI_ prefix)
    4. Default values
    
    Variable Substitution:
        ${paths.models} -> Resolved from paths section
        ${project.name} -> Resolved from project section
        ${env.HOME} -> Resolved from environment variable
    """
    
    ENV_PREFIX = "XI_"
    CONFIG_DIR = ".xi"
    CONFIG_FILE = "xi.toml"
    COMMANDS_DIR = "commands"
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the configuration loader.
        
        Args:
            project_root: Path to project root directory. If None, uses current directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config_dir = self.project_root / self.CONFIG_DIR
        self.config_file = self.config_dir / self.CONFIG_FILE
        self.logger = XiLogger("Xi.ConfigLoader", enable_file=True)
        self._config: Optional[XiConfig] = None
        self._raw_config: Dict[str, Any] = {}
    
    def load(self) -> XiConfig:
        """
        Load configuration from all sources.
        
        Returns:
            XiConfig: Loaded and resolved configuration
        """
        if self._config is not None:
            return self._config
        
        self._raw_config = self._load_toml()
        
        if not self._raw_config:
            self.logger.info("No configuration file found, using defaults", event="xi.config.defaults")
            self._raw_config = {}
        
        self._apply_env_overrides()
        
        config = self._build_config()
        
        self._config = config
        self.logger.info(f"Configuration loaded from {self.config_file}", event="xi.config.loaded")
        
        return config
    
    def _load_toml(self) -> Dict[str, Any]:
        """
        Load configuration from TOML file.
        
        Returns:
            Dictionary of configuration values
        """
        if not self.config_file.exists():
            self.logger.warning(
                f"Configuration file not found: {self.config_file}",
                event="xi.config.not_found"
            )
            return {}
        
        try:
            with open(self.config_file, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            self.logger.error(
                f"Failed to load configuration: {e}",
                event="xi.config.load_error"
            )
            return {}
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables are mapped as:
        XI_PROJECT_NAME -> [project].name
        XI_PATHS_MODELS -> [paths].models
        XI_API_PORT -> [api].port
        """
        env_mappings = {
            "XI_PROJECT_NAME": ("project", "name"),
            "XI_PROJECT_VERSION": ("project", "version"),
            "XI_PROJECT_BACKEND": ("project", "backend"),
            "XI_PATHS_MODELS": ("paths", "models"),
            "XI_PATHS_CHECKPOINTS": ("paths", "checkpoints"),
            "XI_PATHS_DATA": ("paths", "data"),
            "XI_PATHS_OUTPUTS": ("paths", "outputs"),
            "XI_PATHS_LOGS": ("paths", "logs"),
            "XI_PATHS_CACHE": ("paths", "cache"),
            "XI_PATHS_TEMP": ("paths", "temp"),
            "XI_API_HOST": ("api", "host"),
            "XI_API_PORT": ("api", "port"),
            "XI_API_TIMEOUT": ("api", "timeout"),
            "XI_UI_THEME": ("ui", "theme"),
            "XI_UI_LANGUAGE": ("ui", "language"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if section not in self._raw_config:
                    self._raw_config[section] = {}
                self._raw_config[section][key] = self._parse_env_value(value)
                self.logger.debug(
                    f"Applied env override: {env_var}",
                    event="xi.config.env_override"
                )
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment
            
        Returns:
            Parsed value (int, bool, list, or string)
        """
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        try:
            return int(value)
        except ValueError:
            pass
        
        try:
            return float(value)
        except ValueError:
            pass
        
        if value.startswith("[") and value.endswith("]"):
            try:
                import json
                return json.loads(value)
            except Exception:
                pass
        
        return value
    
    def _build_config(self) -> XiConfig:
        """
        Build XiConfig from raw configuration dictionary.
        
        Returns:
            XiConfig object with all sub-configurations
        """
        project_config = self._build_project_config()
        paths_config = self._build_paths_config()
        api_config = self._build_api_config()
        ui_config = self._build_ui_config()
        notification_config = self._build_notification_config()
        environment_config = self._build_environment_config()
        commands_config = self._load_commands()
        
        return XiConfig(
            project=project_config,
            paths=paths_config,
            api=api_config,
            ui=ui_config,
            notifications=notification_config,
            environment=environment_config,
            commands=commands_config,
            config_dir=self.config_dir,
            project_root=self.project_root,
        )
    
    def _build_project_config(self) -> XiProjectConfig:
        """Build project configuration."""
        data = self._raw_config.get("project", {})
        
        commands_data = data.get("commands", {})
        commands_config = None
        if commands_data:
            commands_config = XiProjectCommandsConfig(
                enabled=commands_data.get("enabled", []),
            )
        
        return XiProjectConfig(
            name=data.get("name", "xi-project"),
            version=data.get("version", "1.0.0"),
            backend=data.get("backend", "piscesl1"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            first_launch=data.get("first_launch", True),
            commands=commands_config,
        )
    
    def _build_paths_config(self) -> XiPathsConfig:
        """Build paths configuration."""
        data = self._raw_config.get("paths", {})
        return XiPathsConfig(
            root=data.get("root", "."),
            models=data.get("models", ".pisceslx/models"),
            checkpoints=data.get("checkpoints", ".pisceslx/checkpoints"),
            data=data.get("data", ".pisceslx/data"),
            outputs=data.get("outputs", ".pisceslx/outputs"),
            logs=data.get("logs", ".pisceslx/logs"),
            cache=data.get("cache", ".pisceslx/cache"),
            temp=data.get("temp", ".pisceslx/temp"),
            configs=data.get("configs", "configs"),
        )
    
    def _build_api_config(self) -> XiApiConfig:
        """Build API configuration."""
        data = self._raw_config.get("api", {})
        
        handshake_data = data.get("handshake", {})
        handshake_config = None
        if handshake_data:
            handshake_config = XiApiHandshakeConfig(
                enabled=handshake_data.get("enabled", True),
                timeout=handshake_data.get("timeout", 5),
            )
        
        return XiApiConfig(
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 3140),
            cors_origins=data.get("cors_origins", ["http://localhost:3000"]),
            timeout=data.get("timeout", 120),
            max_workers=data.get("max_workers", 4),
            handshake=handshake_config,
        )
    
    def _build_ui_config(self) -> XiUiConfig:
        """Build UI configuration."""
        data = self._raw_config.get("ui", {})
        return XiUiConfig(
            theme=data.get("theme", "system"),
            language=data.get("language", "en"),
            sidebar_collapsed=data.get("sidebar_collapsed", False),
        )
    
    def _build_notification_config(self) -> XiNotificationConfig:
        """Build notification configuration."""
        data = self._raw_config.get("notifications", {})
        return XiNotificationConfig(
            enabled=data.get("enabled", True),
            retention_days=data.get("retention_days", 30),
            max_count=data.get("max_count", 1000),
            sound=data.get("sound", False),
        )
    
    def _build_environment_config(self) -> XiEnvironmentConfig:
        """Build environment configuration."""
        data = self._raw_config.get("environment", {})
        
        requirements = []
        for req_data in data.get("requirements", []):
            requirements.append(XiRequirementConfig(
                name=req_data.get("name", "main"),
                path=req_data.get("path", "requirements.txt"),
                required=req_data.get("required", True),
            ))
        
        virtualenv_data = data.get("virtualenv", {})
        virtualenv_config = None
        if virtualenv_data:
            virtualenv_config = XiVirtualenvConfig(
                enabled=virtualenv_data.get("enabled", False),
                path=virtualenv_data.get("path", ".venv"),
                create_if_missing=virtualenv_data.get("create_if_missing", False),
            )
        
        return XiEnvironmentConfig(
            python_path=data.get("python_path", "python"),
            cuda_required=data.get("cuda_required", False),
            cuda_version=data.get("cuda_version", ""),
            requirements=requirements,
            virtualenv=virtualenv_config,
        )
    
    def _load_commands(self) -> Dict[str, XiCommandConfig]:
        """
        Load command definitions from .xi/commands/*.toml files.
        
        Returns:
            Dictionary of command name to XiCommandConfig
        """
        commands = {}
        commands_dir = self.config_dir / self.COMMANDS_DIR
        
        if not commands_dir.exists():
            self.logger.info(
                f"Commands directory not found: {commands_dir}",
                event="xi.commands.not_found"
            )
            return commands
        
        for toml_file in commands_dir.glob("*.toml"):
            try:
                with open(toml_file, "rb") as f:
                    cmd_data = tomllib.load(f)
                
                for cmd_name, cmd_config in cmd_data.items():
                    schema = self._build_command_schema(cmd_config.get("schema", {}))
                    commands[cmd_name] = XiCommandConfig(
                        executable=cmd_config.get("executable", "python"),
                        script=cmd_config.get("script", "manage.py"),
                        args=cmd_config.get("args", []),
                        env=cmd_config.get("env", {}),
                        cwd=cmd_config.get("cwd", "${paths.root}"),
                        timeout=cmd_config.get("timeout", 3600),
                        background=cmd_config.get("background", True),
                        defaults=cmd_config.get("defaults", {}),
                        schema=schema,
                    )
                    self.logger.debug(
                        f"Loaded command: {cmd_name}",
                        event="xi.command.loaded"
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to load command file {toml_file}: {e}",
                    event="xi.command.load_error"
                )
        
        return commands
    
    def _build_command_schema(self, schema_data: Dict[str, Any]) -> Optional[XiCommandSchema]:
        """
        Build command schema from TOML data.
        
        Args:
            schema_data: Schema section from TOML file
            
        Returns:
            XiCommandSchema object or None if no schema defined
        """
        if not schema_data:
            return None
        
        description = schema_data.get("description", "")
        parameters = []
        tabs = []
        
        for tab_data in schema_data.get("tabs", []):
            tab = XiTabSchema(
                name=tab_data.get("name", ""),
                label=tab_data.get("label", ""),
                available=tab_data.get("available", True),
                unavailable_reason=tab_data.get("unavailable_reason", ""),
            )
            tabs.append(tab)
        
        param_tabs = []
        for param_data in schema_data.get("parameters", []):
            tab_name = param_data.get("tab", "basic")
            if tab_name not in param_tabs:
                param_tabs.append(tab_name)
        
        existing_tab_names = {t.name for t in tabs}
        for tab_name in param_tabs:
            if tab_name not in existing_tab_names:
                tabs.append(XiTabSchema(
                    name=tab_name,
                    label=tab_name.replace("_", " ").title(),
                    available=True,
                    unavailable_reason="",
                ))
        
        for param_data in schema_data.get("parameters", []):
            widget = self._build_widget_config(param_data.get("widget", {}))
            value_mapping = self._build_value_mapping(param_data.get("value_mapping", {}))
            
            param = XiParameterSchema(
                name=param_data.get("name", ""),
                type=param_data.get("type", "string"),
                description=param_data.get("description", ""),
                required=param_data.get("required", False),
                default=param_data.get("default"),
                options=param_data.get("options", []),
                min=param_data.get("min"),
                max=param_data.get("max"),
                source=param_data.get("source", ""),
                source_type=param_data.get("source_type", ""),
                filter=param_data.get("filter", ""),
                available=param_data.get("available", True),
                unavailable_reason=param_data.get("unavailable_reason", ""),
                tab=param_data.get("tab", "basic"),
                widget=widget,
                value_mapping=value_mapping,
            )
            parameters.append(param)
        
        return XiCommandSchema(
            description=description,
            parameters=parameters,
            tabs=tabs,
            available=schema_data.get("available", True),
            unavailable_reason=schema_data.get("unavailable_reason", ""),
        )
    
    def _build_widget_config(self, widget_data: Dict[str, Any]) -> Optional[XiWidgetConfig]:
        """
        Build widget configuration from TOML data.
        
        Args:
            widget_data: Widget section from TOML file
            
        Returns:
            XiWidgetConfig object or None if no widget defined
        """
        if not widget_data:
            return None
        
        style_data = widget_data.get("style", {})
        style = XiWidgetStyle(
            width=style_data.get("width", "full"),
            height=style_data.get("height"),
            placeholder=style_data.get("placeholder", ""),
            prefix=style_data.get("prefix", ""),
            suffix=style_data.get("suffix", ""),
            class_name=style_data.get("class_name", ""),
        )
        
        validation_data = widget_data.get("validation", {})
        validation = XiWidgetValidation(
            pattern=validation_data.get("pattern", ""),
            message=validation_data.get("message", ""),
            min_length=validation_data.get("min_length"),
            max_length=validation_data.get("max_length"),
        )
        
        return XiWidgetConfig(
            type=widget_data.get("type", "text"),
            style=style,
            validation=validation,
            props=widget_data.get("props", {}),
            depends_on=widget_data.get("depends_on", []),
            show_if=widget_data.get("show_if", ""),
            disabled_if=widget_data.get("disabled_if", ""),
            custom_component=widget_data.get("custom_component", ""),
            custom_props=widget_data.get("custom_props", {}),
        )
    
    def _build_value_mapping(self, mapping_data: Dict[str, Any]) -> Optional[XiValueMapping]:
        """
        Build value mapping configuration from TOML data.
        
        Args:
            mapping_data: Value mapping section from TOML file
            
        Returns:
            XiValueMapping object or None if no mapping defined
        """
        if not mapping_data:
            return None
        
        return XiValueMapping(
            arg_format=mapping_data.get("arg_format", ""),
            arg_prefix=mapping_data.get("arg_prefix", "--"),
            arg_separator=mapping_data.get("arg_separator", " "),
            skip_if=mapping_data.get("skip_if", ""),
            transform=mapping_data.get("transform", ""),
            default_if_empty=mapping_data.get("default_if_empty"),
            join_with=mapping_data.get("join_with", ","),
            wrap_value=mapping_data.get("wrap_value", False),
            template=mapping_data.get("template", ""),
        )
    
    def resolve_path(self, path_str: str) -> Path:
        """
        Resolve a path string with variable substitution.
        
        Args:
            path_str: Path string that may contain variables
            
        Returns:
            Resolved Path object
        """
        if not self._config:
            self.load()
        
        resolved = self._substitute_variables(path_str)
        path = Path(resolved)
        
        if not path.is_absolute():
            path = self.project_root / path
        
        return path
    
    def _substitute_variables(self, text: str) -> str:
        """
        Substitute variables in text.
        
        Supported variables:
        - ${paths.xxx} -> paths configuration
        - ${project.xxx} -> project configuration
        - ${env.XXX} -> environment variable
        
        Args:
            text: Text containing variable placeholders
            
        Returns:
            Text with variables substituted
        """
        if not self._config:
            return text
        
        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split(".")
            
            if len(parts) < 2:
                return match.group(0)
            
            section = parts[0]
            key = parts[1]
            
            if section == "paths":
                value = getattr(self._config.paths, key, None)
                if value:
                    return str(value)
            elif section == "project":
                value = getattr(self._config.project, key, None)
                if value:
                    return str(value)
            elif section == "env":
                env_key = ".".join(parts[1:])
                return os.environ.get(env_key, "")
            
            return match.group(0)
        
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, text)
    
    def get_config(self) -> XiConfig:
        """
        Get loaded configuration or load if not already loaded.
        
        Returns:
            XiConfig object
        """
        if self._config is None:
            return self.load()
        return self._config
    
    def reload(self) -> XiConfig:
        """
        Force reload configuration from files.
        
        Returns:
            Newly loaded XiConfig object
        """
        self._config = None
        self._raw_config = {}
        return self.load()


_config_loader: Optional[XiConfigLoader] = None


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory by searching for .xi directory.

    Searches upward from the start path until a directory containing
    the .xi subdirectory is found.

    Args:
        start_path: Starting path for search. Defaults to current directory.

    Returns:
        Path to the project root directory, or start_path if not found.
    """
    current = Path(start_path) if start_path else Path.cwd()

    search_path = current
    while True:
        xi_dir = search_path / ".xi"
        if xi_dir.exists() and xi_dir.is_dir():
            return search_path
        if search_path == search_path.parent:
            break
        search_path = search_path.parent

    return Path(start_path) if start_path else Path.cwd()


def get_config_loader(project_root: Optional[Path] = None) -> XiConfigLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        project_root: Path to project root. Only used on first call.
        
    Returns:
        XiConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        if project_root is None:
            project_root = find_project_root()
        _config_loader = XiConfigLoader(project_root)
    return _config_loader


def get_xi_config() -> XiConfig:
    """
    Get the loaded Xi configuration.
    
    Returns:
        XiConfig object
    """
    return get_config_loader().get_config()
