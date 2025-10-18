#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

class PiscesLxCoreConfigLoader:
    """A zero-configuration loader that merges configurations from multiple sources.

    This loader merges configurations in the following priority order (low to high):
    defaults < .env file < environment variables.
    It also supports basic schema validation for types and required keys.
    """

    def __init__(self, project_root: Optional[Path] = None, env_file: Optional[str] = None) -> None:
        """Initialize the configuration loader.

        Args:
            project_root (Optional[Path]): The root path of the project. 
                                         If None, it will be detected automatically.
            env_file (Optional[str]): The path to the .env file. 
                                   If None, it defaults to `.env` in the project root.
        """
        self._project_root = project_root or self._detect_project_root()
        self._env_file = env_file or str(self._project_root / ".env")

    def _detect_project_root(self) -> Path:
        """Detect the project root by searching for common marker files or directories.

        This method first checks the current working directory for markers.
        If not found, it searches parent directories.
        If no markers are found, it returns the current working directory.

        Returns:
            Path: The detected project root path.
        """
        current = Path.cwd()
        markers = [".git", "pyproject.toml", "setup.py", "requirements.txt", "PiscesL1"]
        
        for marker in markers:
            if (current / marker).exists():
                return current
        
        # Fallback: search parent directories
        for parent in current.parents:
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        
        return current

    def _coerce(self, value: str) -> Union[str, int, float, bool]:
        """Coerce a string value to an appropriate Python type.

        Attempts to convert the input string to a boolean, integer, float, or returns it as a string.

        Args:
            value (str): The string value to coerce.

        Returns:
            Union[str, int, float, bool]: The coerced value.
        """
        value = value.strip()
        
        # Convert to boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Convert to numeric type
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _validate(self, config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Perform basic schema validation on the configuration.

        Checks if all required keys are present in the configuration and if values match the expected types.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.
            schema (Dict[str, Any]): The validation schema containing 'required' and 'properties'.

        Raises:
            ValueError: If a required configuration key is missing.
            TypeError: If a configuration value does not match the expected type.
        """
        if not schema:
            return
            
        # Check for required keys
        required = schema.get("required", [])
        for key in required:
            if key not in config:
                raise ValueError(f"Required configuration key '{key}' is missing")
        
        # Check value types
        properties = schema.get("properties", {})
        for key, expected_type in properties.items():
            if key in config and not isinstance(config[key], expected_type):
                actual_type = type(config[key]).__name__
                expected_type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                raise TypeError(f"Configuration key '{key}' should be {expected_type_name}, got {actual_type}")

    def load_from_env(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from .env file and environment variables.

        Merges the provided default values with configurations from the .env file,
        then overrides with environment variables if they exist in defaults.

        Args:
            defaults (Dict[str, Any]): Default configuration values.

        Returns:
            Dict[str, Any]: The merged configuration dictionary.
        """
        cfg: Dict[str, Any] = dict(defaults or {})
        
        # Load configuration from .env file
        env_path = Path(self._env_file)
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                cfg[k.strip()] = self._coerce(v.strip())
        
        # Override with environment variables
        for k, v in os.environ.items():
            if k in defaults:
                cfg[k] = self._coerce(v)
        
        return cfg

    def load(self, defaults: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load and merge configuration from multiple sources, then validate it.

        Args:
            defaults (Dict[str, Any]): Default configuration values.
            schema (Optional[Dict[str, Any]]): Schema for validation. Defaults to None.

        Returns:
            Dict[str, Any]: The merged and validated configuration dictionary.
        """
        cfg = self.load_from_env(defaults)
        
        # Validate the configuration against the schema if provided
        if schema:
            self._validate(cfg, schema)
        
        return cfg

    def load_from_file(self, config_path: Union[str, Path], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from a JSON or YAML file.

        Args:
            config_path (Union[str, Path]): Path to the configuration file.
            defaults (Optional[Dict[str, Any]]): Default values to merge with the file content.

        Returns:
            Dict[str, Any]: The configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file format is not supported.
            RuntimeError: If there is an error loading the configuration file.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        config_data = {}
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {path}: {e}")
        
        # Merge with defaults if provided
        if defaults:
            merged = dict(defaults)
            merged.update(config_data)
            return merged
        
        return config_data


def create_config_manager():
    """Factory function to create a unified configuration manager.

    Returns:
        PiscesLxCoreConfigManager: An instance of the unified configuration manager.
    """
    from utils.config.manager import PiscesLxCoreConfigManager
    return PiscesLxCoreConfigManager()


def load_config_from_file(config_path: Union[str, Path], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to load configuration from a JSON or YAML file.

    Args:
        config_path (Union[str, Path]): Path to the configuration file.
        defaults (Optional[Dict[str, Any]]): Default values to merge with the file content.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    loader = PiscesLxCoreConfigLoader()
    return loader.load_from_file(config_path, defaults)
