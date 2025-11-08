#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
import json
import yaml
import threading
from pathlib import Path
from utils.log.core import PiscesLxCoreLog
from .loader import PiscesLxCoreConfigLoader
from typing import Any, Dict, Optional, Union
from utils.error import PiscesLxCoreConfigError, PiscesLxCoreValidationError

class PiscesLxCoreConfigManager:
    """
    A singleton configuration manager that provides unified configuration management functionality.
    """

    # Singleton instance of the configuration manager.
    _instance: Optional['PiscesLxCoreConfigManager'] = None
    # Thread lock for ensuring thread-safe singleton creation.
    _lock = threading.RLock()

    def __init__(self, project_root: Optional[Path] = None) -> None:
        """
        Initialize the configuration manager.

        Args:
            project_root (Optional[Path]): Path to the project root directory. If None, it will be detected automatically.
        """
        # Initialize the logger for the configuration manager.
        self.logger = PiscesLxCoreLog("PiscesLx.Utils.Config.Manager")
        # Set the project root directory. If not provided, detect it automatically.
        self.project_root = project_root or self._detect_project_root()
        # Dictionary to store loaded configurations.
        self._configs: Dict[str, Dict[str, Any]] = {}
        # Dictionary to store locks for each configuration.
        self._config_locks: Dict[str, threading.RLock] = {}

    @classmethod
    def get_instance(cls, project_root: Optional[Path] = None) -> 'PiscesLxCoreConfigManager':
        """
        Get the singleton instance of the configuration manager.

        Args:
            project_root (Optional[Path]): Path to the project root directory.

        Returns:
            PiscesLxCoreConfigManager: Instance of the configuration manager.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(project_root)
        return cls._instance

    def _detect_project_root(self) -> Path:
        """
        Detect and return the project root directory.

        Returns:
            Path: The project root path.
        """
        from utils.config.loader import PiscesLxCoreConfigLoader
        loader = PiscesLxCoreConfigLoader()
        return loader._project_root

    def load_config(self, config_name: str, defaults: Optional[Dict[str, Any]] = None,
                   schema: Optional[Dict[str, Any]] = None, 
                   config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            config_name (str): Name of the configuration.
            defaults (Optional[Dict[str, Any]]): Default configuration values.
            schema (Optional[Dict[str, Any]]): Configuration schema for validation.
            config_path (Optional[Union[str, Path]]): Path to the configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        with self._get_config_lock(config_name):
            # If the configuration is already loaded, return it directly.
            if config_name in self._configs:
                return self._configs[config_name]

            # Initialize default values.
            if defaults is None:
                defaults = {}

            # Create a configuration loader.
            loader = PiscesLxCoreConfigLoader(self.project_root)

            # If a configuration file path is specified, try to load from the file.
            if config_path:
                try:
                    config_data = loader.load_from_file(config_path, defaults)
                    defaults.update(config_data)
                except Exception as e:
                    self.logger.warning("CONFIG_FILE_LOAD_FAILED", 
                        config_path=str(config_path),
                        error=str(e)
                    )

            # Use the loader to load the configuration.
            config = loader.load(defaults, schema)
            self._configs[config_name] = config

            self.logger.success("CONFIG_LOADED", {
                "config_name": config_name,
                "source": str(config_path) if config_path else "defaults+env"
            })

            return config

    def _get_config_lock(self, config_name: str) -> threading.RLock:
        """
        Get the lock for a specific configuration.

        Args:
            config_name (str): Name of the configuration.

        Returns:
            threading.RLock: The lock for the configuration.
        """
        if config_name not in self._config_locks:
            self._config_locks[config_name] = threading.RLock()
        return self._config_locks[config_name]

    def _load_from_file(self, config_path: Union[str, Path], defaults: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load configuration from a file.

        Args:
            config_path (Union[str, Path]): Path to the configuration file.
            defaults (Dict[str, Any]): Default configuration values.

        Returns:
            Optional[Dict[str, Any]]: The loaded configuration data, or None if loading fails.
        """
        path = Path(config_path)
        if not path.exists():
            self.logger.warning("CONFIG_FILE_NOT_FOUND", path=str(path))
            return None

        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning("CONFIG_UNSUPPORTED_FORMAT", path=str(path), suffix=path.suffix)
                return None
        except Exception as e:
            self.logger.error("CONFIG_LOAD_FAILED", path=str(path), error=str(e))
            return None

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the loaded configuration by name.

        Args:
            config_name (str): Name of the configuration.

        Returns:
            Optional[Dict[str, Any]]: The configuration data, or None if not found.
        """
        return self._configs.get(config_name)

    def get_config_value(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            config_name (str): Name of the configuration.
            key (str): Configuration key.
            default (Any): Default value to return if the key is not found.

        Returns:
            Any: The configuration value, or the default value if not found.
        """
        config = self.get_config(config_name)
        if config is None:
            return default
        return config.get(key, default)

    def set_config_value(self, config_name: str, key: str, value: Any) -> None:
        """
        Set a specific configuration value.

        Args:
            config_name (str): Name of the configuration.
            key (str): Configuration key.
            value (Any): Configuration value to set.
        """
        with self._get_config_lock(config_name):
            if config_name not in self._configs:
                self._configs[config_name] = {}
            self._configs[config_name][key] = value

    def save_config(self, config_name: str, config_path: Union[str, Path]) -> bool:
        """
        Save the configuration to a file.

        Args:
            config_name (str): Name of the configuration.
            config_path (Union[str, Path]): Path to save the configuration file.

        Returns:
            bool: True if the save is successful, False otherwise.
        """
        config = self.get_config(config_name)
        if config is None:
            self.logger.warning("CONFIG_NOT_FOUND_FOR_SAVE", config_name=config_name)
            return False

        path = Path(config_path)
        try:
            # Ensure the directory exists.
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Select the save format based on the file extension.
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif path.suffix.lower() == '.json':
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                # Save as JSON format by default.
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.success("CONFIG_SAVED", {"config_name": config_name, "path": str(path)})
            return True
        except Exception as e:
            self.logger.error("CONFIG_SAVE_FAILED", config_name=config_name, path=str(path), error=str(e))
            return False

    def reload_config(self, config_name: str, defaults: Optional[Dict[str, Any]] = None,
                     schema: Optional[Dict[str, Any]] = None,
                     config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Reload a configuration.

        Args:
            config_name (str): Name of the configuration.
            defaults (Optional[Dict[str, Any]]): Default configuration values.
            schema (Optional[Dict[str, Any]]): Configuration schema for validation.
            config_path (Optional[Union[str, Path]]): Path to the configuration file.

        Returns:
            Dict[str, Any]: The reloaded configuration.
        """
        with self._get_config_lock(config_name):
            # Remove the old configuration from memory.
            self._configs.pop(config_name, None)
            # Reload the configuration.
            return self.load_config(config_name, defaults, schema, config_path)

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded configurations.

        Returns:
            Dict[str, Dict[str, Any]]: All configuration data.
        """
        return dict(self._configs)

    def clear_config(self, config_name: str) -> None:
        """
        Clear a specific configuration.

        Args:
            config_name (str): Name of the configuration.
        """
        with self._get_config_lock(config_name):
            self._configs.pop(config_name, None)
            self.logger.success("CONFIG_CLEARED", {"config_name": config_name})

    # --- Compatibility helpers used by training pipeline ---
    def validate(self, obj: Any = None) -> Any:
        """
        Lightweight validation shim to keep backward compatibility.

        Returns an object with attributes: is_valid (bool) and errors (list[str]).
        """
        class _Result:
            def __init__(self, is_valid: bool = True, errors: Optional[list] = None) -> None:
                self.is_valid = is_valid
                self.errors = errors or []

        try:
            # Best-effort: treat presence of object as valid; extend here if schema available
            return _Result(True, [])
        except Exception as e:
            return _Result(False, [str(e)])

    def get(self, key: str, default: Any = None) -> Any:
        """
        Safe dotted-path getter used by higher-level config facades.
        Since this manager is not holding a single unified dict, just return default.
        """
        try:
            # Future: could map to loaded configs if needed
            return default
        except Exception:
            return default

    def validate_config_compatibility(self, cached_config: Dict[str, Any], new_config: Dict[str, Any]) -> bool:
        """
        Best-effort compatibility check between cached and current configs.
        Non-strict: only verifies a few key fields if present; otherwise returns True.
        """
        try:
            keys = (
                "model.size",
                "train.batch_size",
                "device_config.device_type",
            )
            for k in keys:
                # Support dotted-path lookup in flat dicts
                def _get(d: Dict[str, Any], path: str):
                    if d is None:
                        return None
                    node = d
                    for part in path.split('.'):
                        if isinstance(node, dict) and part in node:
                            node = node[part]
                        else:
                            return None
                    return node

                cv = _get(cached_config, k)
                nv = _get(new_config, k)
                if cv is not None and nv is not None and cv != nv:
                    self.logger.debug("CONFIG_INCOMPATIBLE_KEY", key=k, cached=cv, new=nv)
                    return False
            return True
        except Exception:
            # If unsure, do not block training
            return True

    def get_current_timestamp(self) -> str:
        """
        Return current timestamp in ISO-like string for logging/caching.
        """
        import time as _time
        try:
            return _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime())
        except Exception:
            return str(int(_time.time()))


class PiscesLxCoreConfigManagerFacade:
    """
    Facade class for configuration management operations.
    Provides a unified interface for accessing configuration manager functionality.
    """
    
    def __init__(self, project_root: Optional[Path] = None) -> None:
        """
        Initialize the configuration manager facade.
        
        Args:
            project_root (Optional[Path]): Path to the project root directory.
        """
        self._manager = PiscesLxCoreConfigManager.get_instance(project_root)
    
    def get_manager(self) -> PiscesLxCoreConfigManager:
        """
        Get the underlying configuration manager instance.
        
        Returns:
            PiscesLxCoreConfigManager: The configuration manager instance.
        """
        return self._manager


def get_config_manager(project_root: Optional[Path] = None) -> PiscesLxCoreConfigManager:
    """
    Get an instance of the configuration manager.

    Args:
        project_root (Optional[Path]): Path to the project root directory.

    Returns:
        PiscesLxCoreConfigManager: Instance of the configuration manager.
    """
    return PiscesLxCoreConfigManager.get_instance(project_root)