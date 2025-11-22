#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
from .core import PiscesLxCoreLog
from typing import Dict, Any, Optional

class PiscesLxCoreLogConfig:
    """Manages log configuration with default and file-based settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the log configuration manager.
        
        Args:
            config_path (Optional[str]): Path to the configuration file. 
                If None, only default configuration will be used.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_default_config()
        self.logger = PiscesLxCoreLog(name="log_config")
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load the default log configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing default log configuration parameters.
        """
        return {
            "default_level": "INFO",
            "console_enabled": True,
            "file_enabled": True,
            "file_path": None,
            "rotate_when": "time",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 7,
            "json_format_console": False,
            "file_format": "text",
            "sampling_rates": {},
            "anomaly_detectors": {},
            "buffer_size": 1000,
            "async_logging": False
        }
    
    def load_config(self, config_path: str) -> None:
        """Load log configuration from a specified file.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        try:
            from utils.config.loader import load_config_from_file
            file_config = load_config_from_file(config_path)
            self.config.update(file_config)
            self.logger.info("LOG_CONFIG_LOADED", {
                "message": "Log configuration loaded successfully.",
                "config_path": config_path
            })
        except Exception as e:
            self.logger.error("LOG_CONFIG_LOAD_FAILED", {
                "message": "Failed to load log configuration.",
                "config_path": config_path,
                "error": str(e)
            })
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save the current log configuration to a file.
        
        Args:
            config_path (Optional[str]): Path to save the configuration file.
                If None, use the path provided during initialization.
        """
        save_path = config_path or self.config_path
        if not save_path:
            self.logger.warning("LOG_CONFIG_SAVE_FAILED", {
                "message": "No configuration file path specified, unable to save configuration."
            })
            return
            
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            self.logger.info("LOG_CONFIG_SAVED", {
                "message": "Log configuration saved successfully.",
                "config_path": save_path
            })
        except Exception as e:
            self.logger.error("LOG_CONFIG_SAVE_FAILED", {
                "message": "Failed to save log configuration.",
                "config_path": save_path,
                "error": str(e)
            })
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key.
        
        Args:
            key (str): Configuration key.
            default (Any): Default value to return if the key is not found.
            
        Returns:
            Any: Configuration value for the key, or default if not found.
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by key.
        
        Args:
            key (str): Configuration key.
            value (Any): Value to set for the key.
        """
        self.config[key] = value
    
    def get_logger_config(self, logger_name: str = "pisces") -> Dict[str, Any]:
        """Get configuration for a specific logger.
        
        Args:
            logger_name (str): Name of the logger. Defaults to "pisces".
            
        Returns:
            Dict[str, Any]: Logger configuration parameters.
        """
        return {
            "name": logger_name,
            "level": self.config.get("default_level", "INFO"),
            "console": self.config.get("console_enabled", True),
            "file_path": self.config.get("file_path"),
            "rotate_when": self.config.get("rotate_when", "time"),
            "max_bytes": self.config.get("max_bytes", 10 * 1024 * 1024),
            "backup_count": self.config.get("backup_count", 7),
            "json_format_console": self.config.get("json_format_console", False),
            "enable_file": self.config.get("file_enabled", True),
            "file_format": self.config.get("file_format", "text")
        }
    
    def set_sampling_rate(self, event: str, rate: float) -> None:
        """Set the sampling rate for a specific event.
        
        Args:
            event (str): Name of the event.
            rate (float): Sampling rate, clamped between 0.0 and 1.0.
        """
        sampling_rates = self.config.setdefault("sampling_rates", {})
        sampling_rates[event] = max(0.0, min(1.0, rate))
    
    def get_sampling_rate(self, event: str) -> float:
        """Get the sampling rate for a specific event.
        
        Args:
            event (str): Name of the event.
            
        Returns:
            float: Sampling rate for the event. Defaults to 1.0 if not set.
        """
        sampling_rates = self.config.get("sampling_rates", {})
        return sampling_rates.get(event, 1.0)
    
    def add_anomaly_detector_config(self, name: str, detector_config: Dict[str, Any]) -> None:
        """Add configuration for an anomaly detector.
        
        Args:
            name (str): Name of the anomaly detector.
            detector_config (Dict[str, Any]): Configuration for the anomaly detector.
        """
        anomaly_detectors = self.config.setdefault("anomaly_detectors", {})
        anomaly_detectors[name] = detector_config
    
    def get_anomaly_detector_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all anomaly detector configurations.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of anomaly detector configurations.
                Returns an empty dict if no configurations exist.
        """
        return self.config.get("anomaly_detectors", {})

class PiscesLxCoreLogConfigBuilder:
    """Build log configuration using a fluent API."""
    
    def __init__(self):
        """Initialize the log configuration builder with an empty configuration."""
        self.config = {}
    
    def set_default_level(self, level: str) -> 'PiscesLxCoreLogConfigBuilder':
        """Set the default log level.
        
        Args:
            level (str): Log level to set.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["default_level"] = level
        return self
    
    def enable_console(self, enabled: bool = True) -> 'PiscesLxCoreLogConfigBuilder':
        """Enable or disable console logging.
        
        Args:
            enabled (bool): Whether to enable console logging. Defaults to True.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["console_enabled"] = enabled
        return self
    
    def enable_file(self, enabled: bool = True) -> 'PiscesLxCoreLogConfigBuilder':
        """Enable or disable file logging.
        
        Args:
            enabled (bool): Whether to enable file logging. Defaults to True.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["file_enabled"] = enabled
        return self
    
    def set_file_path(self, path: str) -> 'PiscesLxCoreLogConfigBuilder':
        """Set the path for log files.
        
        Args:
            path (str): Path to the log file.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["file_path"] = path
        return self
    
    def set_rotation(self, when: str = "time", max_bytes: int = 10 * 1024 * 1024, 
                     backup_count: int = 7) -> 'PiscesLxCoreLogConfigBuilder':
        """Set log rotation configuration.
        
        Args:
            when (str): Rotation trigger ("time" or "size"). Defaults to "time".
            max_bytes (int): Maximum file size in bytes. Defaults to 10MB.
            backup_count (int): Number of backup files to keep. Defaults to 7.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["rotate_when"] = when
        self.config["max_bytes"] = max_bytes
        self.config["backup_count"] = backup_count
        return self
    
    def set_json_format_console(self, enabled: bool = True) -> 'PiscesLxCoreLogConfigBuilder':
        """Set whether to use JSON format for console output.
        
        Args:
            enabled (bool): Whether to enable JSON format. Defaults to True.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["json_format_console"] = enabled
        return self
    
    def set_file_format(self, format_type: str) -> 'PiscesLxCoreLogConfigBuilder':
        """Set the format of log files.
        
        Args:
            format_type (str): Log file format ("text" or "json").
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["file_format"] = format_type
        return self
    
    def set_sampling_rate(self, event: str, rate: float) -> 'PiscesLxCoreLogConfigBuilder':
        """Set the sampling rate for a specific event.
        
        Args:
            event (str): Name of the event.
            rate (float): Sampling rate, clamped between 0.0 and 1.0.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        sampling_rates = self.config.setdefault("sampling_rates", {})
        sampling_rates[event] = max(0.0, min(1.0, rate))
        return self
    
    def enable_async_logging(self, enabled: bool = True) -> 'PiscesLxCoreLogConfigBuilder':
        """Enable or disable asynchronous logging.
        
        Args:
            enabled (bool): Whether to enable asynchronous logging. Defaults to True.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["async_logging"] = enabled
        return self
    
    def set_buffer_size(self, size: int) -> 'PiscesLxCoreLogConfigBuilder':
        """Set the size of the log buffer.
        
        Args:
            size (int): Size of the log buffer.
            
        Returns:
            PiscesLxCoreLogConfigBuilder: Self instance for method chaining.
        """
        self.config["buffer_size"] = size
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return the log configuration.
        
        Returns:
            Dict[str, Any]: Copy of the built log configuration.
        """
        return self.config.copy()
    
    def build_config_object(self) -> PiscesLxCoreLogConfig:
        """Build and return a PiscesLxCoreLogConfig object with the built configuration.
        
        Returns:
            PiscesLxCoreLogConfig: New instance of PiscesLxCoreLogConfig with the built configuration.
        """
        config_obj = PiscesLxCoreLogConfig()
        config_obj.config.update(self.config)
        return config_obj