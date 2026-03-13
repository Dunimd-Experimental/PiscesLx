#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
PiscesLx Developer Mode Manager.

This module implements the global singleton manager for the developer mode,
providing centralized control over developer mode state and settings.

The manager reads/writes the settings file at .pisceslx/settings/settings.yaml
and provides methods to enable, disable, and check the developer mode status.

Architecture:
    The PiscesLxDevModeManager follows the singleton pattern to ensure a single
    global instance throughout the application lifecycle. It manages:
    
    1. Settings Persistence: Reads and writes the settings.yaml file
    2. State Management: Tracks whether developer mode is enabled
    3. UI Integration: Provides reference to the UI component when attached
    4. Trainer Binding: Maintains reference to the training operator

Usage:
    Get the singleton instance:
        >>> from tools.dev import PiscesLxDevModeManager
        >>> manager = PiscesLxDevModeManager.get_instance()
    
    Check if enabled:
        >>> if manager.is_enabled():
        ...     print("Developer mode is active")
    
    Enable/disable:
        >>> manager.enable()   # Writes settings.yaml
        >>> manager.disable()  # Writes settings.yaml
    
    Attach to trainer:
        >>> manager.attach(trainer)
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.paths import get_settings_file, get_log_file
from utils.dc import PiscesLxLogger


_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", file_path=get_log_file("PiscesLx.Tools.Dev"), enable_file=True)


class PiscesLxDevModeManager:
    """
    Global singleton manager for PiscesL1 developer mode.
    
    This class provides centralized control over the developer mode feature,
    managing settings persistence and state across the application.
    
    The manager uses a singleton pattern to ensure only one instance exists,
    which is important for maintaining consistent state across different
    parts of the application.
    
    Attributes:
        _instance: Class-level singleton instance
        _lock: Thread lock for thread-safe singleton creation
        _enabled: Current enabled state of developer mode
        _settings_path: Path to the settings.yaml file
        _trainer: Reference to the attached training operator
        _ui: Reference to the UI component
    
    Example:
        >>> manager = PiscesLxDevModeManager.get_instance()
        >>> manager.enable()
        >>> print(manager.is_enabled())
        True
    """
    
    _instance: Optional['PiscesLxDevModeManager'] = None
    _lock: threading.RLock = threading.RLock()
    
    def __new__(cls) -> 'PiscesLxDevModeManager':
        """
        Create or return the singleton instance.
        
        This method ensures that only one instance of PiscesLxDevModeManager
        exists throughout the application lifecycle, using double-checked
        locking for thread safety.
        
        Returns:
            PiscesLxDevModeManager: The singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the manager instance.
        
        This method is called only once due to the singleton pattern.
        It loads the current settings from the settings file.
        """
        if getattr(self, '_initialized', False):
            return
            
        self._settings_path = get_settings_file()
        self._enabled = False
        self._trainer: Optional[Any] = None
        self._ui: Optional[Any] = None
        self._paused = False
        
        self._load_settings()
        self._initialized = True
        
        _LOG.info("PiscesLxDevModeManager initialized", settings_path=self._settings_path)
    
    @classmethod
    def get_instance(cls) -> 'PiscesLxDevModeManager':
        """
        Get the singleton instance of the manager.
        
        This is the preferred way to access the manager, ensuring
        consistent state across the application.
        
        Returns:
            PiscesLxDevModeManager: The singleton instance
        """
        return cls()
    
    def _load_settings(self) -> None:
        """
        Load settings from the settings.yaml file.
        
        This method reads the settings file and extracts the dev.enabled
        value. If the file doesn't exist or is malformed, defaults to False.
        
        The settings file format is:
            dev:
              enabled: false
        
        Side Effects:
            - Sets self._enabled based on file content
            - Creates default settings file if missing
        """
        if not os.path.exists(self._settings_path):
            self._enabled = False
            self._create_default_settings()
            return
        
        try:
            with open(self._settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f) or {}
                dev_settings = settings.get('dev', {})
                self._enabled = bool(dev_settings.get('enabled', False))
                        
            _LOG.debug("Settings loaded", enabled=self._enabled)
        except Exception as e:
            _LOG.warning("Failed to load settings, using default", error=str(e))
            self._enabled = False
    
    def _create_default_settings(self) -> None:
        """
        Create the default settings file by copying from template.
        
        This method copies the settings template from configs/settings.yaml
        to the user's settings directory if one doesn't exist.
        
        Side Effects:
            - Creates .pisceslx/settings/ directory
            - Copies configs/settings.yaml to settings location
        """
        try:
            template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'configs', 'settings.yaml'
            )
            
            if os.path.exists(template_path):
                import shutil
                os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
                shutil.copy(template_path, self._settings_path)
            else:
                os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
                with open(self._settings_path, 'w', encoding='utf-8') as f:
                    f.write("dev:\n  enabled: false\n")
            
            _LOG.debug("Settings file created from template", path=self._settings_path)
        except Exception as e:
            _LOG.error("Failed to create default settings", error=str(e))
    
    def _save_settings(self) -> None:
        """
        Save current settings to the settings.yaml file.
        
        This method writes the current state to the settings file,
        persisting the enabled/disabled status across sessions.
        
        Side Effects:
            - Writes to .pisceslx/settings/settings.yaml
        """
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            
            settings = {'dev': {'enabled': self._enabled}}
            with open(self._settings_path, 'w', encoding='utf-8') as f:
                yaml.dump(settings, f, default_flow_style=False)
                    
            _LOG.debug("Settings saved", enabled=self._enabled)
        except Exception as e:
            _LOG.error("Failed to save settings", error=str(e))
    
    def is_enabled(self) -> bool:
        """
        Check if developer mode is currently enabled.
        
        Returns:
            bool: True if developer mode is enabled, False otherwise
        """
        return self._enabled
    
    def enable(self) -> None:
        """
        Enable developer mode.
        
        This method sets the developer mode to enabled and persists
        the change to the settings file.
        
        Side Effects:
            - Sets self._enabled to True
            - Writes to settings.yaml
        """
        if not self._enabled:
            self._enabled = True
            self._save_settings()
            _LOG.info("Developer mode enabled")
    
    def disable(self) -> None:
        """
        Disable developer mode.
        
        This method sets the developer mode to disabled and persists
        the change to the settings file.
        
        Side Effects:
            - Sets self._enabled to False
            - Writes to settings.yaml
        """
        if self._enabled:
            self._enabled = False
            self._save_settings()
            _LOG.info("Developer mode disabled")
    
    def attach(self, trainer: Any) -> None:
        """
        Attach the manager to a training operator.
        
        This method binds the manager to a training operator, allowing
        the developer mode to interact with the training process.
        
        Args:
            trainer: The PiscesLxTrainingOperator instance to attach to
        
        Side Effects:
            - Sets self._trainer reference
            - Initializes UI if developer mode is enabled
        """
        self._trainer = trainer
        _LOG.info("Attached to trainer", trainer_type=type(trainer).__name__)
        
        if self._enabled:
            self._init_ui()
    
    def detach(self) -> None:
        """
        Detach the manager from the current training operator.
        
        This method clears the trainer reference and stops the UI.
        
        Side Effects:
            - Clears self._trainer reference
            - Stops UI if running
        """
        self._trainer = None
        if self._ui is not None:
            self._stop_ui()
        _LOG.info("Detached from trainer")
    
    def _init_ui(self) -> None:
        """
        Initialize the developer mode UI.
        
        This method creates and starts the UI component for the
        command-line interface.
        """
        try:
            from .ui import PiscesLxDevModeUI
            self._ui = PiscesLxDevModeUI(self)
            _LOG.info("Developer mode UI initialized")
        except Exception as e:
            _LOG.error("Failed to initialize UI", error=str(e))
            self._ui = None
    
    def _stop_ui(self) -> None:
        """
        Stop the developer mode UI.
        
        This method stops and cleans up the UI component.
        """
        if self._ui is not None:
            try:
                if hasattr(self._ui, 'stop'):
                    self._ui.stop()
            except Exception as e:
                _LOG.warning("Error stopping UI", error=str(e))
            finally:
                self._ui = None
    
    def get_trainer(self) -> Optional[Any]:
        """
        Get the attached training operator.
        
        Returns:
            Optional[Any]: The attached trainer, or None if not attached
        """
        return self._trainer
    
    def get_ui(self) -> Optional[Any]:
        """
        Get the UI component.
        
        Returns:
            Optional[Any]: The UI component, or None if not initialized
        """
        return self._ui
    
    def is_paused(self) -> bool:
        """
        Check if training is paused.
        
        Returns:
            bool: True if training is paused, False otherwise
        """
        return self._paused
    
    def pause(self) -> None:
        """
        Pause the training process.
        
        This method sets the paused flag, which can be checked by
        the training loop to suspend training.
        """
        self._paused = True
        _LOG.info("Training paused via developer mode")
    
    def resume(self) -> None:
        """
        Resume the training process.
        
        This method clears the paused flag, allowing training to continue.
        """
        self._paused = False
        _LOG.info("Training resumed via developer mode")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the developer mode.
        
        Returns:
            Dict[str, Any]: Status dictionary with enabled, paused, and attached info
        """
        return {
            'enabled': self._enabled,
            'paused': self._paused,
            'attached': self._trainer is not None,
            'ui_active': self._ui is not None,
            'settings_path': self._settings_path
        }
