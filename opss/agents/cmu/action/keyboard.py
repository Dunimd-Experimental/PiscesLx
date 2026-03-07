#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to Dunimd Team.
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
CMU Keyboard Controller - Keyboard Input Control

This module provides comprehensive keyboard control for desktop platforms,
supporting typing, hotkeys, and special keys.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUAction,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUScreenState,
)
from .base import POPSSCMUActionExecutor, POPSSCMUActionConfig

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Action.Keyboard", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Action.Keyboard"), enable_file=True)

_HAS_PYAUTOGUI = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    _HAS_PYAUTOGUI = True
except ImportError:
    _LOG.warning("pyautogui_not_available")


@dataclass
class POPSSCMUKeyboardConfig:
    """Keyboard controller configuration."""
    typing_speed: float = 0.05
    key_delay: float = 0.01
    hotkey_delay: float = 0.1


class POPSSCMUKeyboardController(POPSSCMUActionExecutor):
    """
    Keyboard controller for desktop platforms.
    
    Provides:
        - Text typing with configurable speed
        - Key press and release
        - Hotkey combinations
        - Special keys handling
        - Clipboard operations
    
    Attributes:
        keyboard_config: Keyboard-specific configuration
        _held_keys: Currently held keys
    """
    
    SPECIAL_KEYS = {
        "enter": "enter",
        "return": "enter",
        "tab": "tab",
        "space": "space",
        "backspace": "backspace",
        "delete": "delete",
        "escape": "esc",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "home": "home",
        "end": "end",
        "pageup": "pageup",
        "pagedown": "pagedown",
        "f1": "f1",
        "f2": "f2",
        "f3": "f3",
        "f4": "f4",
        "f5": "f5",
        "f6": "f6",
        "f7": "f7",
        "f8": "f8",
        "f9": "f9",
        "f10": "f10",
        "f11": "f11",
        "f12": "f12",
    }
    
    def __init__(
        self,
        config: Optional[POPSSCMUActionConfig] = None,
        keyboard_config: Optional[POPSSCMUKeyboardConfig] = None,
        platform_adapter: Optional[Any] = None,
    ):
        super().__init__(config, platform_adapter)
        
        self.keyboard_config = keyboard_config or POPSSCMUKeyboardConfig()
        self._held_keys: Dict[str, bool] = {}
        
        _LOG.info("keyboard_controller_initialized")
    
    async def execute_action_internal(
        self,
        action: POPSSCMUAction,
    ) -> POPSSCMUActionResult:
        """Execute keyboard action."""
        if not _HAS_PYAUTOGUI:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="PyAutoGUI not available",
            )
        
        try:
            if action.action_type.value == "type":
                return await self._handle_type(action)
            elif action.action_type.value == "key_press":
                return await self._handle_key_press(action)
            elif action.action_type.value == "hotkey":
                return await self._handle_hotkey(action)
            elif action.action_type.value == "clipboard_copy":
                return await self._handle_clipboard_copy(action)
            elif action.action_type.value == "clipboard_paste":
                return await self._handle_clipboard_paste(action)
            else:
                return POPSSCMUActionResult(
                    action_id=action.action_id,
                    status=POPSSCMUActionResultStatus.FAILED,
                    error_message=f"Unsupported action type: {action.action_type.value}",
                )
        except Exception as e:
            _LOG.error("keyboard_action_failed", action_type=action.action_type.value, error=str(e))
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )
    
    async def verify_action_result(
        self,
        action: POPSSCMUAction,
        result: POPSSCMUActionResult,
    ) -> bool:
        """Verify keyboard action result."""
        if result.status != POPSSCMUActionResultStatus.SUCCESS:
            return False
        
        if action.action_type.value == "type":
            typed_length = action.params.get("text", "")
            return result.output.get("typed_length", 0) == len(typed_length)
        
        if action.action_type.value in ["key_press", "hotkey"]:
            return True
        
        return True
    
    async def _handle_type(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle text typing."""
        text = action.params.get("text", "")
        typing_speed = action.params.get("typing_speed", self.keyboard_config.typing_speed)
        
        if not text:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="No text to type",
            )
        
        pyautogui.typewrite(text, interval=typing_speed)
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"text": text[:50] + "..." if len(text) > 50 else text, "typed_length": len(text)},
        )
    
    async def _handle_key_press(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle single key press."""
        key = action.params.get("key", "")
        
        if not key:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="No key specified",
            )
        
        normalized_key = self._normalize_key(key)
        pyautogui.press(normalized_key)
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"key": normalized_key},
        )
    
    async def _handle_hotkey(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle hotkey combination."""
        keys = action.params.get("keys", [])
        
        if not keys or len(keys) < 2:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Hotkey requires at least 2 keys",
            )
        
        normalized_keys = [self._normalize_key(k) for k in keys]
        pyautogui.hotkey(*normalized_keys, interval=self.keyboard_config.hotkey_delay)
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"keys": normalized_keys},
        )
    
    async def _handle_clipboard_copy(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle clipboard copy."""
        try:
            pyautogui.hotkey("ctrl", "c")
            await asyncio.sleep(0.1)
            
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS,
                output={"operation": "copy"},
            )
        except Exception as e:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )
    
    async def _handle_clipboard_paste(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle clipboard paste."""
        try:
            pyautogui.hotkey("ctrl", "v")
            await asyncio.sleep(0.1)
            
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.SUCCESS,
                output={"operation": "paste"},
            )
        except Exception as e:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )
    
    async def type_text(
        self,
        text: str,
        typing_speed: Optional[float] = None,
    ) -> POPSSCMUActionResult:
        """Type text."""
        speed = typing_speed or self.keyboard_config.typing_speed
        
        if not _HAS_PYAUTOGUI:
            return POPSSCMUActionResult(
                action_id="",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="PyAutoGUI not available",
            )
        
        pyautogui.typewrite(text, interval=speed)
        
        return POPSSCMUActionResult(
            action_id="",
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"text": text[:50] + "..." if len(text) > 50 else text, "typed_length": len(text)},
        )
    
    async def press_key(
        self,
        key: str,
        presses: int = 1,
    ) -> POPSSCMUActionResult:
        """Press a key multiple times."""
        normalized_key = self._normalize_key(key)
        
        if not _HAS_PYAUTOGUI:
            return POPSSCMUActionResult(
                action_id="",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="PyAutoGUI not available",
            )
        
        for _ in range(presses):
            pyautogui.press(normalized_key)
            await asyncio.sleep(self.keyboard_config.key_delay)
        
        return POPSSCMUActionResult(
            action_id="",
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"key": normalized_key, "presses": presses},
        )
    
    async def hold_key(self, key: str) -> None:
        """Hold a key down."""
        normalized_key = self._normalize_key(key)
        
        if _HAS_PYAUTOGUI:
            pyautogui.keyDown(normalized_key)
            self._held_keys[normalized_key] = True
    
    async def release_key(self, key: str) -> None:
        """Release a held key."""
        normalized_key = self._normalize_key(key)
        
        if _HAS_PYAUTOGUI:
            pyautogui.keyUp(normalized_key)
            self._held_keys[normalized_key] = False
    
    async def release_all_keys(self) -> None:
        """Release all held keys."""
        for key in list(self._held_keys.keys()):
            await self.release_key(key)
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key name."""
        key_lower = key.lower()
        
        if key_lower in self.SPECIAL_KEYS:
            return self.SPECIAL_KEYS[key_lower]
        
        if len(key) == 1:
            return key_lower
        
        modifiers = ["ctrl", "alt", "shift", "win", "cmd"]
        for mod in modifiers:
            if key_lower.startswith(mod):
                return mod + key[len(mod):]
        
        return key_lower
    
    def get_held_keys(self) -> List[str]:
        """Get list of currently held keys."""
        return list(self._held_keys.keys())
