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
PiscesLx Developer Mode UI.

This module implements the terminal user interface for the developer mode,
providing a vim-style command bar at the bottom of the training display.

Architecture:
    The UI system consists of:
    
    1. PiscesLxDevModeUI: Main UI renderer using Rich library
       - Manages the command bar display
       - Handles keyboard input
       - Coordinates overlay display
    
    2. Integration with Training:
       - Attaches to training loop via the manager
       - Preserves original training logs
       - Adds command bar at bottom (3 lines height)

Layout:
    +------------------------------------------+
    |                                          |
    |         Original Training Logs           |
    |         (Preserved, scrollable)          |
    |                                          |
    +------------------------------------------+
    | > _                                      |  <- Command bar (3 lines)
    | [Dev Mode] Type :help for commands       |
    +------------------------------------------+

Commands:
    All commands start with ':' (vim-style):
    - :mem [module]     - Memory details
    - :layer <n>        - Layer information
    - :grad             - Gradient statistics
    - :pause/:resume    - Training control
    - :help             - Show all commands
    - :q                - Close overlay

Usage:
    The UI is typically created by the PiscesLxDevModeManager:
    
    >>> from tools.dev import PiscesLxDevModeManager
    >>> manager = PiscesLxDevModeManager.get_instance()
    >>> manager.attach(trainer)
    >>> # UI is automatically initialized if dev mode is enabled
"""

import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.live import Live

from utils.paths import get_log_file
from utils.dc import PiscesLxLogger


_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", file_path=get_log_file("PiscesLx.Tools.Dev"), enable_file=True)


class PiscesLxDevModeUI:
    """
    Terminal UI renderer for developer mode.
    
    This class provides a vim-style command interface at the bottom of
    the training display, allowing real-time debugging and control.
    
    Attributes:
        _manager: Reference to the PiscesLxDevModeManager
        _console: Rich console for rendering
        _command_buffer: Current command being typed
        _history: Command history for navigation
        _history_index: Current position in history
        _running: Whether the UI is active
        _overlay_active: Whether an overlay is displayed
        _overlay_content: Current overlay content
        _input_thread: Thread for keyboard input
        _log_buffer: Recent log lines for display
        _status_message: Current status message
        _callbacks: Registered callback functions
    
    Example:
        >>> ui = PiscesLxDevModeUI(manager)
        >>> ui.start()
        >>> # UI is now running and accepting input
        >>> ui.stop()
    """
    
    COMMAND_BAR_HEIGHT = 3
    MAX_LOG_LINES = 100
    MAX_HISTORY = 50
    
    def __init__(self, manager: Any):
        """
        Initialize the UI renderer.
        
        Args:
            manager: The PiscesLxDevModeManager instance
        """
        self._manager = manager
        self._console: Optional[Any] = None
        self._command_buffer = ""
        self._history: Deque[str] = deque(maxlen=self.MAX_HISTORY)
        self._history_index = -1
        self._running = False
        self._overlay_active = False
        self._overlay_content = ""
        self._input_thread: Optional[threading.Thread] = None
        self._log_buffer: Deque[str] = deque(maxlen=self.MAX_LOG_LINES)
        self._status_message = ""
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._last_render_time = 0.0
        self._render_interval = 0.1
        self._cursor_visible = True
        self._cursor_blink_time = 0.5
        self._last_cursor_toggle = 0.0
        
        self._console = Console()
        
        _LOG.info("PiscesLxDevModeUI initialized")
    
    def start(self) -> None:
        """
        Start the UI input loop.
        
        This method starts the keyboard input thread and begins
        rendering the command bar.
        """
        if self._running:
            return
        
        self._running = True
        self._input_thread = threading.Thread(
            target=self._input_loop,
            daemon=True,
            name="DevModeUI-Input"
        )
        self._input_thread.start()
        _LOG.info("Developer mode UI started")
    
    def stop(self) -> None:
        """
        Stop the UI and clean up resources.
        
        This method stops the input thread and clears the display.
        """
        self._running = False
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
            self._input_thread = None
        _LOG.info("Developer mode UI stopped")
    
    def is_running(self) -> bool:
        """
        Check if the UI is running.
        
        Returns:
            bool: True if the UI is active
        """
        return self._running
    
    def add_log(self, message: str) -> None:
        """
        Add a log message to the display buffer.
        
        Args:
            message: Log message to add
        """
        with self._lock:
            self._log_buffer.append(message)
    
    def set_status(self, message: str) -> None:
        """
        Set the status message displayed in the command bar.
        
        Args:
            message: Status message to display
        """
        with self._lock:
            self._status_message = message
    
    def show_overlay(self, content: str) -> None:
        """
        Display an overlay with the given content.
        
        Args:
            content: Text content to display in the overlay
        """
        with self._lock:
            self._overlay_active = True
            self._overlay_content = content
            self._render()
    
    def hide_overlay(self) -> None:
        """
        Hide the current overlay and return to main view.
        """
        with self._lock:
            self._overlay_active = False
            self._overlay_content = ""
            self._render()
    
    def is_overlay_active(self) -> bool:
        """
        Check if an overlay is currently displayed.
        
        Returns:
            bool: True if overlay is active
        """
        return self._overlay_active
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a specific event.
        
        Args:
            event: Event name (e.g., 'command', 'quit')
            callback: Function to call when event occurs
        """
        self._callbacks[event] = callback
    
    def _input_loop(self) -> None:
        """
        Main input handling loop.
        
        This method runs in a separate thread and handles
        keyboard input for the command bar.
        """
        while self._running:
            try:
                if self._check_input():
                    self._render()
                time.sleep(0.01)
            except Exception as e:
                _LOG.debug("Input loop error", error=str(e))
    
    def _check_input(self) -> bool:
        """
        Check for and process keyboard input.
        
        Returns:
            bool: True if input was processed
        """
        try:
            if sys.platform == 'win32':
                return self._check_input_windows()
            else:
                return self._check_input_unix()
        except Exception:
            return False
    
    def _check_input_windows(self) -> bool:
        """
        Check for keyboard input on Windows.
        
        Returns:
            bool: True if input was processed
        """
        try:
            import msvcrt
            if not msvcrt.kbhit():
                return False
            
            ch = msvcrt.getch()
            
            if ch == b'\x03':
                self._handle_quit()
                return True
            
            if ch == b'\xe0':
                ch2 = msvcrt.getch()
                return self._handle_special_key_windows(ch2)
            
            try:
                char = ch.decode('utf-8')
                return self._handle_char(char)
            except UnicodeDecodeError:
                return False
        except Exception:
            return False
    
    def _check_input_unix(self) -> bool:
        """
        Check for keyboard input on Unix-like systems.
        
        Returns:
            bool: True if input was processed
        """
        try:
            import select
            import termios
            import tty
            
            if not select.select([sys.stdin], [], [], 0)[0]:
                return False
            
            ch = sys.stdin.read(1)
            
            if ch == '\x03':
                self._handle_quit()
                return True
            
            if ch == '\x1b':
                import sys
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        return self._handle_special_key_unix(ch3)
                return False
            
            return self._handle_char(ch)
        except Exception:
            return False
    
    def _handle_special_key_windows(self, ch: bytes) -> bool:
        """
        Handle special keys on Windows.
        
        Args:
            ch: The special key code
        
        Returns:
            bool: True if key was handled
        """
        if ch == b'H':
            self._history_up()
            return True
        elif ch == b'P':
            self._history_down()
            return True
        return False
    
    def _handle_special_key_unix(self, ch: str) -> bool:
        """
        Handle special keys on Unix-like systems.
        
        Args:
            ch: The special key character
        
        Returns:
            bool: True if key was handled
        """
        if ch == 'A':
            self._history_up()
            return True
        elif ch == 'B':
            self._history_down()
            return True
        return False
    
    def _handle_char(self, char: str) -> bool:
        """
        Handle a regular character input.
        
        Args:
            char: The input character
        
        Returns:
            bool: True if character was handled
        """
        if char == '\r' or char == '\n':
            self._execute_command()
            return True
        elif char == '\x7f' or char == '\x08':
            self._backspace()
            return True
        elif char == '\x1b':
            if self._overlay_active:
                self.hide_overlay()
                return True
            return False
        elif char.isprintable():
            self._command_buffer += char
            return True
        return False
    
    def _backspace(self) -> None:
        """
        Remove the last character from the command buffer.
        """
        if self._command_buffer:
            self._command_buffer = self._command_buffer[:-1]
    
    def _history_up(self) -> None:
        """
        Navigate up in command history.
        """
        if not self._history:
            return
        
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._command_buffer = list(self._history)[-(self._history_index + 1)]
    
    def _history_down(self) -> None:
        """
        Navigate down in command history.
        """
        if self._history_index > 0:
            self._history_index -= 1
            self._command_buffer = list(self._history)[-(self._history_index + 1)]
        elif self._history_index == 0:
            self._history_index = -1
            self._command_buffer = ""
    
    def _execute_command(self) -> None:
        """
        Execute the current command in the buffer.
        """
        command = self._command_buffer.strip()
        self._command_buffer = ""
        self._history_index = -1
        
        if not command:
            return
        
        if command not in self._history:
            self._history.append(command)
        
        if command.lower() in ('q', ':q', 'quit', ':quit'):
            if self._overlay_active:
                self.hide_overlay()
            return
        
        if 'command' in self._callbacks:
            try:
                self._callbacks['command'](command)
            except Exception as e:
                _LOG.error("Callback error", error=str(e))
    
    def _handle_quit(self) -> None:
        """
        Handle quit signal (Ctrl+C).
        """
        if 'quit' in self._callbacks:
            try:
                self._callbacks['quit']()
            except Exception:
                pass
    
    def _render(self) -> None:
        """
        Render the current UI state.
        """
        current_time = time.time()
        if current_time - self._last_render_time < self._render_interval:
            return
        self._last_render_time = current_time
        
        if current_time - self._last_cursor_toggle > self._cursor_blink_time:
            self._cursor_visible = not self._cursor_visible
            self._last_cursor_toggle = current_time
        
        self._render_rich()
    
    def _render_rich(self) -> None:
        """
        Render UI using Rich library.
        """
        if self._console is None:
            return
        
        if self._overlay_active:
            self._render_overlay_rich()
        else:
            self._render_command_bar_rich()
    
    def _render_overlay_rich(self) -> None:
        """
        Render the overlay using Rich.
        """
        if self._console is None:
            return
        
        panel = Panel(
            self._overlay_content,
            title="[bold cyan]Developer Mode[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self._console.print(panel)
        self._console.print("[dim]Press 'q' or Escape to return to main view[/dim]")
    
    def _render_command_bar_rich(self) -> None:
        """
        Render the command bar using Rich.
        """
        if self._console is None:
            return
        
        cursor_char = "\u2588" if self._cursor_visible else " "
        
        command_text = Text()
        command_text.append("> ")
        command_text.append(self._command_buffer)
        command_text.append(cursor_char, style=Style(blink=True, bold=True))
        
        status = self._status_message or "Type :help for commands"
        status_text = Text()
        status_text.append("[Dev Mode] ", style=Style(color="cyan", bold=True))
        status_text.append(status, style=Style(color="dim"))
        
        panel = Panel(
            Text.assemble(command_text, "\n", status_text),
            title="[bold]Command Bar[/bold]",
            border_style="blue",
            height=self.COMMAND_BAR_HEIGHT
        )
        
        self._console.print(panel)
    
    def get_command_buffer(self) -> str:
        """
        Get the current command buffer content.
        
        Returns:
            str: Current command being typed
        """
        return self._command_buffer
    
    def clear_command_buffer(self) -> None:
        """
        Clear the command buffer.
        """
        self._command_buffer = ""
    
    def get_history(self) -> List[str]:
        """
        Get the command history.
        
        Returns:
            List[str]: List of previous commands
        """
        return list(self._history)
    
    def get_log_buffer(self) -> List[str]:
        """
        Get the log buffer content.
        
        Returns:
            List[str]: Recent log lines
        """
        return list(self._log_buffer)
    
    def render_full_display(self, logs: Optional[List[str]] = None) -> None:
        """
        Render the full display including logs and command bar.
        
        Args:
            logs: Optional list of log lines to display
        """
        if logs:
            with self._lock:
                for log in logs[-self.MAX_LOG_LINES:]:
                    self._log_buffer.append(log)
        
        self._render()
