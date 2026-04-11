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
PiscesLx Developer Mode UI - Refactored with Rich Live + Layout.

This module implements a professional terminal UI for developer mode using
Rich's Live display with Layout for true split-screen rendering.

Architecture:
    The UI uses Rich's Layout system to create a persistent split-screen:
    
    +------------------------------------------+
    |                                          |
    |         Training Logs (scrollable)       |
    |         Layout: logs (ratio=4)           |
    |                                          |
    +------------------------------------------+
    | > _                                      |
    | [Dev Mode] Type :help for commands       |
    | Layout: command (size=3, fixed)          |
    +------------------------------------------+

Key Improvements:
    1. Live Display: Persistent UI that won't be overwritten by logs
    2. Split Layout: Logs and command bar in separate regions
    3. Blocking Input: Reliable keyboard capture via queue
    4. Log Handler: Captures training logs for display

Usage:
    >>> from tools.dev import PiscesLxDevModeManager
    >>> manager = PiscesLxDevModeManager.get_instance()
    >>> manager.attach(trainer)
    >>> # UI automatically starts with Live display
"""

import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from utils.paths import get_log_file
from utils.dc import PiscesLxLogger


_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", file_path=get_log_file("PiscesLx.Tools.Dev"), enable_file=True)


class PiscesLxDevModeLogHandler:
    """
    Log handler that captures training logs for display in the UI.
    
    This handler intercepts log records and forwards them to the UI
    for real-time display in the logs panel.
    
    Attributes:
        _ui: Reference to the PiscesLxDevModeUI instance
        _logs: Buffer of recent log entries
        _max_logs: Maximum number of logs to keep
    """
    
    def __init__(self, ui: 'PiscesLxDevModeUI', max_logs: int = 200):
        self._ui = ui
        self._logs: Deque[str] = deque(maxlen=max_logs)
        self._max_logs = max_logs
        self._lock = threading.Lock()
    
    def emit(self, record: str) -> None:
        """
        Add a log record to the buffer and update UI.
        
        Args:
            record: The log message to add
        """
        with self._lock:
            self._logs.append(record)
            self._ui.update_logs(list(self._logs))
    
    def add_log(self, message: str) -> None:
        """
        Add a log message directly.
        
        Args:
            message: The log message to add
        """
        self.emit(message)
    
    def get_logs(self) -> List[str]:
        """
        Get all buffered logs.
        
        Returns:
            List[str]: List of log messages
        """
        with self._lock:
            return list(self._logs)
    
    def clear(self) -> None:
        """Clear all buffered logs."""
        with self._lock:
            self._logs.clear()


class PiscesLxDevModeUI:
    """
    Professional terminal UI for developer mode with Live display.
    
    This class provides a vim-style command interface using Rich's Live
    display system, ensuring the UI remains visible during training.
    
    The UI uses a split-screen layout:
    - Top section: Training logs (scrollable, auto-updating)
    - Bottom section: Command bar (fixed, always visible)
    
    Attributes:
        _manager: Reference to the PiscesLxDevModeManager
        _console: Rich console for rendering
        _layout: Split-screen layout
        _live: Live display instance
        _command_buffer: Current command being typed
        _history: Command history for navigation
        _history_index: Current position in history
        _running: Whether the UI is active
        _overlay_active: Whether an overlay is displayed
        _overlay_content: Current overlay content
        _input_thread: Thread for keyboard input
        _input_queue: Queue for input events
        _log_buffer: Recent log lines for display
        _status_message: Current status message
        _callbacks: Registered callback functions
        _log_handler: Log handler instance
    
    Example:
        >>> ui = PiscesLxDevModeUI(manager)
        >>> ui.start()
        >>> # UI is now running with Live display
        >>> ui.stop()
    """
    
    COMMAND_BAR_HEIGHT = 3
    MAX_LOG_LINES = 200
    MAX_HISTORY = 50
    REFRESH_RATE = 10
    
    def __init__(self, manager: Any):
        """
        Initialize the UI renderer with Live display support.
        
        Args:
            manager: The PiscesLxDevModeManager instance
        """
        self._manager = manager
        self._console = Console(force_terminal=True)
        self._layout = Layout()
        self._live: Optional[Live] = None
        
        self._command_buffer = ""
        self._history: Deque[str] = deque(maxlen=self.MAX_HISTORY)
        self._history_index = -1
        self._running = False
        self._overlay_active = False
        self._overlay_content = ""
        
        self._input_thread: Optional[threading.Thread] = None
        self._input_queue: queue.Queue = queue.Queue()
        
        self._log_buffer: Deque[str] = deque(maxlen=self.MAX_LOG_LINES)
        self._status_message = ""
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
        self._cursor_visible = True
        self._last_cursor_toggle = 0.0
        self._cursor_blink_interval = 0.5
        
        self._log_handler: Optional[PiscesLxDevModeLogHandler] = None
        
        self._setup_layout()
        
        _LOG.info("PiscesLxDevModeUI initialized with Live display")
    
    def _setup_layout(self) -> None:
        """
        Setup the split-screen layout.
        
        Creates a layout with:
        - logs: Top section for training logs (ratio=4)
        - command: Bottom section for command bar (fixed size=3)
        """
        self._layout.split_column(
            Layout(name="logs", ratio=4),
            Layout(name="command", size=self.COMMAND_BAR_HEIGHT)
        )
    
    def start(self) -> None:
        """
        Start the UI with Live display.
        
        This method starts the keyboard input thread and begins
        the Live display, which will persist throughout training.
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
        
        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=self.REFRESH_RATE,
            screen=True,
            transient=False
        )
        self._live.start()
        
        _LOG.info("Developer mode UI started with Live display")
    
    def stop(self) -> None:
        """
        Stop the UI and clean up resources.
        
        This method stops the Live display and input thread.
        """
        self._running = False
        
        if self._live is not None:
            self._live.stop()
            self._live = None
        
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
    
    def get_log_handler(self) -> PiscesLxDevModeLogHandler:
        """
        Get or create the log handler for this UI.
        
        Returns:
            PiscesLxDevModeLogHandler: The log handler instance
        """
        if self._log_handler is None:
            self._log_handler = PiscesLxDevModeLogHandler(self, self.MAX_LOG_LINES)
        return self._log_handler
    
    def update_logs(self, logs: List[str]) -> None:
        """
        Update the logs display with new log entries.
        
        Args:
            logs: List of log messages to display
        """
        with self._lock:
            self._log_buffer.clear()
            for log in logs[-self.MAX_LOG_LINES:]:
                self._log_buffer.append(log)
            self._refresh_display()
    
    def add_log(self, message: str) -> None:
        """
        Add a single log message to the display.
        
        Args:
            message: Log message to add
        """
        with self._lock:
            self._log_buffer.append(message)
            self._refresh_display()
    
    def set_status(self, message: str) -> None:
        """
        Set the status message displayed in the command bar.
        
        Args:
            message: Status message to display
        """
        with self._lock:
            self._status_message = message
            self._refresh_display()
    
    def show_overlay(self, content: str) -> None:
        """
        Display an overlay with the given content.
        
        Args:
            content: Text content to display in the overlay
        """
        with self._lock:
            self._overlay_active = True
            self._overlay_content = content
            self._refresh_display()
    
    def hide_overlay(self) -> None:
        """Hide the current overlay and return to main view."""
        with self._lock:
            self._overlay_active = False
            self._overlay_content = ""
            self._refresh_display()
    
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
        Main input handling loop running in separate thread.
        
        Uses blocking input read to ensure no key presses are missed.
        """
        while self._running:
            try:
                char = self._read_char_blocking()
                if char is not None:
                    self._input_queue.put(char)
                    self._process_input_queue()
            except Exception as e:
                _LOG.debug("Input loop error", error=str(e))
                time.sleep(0.01)
    
    def _read_char_blocking(self) -> Optional[str]:
        """
        Read a character with blocking wait.
        
        Returns:
            Optional[str]: The character read, or None on error
        """
        try:
            if sys.platform == 'win32':
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b'\xe0':
                        ch2 = msvcrt.getch()
                        if ch2 == b'H':
                            return '\x1b[A'
                        elif ch2 == b'P':
                            return '\x1b[B'
                        return None
                    if ch == b'\x03':
                        return '\x03'
                    try:
                        return ch.decode('utf-8')
                    except UnicodeDecodeError:
                        return None
                time.sleep(0.01)
                return None
            else:
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                return f'\x1b[{ch3}'
                        return '\x1b'
                    return ch
                return None
        except Exception:
            return None
    
    def _process_input_queue(self) -> None:
        """Process all pending input events from the queue."""
        while not self._input_queue.empty():
            try:
                char = self._input_queue.get_nowait()
                self._handle_char(char)
            except queue.Empty:
                break
            except Exception as e:
                _LOG.debug("Input processing error", error=str(e))
        
        self._refresh_display()
    
    def _handle_char(self, char: str) -> None:
        """
        Handle a character input.
        
        Args:
            char: The input character or escape sequence
        """
        if char == '\x1b[A':
            self._history_up()
        elif char == '\x1b[B':
            self._history_down()
        elif char in ('\r', '\n'):
            self._execute_command()
        elif char in ('\x7f', '\x08'):
            self._backspace()
        elif char == '\x1b':
            if self._overlay_active:
                self.hide_overlay()
        elif char == '\x03':
            self._handle_quit()
        elif char and char.isprintable():
            self._command_buffer += char
    
    def _backspace(self) -> None:
        """Remove the last character from the command buffer."""
        if self._command_buffer:
            self._command_buffer = self._command_buffer[:-1]
    
    def _history_up(self) -> None:
        """Navigate up in command history."""
        if not self._history:
            return
        
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._command_buffer = list(self._history)[-(self._history_index + 1)]
    
    def _history_down(self) -> None:
        """Navigate down in command history."""
        if self._history_index > 0:
            self._history_index -= 1
            self._command_buffer = list(self._history)[-(self._history_index + 1)]
        elif self._history_index == 0:
            self._history_index = -1
            self._command_buffer = ""
    
    def _execute_command(self) -> None:
        """Execute the current command in the buffer."""
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
        """Handle quit signal (Ctrl+C)."""
        if 'quit' in self._callbacks:
            try:
                self._callbacks['quit']()
            except Exception:
                pass
    
    def _refresh_display(self) -> None:
        """Refresh the Live display with current state."""
        if self._live is None or not self._running:
            return
        
        current_time = time.time()
        if current_time - self._last_cursor_toggle > self._cursor_blink_interval:
            self._cursor_visible = not self._cursor_visible
            self._last_cursor_toggle = current_time
        
        self._update_layout()
    
    def _update_layout(self) -> None:
        """Update the layout sections with current content."""
        if self._overlay_active:
            self._render_overlay_to_layout()
        else:
            self._render_logs_to_layout()
            self._render_command_to_layout()
    
    def _render_logs_to_layout(self) -> None:
        """Render the logs section."""
        logs_text = Text()
        
        with self._lock:
            logs = list(self._log_buffer)
        
        if logs:
            visible_logs = logs[-30:]
            for log in visible_logs:
                logs_text.append(log)
                logs_text.append("\n")
        else:
            logs_text.append("[Waiting for training logs...]", style=Style(color="dim"))
        
        panel = Panel(
            logs_text,
            title="[bold cyan]Training Logs[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )
        
        self._layout["logs"].update(panel)
    
    def _render_command_to_layout(self) -> None:
        """Render the command bar section."""
        cursor_char = "\u2588" if self._cursor_visible else " "
        
        command_text = Text()
        command_text.append("> ", style=Style(color="green", bold=True))
        command_text.append(self._command_buffer)
        command_text.append(cursor_char, style=Style(blink=True, bold=True))
        
        status = self._status_message or "Type :help for commands"
        status_text = Text()
        status_text.append("[Dev Mode] ", style=Style(color="cyan", bold=True))
        status_text.append(status, style=Style(color="dim"))
        
        content = Group(command_text, status_text)
        
        panel = Panel(
            content,
            title="[bold]Command Bar[/bold]",
            border_style="blue",
            padding=(0, 1)
        )
        
        self._layout["command"].update(panel)
    
    def _render_overlay_to_layout(self) -> None:
        """Render overlay content in the logs section."""
        panel = Panel(
            self._overlay_content,
            title="[bold cyan]Developer Mode - Overlay[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self._layout["logs"].update(panel)
        
        hint_text = Text()
        hint_text.append("Press ", style=Style(color="dim"))
        hint_text.append("'q'", style=Style(color="yellow", bold=True))
        hint_text.append(" or ", style=Style(color="dim"))
        hint_text.append("Escape", style=Style(color="yellow", bold=True))
        hint_text.append(" to return", style=Style(color="dim"))
        
        hint_panel = Panel(
            hint_text,
            border_style="dim",
            padding=(0, 1)
        )
        
        self._layout["command"].update(hint_panel)
    
    def get_command_buffer(self) -> str:
        """
        Get the current command buffer content.
        
        Returns:
            str: Current command being typed
        """
        return self._command_buffer
    
    def clear_command_buffer(self) -> None:
        """Clear the command buffer."""
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
        with self._lock:
            return list(self._log_buffer)
    
    def render_full_display(self, logs: Optional[List[str]] = None) -> None:
        """
        Render the full display including logs and command bar.
        
        Args:
            logs: Optional list of log lines to display
        """
        if logs:
            self.update_logs(logs)
        else:
            self._refresh_display()
