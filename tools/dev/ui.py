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
PiscesLx Developer Mode UI - Vim-style Terminal UI using Rich.

This module implements a vim-style terminal UI using the Rich library,
providing cross-platform compatibility including Docker, SSH, and telnet.

Architecture:
    - Rich-based split screen: logs (top) + command bar (bottom)
    - Keyboard-driven vim-style commands (j/k/gg/G//)
    - Real-time log capture and display with Live
    - Search highlighting
"""

import sys
import time
import traceback
import threading
from collections import deque
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text
from rich.live import Live

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", file_path=get_log_file("PiscesLx.Tools.Dev"), enable_file=True)


@dataclass
class PiscesLxDevModeLogHandler:
    _instance = None

    def __init__(self):
        self._buffer: deque = deque(maxlen=1000)
        self._callbacks: List[Callable] = []
        self._search_term: str = ""
        self._search_matches: List[int] = []
        self._current_match: int = 0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def emit(self, message: str):
        if message is None:
            return
        self._buffer.append(message)
        if self._search_term:
            self._update_search_matches()
        for cb in self._callbacks:
            try:
                cb(message)
            except Exception:
                pass

    def register_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def get_all(self) -> List[str]:
        return list(self._buffer)

    def get_recent(self, count: int = 100) -> List[str]:
        return list(self._buffer)[-count:]

    def set_search(self, term: str):
        self._search_term = term
        self._update_search_matches()

    def _update_search_matches(self):
        self._search_matches = [
            i for i, msg in enumerate(self._buffer)
            if msg is not None and self._search_term.lower() in str(msg).lower()
        ]
        self._current_match = 0

    def next_match(self):
        if self._search_matches:
            self._current_match = (self._current_match + 1) % len(self._search_matches)
            return self._search_matches[self._current_match]
        return None

    def prev_match(self):
        if self._search_matches:
            self._current_match = (self._current_match - 1) % len(self._search_matches)
            return self._search_matches[self._current_match]
        return None

    def get_current_match_index(self) -> Optional[int]:
        if self._search_matches:
            return self._search_matches[self._current_match]
        return None

    def clear_search(self):
        self._search_term = ""
        self._search_matches = []
        self._current_match = 0


class PiscesLxDevModeUI:
    """
    Vim-style Developer Mode UI using Rich library.

    Supported Vim-style commands:
        j, k          - Navigate down/up one line
        gg, G         - Jump to top/bottom
        /             - Search forward
        n, N          - Next/previous search match
        q, quit       - Exit
        pause, resume - Pause/resume training
        ?             - Show help
    """

    KEY_UP = "up"
    KEY_DOWN = "down"
    KEY_LEFT = "left"
    KEY_RIGHT = "right"
    KEY_ENTER = "enter"
    KEY_ESCAPE = "escape"
    KEY_BACKSPACE = "backspace"
    KEY_DELETE = "delete"
    KEY_HOME = "home"
    KEY_END = "end"
    KEY_PAGE_UP = "pageup"
    KEY_PAGE_DOWN = "pagedown"

    REFRESH_RATE = 10
    COMMAND_BAR_HEIGHT = 3
    MAX_HISTORY = 100
    MAX_LOG_LINES = 1000

    def __init__(self, manager):
        self._manager = manager
        self._log_handler = PiscesLxDevModeLogHandler.get_instance()
        self._log_handler.register_callback(self._on_log_message)

        self._console: Optional[Console] = None
        self._live: Optional[Live] = None
        self._running = False
        self._paused = False

        self._input_buffer = ""
        self._command_mode = False
        self._search_mode = False
        self._vim_buffer = ""

        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._history_index = -1

        self._offset = 0
        self._max_display = 30
        self._total_lines = 0

        self._status_message = "Ready"
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()

        self._help_lines: List[str] = [
            "=== PiscesLx Developer Mode (Vim-style) ===",
            "Navigation: j/k (down/up), gg/G (top/bottom)",
            "Search: / (forward), n/N (next/prev match)",
            "Commands: q(quit), pause, resume, status",
            "==========================================",
        ]

        self._console = Console()
        _LOG.info("PiscesLxDevModeUI initialized with Rich (Vim-style)")

    def register_callback(self, event: str, callback: Callable):
        with self._lock:
            self._callbacks[event] = callback

    def _on_log_message(self, message: str):
        pass

    def _create_layout(self):
        log_lines = self._log_handler.get_all()
        self._total_lines = len(log_lines)

        try:
            terminal_height = self._console.size.height
        except Exception:
            terminal_height = 24

        if terminal_height <= 0:
            terminal_height = 24

        log_area_height = max(1, terminal_height - 1)
        self._max_display = log_area_height

        visible_lines = log_lines[self._offset:self._offset + log_area_height]

        search_term = self._log_handler._search_term
        rendered_lines = []

        for raw_line in visible_lines:
            if raw_line is None:
                rendered_lines.append(Text(""))
                continue
            line_str = str(raw_line)
            if not line_str:
                rendered_lines.append(Text(""))
                continue
            if search_term:
                rendered_lines.append(self._highlight_search(line_str, search_term))
            else:
                rendered_lines.append(Text(line_str, style="white"))

        while len(rendered_lines) < log_area_height:
            rendered_lines.append(Text(""))

        combined = Text()
        for i, line_text in enumerate(rendered_lines):
            if i > 0:
                combined.append("\n")
            combined.append_text(line_text)

        combined.append("\n")
        if self._command_mode:
            combined.append(f":{self._input_buffer}_")
        elif self._search_mode:
            combined.append(f"/{self._input_buffer}_")
        else:
            combined.append(":")

        return combined

    def _highlight_search(self, text: str, term: str) -> Text:
        result = Text()
        if not text or not term:
            return Text(str(text) if text else "")
        term_lower = term.lower()
        text_lower = text.lower()
        idx = 0
        while True:
            pos = text_lower.find(term_lower, idx)
            if pos == -1:
                result.append(text[idx:])
                break
            result.append(text[idx:pos])
            result.append(text[pos:pos + len(term)], style="bold yellow on red")
            idx = pos + len(term)
        return result

    def _read_key(self) -> Optional[str]:
        import select as _select

        if not _select.select([sys.stdin], [], [], 0.05)[0]:
            return None

        ch = sys.stdin.read(1)
        if not ch:
            return None

        if ch == '\x1b':
            return self._read_escape_seq()

        if ch in ('\r', '\n'):
            return "enter"
        if ch in ('\x7f', '\x08'):
            return "backspace"
        if ch == '\x03':
            raise KeyboardInterrupt

        return ch

    def _read_escape_seq(self) -> str:
        import select as _select

        if not _select.select([sys.stdin], [], [], 0.02)[0]:
            return "escape"

        ch = sys.stdin.read(1)
        if ch != '[':
            return "escape"

        if not _select.select([sys.stdin], [], [], 0.02)[0]:
            return "escape"

        ch = sys.stdin.read(1)

        seq_map = {
            'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left',
            'H': 'home', 'F': 'end',
        }

        if ch in seq_map:
            return seq_map[ch]

        if ch in ('1', '3', '5', '6', '4'):
            if _select.select([sys.stdin], [], [], 0.02)[0]:
                ch2 = sys.stdin.read(1)
                ext_map = {
                    '3~': 'delete', '5~': 'pageup', '6~': 'pagedown',
                    '4~': 'end', '1~': 'home',
                }
                return ext_map.get(ch + ch2, "escape")

        return "escape"

    def _handle_vim_key(self, key: str) -> bool:
        key_lower = key.lower()

        if self._search_mode:
            if key == "escape":
                self._search_mode = False
                self._input_buffer = ""
                self._vim_buffer = ""
                return True
            elif key == "enter":
                if self._input_buffer.startswith('/'):
                    search_term = self._input_buffer[1:]
                    self._log_handler.set_search(search_term)
                self._search_mode = False
                self._input_buffer = ""
                self._vim_buffer = ""
                return True
            elif key == "backspace":
                if len(self._input_buffer) > 1:
                    self._input_buffer = self._input_buffer[:-1]
                else:
                    self._search_mode = False
                    self._input_buffer = ""
                    self._vim_buffer = ""
                return True
            elif len(key) == 1:
                self._input_buffer += key
            return True

        if self._command_mode:
            if key == "escape":
                self._command_mode = False
                self._input_buffer = ""
                self._vim_buffer = ""
                return True
            elif key == "enter":
                if len(self._input_buffer) > 1 and self._input_buffer.startswith(':'):
                    cmd = self._input_buffer[1:].strip()
                    if cmd:
                        self._process_command(cmd)
                self._command_mode = False
                self._input_buffer = ""
                self._vim_buffer = ""
                return True
            elif key == "backspace":
                if len(self._input_buffer) > 1:
                    self._input_buffer = self._input_buffer[:-1]
                else:
                    self._command_mode = False
                    self._input_buffer = ""
                    self._vim_buffer = ""
                return True
            elif len(key) == 1:
                self._input_buffer += key
            return True

        if key == 'escape':
            self._offset = max(0, self._total_lines - self._max_display)
            self._vim_buffer = ""
            self._input_buffer = ""
            return True

        if len(key) != 1:
            if key == "up":
                self._offset = max(0, self._offset - 1)
                self._vim_buffer = ""
                return True
            elif key == "down":
                self._offset = min(self._offset + 1, max(0, self._total_lines - self._max_display))
                self._vim_buffer = ""
                return True
            elif key == "pageup":
                self._offset = max(0, self._offset - self._max_display)
                self._vim_buffer = ""
                return True
            elif key == "pagedown":
                self._offset = min(self._offset + self._max_display, max(0, self._total_lines - self._max_display))
                self._vim_buffer = ""
                return True
            elif key == "home":
                self._offset = 0
                self._vim_buffer = ""
                return True
            elif key == "end":
                self._offset = max(0, self._total_lines - self._max_display)
                self._vim_buffer = ""
                return True
            return False

        if key_lower == 'j':
            self._offset = min(self._offset + 1, max(0, self._total_lines - self._max_display))
            self._vim_buffer = ""
            return True
        elif key_lower == 'k':
            self._offset = max(0, self._offset - 1)
            self._vim_buffer = ""
            return True
        elif key_lower == 'g':
            if self._vim_buffer == 'g':
                self._offset = 0
                self._vim_buffer = ""
            else:
                self._vim_buffer = "g"
            return True
        elif key == '/':
            self._search_mode = True
            self._input_buffer = "/"
            self._vim_buffer = ""
            return True
        elif key == ':':
            self._command_mode = True
            self._input_buffer = ":"
            self._vim_buffer = ""
            return True
        elif key == '?':
            self._show_help()
            self._vim_buffer = ""
            return True
        elif key_lower == 'n':
            match_idx = self._log_handler.next_match()
            if match_idx is not None:
                self._offset = max(0, match_idx - self._max_display // 2)
            self._vim_buffer = ""
            return True
        elif key == 'N':
            match_idx = self._log_handler.prev_match()
            if match_idx is not None:
                self._offset = max(0, match_idx - self._max_display // 2)
            self._vim_buffer = ""
            return True
        elif key_lower == 'q':
            self._running = False
            return True
        elif key == 'G':
            self._offset = max(0, self._total_lines - self._max_display)
            self._vim_buffer = ""
            return True

        return False

    def _process_command(self, cmd: str):
        if not cmd:
            return
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ('q', 'quit', 'exit'):
            self._running = False
        elif cmd_lower in ('pause', 'suspend'):
            self._paused = True
            self._status_message = "PAUSED"
        elif cmd_lower in ('resume', 'continue', 'start'):
            self._paused = False
            self._status_message = "Running"
        elif cmd_lower in ('status', 'st'):
            self._status_message = "Status: Running" if not self._paused else "Status: Paused"
        elif cmd_lower in ('help', '?', 'h'):
            self._show_help()
        elif cmd_lower.startswith('log '):
            try:
                count = int(cmd_lower.split()[1])
                self._show_logs(count)
            except (ValueError, IndexError):
                pass
        elif cmd_lower == 'clear':
            pass
        else:
            with self._lock:
                if "command" in self._callbacks:
                    try:
                        self._callbacks["command"](cmd)
                    except Exception as e:
                        self._log(f"Command error: {e}")

    def _show_help(self):
        for line in self._help_lines:
            self._log_handler.emit(line)

    def _show_logs(self, count: int):
        logs = self._log_handler.get_recent(count)
        self._log_handler.emit(f"=== Last {len(logs)} log entries ===")
        for log in logs:
            self._log_handler.emit(log)
        self._log_handler.emit("=" * 40)

    def _log(self, message: str):
        if self._console:
            try:
                self._console.print(message, end="\n")
            except Exception:
                pass

    def start(self) -> None:
        if self._running:
            return

        self._running = True

        self._input_thread = threading.Thread(
            target=self._run_input_loop,
            daemon=True,
            name="DevModeUI-Rich"
        )
        self._input_thread.start()

        _LOG.info("Developer mode UI started with Rich (Vim-style)")

    def _run_input_loop(self) -> None:
        if not self._console:
            return

        if not sys.stdin.isatty():
            try:
                with Live(self._create_layout(), console=self._console, refresh_per_second=10, screen=False) as live:
                    self._live = live
                    while self._running:
                        time.sleep(0.1)
                        self._live.update(self._create_layout())
            except Exception as e:
                _LOG.error(f"UI Error (non-tty): {e}")
            return

        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setcbreak(sys.stdin.fileno())

            with Live(self._create_layout(), console=self._console, refresh_per_second=10, screen=True) as live:
                self._live = live

                while self._running:
                    try:
                        key = self._read_key()
                        if key is not None:
                            self._handle_vim_key(key)
                        self._live.update(self._create_layout())

                    except KeyboardInterrupt:
                        self._running = False
                        break
                    except Exception:
                        time.sleep(0.05)

        except Exception as e:
            _LOG.error(f"UI Error: {e}\n{traceback.format_exc()}")
        finally:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

        _LOG.info("Developer mode UI stopped")

    def is_running(self) -> bool:
        return self._running

    def is_paused(self) -> bool:
        return self._paused

    def update_status(self, message: str) -> None:
        self._status_message = message

    def add_log(self, message: str) -> None:
        if message is not None:
            self._log_handler.emit(message)

    def show_overlay(self, text: str) -> None:
        if text is not None:
            self._log_handler.emit(str(text))

    def set_status(self, status: str) -> None:
        if status is not None:
            self._status_message = str(status)
