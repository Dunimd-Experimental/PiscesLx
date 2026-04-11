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
PiscesLx Developer Mode UI - Curses-based Terminal UI.

This module implements a vim-style terminal UI using curses/ncurses,
providing full terminal compatibility including SSH, Docker, and telnet.

Architecture:
    - Curses-based split screen: logs (top) + command bar (bottom)
    - Keyboard-driven vim-style commands
    - Real-time log capture and display
    - Non-blocking input with escape sequence support
"""

import os
import sys
import time
import queue
import threading
import curses
from collections import deque
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", enable_console=True, enable_file=False)


@dataclass
class PiscesLxDevModeLogHandler:
    _instance = None

    def __init__(self):
        self._buffer: deque = deque(maxlen=1000)
        self._callbacks: List[Callable] = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def emit(self, message: str):
        self._buffer.append(message)
        for cb in self._callbacks:
            try:
                cb(message)
            except Exception:
                pass

    def register_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def get_recent(self, count: int = 100) -> List[str]:
        return list(self._buffer)[-count:]


class PiscesLxDevModeUI:
    KEY_UP = curses.KEY_UP
    KEY_DOWN = curses.KEY_DOWN
    KEY_LEFT = curses.KEY_LEFT
    KEY_RIGHT = curses.KEY_RIGHT
    KEY_ENTER = curses.KEY_ENTER
    KEY_ESCAPE = 27
    KEY_BACKSPACE = curses.KEY_BACKSPACE
    KEY_DELETE = curses.KEY_DC
    KEY_HOME = curses.KEY_HOME
    KEY_END = curses.KEY_END
    KEY_PAGE_UP = curses.KEY_PPAGE
    KEY_PAGE_DOWN = curses.KEY_NPAGE

    REFRESH_RATE = 10
    COMMAND_BAR_HEIGHT = 3
    MAX_HISTORY = 100
    MAX_LOG_LINES = 1000

    def __init__(self, manager, use_curses: Optional[bool] = None):
        self._manager = manager
        self._log_handler = PiscesLxDevModeLogHandler.get_instance()
        self._log_handler.register_callback(self._on_log_message)

        self._command_buffer = ""
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._history_index = -1
        self._running = False
        self._paused = False

        self._input_thread: Optional[threading.Thread] = None
        self._input_queue: queue.Queue = queue.Queue()

        self._status_message = "Ready"
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()

        self._window: Optional[curses.window] = None
        self._logs_win: Optional[curses.window] = None
        self._cmd_win: Optional[curses.window] = None
        self._status_win: Optional[curses.window] = None

        self._log_offset = 0
        self._cmd_pos = 0

        self._use_curses = True

        _LOG.info("PiscesLxDevModeUI initialized with curses")

    def _check_curses_support(self) -> bool:
        if not sys.stdout.isatty():
            return False

        term = os.environ.get("TERM", "")
        if term == "dumb":
            return False

        try:
            curses.initscr()
            curses.endwin()
            return True
        except Exception:
            return False

    def register_callback(self, event: str, callback: Callable):
        with self._lock:
            self._callbacks[event] = callback

    def _on_log_message(self, message: str):
        pass

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._input_thread = threading.Thread(
            target=self._curses_main,
            daemon=True,
            name="DevModeUI-Curses"
        )
        self._input_thread.start()

        _LOG.info("Developer mode UI started with curses")

    def stop(self) -> None:
        self._running = False

        if self._window is not None:
            try:
                curses.nocbreak()
                self._window.keypad(False)
                curses.echo()
                curses.endwin()
            except Exception:
                pass
            self._window = None

        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
            self._input_thread = None

        _LOG.info("Developer mode UI stopped")

    def _curses_main(self, stdscr: curses.window) -> None:
        curses.use_default_colors()
        curses.curs_set(1)
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(True)
        stdscr.keypad(True)

        self._window = stdscr
        max_y, max_x = stdscr.getmaxyx()

        log_height = max_y - self.COMMAND_BAR_HEIGHT - 1
        self._logs_win = curses.newwin(log_height, max_x, 0, 0)
        self._logs_win.scrollok(True)
        self._logs_win.idlok(True)

        self._status_win = curses.newwin(1, max_x, log_height, 0)

        self._cmd_win = curses.newwin(self.COMMAND_BAR_HEIGHT, max_x, log_height + 1, 0)
        self._cmd_win.keypad(True)

        self._draw_border()
        self._refresh_all()

        last_refresh = time.time()
        refresh_interval = 1.0 / self.REFRESH_RATE

        while self._running:
            try:
                key = self._cmd_win.getch()

                if key != curses.ERR:
                    self._handle_input(key)

                now = time.time()
                if now - last_refresh >= refresh_interval:
                    self._refresh_all()
                    last_refresh = now

                time.sleep(0.01)

            except curses.error:
                time.sleep(0.01)
                continue
            except Exception:
                time.sleep(0.01)
                continue

    def _draw_border(self) -> None:
        if self._window is None:
            return

        max_y, max_x = self._window.getmaxyx()
        try:
            self._window.clear()
            self._window.border()
            title = " PiscesL1 Developer Mode - Press ? for help "
            self._window.addstr(0, (max_x - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(1))
            self._window.refresh()
        except curses.error:
            pass

    def _refresh_all(self) -> None:
        if self._logs_win:
            try:
                self._logs_win.refresh()
            except curses.error:
                pass

        if self._status_win:
            try:
                max_y, max_x = self._status_win.getmaxyx()
                self._status_win.clear()
                status_text = f" {self._status_message} "
                self._status_win.addstr(0, 0, status_text, curses.A_REVERSE)
                self._status_win.clrtoeol()
                self._status_win.refresh()
            except curses.error:
                pass

        if self._cmd_win:
            try:
                max_y, max_x = self._cmd_win.getmaxyx()
                self._cmd_win.clear()
                self._cmd_win.border()

                prompt = ": "
                self._cmd_win.addstr(1, 2, prompt, curses.A_BOLD | curses.color_pair(2))

                display_cmd = self._command_buffer[:max_x - len(prompt) - 4]
                if self._cmd_pos < len(display_cmd):
                    cursor_x = 2 + len(prompt) + self._cmd_pos
                else:
                    cursor_x = 2 + len(prompt) + len(display_cmd)

                self._cmd_win.addstr(1, 2 + len(prompt), display_cmd, curses.color_pair(3))
                self._cmd_win.move(1, min(cursor_x, max_x - 2))
                self._cmd_win.refresh()
            except curses.error:
                pass

    def _handle_input(self, key: int) -> None:
        if key == curses.KEY_UP:
            self._navigate_history(-1)
        elif key == curses.KEY_DOWN:
            self._navigate_history(1)
        elif key == curses.KEY_LEFT:
            self._cmd_pos = max(0, self._cmd_pos - 1)
        elif key == curses.KEY_RIGHT:
            self._cmd_pos = min(len(self._command_buffer), self._cmd_pos + 1)
        elif key == curses.KEY_HOME:
            self._cmd_pos = 0
        elif key == curses.KEY_END:
            self._cmd_pos = len(self._command_buffer)
        elif key == curses.KEY_BACKSPACE:
            if self._cmd_pos > 0 and self._command_buffer:
                self._command_buffer = (
                    self._command_buffer[: self._cmd_pos - 1]
                    + self._command_buffer[self._cmd_pos :]
                )
                self._cmd_pos -= 1
        elif key == curses.KEY_DC:
            if self._cmd_pos < len(self._command_buffer):
                self._command_buffer = (
                    self._command_buffer[: self._cmd_pos]
                    + self._command_buffer[self._cmd_pos + 1 :]
                )
        elif key == curses.KEY_ENTER or key in (10, 13):
            self._execute_command()
        elif key == 27:
            self._command_buffer = ""
            self._cmd_pos = 0
        elif 32 <= key <= 126:
            ch = chr(key)
            self._command_buffer = (
                self._command_buffer[: self._cmd_pos]
                + ch
                + self._command_buffer[self._cmd_pos :]
            )
            self._cmd_pos += 1

        self._refresh_all()

    def _navigate_history(self, direction: int) -> None:
        if not self._history:
            return

        new_index = self._history_index + direction

        if new_index < 0:
            new_index = 0
        elif new_index >= len(self._history):
            new_index = len(self._history) - 1

        if new_index != self._history_index:
            self._history_index = new_index
            self._command_buffer = self._history[new_index]
            self._cmd_pos = len(self._command_buffer)

    def _execute_command(self) -> None:
        cmd = self._command_buffer.strip()

        if cmd:
            self._history.append(cmd)
            self._history_index = len(self._history)

        self._command_buffer = ""
        self._cmd_pos = 0

        if cmd:
            self._process_command(cmd)

    def _process_command(self, cmd: str) -> None:
        with self._lock:
            if "command" in self._callbacks:
                try:
                    self._callbacks["command"](cmd)
                except Exception as e:
                    self._log(f"Command error: {e}")
            else:
                self._log(f"Command received: {cmd}")

        if cmd == "q" or cmd == "quit":
            self._running = False
        elif cmd == "?" or cmd == "help":
            self._show_help()
        elif cmd == "pause":
            self._paused = True
            self._status_message = "PAUSED - Press resume to continue"
            if "pause" in self._callbacks:
                self._callbacks["pause"]()
        elif cmd == "resume" or cmd == "continue":
            self._paused = False
            self._status_message = "Running"
            if "resume" in self._callbacks:
                self._callbacks["resume"]()
        elif cmd == "status":
            self._status_message = "Status: Running"
        elif cmd.startswith("log"):
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    count = int(parts[1])
                    self._show_logs(count)
                except ValueError:
                    self._log("Usage: log <count>")
            else:
                self._show_logs(20)

    def _log(self, message: str) -> None:
        if self._logs_win:
            try:
                max_y, max_x = self._logs_win.getmaxyx()
                self._logs_win.addstr(f"{message}\n")
                self._logs_win.clrtoeol()
                self._logs_win.refresh()
            except curses.error:
                pass

    def _show_help(self) -> None:
        help_text = [
            "=== PiscesL1 Developer Mode Commands ===",
            "q, quit           Exit developer mode",
            "pause             Pause training",
            "resume, continue  Resume training",
            "status            Show training status",
            "log <n>           Show last n log entries",
            "h, history        Show command history",
            "?, help           Show this help",
            "========================================",
        ]

        for line in help_text:
            self._log(line)

    def _show_logs(self, count: int) -> None:
        logs = self._log_handler.get_recent(count)
        self._log(f"=== Last {len(logs)} log entries ===")
        for log in logs[-count:]:
            self._log(log)
        self._log("=" * 40)

    def is_running(self) -> bool:
        return self._running

    def is_paused(self) -> bool:
        return self._paused

    def update_status(self, message: str) -> None:
        self._status_message = message

    def add_log(self, message: str) -> None:
        self._log_handler.emit(message)
        self._log(message)
