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
PiscesLx Developer Mode Overlay.

This module implements the temporary overlay display for developer mode,
providing a full-screen view for command results that can be dismissed
by pressing 'q'.

Architecture:
    The overlay system provides:
    
    1. PiscesLxDevModeOverlay: Full-screen overlay renderer
       - Displays command results in a formatted panel
       - Supports scrolling for long content
       - Handles keyboard input for dismissal
    
    2. Integration with UI:
       - Triggered by commands that return is_overlay=True
       - Dismissed by pressing 'q' or Escape
       - Returns to main view with command bar

Layout:
    +------------------------------------------+
    | [Developer Mode]                         |
    |------------------------------------------|
    |                                          |
    |         Command Result Content           |
    |         (Scrollable if long)             |
    |                                          |
    |------------------------------------------|
    | Press 'q' or Escape to return            |
    +------------------------------------------+

Usage:
    The overlay is typically managed by PiscesLxDevModeUI:
    
    >>> overlay = PiscesLxDevModeOverlay()
    >>> overlay.show("Memory details...")
    >>> # User presses 'q' to dismiss
    >>> overlay.hide()
"""

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.table import Table
from rich.layout import Layout

from utils.paths import get_log_file
from utils.dc import PiscesLxLogger


_LOG = PiscesLxLogger("PiscesLx.Tools.Dev", file_path=get_log_file("PiscesLx.Tools.Dev"), enable_file=True)


class PiscesLxDevModeOverlay:
    """
    Temporary overlay display for developer mode command results.
    
    This class provides a full-screen overlay that displays command output
    and can be dismissed by pressing 'q' or Escape.
    
    Attributes:
        _console: Rich console for rendering
        _content: Current overlay content
        _title: Overlay title
        _visible: Whether the overlay is currently visible
        _scroll_offset: Current scroll position for long content
        _max_height: Maximum display height
        _on_dismiss: Callback when overlay is dismissed
    
    Example:
        >>> overlay = PiscesLxDevModeOverlay()
        >>> overlay.show(":mem output", "Memory Details")
        >>> # User presses 'q'
        >>> overlay.hide()
    """
    
    DEFAULT_TITLE = "Developer Mode"
    FOOTER_HEIGHT = 2
    HEADER_HEIGHT = 2
    
    def __init__(self, max_height: int = 40):
        """
        Initialize the overlay renderer.
        
        Args:
            max_height: Maximum height for content display
        """
        self._console: Optional[Any] = None
        self._content = ""
        self._title = self.DEFAULT_TITLE
        self._visible = False
        self._scroll_offset = 0
        self._max_height = max_height
        self._on_dismiss: Optional[Callable] = None
        self._content_lines: List[str] = []
        
        self._console = Console()
        
        _LOG.debug("PiscesLxDevModeOverlay initialized")
    
    def show(self, content: str, title: Optional[str] = None) -> None:
        """
        Display the overlay with the given content.
        
        Args:
            content: Text content to display
            title: Optional title for the overlay
        """
        self._content = content
        self._title = title or self.DEFAULT_TITLE
        self._content_lines = content.split('\n')
        self._scroll_offset = 0
        self._visible = True
        self._render()
        _LOG.debug("Overlay shown", title=self._title, lines=len(self._content_lines))
    
    def hide(self) -> None:
        """
        Hide the overlay and return to main view.
        """
        self._visible = False
        self._content = ""
        self._content_lines = []
        self._scroll_offset = 0
        
        if self._on_dismiss:
            try:
                self._on_dismiss()
            except Exception as e:
                _LOG.warning("Dismiss callback error", error=str(e))
        
        _LOG.debug("Overlay hidden")
    
    def is_visible(self) -> bool:
        """
        Check if the overlay is currently visible.
        
        Returns:
            bool: True if overlay is visible
        """
        return self._visible
    
    def scroll_up(self, lines: int = 5) -> None:
        """
        Scroll the content up.
        
        Args:
            lines: Number of lines to scroll
        """
        if not self._visible:
            return
        
        self._scroll_offset = max(0, self._scroll_offset - lines)
        self._render()
    
    def scroll_down(self, lines: int = 5) -> None:
        """
        Scroll the content down.
        
        Args:
            lines: Number of lines to scroll
        """
        if not self._visible:
            return
        
        max_offset = max(0, len(self._content_lines) - self._get_visible_lines())
        self._scroll_offset = min(max_offset, self._scroll_offset + lines)
        self._render()
    
    def scroll_to_top(self) -> None:
        """
        Scroll to the top of the content.
        """
        if not self._visible:
            return
        
        self._scroll_offset = 0
        self._render()
    
    def scroll_to_bottom(self) -> None:
        """
        Scroll to the bottom of the content.
        """
        if not self._visible:
            return
        
        max_offset = max(0, len(self._content_lines) - self._get_visible_lines())
        self._scroll_offset = max_offset
        self._render()
    
    def set_dismiss_callback(self, callback: Callable) -> None:
        """
        Set the callback for when overlay is dismissed.
        
        Args:
            callback: Function to call when dismissed
        """
        self._on_dismiss = callback
    
    def _get_visible_lines(self) -> int:
        """
        Calculate the number of visible content lines.
        
        Returns:
            int: Number of lines that can be displayed
        """
        return self._max_height - self.HEADER_HEIGHT - self.FOOTER_HEIGHT
    
    def _render(self) -> None:
        """
        Render the overlay.
        """
        self._render_rich()
    
    def _render_rich(self) -> None:
        """
        Render overlay using Rich library.
        """
        if self._console is None:
            return
        
        visible_lines = self._get_visible_lines()
        start = self._scroll_offset
        end = min(start + visible_lines, len(self._content_lines))
        
        visible_content = '\n'.join(self._content_lines[start:end])
        
        footer_text = "Press 'q' or Escape to return"
        if len(self._content_lines) > visible_lines:
            footer_text = f"Lines {start + 1}-{end} of {len(self._content_lines)} | {footer_text}"
        
        panel = Panel(
            visible_content,
            title=f"[bold cyan]{self._title}[/bold cyan]",
            subtitle=f"[dim]{footer_text}[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        
        self._console.clear()
        self._console.print(panel)
    
    def handle_key(self, key: str) -> bool:
        """
        Handle a keyboard input while overlay is visible.
        
        Args:
            key: The key that was pressed
        
        Returns:
            bool: True if the key was handled
        """
        if not self._visible:
            return False
        
        key_lower = key.lower()
        
        if key_lower in ('q', 'escape', '\x1b'):
            self.hide()
            return True
        
        if key_lower in ('up', 'k'):
            self.scroll_up(1)
            return True
        
        if key_lower in ('down', 'j'):
            self.scroll_down(1)
            return True
        
        if key_lower in ('pageup', 'b'):
            self.scroll_up(self._get_visible_lines())
            return True
        
        if key_lower in ('pagedown', ' '):
            self.scroll_down(self._get_visible_lines())
            return True
        
        if key_lower in ('home', 'g'):
            self.scroll_to_top()
            return True
        
        if key_lower in ('end', 'G'):
            self.scroll_to_bottom()
            return True
        
        return False
    
    def get_content(self) -> str:
        """
        Get the current overlay content.
        
        Returns:
            str: Current content text
        """
        return self._content
    
    def get_title(self) -> str:
        """
        Get the current overlay title.
        
        Returns:
            str: Current title
        """
        return self._title
    
    def get_scroll_position(self) -> Tuple[int, int, int]:
        """
        Get the current scroll position.
        
        Returns:
            Tuple[int, int, int]: (current_offset, visible_lines, total_lines)
        """
        return (self._scroll_offset, self._get_visible_lines(), len(self._content_lines))


class PiscesLxDevModeOverlayManager:
    """
    Manager for coordinating multiple overlay displays.
    
    This class manages overlay state and provides utilities for
    formatting different types of content for display.
    
    Attributes:
        _overlays: Stack of overlay instances
        _current: Currently active overlay
        _formatters: Registered content formatters
    
    Example:
        >>> manager = PiscesLxDevModeOverlayManager()
        >>> manager.show_result(":mem", memory_info, "Memory Details")
    """
    
    def __init__(self):
        """Initialize the overlay manager."""
        self._overlays: List[PiscesLxDevModeOverlay] = []
        self._current: Optional[PiscesLxDevModeOverlay] = None
        self._formatters: Dict[str, Callable] = {}
        
        self._register_default_formatters()
    
    def _register_default_formatters(self) -> None:
        """Register default content formatters."""
        self._formatters['memory'] = self._format_memory
        self._formatters['layer'] = self._format_layer
        self._formatters['gradient'] = self._format_gradient
        self._formatters['config'] = self._format_config
        self._formatters['help'] = self._format_help
    
    def create_overlay(self, max_height: int = 40) -> PiscesLxDevModeOverlay:
        """
        Create a new overlay instance.
        
        Args:
            max_height: Maximum height for the overlay
        
        Returns:
            PiscesLxDevModeOverlay: New overlay instance
        """
        overlay = PiscesLxDevModeOverlay(max_height=max_height)
        return overlay
    
    def show_result(self, command: str, result: str, title: Optional[str] = None) -> None:
        """
        Show a command result in an overlay.
        
        Args:
            command: The command that produced the result
            result: The result text to display
            title: Optional title for the overlay
        """
        if self._current is None:
            self._current = self.create_overlay()
        
        formatted_result = self._format_result(command, result)
        self._current.show(formatted_result, title)
    
    def hide_current(self) -> None:
        """Hide the current overlay."""
        if self._current is not None:
            self._current.hide()
    
    def is_visible(self) -> bool:
        """
        Check if any overlay is visible.
        
        Returns:
            bool: True if an overlay is visible
        """
        return self._current is not None and self._current.is_visible()
    
    def handle_key(self, key: str) -> bool:
        """
        Handle keyboard input for the current overlay.
        
        Args:
            key: The key that was pressed
        
        Returns:
            bool: True if the key was handled
        """
        if self._current is not None:
            return self._current.handle_key(key)
        return False
    
    def _format_result(self, command: str, result: str) -> str:
        """
        Format a command result for display.
        
        Args:
            command: The command that produced the result
            result: The raw result text
        
        Returns:
            str: Formatted result text
        """
        cmd_base = command.lstrip(':').split()[0].lower() if command else ''
        
        formatter = self._formatters.get(cmd_base)
        if formatter:
            try:
                return formatter(result)
            except Exception:
                pass
        
        return result
    
    def _format_memory(self, content: str) -> str:
        """Format memory information."""
        return content
    
    def _format_layer(self, content: str) -> str:
        """Format layer information."""
        return content
    
    def _format_gradient(self, content: str) -> str:
        """Format gradient information."""
        return content
    
    def _format_config(self, content: str) -> str:
        """Format configuration information."""
        return content
    
    def _format_help(self, content: str) -> str:
        """Format help information."""
        return content
    
    def register_formatter(self, command: str, formatter: Callable) -> None:
        """
        Register a custom formatter for a command.
        
        Args:
            command: Command name (without ':')
            formatter: Function to format the result
        """
        self._formatters[command.lower()] = formatter
