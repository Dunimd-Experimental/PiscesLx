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
CMU Type Definitions - Computer Use Agent Type System

This module provides comprehensive type definitions for the Computer Use Agent (CMU),
including enumerations, dataclasses, and type aliases for cross-platform device control.

Module Components:
    1. Enumerations:
       - POPSSCMUActionType: Action type enumeration
       - POPSSCMUPlatform: Platform type enumeration
       - POPSSCMUSafetyLevel: Safety level enumeration
       - POPSSCMUElementState: UI element state enumeration
       - POPSSCMUActionResultStatus: Action result status enumeration

    2. Dataclasses:
       - POPSSCMUCoordinate: 2D coordinate representation
       - POPSSCMURectangle: Rectangle region representation
       - POPSSCMUTarget: Action target specification
       - POPSSCMUAction: Unified action representation
       - POPSSCMUScreenState: Screen state snapshot
       - POPSSCMUActionResult: Action execution result
       - POPSSCMUElement: UI element representation
       - POPSSCMUTaskContext: Task execution context

Key Features:
    - Type-safe action representation
    - Cross-platform coordinate system
    - Safety level classification
    - Comprehensive action result tracking
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class POPSSCMUActionType(Enum):
    """
    Enumeration of supported action types for Computer Use Agent.
    
    Action Types:
        CLICK: Single click action (mouse or touch)
        DOUBLE_CLICK: Double click action
        RIGHT_CLICK: Right click action (mouse only)
        MIDDLE_CLICK: Middle click action (mouse only)
        TYPE: Text typing action
        KEY_PRESS: Single key press action
        HOTKEY: Keyboard hotkey combination
        SCROLL: Scroll action (mouse wheel or touch)
        DRAG: Drag action (mouse or touch)
        SWIPE: Swipe gesture (touch only)
        PINCH: Pinch gesture (touch only)
        ZOOM: Zoom gesture (touch only)
        ROTATE: Rotation gesture (touch only)
        HOVER: Mouse hover action
        WAIT: Wait/delay action
        SCREENSHOT: Screen capture action
        CLIPBOARD_COPY: Copy to clipboard
        CLIPBOARD_PASTE: Paste from clipboard
        APP_LAUNCH: Launch application
        APP_CLOSE: Close application
        WINDOW_SWITCH: Switch window
        CUSTOM: Custom action type
    """
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    TYPE = "type"
    KEY_PRESS = "key_press"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    DRAG = "drag"
    SWIPE = "swipe"
    PINCH = "pinch"
    ZOOM = "zoom"
    ROTATE = "rotate"
    HOVER = "hover"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    CLIPBOARD_COPY = "clipboard_copy"
    CLIPBOARD_PASTE = "clipboard_paste"
    APP_LAUNCH = "app_launch"
    APP_CLOSE = "app_close"
    WINDOW_SWITCH = "window_switch"
    CUSTOM = "custom"


class POPSSCMUPlatform(Enum):
    """
    Enumeration of supported platforms for Computer Use Agent.
    
    Platforms:
        DESKTOP_WINDOWS: Windows desktop platform
        DESKTOP_MACOS: macOS desktop platform
        DESKTOP_LINUX: Linux desktop platform
        ANDROID_PHONE: Android phone platform
        ANDROID_TABLET: Android tablet platform
        IOS_IPHONE: iOS iPhone platform
        IOS_IPAD: iOS iPad platform
        WEB_CHROME: Chrome browser platform
        WEB_FIREFOX: Firefox browser platform
        WEB_SAFARI: Safari browser platform
        WEB_EDGE: Edge browser platform
        UNKNOWN: Unknown platform
    """
    DESKTOP_WINDOWS = "desktop_windows"
    DESKTOP_MACOS = "desktop_macos"
    DESKTOP_LINUX = "desktop_linux"
    ANDROID_PHONE = "android_phone"
    ANDROID_TABLET = "android_tablet"
    IOS_IPHONE = "ios_iphone"
    IOS_IPAD = "ios_ipad"
    WEB_CHROME = "web_chrome"
    WEB_FIREFOX = "web_firefox"
    WEB_SAFARI = "web_safari"
    WEB_EDGE = "web_edge"
    UNKNOWN = "unknown"

    @property
    def is_desktop(self) -> bool:
        """Check if platform is a desktop platform."""
        return self in (
            POPSSCMUPlatform.DESKTOP_WINDOWS,
            POPSSCMUPlatform.DESKTOP_MACOS,
            POPSSCMUPlatform.DESKTOP_LINUX,
        )

    @property
    def is_mobile(self) -> bool:
        """Check if platform is a mobile platform."""
        return self in (
            POPSSCMUPlatform.ANDROID_PHONE,
            POPSSCMUPlatform.IOS_IPHONE,
        )

    @property
    def is_tablet(self) -> bool:
        """Check if platform is a tablet platform."""
        return self in (
            POPSSCMUPlatform.ANDROID_TABLET,
            POPSSCMUPlatform.IOS_IPAD,
        )

    @property
    def is_web(self) -> bool:
        """Check if platform is a web browser platform."""
        return self in (
            POPSSCMUPlatform.WEB_CHROME,
            POPSSCMUPlatform.WEB_FIREFOX,
            POPSSCMUPlatform.WEB_SAFARI,
            POPSSCMUPlatform.WEB_EDGE,
        )

    @property
    def is_touch_enabled(self) -> bool:
        """Check if platform supports touch input."""
        return self.is_mobile or self.is_tablet


class POPSSCMUSafetyLevel(Enum):
    """
    Enumeration of safety levels for action classification.
    
    Safety Levels:
        SAFE: Safe action, no user confirmation required
        MODERATE: Moderate risk action, optional confirmation
        CONFIRM: Requires user confirmation before execution
        DANGEROUS: High risk action, mandatory confirmation and logging
        RESTRICTED: Restricted action, blocked by default
    """
    SAFE = "safe"
    MODERATE = "moderate"
    CONFIRM = "confirm"
    DANGEROUS = "dangerous"
    RESTRICTED = "restricted"


class POPSSCMUElementState(Enum):
    """
    Enumeration of UI element states.
    
    States:
        VISIBLE: Element is visible and can be interacted with
        HIDDEN: Element is hidden but exists in DOM
        DISABLED: Element is visible but disabled
        FOCUSED: Element has focus
        SELECTED: Element is selected
        EXPANDED: Element is expanded (e.g., dropdown)
        COLLAPSED: Element is collapsed
        SCROLLABLE: Element is scrollable
    """
    VISIBLE = "visible"
    HIDDEN = "hidden"
    DISABLED = "disabled"
    FOCUSED = "focused"
    SELECTED = "selected"
    EXPANDED = "expanded"
    COLLAPSED = "collapsed"
    SCROLLABLE = "scrollable"


class POPSSCMUActionResultStatus(Enum):
    """
    Enumeration of action execution result status.
    
    Status Types:
        SUCCESS: Action executed successfully
        PARTIAL: Action partially succeeded
        FAILED: Action execution failed
        TIMEOUT: Action execution timed out
        CANCELLED: Action was cancelled by user
        BLOCKED: Action was blocked by safety policy
        RETRY: Action needs retry
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    RETRY = "retry"


class POPSSCMUScrollDirection(Enum):
    """
    Enumeration of scroll directions.
    """
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class POPSSCMUSwipeDirection(Enum):
    """
    Enumeration of swipe directions.
    """
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


class POPSSCMUMouseButton(Enum):
    """
    Enumeration of mouse buttons.
    """
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class POPSSCMUCoordinate:
    """
    2D coordinate representation for screen positions.
    
    Attributes:
        x: X coordinate (horizontal position)
        y: Y coordinate (vertical position)
        is_relative: Whether coordinates are relative (0-1) or absolute (pixels)
    """
    x: float
    y: float
    is_relative: bool = False

    def to_absolute(self, width: int, height: int) -> POPSSCMUCoordinate:
        """Convert relative coordinates to absolute."""
        if not self.is_relative:
            return self
        return POPSSCMUCoordinate(
            x=self.x * width,
            y=self.y * height,
            is_relative=False
        )

    def to_relative(self, width: int, height: int) -> POPSSCMUCoordinate:
        """Convert absolute coordinates to relative."""
        if self.is_relative:
            return self
        return POPSSCMUCoordinate(
            x=self.x / width if width > 0 else 0,
            y=self.y / height if height > 0 else 0,
            is_relative=True
        )

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to integer tuple representation."""
        return (int(self.x), int(self.y))


@dataclass
class POPSSCMURectangle:
    """
    Rectangle region representation for screen areas.
    
    Attributes:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of rectangle
        height: Height of rectangle
        is_relative: Whether coordinates are relative or absolute
    """
    x: float
    y: float
    width: float
    height: float
    is_relative: bool = False

    @property
    def center(self) -> POPSSCMUCoordinate:
        """Get center coordinate of rectangle."""
        return POPSSCMUCoordinate(
            x=self.x + self.width / 2,
            y=self.y + self.height / 2,
            is_relative=self.is_relative
        )

    @property
    def top_left(self) -> POPSSCMUCoordinate:
        """Get top-left coordinate."""
        return POPSSCMUCoordinate(x=self.x, y=self.y, is_relative=self.is_relative)

    @property
    def bottom_right(self) -> POPSSCMUCoordinate:
        """Get bottom-right coordinate."""
        return POPSSCMUCoordinate(
            x=self.x + self.width,
            y=self.y + self.height,
            is_relative=self.is_relative
        )

    def contains(self, coord: POPSSCMUCoordinate) -> bool:
        """Check if coordinate is within rectangle."""
        return (
            self.x <= coord.x <= self.x + self.width and
            self.y <= coord.y <= self.y + self.height
        )

    def to_absolute(self, screen_width: int, screen_height: int) -> POPSSCMURectangle:
        """Convert relative rectangle to absolute."""
        if not self.is_relative:
            return self
        return POPSSCMURectangle(
            x=self.x * screen_width,
            y=self.y * screen_height,
            width=self.width * screen_width,
            height=self.height * screen_height,
            is_relative=False
        )


@dataclass
class POPSSCMUTarget:
    """
    Action target specification for identifying UI elements.
    
    Attributes:
        coordinate: Direct coordinate target (optional)
        rectangle: Rectangle region target (optional)
        element_description: Natural language description of target element
        element_id: Unique identifier of target element
        element_type: Type of element (button, input, etc.)
        image_path: Path to reference image for visual matching
        text: Text content to match
        selector: CSS/XPath selector for web elements
        confidence_threshold: Minimum confidence for matching
    """
    coordinate: Optional[POPSSCMUCoordinate] = None
    rectangle: Optional[POPSSCMURectangle] = None
    element_description: Optional[str] = None
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    image_path: Optional[str] = None
    text: Optional[str] = None
    selector: Optional[str] = None
    confidence_threshold: float = 0.8

    def has_coordinate_target(self) -> bool:
        """Check if target has coordinate specification."""
        return self.coordinate is not None

    def has_element_target(self) -> bool:
        """Check if target has element specification."""
        return any([
            self.element_description,
            self.element_id,
            self.element_type,
            self.text,
            self.selector,
        ])

    def has_visual_target(self) -> bool:
        """Check if target has visual matching specification."""
        return self.image_path is not None


@dataclass
class POPSSCMUAction:
    """
    Unified action representation for Computer Use Agent.
    
    Attributes:
        action_id: Unique identifier for the action
        action_type: Type of action to perform
        target: Target specification for the action
        platform: Target platform for the action
        params: Additional parameters for the action
        safety_level: Safety classification of the action
        timeout: Timeout in seconds for action execution
        retry_count: Number of retry attempts allowed
        delay_before: Delay before execution in seconds
        delay_after: Delay after execution in seconds
        description: Human-readable description of the action
        expected_result: Expected result description
        depends_on: List of action IDs this action depends on
        created_at: Creation timestamp
    """
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: POPSSCMUActionType = POPSSCMUActionType.CLICK
    target: POPSSCMUTarget = field(default_factory=POPSSCMUTarget)
    platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS
    params: Dict[str, Any] = field(default_factory=dict)
    safety_level: POPSSCMUSafetyLevel = POPSSCMUSafetyLevel.SAFE
    timeout: float = 30.0
    retry_count: int = 3
    delay_before: float = 0.0
    delay_after: float = 0.1
    description: str = ""
    expected_result: str = ""
    depends_on: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target": {
                "coordinate": self.target.coordinate.to_tuple() if self.target.coordinate else None,
                "element_description": self.target.element_description,
                "element_id": self.target.element_id,
                "text": self.target.text,
            },
            "platform": self.platform.value,
            "params": self.params,
            "safety_level": self.safety_level.value,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "description": self.description,
        }

    @classmethod
    def click(
        cls,
        target: POPSSCMUTarget,
        button: POPSSCMUMouseButton = POPSSCMUMouseButton.LEFT,
        double: bool = False,
        **kwargs
    ) -> POPSSCMUAction:
        """Create a click action."""
        action_type = POPSSCMUActionType.DOUBLE_CLICK if double else POPSSCMUActionType.CLICK
        if button == POPSSCMUMouseButton.RIGHT:
            action_type = POPSSCMUActionType.RIGHT_CLICK
        elif button == POPSSCMUMouseButton.MIDDLE:
            action_type = POPSSCMUActionType.MIDDLE_CLICK

        return cls(
            action_type=action_type,
            target=target,
            params={"button": button.value},
            **kwargs
        )

    @classmethod
    def type_text(
        cls,
        text: str,
        target: Optional[POPSSCMUTarget] = None,
        typing_speed: float = 0.05,
        **kwargs
    ) -> POPSSCMUAction:
        """Create a type action."""
        return cls(
            action_type=POPSSCMUActionType.TYPE,
            target=target or POPSSCMUTarget(),
            params={"text": text, "typing_speed": typing_speed},
            **kwargs
        )

    @classmethod
    def scroll(
        cls,
        direction: POPSSCMUScrollDirection,
        amount: int = 3,
        target: Optional[POPSSCMUTarget] = None,
        **kwargs
    ) -> POPSSCMUAction:
        """Create a scroll action."""
        return cls(
            action_type=POPSSCMUActionType.SCROLL,
            target=target or POPSSCMUTarget(),
            params={"direction": direction.value, "amount": amount},
            **kwargs
        )

    @classmethod
    def drag(
        cls,
        start: POPSSCMUCoordinate,
        end: POPSSCMUCoordinate,
        duration: float = 0.5,
        **kwargs
    ) -> POPSSCMUAction:
        """Create a drag action."""
        return cls(
            action_type=POPSSCMUActionType.DRAG,
            target=POPSSCMUTarget(coordinate=start),
            params={"end": end.to_tuple(), "duration": duration},
            **kwargs
        )

    @classmethod
    def swipe(
        cls,
        direction: POPSSCMUSwipeDirection,
        distance: float = 0.3,
        duration: float = 0.3,
        target: Optional[POPSSCMUTarget] = None,
        **kwargs
    ) -> POPSSCMUAction:
        """Create a swipe action."""
        return cls(
            action_type=POPSSCMUActionType.SWIPE,
            target=target or POPSSCMUTarget(),
            params={"direction": direction.value, "distance": distance, "duration": duration},
            **kwargs
        )

    @classmethod
    def hotkey(
        cls,
        keys: List[str],
        **kwargs
    ) -> POPSSCMUAction:
        """Create a hotkey action."""
        return cls(
            action_type=POPSSCMUActionType.HOTKEY,
            target=POPSSCMUTarget(),
            params={"keys": keys},
            safety_level=POPSSCMUSafetyLevel.MODERATE,
            **kwargs
        )

    @classmethod
    def wait(
        cls,
        duration: float = 1.0,
        **kwargs
    ) -> POPSSCMUAction:
        """Create a wait action."""
        return cls(
            action_type=POPSSCMUActionType.WAIT,
            target=POPSSCMUTarget(),
            params={"duration": duration},
            **kwargs
        )


@dataclass
class POPSSCMUElement:
    """
    UI element representation for element detection.
    
    Attributes:
        element_id: Unique identifier for the element
        element_type: Type of element (button, input, etc.)
        bounds: Bounding rectangle of the element
        text: Text content of the element
        label: Accessible label of the element
        value: Current value of the element
        state: Current state of the element
        attributes: Additional element attributes
        confidence: Detection confidence score
        parent_id: ID of parent element
        children_ids: IDs of child elements
    """
    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_type: str = "unknown"
    bounds: POPSSCMURectangle = field(default_factory=POPSSCMURectangle)
    text: str = ""
    label: str = ""
    value: str = ""
    state: POPSSCMUElementState = POPSSCMUElementState.VISIBLE
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    @property
    def center(self) -> POPSSCMUCoordinate:
        """Get center coordinate of element."""
        return self.bounds.center

    @property
    def is_interactable(self) -> bool:
        """Check if element is interactable."""
        return self.state in (
            POPSSCMUElementState.VISIBLE,
            POPSSCMUElementState.FOCUSED,
            POPSSCMUElementState.SELECTED,
        )

    def to_target(self) -> POPSSCMUTarget:
        """Convert element to action target."""
        return POPSSCMUTarget(
            coordinate=self.center,
            rectangle=self.bounds,
            element_id=self.element_id,
            element_type=self.element_type,
            text=self.text,
            confidence_threshold=self.confidence,
        )


@dataclass
class POPSSCMUScreenState:
    """
    Screen state snapshot for perception and analysis.
    
    Attributes:
        state_id: Unique identifier for the screen state
        screenshot_path: Path to saved screenshot file
        image_data: Raw image data (PIL Image or numpy array)
        width: Screen width in pixels
        height: Screen height in pixels
        platform: Platform type
        elements: Detected UI elements
        text_blocks: Detected text blocks with positions
        timestamp: Capture timestamp
        cursor_position: Current cursor position
        active_window: Active window title
        focused_element: Currently focused element
    """
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    screenshot_path: Optional[str] = None
    image_data: Optional[Any] = None
    width: int = 1920
    height: int = 1080
    platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS
    elements: List[POPSSCMUElement] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    cursor_position: Optional[POPSSCMUCoordinate] = None
    active_window: str = ""
    focused_element: Optional[POPSSCMUElement] = None

    def find_element(
        self,
        text: Optional[str] = None,
        element_type: Optional[str] = None,
        element_id: Optional[str] = None,
    ) -> Optional[POPSSCMUElement]:
        """Find element by criteria."""
        for element in self.elements:
            if element_id and element.element_id == element_id:
                return element
            if text and text.lower() in element.text.lower():
                return element
            if element_type and element.element_type == element_type:
                return element
        return None

    def find_elements_by_type(self, element_type: str) -> List[POPSSCMUElement]:
        """Find all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]

    def get_interactable_elements(self) -> List[POPSSCMUElement]:
        """Get all interactable elements."""
        return [e for e in self.elements if e.is_interactable]


@dataclass
class POPSSCMUActionResult:
    """
    Action execution result for tracking and verification.
    
    Attributes:
        result_id: Unique identifier for the result
        action_id: ID of the executed action
        status: Execution status
        screen_state_before: Screen state before action
        screen_state_after: Screen state after action
        output: Output data from the action
        error_message: Error message if failed
        execution_time: Execution time in seconds
        retry_count: Number of retries performed
        verified: Whether result was verified
        verification_result: Verification result details
        timestamp: Execution timestamp
        metadata: Additional metadata
    """
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_id: str = ""
    status: POPSSCMUActionResultStatus = POPSSCMUActionResultStatus.SUCCESS
    screen_state_before: Optional[POPSSCMUScreenState] = None
    screen_state_after: Optional[POPSSCMUScreenState] = None
    output: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    retry_count: int = 0
    verified: bool = False
    verification_result: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if action was successful."""
        return self.status == POPSSCMUActionResultStatus.SUCCESS

    @property
    def needs_retry(self) -> bool:
        """Check if action needs retry."""
        return self.status == POPSSCMUActionResultStatus.RETRY

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "result_id": self.result_id,
            "action_id": self.action_id,
            "status": self.status.value,
            "output": self.output,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "verified": self.verified,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class POPSSCMUTaskContext:
    """
    Task execution context for tracking multi-step tasks.
    
    Attributes:
        task_id: Unique identifier for the task
        goal: Original goal/task description
        platform: Target platform
        actions: List of planned actions
        results: List of execution results
        current_action_index: Index of current action
        status: Task execution status
        created_at: Task creation timestamp
        updated_at: Last update timestamp
        metadata: Additional task metadata
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS
    actions: List[POPSSCMUAction] = field(default_factory=list)
    results: List[POPSSCMUActionResult] = field(default_factory=list)
    current_action_index: int = 0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_action(self) -> Optional[POPSSCMUAction]:
        """Get current action."""
        if 0 <= self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None

    @property
    def progress(self) -> float:
        """Get task progress as percentage."""
        if not self.actions:
            return 0.0
        return (self.current_action_index / len(self.actions)) * 100

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in ("completed", "failed", "cancelled")

    def add_action(self, action: POPSSCMUAction) -> None:
        """Add an action to the task."""
        self.actions.append(action)
        self.updated_at = datetime.now()

    def add_result(self, result: POPSSCMUActionResult) -> None:
        """Add a result to the task."""
        self.results.append(result)
        self.current_action_index += 1
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert task context to dictionary."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "platform": self.platform.value,
            "action_count": len(self.actions),
            "result_count": len(self.results),
            "current_action_index": self.current_action_index,
            "progress": self.progress,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class POPSSCMUSafetyPolicy:
    """
    Safety policy definition for action validation.
    
    Attributes:
        policy_id: Unique identifier for the policy
        name: Policy name
        description: Policy description
        allowed_actions: Set of allowed action types
        blocked_actions: Set of blocked action types
        restricted_areas: List of restricted screen areas
        max_actions_per_task: Maximum actions allowed per task
        require_confirmation_for: Actions requiring confirmation
        audit_enabled: Whether to audit all actions
    """
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    description: str = "Default safety policy"
    allowed_actions: set = field(default_factory=lambda: set(POPSSCMUActionType))
    blocked_actions: set = field(default_factory=set)
    restricted_areas: List[POPSSCMURectangle] = field(default_factory=list)
    max_actions_per_task: int = 100
    require_confirmation_for: set = field(
        default_factory=lambda: {
            POPSSCMUActionType.APP_CLOSE,
            POPSSCMUActionType.APP_LAUNCH,
        }
    )
    audit_enabled: bool = True

    def is_action_allowed(self, action: POPSSCMUAction) -> Tuple[bool, str]:
        """Check if action is allowed under this policy."""
        if action.action_type in self.blocked_actions:
            return False, f"Action type {action.action_type.value} is blocked"

        if action.action_type not in self.allowed_actions:
            return False, f"Action type {action.action_type.value} is not allowed"

        if action.target.coordinate:
            for area in self.restricted_areas:
                if area.contains(action.target.coordinate):
                    return False, "Action targets restricted area"

        return True, "Action allowed"

    def requires_confirmation(self, action: POPSSCMUAction) -> bool:
        """Check if action requires user confirmation."""
        return action.action_type in self.require_confirmation_for
