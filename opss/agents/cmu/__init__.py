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
CMU (Computer Use Agent) Module

A flagship-level Computer Use Agent for cross-platform device control
with enterprise-grade security and visual perception.

Features:
    - Cross-platform support (Desktop, Mobile, Tablet, Web)
    - Three-layer security model
    - Visual perception via YvVisionEncoder
    - Task planning and decomposition
    - Action execution with verification
"""

from .types import (
    POPSSCMUAction,
    POPSSCMUActionType,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUCoordinate,
    POPSSCMUElement,
    POPSSCMUElementState,
    POPSSCMUMouseButton,
    POPSSCMUPlatform,
    POPSSCMURectangle,
    POPSSCMUSafetyLevel,
    POPSSCMUSafetyPolicy,
    POPSSCMUScreenState,
    POPSSCMUScrollDirection,
    POPSSCMUSwipeDirection,
    POPSSCMUTarget,
    POPSSCMUTaskContext,
)

from .safety import (
    POPSSCMUAuditEvent,
    POPSSCMUAuditEventType,
    POPSSCMUAuditLogger,
    POPSSCMUEmergencyStop,
    POPSSCMUPermission,
    POPSSCMUPermissionManager,
    POPSSCMURiskLevel,
    POPSSCMUSafetySystem,
    POPSSCMUSafetyValidator,
    POPSSCMUSandbox,
    POPSSCMUStateSnapshot,
)

from .core import (
    POPSSCMUCapability,
    POPSSCMUConfig,
    POPSSCMUEngine,
)

from .platform import (
    POPSSCMUPlatformAdapter,
    POPSSCMUPlatformInfo,
    POPSSCMUDesktop,
    POPSSCMUWeb,
    POPSSCMUMobile,
    POPSSCMUTablet,
)

from .perception import (
    POPSSCMUScreenCapture,
    POPSSCMUCaptureConfig,
    POPSSCMUElementDetector,
    POPSSCMUDetectionResult,
    POPSSCMUOCRReader,
    POPSSCMUTextBlock,
)

from .action import (
    POPSSCMUActionExecutor,
    POPSSCMUActionConfig,
    POPSSCMUMouseController,
    POPSSCMUMouseConfig,
    POPSSCMUKeyboardController,
    POPSSCMUKeyboardConfig,
    POPSSCMUTouchController,
    POPSSCMUTouchConfig,
    POPSSCMUGestureController,
    POPSSCMUGestureConfig,
)

from .planning import (
    POPSSCMUTaskPlanner,
    POPSSCMUTaskNode,
    POPSSCMUPlanResult,
    POPSSCMUActionSequence,
    POPSSCMUSequenceState,
    POPSSCMUCondition,
)

__all__ = [
    "POPSSCMUAction",
    "POPSSCMUActionType",
    "POPSSCMUActionResult",
    "POPSSCMUActionResultStatus",
    "POPSSCMUCoordinate",
    "POPSSCMUElement",
    "POPSSCMUElementState",
    "POPSSCMUMouseButton",
    "POPSSCMUPlatform",
    "POPSSCMURectangle",
    "POPSSCMUSafetyLevel",
    "POPSSCMUSafetyPolicy",
    "POPSSCMUScreenState",
    "POPSSCMUScrollDirection",
    "POPSSCMUSwipeDirection",
    "POPSSCMUTarget",
    "POPSSCMUTaskContext",
    "POPSSCMUAuditEvent",
    "POPSSCMUAuditEventType",
    "POPSSCMUAuditLogger",
    "POPSSCMUEmergencyStop",
    "POPSSCMUPermission",
    "POPSSCMUPermissionManager",
    "POPSSCMURiskLevel",
    "POPSSCMUSafetySystem",
    "POPSSCMUSafetyValidator",
    "POPSSCMUSandbox",
    "POPSSCMUStateSnapshot",
    "POPSSCMUCapability",
    "POPSSCMUConfig",
    "POPSSCMUEngine",
    "POPSSCMUPlatformAdapter",
    "POPSSCMUPlatformInfo",
    "POPSSCMUDesktop",
    "POPSSCMUWeb",
    "POPSSCMUMobile",
    "POPSSCMUTablet",
    "POPSSCMUScreenCapture",
    "POPSSCMUCaptureConfig",
    "POPSSCMUElementDetector",
    "POPSSCMUDetectionResult",
    "POPSSCMUOCRReader",
    "POPSSCMUTextBlock",
    "POPSSCMUActionExecutor",
    "POPSSCMUActionConfig",
    "POPSSCMUMouseController",
    "POPSSCMUMouseConfig",
    "POPSSCMUKeyboardController",
    "POPSSCMUKeyboardConfig",
    "POPSSCMUTouchController",
    "POPSSCMUTouchConfig",
    "POPSSCMUGestureController",
    "POPSSCMUGestureConfig",
    "POPSSCMUTaskPlanner",
    "POPSSCMUTaskNode",
    "POPSSCMUPlanResult",
    "POPSSCMUActionSequence",
    "POPSSCMUSequenceState",
    "POPSSCMUCondition",
]
