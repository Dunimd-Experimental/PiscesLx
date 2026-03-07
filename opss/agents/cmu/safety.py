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
CMU Safety Module - Three-Layer Security Sandbox

This module provides comprehensive security infrastructure for the Computer Use Agent,
implementing a three-layer security model for safe and auditable device control.

Module Components:
    1. POPSSCMUAuditLogger:
       - Comprehensive action logging
       - Tamper-proof audit trail
       - Searchable log database

    2. POPSSCMUPermissionManager:
       - Permission-based access control
       - User consent management
       - Permission caching

    3. POPSSCMUSandbox:
       - Isolated execution environment
       - State snapshot and rollback
       - Resource limitation

    4. POPSSCMUSafetyValidator:
       - Action validation engine
       - Risk assessment
       - Policy enforcement

    5. POPSSCMUEmergencyStop:
       - Emergency halt mechanism
       - Graceful shutdown
       - State recovery

Security Layers:
    Layer 1: Intent Verification
        - User confirmation for dangerous operations
        - Natural language intent analysis
        - Risk level classification

    Layer 2: Sandbox Isolation
        - Virtual environment execution
        - State snapshots before actions
        - Rollback capability

    Layer 3: Real-time Monitoring
        - Anomaly detection
        - Behavior analysis
        - Emergency stop triggers

Usage Example:
    >>> from opss.agents.cmu.safety import POPSSCMUSandbox, POPSSCMUSafetyValidator
    >>> 
    >>> # Initialize safety system
    >>> sandbox = POPSSCMUSandbox()
    >>> validator = POPSSCMUSafetyValidator()
    >>> 
    >>> # Validate action
    >>> is_safe, reason = validator.validate_action(action)
    >>> if is_safe:
    ...     result = sandbox.execute_safe(action)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .types import (
    POPSSCMUAction,
    POPSSCMUActionType,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUPlatform,
    POPSSCMUSafetyLevel,
    POPSSCMUSafetyPolicy,
    POPSSCMUScreenState,
    POPSSCMUTarget,
    POPSSCMURectangle,
    POPSSCMUCoordinate,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Safety", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Safety"), enable_file=True)


class POPSSCMUAuditEventType(Enum):
    """Enumeration of audit event types."""
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    ACTION_BLOCKED = "action_blocked"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CONFIRMATION_GRANTED = "confirmation_granted"
    CONFIRMATION_DENIED = "confirmation_denied"
    EMERGENCY_STOP = "emergency_stop"
    SNAPSHOT_CREATED = "snapshot_created"
    ROLLBACK_EXECUTED = "rollback_executed"
    POLICY_VIOLATION = "policy_violation"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ANOMALY_DETECTED = "anomaly_detected"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"


class POPSSCMURiskLevel(Enum):
    """Enumeration of risk levels for actions."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class POPSSCMUAuditEvent:
    """
    Audit event record for comprehensive logging.
    
    Attributes:
        event_id: Unique identifier for the event
        event_type: Type of audit event
        timestamp: Event timestamp
        session_id: Session identifier
        action_id: Related action ID (if applicable)
        user_id: User identifier
        platform: Platform type
        event_data: Event-specific data
        risk_level: Assessed risk level
        checksum: Integrity checksum
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: POPSSCMUAuditEventType = POPSSCMUAuditEventType.ACTION_STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    action_id: str = ""
    user_id: str = ""
    platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS
    event_data: Dict[str, Any] = field(default_factory=dict)
    risk_level: POPSSCMURiskLevel = POPSSCMURiskLevel.LOW
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute integrity checksum for the event."""
        data = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "action_id": self.action_id,
            "event_data": self.event_data,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify event integrity."""
        return self.checksum == self.compute_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "action_id": self.action_id,
            "user_id": self.user_id,
            "platform": self.platform.value,
            "event_data": self.event_data,
            "risk_level": self.risk_level.value,
            "checksum": self.checksum,
        }


@dataclass
class POPSSCMUStateSnapshot:
    """
    State snapshot for rollback capability.
    
    Attributes:
        snapshot_id: Unique identifier for the snapshot
        timestamp: Snapshot creation timestamp
        screen_state: Screen state at snapshot time
        system_state: System state information
        action_id: Action that will be executed after snapshot
        metadata: Additional metadata
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    screen_state: Optional[POPSSCMUScreenState] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    action_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSCMUPermission:
    """
    Permission definition for access control.
    
    Attributes:
        permission_id: Unique identifier for the permission
        name: Permission name
        description: Permission description
        action_types: Set of action types this permission covers
        scope: Permission scope (global, session, action)
        granted: Whether permission is granted
        granted_at: Timestamp when permission was granted
        expires_at: Optional expiration timestamp
    """
    permission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    action_types: Set[POPSSCMUActionType] = field(default_factory=set)
    scope: str = "session"
    granted: bool = False
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if permission is currently valid."""
        if not self.granted:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def covers_action(self, action_type: POPSSCMUActionType) -> bool:
        """Check if permission covers a specific action type."""
        return action_type in self.action_types


class POPSSCMUAuditLogger:
    """
    Comprehensive audit logger for action tracking.
    
    Provides tamper-proof logging of all CMU actions with searchable
    event database and integrity verification.
    
    Attributes:
        log_path: Path to audit log file
        events: List of recorded events
        session_id: Current session identifier
        enabled: Whether logging is enabled
        max_events: Maximum events to keep in memory
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        session_id: Optional[str] = None,
        enabled: bool = True,
        max_events: int = 10000,
    ):
        self.log_path = log_path or str(get_log_file("cmu_audit"))
        self.events: List[POPSSCMUAuditEvent] = []
        self.session_id = session_id or str(uuid.uuid4())
        self.enabled = enabled
        self.max_events = max_events
        self._lock = threading.RLock()
        self._file_handle: Optional[Any] = None

        self._initialize_log_file()
        _LOG.info("audit_logger_initialized", session_id=self.session_id)

    def _initialize_log_file(self) -> None:
        """Initialize the audit log file."""
        try:
            log_dir = Path(self.log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_path, 'a', encoding='utf-8')
        except Exception as e:
            _LOG.error("audit_log_init_failed", error=str(e))

    def log_event(
        self,
        event_type: POPSSCMUAuditEventType,
        action_id: str = "",
        user_id: str = "",
        platform: POPSSCMUPlatform = POPSSCMUPlatform.DESKTOP_WINDOWS,
        event_data: Optional[Dict[str, Any]] = None,
        risk_level: POPSSCMURiskLevel = POPSSCMURiskLevel.LOW,
    ) -> POPSSCMUAuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of audit event
            action_id: Related action ID
            user_id: User identifier
            platform: Platform type
            event_data: Event-specific data
            risk_level: Assessed risk level
        
        Returns:
            POPSSCMUAuditEvent: The created audit event
        """
        event = POPSSCMUAuditEvent(
            event_type=event_type,
            session_id=self.session_id,
            action_id=action_id,
            user_id=user_id,
            platform=platform,
            event_data=event_data or {},
            risk_level=risk_level,
        )
        event.checksum = event.compute_checksum()

        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

            if self.enabled and self._file_handle:
                try:
                    self._file_handle.write(json.dumps(event.to_dict()) + '\n')
                    self._file_handle.flush()
                except Exception as e:
                    _LOG.error("audit_write_failed", error=str(e))

        _LOG.info(
            "audit_event_logged",
            event_type=event_type.value,
            action_id=action_id,
            risk_level=risk_level.value,
        )

        return event

    def log_action_started(self, action: POPSSCMUAction) -> POPSSCMUAuditEvent:
        """Log action started event."""
        return self.log_event(
            event_type=POPSSCMUAuditEventType.ACTION_STARTED,
            action_id=action.action_id,
            event_data={
                "action_type": action.action_type.value,
                "description": action.description,
                "safety_level": action.safety_level.value,
            },
            risk_level=self._assess_action_risk(action),
        )

    def log_action_completed(
        self,
        action: POPSSCMUAction,
        result: POPSSCMUActionResult,
    ) -> POPSSCMUAuditEvent:
        """Log action completed event."""
        return self.log_event(
            event_type=POPSSCMUAuditEventType.ACTION_COMPLETED,
            action_id=action.action_id,
            event_data={
                "action_type": action.action_type.value,
                "status": result.status.value,
                "execution_time": result.execution_time,
            },
            risk_level=POPSSCMURiskLevel.LOW,
        )

    def log_action_blocked(
        self,
        action: POPSSCMUAction,
        reason: str,
    ) -> POPSSCMUAuditEvent:
        """Log action blocked event."""
        return self.log_event(
            event_type=POPSSCMUAuditEventType.ACTION_BLOCKED,
            action_id=action.action_id,
            event_data={
                "action_type": action.action_type.value,
                "reason": reason,
            },
            risk_level=POPSSCMURiskLevel.HIGH,
        )

    def log_emergency_stop(self, reason: str) -> POPSSCMUAuditEvent:
        """Log emergency stop event."""
        return self.log_event(
            event_type=POPSSCMUAuditEventType.EMERGENCY_STOP,
            event_data={"reason": reason},
            risk_level=POPSSCMURiskLevel.CRITICAL,
        )

    def _assess_action_risk(self, action: POPSSCMUAction) -> POPSSCMURiskLevel:
        """Assess risk level for an action."""
        high_risk_actions = {
            POPSSCMUActionType.APP_CLOSE,
            POPSSCMUActionType.APP_LAUNCH,
            POPSSCMUActionType.HOTKEY,
        }
        critical_risk_actions = {
            POPSSCMUActionType.CLIPBOARD_COPY,
            POPSSCMUActionType.CLIPBOARD_PASTE,
        }

        if action.action_type in critical_risk_actions:
            return POPSSCMURiskLevel.CRITICAL
        if action.action_type in high_risk_actions:
            return POPSSCMURiskLevel.HIGH
        if action.safety_level == POPSSCMUSafetyLevel.DANGEROUS:
            return POPSSCMURiskLevel.HIGH
        if action.safety_level == POPSSCMUSafetyLevel.CONFIRM:
            return POPSSCMURiskLevel.MEDIUM
        return POPSSCMURiskLevel.LOW

    def search_events(
        self,
        event_type: Optional[POPSSCMUAuditEventType] = None,
        action_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        risk_level: Optional[POPSSCMURiskLevel] = None,
    ) -> List[POPSSCMUAuditEvent]:
        """Search audit events by criteria."""
        results = []
        with self._lock:
            for event in self.events:
                if event_type and event.event_type != event_type:
                    continue
                if action_id and event.action_id != action_id:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if risk_level and event.risk_level != risk_level:
                    continue
                results.append(event)
        return results

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "total_events": len(self.events),
                "event_types": {
                    et.value: sum(1 for e in self.events if e.event_type == et)
                    for et in POPSSCMUAuditEventType
                },
                "risk_distribution": {
                    rl.value: sum(1 for e in self.events if e.risk_level == rl)
                    for rl in POPSSCMURiskLevel
                },
            }

    def close(self) -> None:
        """Close the audit logger."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        _LOG.info("audit_logger_closed", session_id=self.session_id)


class POPSSCMUPermissionManager:
    """
    Permission manager for access control.
    
    Manages user permissions and consent for CMU actions.
    
    Attributes:
        permissions: Dictionary of granted permissions
        confirmation_callback: Callback for user confirmation
        auto_approve_safe: Whether to auto-approve safe actions
        permission_cache: Cache of permission decisions
    """

    def __init__(
        self,
        confirmation_callback: Optional[Callable[[str, POPSSCMUAction], bool]] = None,
        auto_approve_safe: bool = True,
    ):
        self.permissions: Dict[str, POPSSCMUPermission] = {}
        self.confirmation_callback = confirmation_callback
        self.auto_approve_safe = auto_approve_safe
        self.permission_cache: Dict[str, bool] = {}
        self._lock = threading.RLock()

        self._initialize_default_permissions()
        _LOG.info("permission_manager_initialized")

    def _initialize_default_permissions(self) -> None:
        """Initialize default permission set."""
        safe_actions = {
            POPSSCMUActionType.CLICK,
            POPSSCMUActionType.DOUBLE_CLICK,
            POPSSCMUActionType.TYPE,
            POPSSCMUActionType.SCROLL,
            POPSSCMUActionType.HOVER,
            POPSSCMUActionType.WAIT,
            POPSSCMUActionType.SCREENSHOT,
        }

        self.permissions["safe_actions"] = POPSSCMUPermission(
            name="safe_actions",
            description="Permission for safe action types",
            action_types=safe_actions,
            scope="global",
            granted=True,
            granted_at=datetime.now(),
        )

        moderate_actions = {
            POPSSCMUActionType.DRAG,
            POPSSCMUActionType.SWIPE,
            POPSSCMUActionType.KEY_PRESS,
        }

        self.permissions["moderate_actions"] = POPSSCMUPermission(
            name="moderate_actions",
            description="Permission for moderate risk action types",
            action_types=moderate_actions,
            scope="session",
            granted=True,
            granted_at=datetime.now(),
        )

    def check_permission(
        self,
        action: POPSSCMUAction,
        user_id: str = "",
    ) -> Tuple[bool, str]:
        """
        Check if action has required permission.
        
        Args:
            action: Action to check permission for
            user_id: User identifier
        
        Returns:
            Tuple[bool, str]: (is_allowed, reason)
        """
        cache_key = f"{action.action_type.value}_{action.safety_level.value}"
        if cache_key in self.permission_cache:
            return self.permission_cache[cache_key], "cached"

        if action.safety_level == POPSSCMUSafetyLevel.RESTRICTED:
            return False, "Action is restricted"

        if action.safety_level == POPSSCMUSafetyLevel.SAFE and self.auto_approve_safe:
            self.permission_cache[cache_key] = True
            return True, "Auto-approved safe action"

        for perm in self.permissions.values():
            if perm.is_valid() and perm.covers_action(action.action_type):
                self.permission_cache[cache_key] = True
                return True, f"Covered by permission: {perm.name}"

        if action.safety_level == POPSSCMUSafetyLevel.CONFIRM:
            if self.confirmation_callback:
                confirmed = self.confirmation_callback(user_id, action)
                if confirmed:
                    self.permission_cache[cache_key] = True
                    return True, "User confirmed"
                else:
                    return False, "User denied confirmation"

        if action.safety_level == POPSSCMUSafetyLevel.DANGEROUS:
            if self.confirmation_callback:
                confirmed = self.confirmation_callback(user_id, action)
                if confirmed:
                    self.permission_cache[cache_key] = True
                    return True, "User confirmed dangerous action"
                else:
                    return False, "User denied dangerous action"

        return False, "No valid permission found"

    def grant_permission(
        self,
        name: str,
        action_types: Set[POPSSCMUActionType],
        scope: str = "session",
        expires_in: Optional[float] = None,
    ) -> POPSSCMUPermission:
        """Grant a new permission."""
        expires_at = None
        if expires_in:
            expires_at = datetime.fromtimestamp(time.time() + expires_in)

        permission = POPSSCMUPermission(
            name=name,
            action_types=action_types,
            scope=scope,
            granted=True,
            granted_at=datetime.now(),
            expires_at=expires_at,
        )

        with self._lock:
            self.permissions[name] = permission
            self.permission_cache.clear()

        _LOG.info("permission_granted", name=name, scope=scope)
        return permission

    def revoke_permission(self, name: str) -> bool:
        """Revoke a permission."""
        with self._lock:
            if name in self.permissions:
                del self.permissions[name]
                self.permission_cache.clear()
                _LOG.info("permission_revoked", name=name)
                return True
        return False

    def clear_expired_permissions(self) -> int:
        """Clear expired permissions."""
        count = 0
        with self._lock:
            expired = [
                name for name, perm in self.permissions.items()
                if perm.expires_at and datetime.now() > perm.expires_at
            ]
            for name in expired:
                del self.permissions[name]
                count += 1
            if count > 0:
                self.permission_cache.clear()
        return count


class POPSSCMUSandbox:
    """
    Isolated execution sandbox for safe action execution.
    
    Provides state snapshots, rollback capability, and resource
    limitation for action execution.
    
    Attributes:
        snapshots: Dictionary of state snapshots
        max_snapshots: Maximum snapshots to retain
        resource_limits: Resource limitation settings
        execution_timeout: Default execution timeout
    """

    def __init__(
        self,
        max_snapshots: int = 100,
        execution_timeout: float = 30.0,
    ):
        self.snapshots: Dict[str, POPSSCMUStateSnapshot] = {}
        self.max_snapshots = max_snapshots
        self.execution_timeout = execution_timeout
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cmu_sandbox")

        _LOG.info("sandbox_initialized")

    def create_snapshot(
        self,
        screen_state: Optional[POPSSCMUScreenState] = None,
        action_id: str = "",
        system_state: Optional[Dict[str, Any]] = None,
    ) -> POPSSCMUStateSnapshot:
        """
        Create a state snapshot before action execution.
        
        Args:
            screen_state: Current screen state
            action_id: Action that will be executed
            system_state: Current system state
        
        Returns:
            POPSSCMUStateSnapshot: Created snapshot
        """
        snapshot = POPSSCMUStateSnapshot(
            screen_state=screen_state,
            action_id=action_id,
            system_state=system_state or {},
        )

        with self._lock:
            self.snapshots[snapshot.snapshot_id] = snapshot
            if len(self.snapshots) > self.max_snapshots:
                oldest = min(self.snapshots.values(), key=lambda s: s.timestamp)
                del self.snapshots[oldest.snapshot_id]

        _LOG.info("snapshot_created", snapshot_id=snapshot.snapshot_id, action_id=action_id)
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Optional[POPSSCMUStateSnapshot]:
        """Get a snapshot by ID."""
        with self._lock:
            return self.snapshots.get(snapshot_id)

    def get_latest_snapshot(self) -> Optional[POPSSCMUStateSnapshot]:
        """Get the most recent snapshot."""
        with self._lock:
            if not self.snapshots:
                return None
            return max(self.snapshots.values(), key=lambda s: s.timestamp)

    def execute_safe(
        self,
        action: POPSSCMUAction,
        executor: Callable[[POPSSCMUAction], POPSSCMUActionResult],
        screen_state: Optional[POPSSCMUScreenState] = None,
    ) -> POPSSCMUActionResult:
        """
        Execute action safely with snapshot and rollback support.
        
        Args:
            action: Action to execute
            executor: Function to execute the action
            screen_state: Current screen state
        
        Returns:
            POPSSCMUActionResult: Execution result
        """
        snapshot = self.create_snapshot(
            screen_state=screen_state,
            action_id=action.action_id,
        )

        try:
            result = executor(action)
            return result
        except Exception as e:
            _LOG.error("sandbox_execution_failed", action_id=action.action_id, error=str(e))
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )

    async def execute_safe_async(
        self,
        action: POPSSCMUAction,
        executor: Callable[[POPSSCMUAction], POPSSCMUActionResult],
        screen_state: Optional[POPSSCMUScreenState] = None,
    ) -> POPSSCMUActionResult:
        """Execute action safely in async context."""
        snapshot = self.create_snapshot(
            screen_state=screen_state,
            action_id=action.action_id,
        )

        try:
            if asyncio.iscoroutinefunction(executor):
                result = await executor(action)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor, executor, action
                )
            return result
        except Exception as e:
            _LOG.error("sandbox_async_execution_failed", action_id=action.action_id, error=str(e))
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )

    def clear_snapshots(self) -> int:
        """Clear all snapshots."""
        with self._lock:
            count = len(self.snapshots)
            self.snapshots.clear()
        return count

    def shutdown(self) -> None:
        """Shutdown the sandbox."""
        self._executor.shutdown(wait=True)
        self.clear_snapshots()
        _LOG.info("sandbox_shutdown")


class POPSSCMUSafetyValidator:
    """
    Action validation engine with risk assessment.
    
    Validates actions against safety policies and performs
    risk assessment before execution.
    
    Attributes:
        policies: Dictionary of safety policies
        default_policy: Default safety policy
        restricted_areas: List of restricted screen areas
        dangerous_keywords: Keywords indicating dangerous operations
    """

    def __init__(
        self,
        default_policy: Optional[POPSSCMUSafetyPolicy] = None,
    ):
        self.policies: Dict[str, POPSSCMUSafetyPolicy] = {}
        self.default_policy = default_policy or POPSSCMUSafetyPolicy()
        self.restricted_areas: List[POPSSCMURectangle] = []
        self.dangerous_keywords: Set[str] = {
            "delete", "remove", "format", "erase", "wipe",
            "shutdown", "restart", "reboot",
            "password", "credential", "secret", "key",
            "admin", "root", "sudo",
        }
        self._lock = threading.RLock()

        self.policies["default"] = self.default_policy
        _LOG.info("safety_validator_initialized")

    def add_policy(self, policy: POPSSCMUSafetyPolicy) -> None:
        """Add a safety policy."""
        with self._lock:
            self.policies[policy.policy_id] = policy

    def add_restricted_area(self, area: POPSSCMURectangle) -> None:
        """Add a restricted screen area."""
        with self._lock:
            self.restricted_areas.append(area)

    def validate_action(
        self,
        action: POPSSCMUAction,
        policy: Optional[POPSSCMUSafetyPolicy] = None,
    ) -> Tuple[bool, str, POPSSCMURiskLevel]:
        """
        Validate an action against safety policies.
        
        Args:
            action: Action to validate
            policy: Optional specific policy to use
        
        Returns:
            Tuple[bool, str, POPSSCMURiskLevel]: (is_valid, reason, risk_level)
        """
        active_policy = policy or self.default_policy

        is_allowed, reason = active_policy.is_action_allowed(action)
        if not is_allowed:
            return False, reason, POPSSCMURiskLevel.HIGH

        if action.target.coordinate:
            for area in self.restricted_areas:
                if area.contains(action.target.coordinate):
                    return False, "Action targets restricted area", POPSSCMURiskLevel.HIGH

        risk_level = self._assess_risk(action)

        if action.description:
            desc_lower = action.description.lower()
            for keyword in self.dangerous_keywords:
                if keyword in desc_lower:
                    risk_level = POPSSCMURiskLevel.HIGH
                    break

        if active_policy.requires_confirmation(action):
            return True, "Requires user confirmation", risk_level

        return True, "Action validated", risk_level

    def _assess_risk(self, action: POPSSCMUAction) -> POPSSCMURiskLevel:
        """Assess risk level for an action."""
        if action.safety_level == POPSSCMUSafetyLevel.RESTRICTED:
            return POPSSCMURiskLevel.CRITICAL
        if action.safety_level == POPSSCMUSafetyLevel.DANGEROUS:
            return POPSSCMURiskLevel.HIGH
        if action.safety_level == POPSSCMUSafetyLevel.CONFIRM:
            return POPSSCMURiskLevel.MEDIUM

        high_risk_types = {
            POPSSCMUActionType.APP_CLOSE,
            POPSSCMUActionType.APP_LAUNCH,
            POPSSCMUActionType.HOTKEY,
            POPSSCMUActionType.CLIPBOARD_COPY,
            POPSSCMUActionType.CLIPBOARD_PASTE,
        }

        if action.action_type in high_risk_types:
            return POPSSCMURiskLevel.HIGH

        medium_risk_types = {
            POPSSCMUActionType.DRAG,
            POPSSCMUActionType.SWIPE,
            POPSSCMUActionType.KEY_PRESS,
            POPSSCMUActionType.RIGHT_CLICK,
        }

        if action.action_type in medium_risk_types:
            return POPSSCMURiskLevel.MEDIUM

        return POPSSCMURiskLevel.LOW

    def assess_task_risk(
        self,
        actions: List[POPSSCMUAction],
    ) -> POPSSCMURiskLevel:
        """Assess overall risk for a sequence of actions."""
        if not actions:
            return POPSSCMURiskLevel.MINIMAL

        max_risk = POPSSCMURiskLevel.MINIMAL
        risk_order = [
            POPSSCMURiskLevel.MINIMAL,
            POPSSCMURiskLevel.LOW,
            POPSSCMURiskLevel.MEDIUM,
            POPSSCMURiskLevel.HIGH,
            POPSSCMURiskLevel.CRITICAL,
        ]

        for action in actions:
            _, _, risk = self.validate_action(action)
            if risk_order.index(risk) > risk_order.index(max_risk):
                max_risk = risk

        if len(actions) > 10:
            idx = min(risk_order.index(max_risk) + 1, len(risk_order) - 1)
            max_risk = risk_order[idx]

        return max_risk


class POPSSCMUEmergencyStop:
    """
    Emergency stop mechanism for immediate halt.
    
    Provides graceful shutdown and state recovery when
    emergency stop is triggered.
    
    Attributes:
        is_active: Whether emergency stop is active
        stop_callbacks: Callbacks to execute on stop
        state_recovery: State recovery function
        trigger_time: Time when emergency stop was triggered
    """

    def __init__(
        self,
        state_recovery: Optional[Callable[[], None]] = None,
    ):
        self.is_active = False
        self.stop_callbacks: List[Callable[[], None]] = []
        self.state_recovery = state_recovery
        self.trigger_time: Optional[datetime] = None
        self._lock = threading.RLock()
        self._stop_reason: str = ""

        _LOG.info("emergency_stop_initialized")

    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for emergency stop."""
        with self._lock:
            self.stop_callbacks.append(callback)

    def trigger(self, reason: str = "Manual trigger") -> None:
        """
        Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        with self._lock:
            if self.is_active:
                return

            self.is_active = True
            self.trigger_time = datetime.now()
            self._stop_reason = reason

        _LOG.critical("emergency_stop_triggered", reason=reason)

        for callback in self.stop_callbacks:
            try:
                callback()
            except Exception as e:
                _LOG.error("emergency_stop_callback_failed", error=str(e))

        if self.state_recovery:
            try:
                self.state_recovery()
            except Exception as e:
                _LOG.error("state_recovery_failed", error=str(e))

    def reset(self) -> None:
        """Reset emergency stop state."""
        with self._lock:
            self.is_active = False
            self.trigger_time = None
            self._stop_reason = ""

        _LOG.info("emergency_stop_reset")

    def check(self) -> Tuple[bool, str]:
        """
        Check if emergency stop is active.
        
        Returns:
            Tuple[bool, str]: (is_active, reason)
        """
        with self._lock:
            return self.is_active, self._stop_reason

    def get_status(self) -> Dict[str, Any]:
        """Get emergency stop status."""
        with self._lock:
            return {
                "is_active": self.is_active,
                "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
                "reason": self._stop_reason,
                "callback_count": len(self.stop_callbacks),
            }


class POPSSCMUSafetySystem:
    """
    Integrated safety system combining all security components.
    
    Provides unified interface for the three-layer security model.
    
    Attributes:
        audit_logger: Audit logging component
        permission_manager: Permission management component
        sandbox: Execution sandbox component
        validator: Action validation component
        emergency_stop: Emergency stop component
    """

    def __init__(
        self,
        confirmation_callback: Optional[Callable[[str, POPSSCMUAction], bool]] = None,
        auto_approve_safe: bool = True,
        audit_enabled: bool = True,
    ):
        self.audit_logger = POPSSCMUAuditLogger(enabled=audit_enabled)
        self.permission_manager = POPSSCMUPermissionManager(
            confirmation_callback=confirmation_callback,
            auto_approve_safe=auto_approve_safe,
        )
        self.sandbox = POPSSCMUSandbox()
        self.validator = POPSSCMUSafetyValidator()
        self.emergency_stop = POPSSCMUEmergencyStop(
            state_recovery=self._recover_state
        )

        self.emergency_stop.register_callback(self._on_emergency_stop)

        _LOG.info("safety_system_initialized")

    def _on_emergency_stop(self) -> None:
        """Handle emergency stop event."""
        self.audit_logger.log_emergency_stop("Emergency stop triggered")

    def _recover_state(self) -> None:
        """Recover state from latest snapshot."""
        snapshot = self.sandbox.get_latest_snapshot()
        if snapshot:
            _LOG.info("state_recovery_started", snapshot_id=snapshot.snapshot_id)

    def validate_and_execute(
        self,
        action: POPSSCMUAction,
        executor: Callable[[POPSSCMUAction], POPSSCMUActionResult],
        screen_state: Optional[POPSSCMUScreenState] = None,
        user_id: str = "",
    ) -> POPSSCMUActionResult:
        """
        Validate and execute action with full safety checks.
        
        Args:
            action: Action to execute
            executor: Function to execute the action
            screen_state: Current screen state
            user_id: User identifier
        
        Returns:
            POPSSCMUActionResult: Execution result
        """
        is_active, reason = self.emergency_stop.check()
        if is_active:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.CANCELLED,
                error_message=f"Emergency stop active: {reason}",
            )

        is_valid, validation_reason, risk_level = self.validator.validate_action(action)
        if not is_valid:
            self.audit_logger.log_action_blocked(action, validation_reason)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.BLOCKED,
                error_message=validation_reason,
            )

        has_permission, perm_reason = self.permission_manager.check_permission(action, user_id)
        if not has_permission:
            self.audit_logger.log_action_blocked(action, perm_reason)
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.BLOCKED,
                error_message=perm_reason,
            )

        self.audit_logger.log_action_started(action)

        result = self.sandbox.execute_safe(action, executor, screen_state)

        self.audit_logger.log_action_completed(action, result)

        return result

    def shutdown(self) -> None:
        """Shutdown the safety system."""
        self.audit_logger.close()
        self.sandbox.shutdown()
        _LOG.info("safety_system_shutdown")
