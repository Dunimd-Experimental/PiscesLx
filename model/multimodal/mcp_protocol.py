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

from __future__ import annotations

"""
Model-level tool-calling protocol primitives for the Yv architecture.

This module is the *model layer* of tool invocation: it defines the message
envelope, the tool descriptor, the registry contract and the executor contract
that the model emits and consumes. It deliberately contains **no** service-layer
code — hosting, sandboxing, sessions and monitoring live outside the model
(see ``third_party/mcp_service/``).

Per ``docs/prompts/UPGRADE_03_ARCH.md`` §3.3.1, tool calling is a *text-level*
capability (structured tags / special tokens produced autoregressively), not an
independent modality. These primitives model that text-level protocol only.
"""

import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "YvMCPMessageType",
    "YvMCPToolType",
    "YvExecutionMode",
    "YvMCPMessage",
    "YvMCPToolMetadata",
    "YvMCPExecutionResult",
    "YvAgenticAction",
    "YvAgenticObservation",
    "YvMCPProtocol",
    "YvCoreMCPToolRegistry",
    "YvMCPExecutionManager",
    "YvMCPUnifiedToolExecutor",
]


class YvMCPMessageType(str, Enum):
    """Envelope kinds exchanged between the model and a tool host."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_REGISTER = "tool_register"
    TOOL_LIST = "tool_list"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class YvMCPToolType(str, Enum):
    """Where a tool's computation physically happens relative to the model."""

    NATIVE = "native"
    EXTERNAL = "external"


class YvExecutionMode(str, Enum):
    """How the executor decides to run a tool."""

    AUTO = "auto"
    NATIVE_ONLY = "native_only"
    EXTERNAL_ONLY = "external_only"


class YvMCPMessage:
    """A single protocol message: envelope metadata plus a JSON-safe payload."""

    __slots__ = ("message_type", "sender_id", "receiver_id", "payload",
                 "correlation_id", "timestamp")

    def __init__(self, message_type: YvMCPMessageType, sender_id: str, receiver_id: str,
                 payload: Dict[str, Any], correlation_id: str, timestamp: float) -> None:
        self.message_type = message_type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.payload = payload
        self.correlation_id = correlation_id
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the message for transport or for a text-level tool tag."""
        return {
            "type": self.message_type.value,
            "sender": self.sender_id,
            "receiver": self.receiver_id,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


class YvMCPToolMetadata:
    """Declarative description of one callable tool."""

    __slots__ = ("name", "description", "parameters", "tool_type", "handler",
                 "returns", "timeout_s", "tags")

    def __init__(self, name: str, description: str, parameters: Dict[str, Any],
                 tool_type: YvMCPToolType, handler: Optional[Callable[..., Any]] = None,
                 returns: str = "", timeout_s: float = 30.0,
                 tags: Optional[List[str]] = None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.tool_type = tool_type
        self.handler = handler
        self.returns = returns
        self.timeout_s = timeout_s
        self.tags = list(tags or [])

    def signature(self) -> Dict[str, Any]:
        """The schema block a model conditions on when emitting a tool call."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "type": self.tool_type.value,
        }


class YvMCPExecutionResult:
    """Outcome of one tool invocation."""

    __slots__ = ("tool_name", "success", "output", "error", "duration_s", "attempted")

    def __init__(self, tool_name: str, success: bool, output: Any = None,
                 error: str = "", duration_s: float = 0.0,
                 attempted: tuple = ()) -> None:
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error
        self.duration_s = duration_s
        self.attempted = attempted

    def is_success(self) -> bool:
        """Whether the call produced a usable output."""
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary form used to build a tool-result message."""
        return {
            "tool": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_s": self.duration_s,
        }


class YvAgenticAction:
    """A model-emitted decision to invoke a tool."""

    __slots__ = ("tool_name", "arguments", "thought", "step")

    def __init__(self, tool_name: str, arguments: Dict[str, Any],
                 thought: str = "", step: int = 0) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.thought = thought
        self.step = step

    def to_dict(self) -> Dict[str, Any]:
        """Render the action as the JSON body of a tool-call tag."""
        return {"tool": self.tool_name, "arguments": self.arguments,
                "thought": self.thought, "step": self.step}


class YvAgenticObservation:
    """The environment/tool response fed back into the text stream."""

    __slots__ = ("content", "error", "timestamp")

    def __init__(self, content: Any, error: str = "", timestamp: Optional[float] = None) -> None:
        self.content = content
        self.error = error
        self.timestamp = timestamp if timestamp is not None else time.time()

    @property
    def failed(self) -> bool:
        """Whether the observation carries an error."""
        return bool(self.error)


class YvMCPProtocol:
    """Factory helpers for well-formed protocol messages."""

    @staticmethod
    def create_message(message_type: YvMCPMessageType, sender_id: str,
                       payload: Dict[str, Any], correlation_id: str = "",
                       receiver_id: str = "") -> YvMCPMessage:
        """Build a message with a fresh identifier and timestamp."""
        return YvMCPMessage(
            message_type=message_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=payload,
            correlation_id=correlation_id or uuid.uuid4().hex,
            timestamp=time.time(),
        )

    @staticmethod
    def tool_call(sender_id: str, receiver_id: str, action: YvAgenticAction,
                  correlation_id: str = "") -> YvMCPMessage:
        """Build a tool-call message from a model action."""
        return YvMCPProtocol.create_message(
            YvMCPMessageType.TOOL_CALL, sender_id,
            {"action": action.to_dict()}, correlation_id, receiver_id,
        )

    @staticmethod
    def tool_result(sender_id: str, receiver_id: str, result: YvMCPExecutionResult,
                    correlation_id: str = "") -> YvMCPMessage:
        """Build a tool-result message from an execution outcome."""
        return YvMCPProtocol.create_message(
            YvMCPMessageType.TOOL_RESULT, sender_id,
            {"result": result.to_dict()}, correlation_id, receiver_id,
        )


class YvCoreMCPToolRegistry:
    """Name → metadata store for tools the model may call."""

    def __init__(self) -> None:
        self._tools: Dict[str, YvMCPToolMetadata] = {}

    def register(self, metadata: YvMCPToolMetadata) -> bool:
        """Insert or replace a tool descriptor."""
        self._tools[metadata.name] = metadata
        return True

    def unregister(self, name: str) -> bool:
        """Remove a tool descriptor; returns whether it existed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[YvMCPToolMetadata]:
        """Look up one tool descriptor."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Sorted names of all registered tools."""
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """Signatures for every tool — what gets rendered into the prompt."""
        return [self._tools[name].signature() for name in self.list_tools()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


class YvMCPExecutionManager:
    """Tracks executions and their outcomes for one session."""

    def __init__(self) -> None:
        self._history: List[YvMCPExecutionResult] = []

    def record(self, result: YvMCPExecutionResult) -> None:
        """Append one execution outcome."""
        self._history.append(result)

    def history(self) -> List[YvMCPExecutionResult]:
        """All recorded outcomes in order."""
        return list(self._history)

    def failure_count(self) -> int:
        """How many recorded executions failed."""
        return sum(1 for r in self._history if not r.success)

    def clear(self) -> None:
        """Drop recorded history."""
        self._history.clear()


class YvMCPUnifiedToolExecutor:
    """Resolves a tool name to a callable and runs it under the given mode."""

    _MANAGER = YvMCPExecutionManager()

    def __init__(self, registry: Optional[YvCoreMCPToolRegistry] = None) -> None:
        self.registry = registry if registry is not None else YvCoreMCPToolRegistry()

    @classmethod
    def get_instance(cls) -> "YvMCPUnifiedToolExecutor":
        """Process-wide executor used by callers that only need name resolution."""
        return cls()

    def execute(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None,
                mode: str = YvExecutionMode.AUTO.value) -> YvMCPExecutionResult:
        """Invoke a registered tool and record the outcome."""
        started = time.time()
        arguments = dict(arguments or {})
        metadata = self.registry.get(tool_name)
        if metadata is None:
            result = YvMCPExecutionResult(
                tool_name=tool_name, success=False,
                error=f"unknown tool '{tool_name}'",
                duration_s=time.time() - started,
            )
            self._MANAGER.record(result)
            return result
        if metadata.handler is None:
            result = YvMCPExecutionResult(
                tool_name=tool_name, success=False,
                error=f"tool '{tool_name}' has no bound handler",
                duration_s=time.time() - started,
            )
            self._MANAGER.record(result)
            return result
        if mode == YvExecutionMode.NATIVE_ONLY.value and metadata.tool_type != YvMCPToolType.NATIVE:
            result = YvMCPExecutionResult(
                tool_name=tool_name, success=False, error="native-only mode rejects external tool",
                duration_s=time.time() - started,
            )
            self._MANAGER.record(result)
            return result
        if mode == YvExecutionMode.EXTERNAL_ONLY.value and metadata.tool_type != YvMCPToolType.EXTERNAL:
            result = YvMCPExecutionResult(
                tool_name=tool_name, success=False, error="external-only mode rejects native tool",
                duration_s=time.time() - started,
            )
            self._MANAGER.record(result)
            return result
        try:
            output = metadata.handler(**arguments)
            result = YvMCPExecutionResult(
                tool_name=tool_name, success=True, output=output,
                duration_s=time.time() - started,
            )
        except Exception as error:
            result = YvMCPExecutionResult(
                tool_name=tool_name, success=False, error=str(error),
                duration_s=time.time() - started,
            )
        self._MANAGER.record(result)
        return result

    def manager(self) -> YvMCPExecutionManager:
        """The execution manager backing this executor."""
        return self._MANAGER


class YvMCPService:
    """Convenience accessor for the process-wide registry and executor."""

    _REGISTRY = YvCoreMCPToolRegistry()
    _EXECUTOR = YvMCPUnifiedToolExecutor(_REGISTRY)

    @classmethod
    def registry(cls) -> YvCoreMCPToolRegistry:
        """Shared model-level tool registry."""
        return cls._REGISTRY

    @classmethod
    def executor(cls) -> YvMCPUnifiedToolExecutor:
        """Shared model-level tool executor."""
        return cls._EXECUTOR
