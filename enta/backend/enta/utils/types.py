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

from dataclasses import dataclass, field
from typing import Any, Literal, Union

FinishReason = Literal["stop", "tool_calls", "error", "max_tokens", "cancelled"]
PermissionMode = Literal["default", "accept_edits", "bypass", "dont_ask", "plan", "spec", "auto", "blacklist"]  # noqa: E501
PermissionBehavior = Literal["allow", "deny", "ask"]
TaskType = Literal["bash", "agent", "workflow"]
TaskStatus = Literal["pending", "running", "completed", "failed", "killed"]


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCallStart:
    name: str
    id: str


@dataclass
class ToolCallDelta:
    id: str
    key: str
    value: str


@dataclass
class ToolCallEnd:
    id: str


@dataclass
class ToolProgress:
    id: str
    tool_name: str
    status: str
    sub_agent_messages: list[dict[str, Any]] | None = None


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool
    sub_agent_messages: list[dict[str, Any]] | None = None
    sub_agent_session_id: str | None = None


@dataclass
class PermissionRequest:
    tool_name: str
    reason: str


@dataclass
class EngineInstallRequest:
    """Sent to frontend when a tool needs a missing engine / driver
    (Playwright bundled Chromium, Edge CDP, msedgedriver, etc.) and
    the LLM should NOT be involved in resolving the choice.

    The frontend is expected to show a native dialog (e.g. Electron
    confirmInstall) and send back an :class:`EngineInstallResponse`
    on the same channel.  The agent run() that yielded this event
    is suspended until the response arrives or the request times out.
    """
    request_id: str
    engine: str  # e.g. "playwright-chromium", "msedgedriver", "edge-cdp"
    title: str
    body: str
    hint: str = ""
    options: list[dict[str, Any]] = field(default_factory=list)
    # Each option is ``{"id": str, "label": str, "description": str,
    # "kind": "primary"|"secondary"}``.  The frontend should default to
    # the first option on Enter.
    # I18n: when these message_code fields are non-empty the frontend
    # resolves the text via t() instead of using the raw string.
    title_code: str = ""
    title_args: dict[str, str] = field(default_factory=dict)
    body_code: str = ""
    body_args: dict[str, str] = field(default_factory=dict)
    hint_code: str = ""
    hint_args: dict[str, str] = field(default_factory=dict)


@dataclass
class EngineInstallProgress:
    """Progress update for an in-flight engine install / connect."""
    request_id: str
    pct: float  # 0..100
    message: str
    sub_message: str = ""
    indeterminate: bool = False
    status: str = "running"  # "running" | "success" | "fail" | "cancelled"
    # I18n: backend sends a message_code instead of a hardcoded string;
    # frontend looks up the translation.  message_args provides template
    # variables (e.g. ``{"pct": "42"}``).
    message_code: str = ""
    message_args: dict[str, str] = field(default_factory=dict)
    sub_message_code: str = ""
    sub_message_args: dict[str, str] = field(default_factory=dict)


AgentEvent = Union[
    "TextDelta", "ThinkingDelta",
    "ToolCallStart", "ToolCallDelta", "ToolCallEnd",
    "ToolProgress", "ToolResult",
    "PermissionRequest", "QuestionRequest",
    "EngineInstallRequest", "EngineInstallProgress",
    "Finish",
]


@dataclass
class QuestionRequest:
    """Sent to frontend when the model asks the user questions."""
    tool_call_id: str
    questions: list[dict[str, Any]]
    """Each dict: {"question": str, "details"?: str, "options"?: [str]}"""


@dataclass
class Finish:
    reason: FinishReason
    usage: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class Artifact:
    artifact: dict[str, Any]


@dataclass
class Reference:
    reference: dict[str, Any]


@dataclass
class PlanUpdate:
    plan_items: list[dict[str, Any]]


@dataclass
class PlanProposal:
    """A pending write action the agent wants to perform in plan mode.

    The desktop UI shows the preview/diff and lets the user approve or
    reject. The agent only executes the underlying tool when the user
    explicitly approves.

    Attributes:
        proposal_id: Stable id so the UI can route approve/reject back.
        tool_call_id: Originating tool call id from the model.
        tool_name: Concrete tool that would run (file_write, file_edit,
            apply_patch, bash, etc.).
        tool_args: The original args the model produced.
        preview: A short human-readable description of the intended
            change (e.g. ``"Modify backend/enta/loop.py: +12 -4"``).
        diff_text: Optional unified diff for file-shaped proposals.
        file_path: Optional file path the proposal targets.
        original: Optional original file content (for diff display).
        proposed: Optional proposed file content.
        added: Lines added (for diff display).
        removed: Lines removed (for diff display).
        risk: Optional risk hint (``"low" | "medium" | "high"``).
    """

    proposal_id: str
    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    preview: str = ""
    diff_text: str = ""
    file_path: str = ""
    original: str = ""
    proposed: str = ""
    added: int = 0
    removed: int = 0
    risk: str = "low"


@dataclass
class PlanModeChanged:
    """Emitted when the agent enters or exits plan mode."""

    active: bool
    reason: str = ""


@dataclass
class PlanResolved:
    """Emitted when the user resolves a pending plan proposal."""

    proposal_id: str
    tool_call_id: str
    approved: bool


@dataclass
class CompactNotification:
    old_count: int
    new_count: int
    old_tokens: int
    new_tokens: int


@dataclass
class EditProposal:
    """A pending file edit the agent proposes but has not yet applied.

    Emitted from the agent loop (typically by the ``file_edit`` tool
    with ``dry_run=True``) and consumed by the desktop UI which
    renders an inline diff and lets the user accept or reject the
    change.  The agent does not write the file until the user accepts.

    Attributes:
        tool_call_id: The originating tool call id -- required so the
            UI can route the user's accept/reject decision back to
            the correct pending request.
        file_path: Absolute path of the file the edit applies to.
        diff_text: Unified diff text (Rust-native ``compute_diff``
            output) for rendering in the UI.
        original: The file content **before** the proposed edit.
        proposed: The file content **after** the proposed edit.
        added: Number of inserted lines.
        removed: Number of deleted lines.
        summary: Optional human-readable summary of the edits
            (e.g. ``"renamed foo to bar in module X"``).
    """

    tool_call_id: str
    file_path: str
    diff_text: str
    original: str
    proposed: str
    added: int
    removed: int
    summary: str = ""


def create_edit_proposal(
    tool_call_id: str,
    file_path: str,
    diff_text: str,
    original: str,
    proposed: str,
    added: int,
    removed: int,
    summary: str = "",
) -> EditProposal:
    return EditProposal(
        tool_call_id=tool_call_id,
        file_path=file_path,
        diff_text=diff_text,
        original=original,
        proposed=proposed,
        added=added,
        removed=removed,
        summary=summary,
    )


@dataclass
class AssistantBoundary:
    pass


@dataclass
class WorkflowStartedEvent:
    workflow_id: str
    goal: str
    total_tasks: int
    task_ids: list[str] = field(default_factory=list)


@dataclass
class WorkflowTaskEvent:
    workflow_id: str
    task_id: str
    task_name: str
    status: str  # started | running | completed | failed | skipped


@dataclass
class WorkflowCompletedEvent:
    workflow_id: str
    goal: str
    success: bool
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    total_duration: float = 0.0


AgentEvent = Union[TextDelta, ThinkingDelta, ToolCallStart, ToolCallDelta, ToolCallEnd, ToolProgress, ToolResult, PermissionRequest, QuestionRequest, Finish, Artifact, PlanUpdate, CompactNotification, EditProposal, AssistantBoundary, WorkflowStartedEvent, WorkflowTaskEvent, WorkflowCompletedEvent]  # noqa: E501


# ── Multimodal Content Blocks ──────────────────────────────────────


@dataclass
class TextContent:
    type: str = "text"
    text: str = ""


@dataclass
class ImageContent:
    type: str = "image"
    data: str = ""           # base64 encoded
    mime_type: str = "image/png"
    source_url: str = ""


@dataclass
class FileContent:
    type: str = "file"
    data: str = ""           # base64 encoded
    filename: str = ""
    mime_type: str = "application/octet-stream"


@dataclass
class ToolUseContent:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] | None = None


@dataclass
class ToolResultContent:
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


ContentBlock = Union[TextContent, ImageContent, FileContent, ToolUseContent, ToolResultContent]


@dataclass
class BackendText:
    text: str


@dataclass
class BackendThinking:
    text: str
    signature_delta: str | None = None


@dataclass
class BackendToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class BackendToolCallDelta:
    index: int
    key: str
    value: str


@dataclass
class BackendFinish:
    reason: str
    usage: dict[str, Any] | None = None


@dataclass
class BackendError:
    error: str


BackendEvent = Union[BackendText, BackendThinking, BackendToolCall, BackendToolCallDelta, BackendFinish, BackendError]  # noqa: E501


@dataclass
class PermissionAllow:
    behavior: PermissionBehavior = "allow"
    reason: str = ""


@dataclass
class PermissionDeny:
    behavior: PermissionBehavior = "deny"
    reason: str = ""


@dataclass
class PermissionAsk:
    behavior: PermissionBehavior = "ask"
    reason: str = ""
    rule: str = ""


PermissionDecision = Union[PermissionAllow, PermissionDeny, PermissionAsk]


@dataclass
class AdaptiveThinking:
    enabled: bool = True
    min_tokens: int = 1024
    max_tokens: int = 8192
    budget_ratio: float = 0.5


@dataclass
class EnabledThinking:
    enabled: bool = True
    budget_tokens: int = 4096


@dataclass
class DisabledThinking:
    enabled: bool = False


ThinkingConfig = Union[AdaptiveThinking, EnabledThinking, DisabledThinking]


def create_text_delta(text: str) -> TextDelta:
    return TextDelta(text=text)


def create_thinking_delta(text: str) -> ThinkingDelta:
    return ThinkingDelta(text=text)


def create_tool_call_start(name: str, id: str) -> ToolCallStart:
    return ToolCallStart(name=name, id=id)


def create_tool_call_delta(id: str, key: str, value: str) -> ToolCallDelta:
    return ToolCallDelta(id=id, key=key, value=value)


def create_tool_call_end(id: str) -> ToolCallEnd:
    return ToolCallEnd(id=id)


def create_tool_progress(
    id: str,
    tool_name: str,
    status: str,
    sub_agent_messages: list[dict[str, Any]] | None = None,
) -> ToolProgress:
    return ToolProgress(
        id=id,
        tool_name=tool_name,
        status=status,
        sub_agent_messages=sub_agent_messages,
    )


def create_tool_result(
    id: str,
    content: str,
    is_error: bool = False,
    sub_agent_messages: list[dict[str, Any]] | None = None,
    sub_agent_session_id: str | None = None,
) -> ToolResult:
    return ToolResult(
        id=id,
        content=content,
        is_error=is_error,
        sub_agent_messages=sub_agent_messages,
        sub_agent_session_id=sub_agent_session_id,
    )


def create_permission_request(tool_name: str, reason: str) -> PermissionRequest:
    return PermissionRequest(tool_name=tool_name, reason=reason)


def create_question_request(tool_call_id: str, questions: list[dict[str, Any]]) -> QuestionRequest:
    return QuestionRequest(tool_call_id=tool_call_id, questions=questions)


def create_finish(reason: FinishReason, usage: dict[str, Any] | None = None, error: str | None = None) -> Finish:  # noqa: E501
    return Finish(reason=reason, usage=usage, error=error)


def create_backend_text(text: str) -> BackendText:
    return BackendText(text=text)


def create_backend_thinking(text: str, signature_delta: str | None = None) -> BackendThinking:
    return BackendThinking(text=text, signature_delta=signature_delta)


def create_backend_tool_call(id: str, name: str, arguments: str) -> BackendToolCall:
    return BackendToolCall(id=id, name=name, arguments=arguments)


def create_backend_tool_call_delta(index: int, key: str, value: str) -> BackendToolCallDelta:
    return BackendToolCallDelta(index=index, key=key, value=value)


def create_backend_finish(reason: str, usage: dict[str, Any] | None = None) -> BackendFinish:
    return BackendFinish(reason=reason, usage=usage)


def create_backend_error(error: str) -> BackendError:
    return BackendError(error=error)


def create_artifact(artifact: dict[str, Any]) -> Artifact:
    return Artifact(artifact=artifact)


def create_assistant_boundary() -> AssistantBoundary:
    return AssistantBoundary()


def create_plan_proposal(
    proposal_id: str,
    tool_call_id: str,
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    preview: str = "",
    diff_text: str = "",
    file_path: str = "",
    original: str = "",
    proposed: str = "",
    added: int = 0,
    removed: int = 0,
    risk: str = "low",
) -> PlanProposal:
    return PlanProposal(
        proposal_id=proposal_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args or {},
        preview=preview,
        diff_text=diff_text,
        file_path=file_path,
        original=original,
        proposed=proposed,
        added=added,
        removed=removed,
        risk=risk,
    )


def create_plan_mode_changed(active: bool, reason: str = "") -> PlanModeChanged:
    return PlanModeChanged(active=active, reason=reason)


def create_plan_resolved(proposal_id: str, tool_call_id: str, approved: bool) -> PlanResolved:
    return PlanResolved(proposal_id=proposal_id, tool_call_id=tool_call_id, approved=approved)


# ── Branch Protocol Types ──────────────────────────────────────────


@dataclass
class BranchMetaData:
    id: str
    parent_branch_id: str | None = None
    fork_point_message_id: str | None = None
    created_at: float = 0.0
    messages_count: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "total": 0})


@dataclass
class BranchUpdated:
    active_branch_id: str
    branches: list[dict[str, Any]]
    messages: list[dict[str, Any]]


@dataclass
class BranchSwitched:
    branch_id: str
    messages: list[dict[str, Any]]
    branches: list[dict[str, Any]]
    tokens: dict[str, int] | None = None


@dataclass
class BranchRolledBack:
    branch_id: str
    removed_message_ids: list[str]


BranchEvent = Union[BranchUpdated, BranchSwitched, BranchRolledBack]
