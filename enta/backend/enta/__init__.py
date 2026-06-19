#!/usr/bin/env python3

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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
Encre (EnTA Core) -- training-grade tool surface for PiscesL1.

This package provides the curated tool palette and inference back-ends
that the model uses during adversarial training.  Every external /
cloud-LLM gateway, web service, IDE hook, and orchestration system has
been removed; what remains is:

* the EnCRE builtin tool set (``enta.tools.builtin``) used for
  file editing, code search, bash, git, task management, web search,
  memory, etc.  These are the tools the trained model learns to wield.
* the sandbox subsystem used to safely execute the model's shell
  commands during training rollouts.
* a small set of inference back-ends (``LocalBackend`` for HF
  transformers, ``OpenAICompatibleBackend`` for any vLLM/SGLang-style
  server).  No cloud provider adapters are bundled.
* a thin Python bridge over the Rust native extension that powers
  fast file I/O, token counting, permission checks, and SIMD search.

The public surface of the package is exposed lazily via ``__getattr__``
so that any module entry can be referenced even when the optional Rust
extension (``enta._native``) is not built.
"""

# ── Monkey-patch subprocess FIRST so subsequent imports see a consistent API ──
import asyncio as _aio
import os as _os
import subprocess as _sp

_CRF = 0x08000000  # CREATE_NO_WINDOW only
_SI = _sp.STARTUPINFO(dwFlags=_sp.STARTF_USESHOWWINDOW, wShowWindow=_sp.SW_HIDE)


def _inj(kw):
    if _os.name == "nt":
        kw.setdefault("creationflags", _CRF)
        kw.setdefault("startupinfo", _SI)
    else:
        kw.setdefault("new_session", True)
    return kw


_sp_run = _sp.run
_sp_popen = _sp.Popen
_aio_exec = _aio.create_subprocess_exec
_aio_shell = _aio.create_subprocess_shell


def _patched_run(*a, **k):
    return _sp_run(*a, **_inj(k))


class _PatchedPopen(_sp_popen):
    def __init__(self, a, **k):
        _sp_popen.__init__(self, a, **_inj(k))


async def _patched_exec(*a, **k):
    return await _aio_exec(*a, **_inj(k))


async def _patched_shell(*a, **k):
    return await _aio_shell(*a, **_inj(k))


_sp.run = _patched_run  # type: ignore[assignment]
_sp.Popen = _PatchedPopen  # type: ignore[assignment]
_sp._EncrePatched = True
_aio.create_subprocess_exec = _patched_exec  # type: ignore[assignment]
_aio.create_subprocess_shell = _patched_shell  # type: ignore[assignment]
_aio._EncrePatched = True

# ── Lazy import map ─────────────────────────────────────────────────────────
# (module_path, attribute_name).  Modules referenced here must exist in
# the slimmed EnCRE package layout.
_LAZY_MAP: dict = {
    # Backends
    "BaseBackend": ("enta.backends.base", "BaseBackend"),
    "LocalBackend": ("enta.backends.local", "LocalBackend"),
    "OpenAICompatibleBackend": ("enta.backends.openai_compatible", "OpenAICompatibleBackend"),
    "OpenAISSEBackend": ("enta.backends.openai_sse", "OpenAISSEBackend"),
    "DEFAULT_RETRY_CONFIG": ("enta.backends.retry", "DEFAULT_RETRY_CONFIG"),
    "RetryConfig": ("enta.backends.retry", "RetryConfig"),
    "retry_with_backoff": ("enta.backends.retry", "retry_with_backoff"),
    "BackendRegistry": ("enta.backends.registry", "BackendRegistry"),
    "resolve_model_info": ("enta.backends.registry", "resolve_model_info"),
    "PROVIDERS": ("enta.backends.catalog", "PROVIDERS"),
    "DEFAULT_MAX_OUTPUT_TOKENS": ("enta.backends.catalog", "DEFAULT_MAX_OUTPUT_TOKENS"),
    "catalog_payload": ("enta.backends.catalog", "catalog_payload"),
    "default_output_tokens": ("enta.backends.catalog", "default_output_tokens"),
    "get_model": ("enta.backends.catalog", "get_model"),
    "get_provider": ("enta.backends.catalog", "get_provider"),
    # Remote teacher + multi-teacher roundtable
    "TeacherSpec": ("enta.backends.remote_teacher", "TeacherSpec"),
    "TeacherAnswer": ("enta.backends.remote_teacher", "TeacherAnswer"),
    "JudgeVerdict": ("enta.backends.remote_teacher", "JudgeVerdict"),
    "RoundtableResult": ("enta.backends.remote_teacher", "RoundtableResult"),
    "RemoteTeacherClient": ("enta.backends.remote_teacher", "RemoteTeacherClient"),
    "TeacherRoundtable": ("enta.backends.remote_teacher", "TeacherRoundtable"),
    "build_roundtable_from_config": (
        "enta.backends.remote_teacher",
        "build_roundtable_from_config",
    ),
    # Native Rust bridge
    "native_apply_diff": ("enta.native", "apply_diff"),
    "native_compute_diff": ("enta.native", "compute_diff"),
    "native_count_tokens": ("enta.native", "count_tokens"),
    "native_glob": ("enta.native", "glob"),
    "native_grep": ("enta.native", "grep"),
    "native_read_file": ("enta.native", "read_file"),
    "native_write_file": ("enta.native", "write_file"),
    "native_shell_execute": ("enta.native", "execute_shell"),
    "native_sandbox_execute": ("enta.native", "sandbox_execute"),
    "native_search_codebase": ("enta.native", "search_codebase"),
    # Tool system
    "EncreTool": ("enta.tools.base", "EncreTool"),
    "build_tool": ("enta.tools.base", "build_tool"),
    "ToolRegistry": ("enta.tools.registry", "ToolRegistry"),
    # Tool palette (EnCRE builtin tools -- preserved for adversarial training)
    "build_default_tool_registry": ("enta.tools.builtin", "build_default_tool_registry"),
    "DEFAULT_BUILTIN_TOOLS": ("enta.tools.builtin", "DEFAULT_BUILTIN_TOOLS"),
    "EncreAgentTool": ("enta.tools.builtin.agent", "EncreAgentTool"),
    "EncreApplyPatchTool": ("enta.tools.builtin.apply_patch", "EncreApplyPatchTool"),
    "EncreBashTool": ("enta.tools.builtin.bash", "EncreBashTool"),
    "EncreBashKillTool": ("enta.tools.builtin.bash_io", "EncreBashKillTool"),
    "EncreBashListTool": ("enta.tools.builtin.bash_io", "EncreBashListTool"),
    "EncreBashOutputTool": ("enta.tools.builtin.bash_io", "EncreBashOutputTool"),
    "EncreDatabaseTool": ("enta.tools.builtin.database", "EncreDatabaseTool"),
    "EncreDeployTool": ("enta.tools.builtin.deploy", "EncreDeployTool"),
    "EncreDockerTool": ("enta.tools.builtin.docker", "EncreDockerTool"),
    "EncreFileEditTool": ("enta.tools.builtin.file_edit", "EncreFileEditTool"),
    "EncreFileReadTool": ("enta.tools.builtin.file_read", "EncreFileReadTool"),
    "EncreFileWriteTool": ("enta.tools.builtin.file_write", "EncreFileWriteTool"),
    "EncreFindToolTool": ("enta.tools.builtin.find_tool", "EncreFindToolTool"),
    "EncreGitTool": ("enta.tools.builtin.git_tool", "EncreGitTool"),
    "EncreGlobTool": ("enta.tools.builtin.glob", "EncreGlobTool"),
    "EncreGrepTool": ("enta.tools.builtin.grep", "EncreGrepTool"),
    "EncreImageTool": ("enta.tools.builtin.image", "EncreImageTool"),
    "EncreLintFormatTool": ("enta.tools.builtin.lint_format", "EncreLintFormatTool"),
    "EncreLSPTool": ("enta.tools.builtin.lsp", "EncreLSPTool"),
    "EncreMemoryCreateTool": ("enta.tools.builtin.memory", "EncreMemoryCreateTool"),
    "EncreMemoryDeleteTool": ("enta.tools.builtin.memory", "EncreMemoryDeleteTool"),
    "EncreMemoryReadTool": ("enta.tools.builtin.memory", "EncreMemoryReadTool"),
    "EncreMemorySearchTool": ("enta.tools.builtin.memory", "EncreMemorySearchTool"),
    "EncreMemoryUpdateTool": ("enta.tools.builtin.memory", "EncreMemoryUpdateTool"),
    "EncrePDFTool": ("enta.tools.builtin.pdf", "EncrePDFTool"),
    "EncreRESTTool": ("enta.tools.builtin.rest_client", "EncreRESTTool"),
    "EncreSpreadsheetTool": ("enta.tools.builtin.spreadsheet", "EncreSpreadsheetTool"),
    "EncreTaskCreateTool": ("enta.tools.builtin.task_create", "EncreTaskCreateTool"),
    "EncreTaskGetTool": ("enta.tools.builtin.task_get", "EncreTaskGetTool"),
    "EncreTaskListTool": ("enta.tools.builtin.task_list", "EncreTaskListTool"),
    "EncreTaskOutputTool": ("enta.tools.builtin.task_output", "EncreTaskOutputTool"),
    "EncreTaskStopTool": ("enta.tools.builtin.task_stop", "EncreTaskStopTool"),
    "EncreTaskUpdateTool": ("enta.tools.builtin.task_update", "EncreTaskUpdateTool"),
    "EncreTestRunTool": ("enta.tools.builtin.test_runner", "EncreTestRunTool"),
    "EncreTodoTool": ("enta.tools.builtin.todo", "EncreTodoTool"),
    "EncreWebFetchTool": ("enta.tools.builtin.web_fetch", "EncreWebFetchTool"),
    "EncreWebSearchTool": ("enta.tools.builtin.web_search", "EncreWebSearchTool"),
    "EncreWorkflowTool": ("enta.tools.builtin.workflow", "EncreWorkflowTool"),
    # Sandbox
    "EncreContainerSandbox": ("enta.sandbox.container", "EncreContainerSandbox"),
    "SandboxConfig": ("enta.sandbox.types", "SandboxConfig"),
    "SandboxMode": ("enta.sandbox.types", "SandboxMode"),
    "SandboxResult": ("enta.sandbox.types", "SandboxResult"),
    "NetworkPolicy": ("enta.sandbox.types", "NetworkPolicy"),
    "ResourceConfig": ("enta.sandbox.types", "ResourceConfig"),
    "SeccompConfig": ("enta.sandbox.types", "SeccompConfig"),
    "SeccompProfile": ("enta.sandbox.types", "SeccompProfile"),
    "FileProtection": ("enta.sandbox.types", "FileProtection"),
    "FileProtectionConfig": ("enta.sandbox.types", "FileProtectionConfig"),
    "EnvConfig": ("enta.sandbox.types", "EnvConfig"),
    "NetworkConfig": ("enta.sandbox.types", "NetworkConfig"),
    "CGroupLimit": ("enta.sandbox.types", "CGroupLimit"),
    # Utils
    "BranchIDGenerator": ("enta.utils.idgen", "BranchIDGenerator"),
    "EncreTaskStore": ("enta.utils.task_store", "EncreTaskStore"),
    "get_task_store": ("enta.utils.task_store", "get_store"),
    "count_message_tokens": ("enta.utils.tokens", "count_message_tokens"),
    "estimate_message_tokens": ("enta.utils.tokens", "estimate_message_tokens"),
    "trim_messages_to_budget": ("enta.utils.tokens", "trim_messages_to_budget"),
    # Event / data types
    "TextDelta": ("enta.utils.types", "TextDelta"),
    "ThinkingDelta": ("enta.utils.types", "ThinkingDelta"),
    "ToolCallStart": ("enta.utils.types", "ToolCallStart"),
    "ToolCallDelta": ("enta.utils.types", "ToolCallDelta"),
    "ToolCallEnd": ("enta.utils.types", "ToolCallEnd"),
    "ToolCallRecord": ("enta.utils.types", "ToolCallRecord"),
    "ToolProgress": ("enta.utils.types", "ToolProgress"),
    "ToolResult": ("enta.utils.types", "ToolResult"),
    "BackendEvent": ("enta.utils.types", "BackendEvent"),
    "BackendError": ("enta.utils.types", "BackendError"),
    "BackendFinish": ("enta.utils.types", "BackendFinish"),
    "BackendText": ("enta.utils.types", "BackendText"),
    "BackendToolCall": ("enta.utils.types", "BackendToolCall"),
    "BackendToolCallDelta": ("enta.utils.types", "BackendToolCallDelta"),
    "Finish": ("enta.utils.types", "Finish"),
    "FinishReason": ("enta.utils.types", "FinishReason"),
    "PlanUpdate": ("enta.utils.types", "PlanUpdate"),
    "create_backend_error": ("enta.utils.types", "create_backend_error"),
    "create_backend_finish": ("enta.utils.types", "create_backend_finish"),
    "create_backend_text": ("enta.utils.types", "create_backend_text"),
    "create_backend_tool_call": ("enta.utils.types", "create_backend_tool_call"),
    "create_backend_tool_call_delta": ("enta.utils.types", "create_backend_tool_call_delta"),
    "create_finish": ("enta.utils.types", "create_finish"),
    "create_text_delta": ("enta.utils.types", "create_text_delta"),
    "create_thinking_delta": ("enta.utils.types", "create_thinking_delta"),
    "create_tool_call_delta": ("enta.utils.types", "create_tool_call_delta"),
    "create_tool_call_end": ("enta.utils.types", "create_tool_call_end"),
    "create_tool_call_start": ("enta.utils.types", "create_tool_call_start"),
    "create_tool_progress": ("enta.utils.types", "create_tool_progress"),
    "create_tool_result": ("enta.utils.types", "create_tool_result"),
}


def __getattr__(name: str):
    """Lazy attribute loader -- import the underlying module on first use."""
    if name in _LAZY_MAP:
        import importlib
        mod_path, attr = _LAZY_MAP[name]
        try:
            return getattr(importlib.import_module(mod_path), attr)
        except Exception as e:
            raise ImportError(
                f"enta.{name} requires module {mod_path} (cause: {type(e).__name__}: {e})"
            ) from e
    raise AttributeError(f"module 'enta' has no attribute {name!r}")


def __dir__():
    """Expose all public names to ``dir()`` / autocompletion."""
    return sorted(set(_LAZY_MAP) | set(globals()))


# ── Convenience helpers for the training pipeline ───────────────────────────

def create_default_backend(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                           device: str = "cpu", **kwargs):
    """Create a :class:`LocalBackend` with sensible training defaults.

    The training loop should normally use this helper instead of importing
    ``LocalBackend`` directly so that the choice of backend can be
    overridden by environment variables in the future without touching
    the call sites.
    """
    from enta.backends.local import LocalBackend
    return LocalBackend(model_name=model_name, device=device, **kwargs)


def build_training_tool_registry():
    """Return a :class:`ToolRegistry` populated with the EnCRE palette.

    The returned registry is what the training loop should hand to the
    model as ``tools=...`` during adversarial rollouts.  It includes
    every preserved builtin tool (file ops, search, git, bash, task
    management, memory, web search, etc.) and nothing else.
    """
    from enta.tools.builtin import build_default_tool_registry
    return build_default_tool_registry()


__all__ = [
    *sorted(_LAZY_MAP),
    "build_training_tool_registry",
    "create_default_backend",
]
