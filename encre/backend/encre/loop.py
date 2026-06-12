#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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
import asyncio
import json
import os
import re
import time
from typing import Any, AsyncGenerator

from encre.backend import create_backend
from encre.backends.base import BaseBackend
from encre.compact.engine import EncreCompactEngine
from encre.config import EncreConfig
from encre.evolution.config import EvolutionConfig
from encre.logging_config import get_logger
from encre.prompts.base import EncrePromptTemplate
from encre.prompts.classifier import classify_intents

logger = get_logger(__name__)
from encre.codebase.indexer import EncreCodeIndex
from encre.codebase.document_manager import EncreDocumentManager
from encre.feedback.learner import EncreFeedbackLearner
from encre.git.repo import EncreGitRepo
from encre.recovery import ErrorRecoveryEngine, RetryableExecutor
from encre.thinking.config import resolve_thinking_config
from encre.safety import EncreSafetyEngine
from encre.utils.tokens import count_message_tokens
from encre.rollback import EncreRollbackGit
from encre.session import EncreSession
from encre.telemetry import EncreTelemetry
from encre.tools.registry import ToolRegistry
from encre.tools.discovery import ToolDiscovery
from encre.hooks.system import EncreHookSystem
from encre.memdir.system import EncreMemorySystem
from encre.profile.system import EncreProfileSystem
from encre.soul.system import EncreSoulSystem
from encre.skills.registry import EncreSkillRegistry
from encre.rules.loader import RulesLoader
from encre.utils.types import (
    AgentEvent,
    Artifact,
    AssistantBoundary,
    BackendError,
    BackendFinish,
    BackendText,
    BackendThinking,
    CompactNotification,
    PlanUpdate,
    BackendToolCall,
    BackendToolCallDelta,
    Finish,
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
    ToolCallEnd,
    ToolProgress,
    ToolResult,
    create_finish,
    create_permission_request,
    create_text_delta,
    create_thinking_delta,
    create_tool_call_delta,
    create_tool_call_end,
    create_tool_call_start,
    create_tool_progress,
    create_tool_result,
    create_assistant_boundary,
    create_question_request,
)

_WRITE_TOOL_NAMES = {"file_write", "file_edit", "write_file", "writeFile", "apply_patch"}
_MICROCOMPACT_THRESHOLD = 3  # turns after which old tool results are trimmed
_PROMPT_CACHE_TTL_SECONDS = 5.0


def _apply_result_budget(
    result: str,
    tool: Any,
    max_chars: int = 100_000,
) -> str:
    """Truncate a tool result if it exceeds the tool's size budget.

    Each tool can declare ``max_result_size_chars``.  The default is
    100 000 characters (≈ 25 000 tokens).  Results beyond that are
    truncated with a count of removed characters.
    """
    budget = getattr(tool, "max_result_size_chars", max_chars) or max_chars
    if len(result) > budget:
        excess = len(result) - budget
        return result[:budget] + f"\n... (truncated {excess} characters)"
    return result


def _microcompact_old_results(
    messages: list[dict[str, Any]],
    keep_recent_turns: int = 3,
    trimmed_result: str = "[Previous tool output cleared]",
) -> list[dict[str, Any]]:
    """Replace tool results older than ``keep_recent_turns`` with a stub.

    This is the "micro-compaction" step — it prevents old tool results
    from consuming context budget without requiring a full LLM summary.
    Only ``tool`` role messages whose content exceeds 200 characters are
    candidates for clearing.

    Returns a **new** list; the original is not mutated.
    """
    if len(messages) < 4:
        return messages

    # Work backwards counting assistant+turns; skip the most recent ones.
    turn_count = 0
    keep_idx = 0
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role", "")
        if role == "assistant":
            turn_count += 1
            if turn_count >= keep_recent_turns:
                keep_idx = i
                break

    if keep_idx == 0:
        return messages

    result = list(messages)
    for i in range(keep_idx):
        if result[i].get("role") == "tool":
            content = result[i].get("content", "")
            if isinstance(content, str) and len(content) > 200:
                result[i] = dict(result[i], content=trimmed_result)
    return result


def _extract_file_path(tool_name: str, result: str) -> str | None:
    if tool_name not in _WRITE_TOOL_NAMES:
        return None

    if tool_name == "apply_patch":
        # Result format: "{summary}\n{json}"
        import json as _json
        try:
            json_part = result.split("\n", 1)[1] if "\n" in result else result
            data = _json.loads(json_part)
            files = data.get("files", []) if isinstance(data, dict) else []
            if files:
                fp = files[0].get("new_path") or files[0].get("old_path", "")
                if fp and os.path.isabs(fp) and os.path.exists(fp):
                    return fp
        except (_json.JSONDecodeError, IndexError, KeyError):
            pass
        return None

    for pattern in [
        r"Successfully wrote \d+ characters to (.+)",
        r"Applied \d+ edit\(s\) to (.+?)\.\s*\n",  # file_edit — \n forces lazy match past file extension periods
        r"Wrote .+ to (.+)",
    ]:
        m = re.search(pattern, result, re.IGNORECASE)
        if m:
            path = m.group(1).strip()
            if os.path.isabs(path) and os.path.exists(path):
                return path
    return None


def _extract_diff_text(tool_name: str, result: str) -> str:
    """Extract the unified diff block from a tool result string."""
    m = re.search(r"```diff\n(.+?)\n```", result, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _args_summary(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, ensure_ascii=False)[:600]
    except Exception:
        return str(args)[:600]


def _permission_reason(tool_name: str) -> str:
    return f"Tool {tool_name} requires permission"


def _extract_apply_patch_paths(result: str) -> list[str]:
    """Extract all successful file paths from an apply_patch result JSON."""
    import json as _json
    try:
        json_part = result.split("\n", 1)[1] if "\n" in result else result
        data = _json.loads(json_part)
        files = data.get("files", []) if isinstance(data, dict) else []
        paths = []
        for f in files:
            if f.get("status") != "ok":
                continue
            fp = f.get("new_path") or f.get("old_path", "")
            if fp and os.path.isabs(fp) and os.path.exists(fp):
                paths.append(fp)
        return paths
    except (_json.JSONDecodeError, IndexError, KeyError):
        return []


_PLAN_STATUS_MAP = {
    "pending": "pending",
    "in_progress": "active",
    "completed": "done",
}


def _ensure_plan_items(tool_name: str, args: dict[str, Any]) -> list[dict[str, Any]] | None:
    if tool_name != "todo":
        return None
    todos = args.get("todos")
    if not todos or not isinstance(todos, list):
        return None
    items: list[dict[str, Any]] = []
    for i, todo in enumerate(todos):
        content = todo.get("content", "")
        if not content:
            continue
        status = _PLAN_STATUS_MAP.get(todo.get("status", "pending"), "pending")
        items.append({
            "id": f"plan-{i}",
            "text": content,
            "status": status,
        })
    return items if items else None


class EncreLoop:
    def __init__(
        self,
        config: EncreConfig,
        session: EncreSession,
        tool_registry: ToolRegistry | None = None,
        hook_system: EncreHookSystem | None = None,
        safety: EncreSafetyEngine | None = None,
        memory_system: EncreMemorySystem | None = None,
        profile_system: EncreProfileSystem | None = None,
        soul_system: EncreSoulSystem | None = None,
        skill_registry: EncreSkillRegistry | None = None,
        telemetry: EncreTelemetry | None = None,
        evolution: EvolutionConfig | None = None,
        recovery: ErrorRecoveryEngine | None = None,
        feedback: EncreFeedbackLearner | None = None,
        code_index: EncreCodeIndex | None = None,
        sub_agent_depth: int = 0,
    ) -> None:
        self.config = config
        self.session = session
        self.tool_registry = tool_registry or ToolRegistry()
        self.discovery = ToolDiscovery(self.tool_registry)
        self.hook_system = hook_system or EncreHookSystem()
        self.memory_system = memory_system
        self.profile_system = profile_system
        self.soul_system = soul_system
        self.skill_registry = skill_registry
        self.telemetry = telemetry or EncreTelemetry(enabled=False)
        self.sub_agent_depth = sub_agent_depth
        evo = evolution or EvolutionConfig.create_default()
        self.learner = evo.learner
        self.optimizer = evo.optimizer
        self.reflex = evo.reflex
        self.meta = evo.meta
        self.recovery_engine = recovery or ErrorRecoveryEngine()
        self.feedback = feedback
        self._code_index: EncreCodeIndex | None = code_index
        self._pending_code_scan: EncreCodeIndex | None = None

        # Auto-resolve thinking config based on model if not explicitly set
        self._thinking_config = config.thinking_config
        if self._thinking_config is None:
            self._thinking_config = resolve_thinking_config(None, config.model)
        self.backend: BaseBackend | None = create_backend(
            config.backend_type,
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            **config.backend_kwargs,
        )
        self.safety = safety or EncreSafetyEngine(config)
        self.compact_engine = EncreCompactEngine()
        self.prompt_builder = EncrePromptTemplate()
        self.rollback = EncreRollbackGit()
        self._permission_event: asyncio.Event | None = None
        self._permission_decision: bool = False
        self._pending_tool_name: str = ""
        self._question_event: asyncio.Event | None = None
        self._question_answers: str = ""
        self._cancel_event = asyncio.Event()
        self._rules_loader = RulesLoader()
        self._document_manager: EncreDocumentManager | None = None
        self._document_manager_data_dir: str | None = None
        self._workspace_info_cache: tuple[str, float, tuple[str, str, str]] | None = None
        self._memory_prompt_cache: tuple[str, float, str] | None = None
        self._soul_prompt_cache: tuple[str, float, str] | None = None
        self._document_prompt_cache: tuple[str, float, str] | None = None
        self._codebase_context_cache: tuple[tuple[str, int, int], float, str] | None = None
        self._profile_prompt_cache: tuple[str, str, float, str] | None = None
        self._rules_prompt_cache: tuple[tuple[str, bool, bool], float, str] | None = None
        self._sanitized_branches: set[str] = set()

    def _cache_fresh(self, built_at: float, ttl: float = _PROMPT_CACHE_TTL_SECONDS) -> bool:
        return (time.time() - built_at) < ttl

    async def aclose(self) -> None:
        """Release backend resources (httpx clients, model memory, etc.)."""
        if self.backend is not None:
            try:
                await self.backend.aclose()
            except Exception as e:
                logger.warning(f"Error closing backend: {e}", extra={"backend": type(self.backend).__name__})

    def resolve_permission(self, decision: bool) -> None:
        """Called by the agent owner to approve or deny a pending permission request."""
        self._permission_decision = decision
        if self._permission_event is not None:
            self._permission_event.set()

    def resolve_question(self, answers: str) -> None:
        """Called when the user answers a pending question."""
        self._question_answers = answers
        if self._question_event is not None:
            self._question_event.set()

    async def _chat_with_timeout(
        self,
        gen: AsyncGenerator[BackendEvent, None],
        timeout: float = 120.0,
    ) -> AsyncGenerator[BackendEvent, None]:
        """Iterate ``gen`` with a per-iteration timeout so a hanging API call
        (wrong key, no network, overloaded provider) surfaces an error instead
        of freezing the UI indefinitely."""
        try:
            while True:
                try:
                    event = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                    yield event
                except StopAsyncIteration:
                    return
        except asyncio.TimeoutError:
            logger.error("[run] backend.chat() timed out after %.0fs — check API key / network", timeout)
            yield BackendError(f"API request timed out after {timeout}s")
        except Exception:
            raise

    def cancel(self) -> None:
        """Signal the agent loop to stop at the next checkpoint."""
        self._cancel_event.set()
        # Allow sanitize to re-run on the next turn so any incomplete
        # assistant+tool_calls message (from an interrupted tool execution)
        # is cleaned up before the backend sees it, preventing 400 errors.
        self._sanitized_branches.clear()

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    _SKILL_PATTERN = re.compile(r"^/(\S+)(?:\s+(.*))?", re.DOTALL)

    async def _activate_skills(self, prompt: str) -> tuple[str, str]:
        """Detect /skill-name invocations in prompt.

        Returns (skill_prompt, stripped_prompt). skill_prompt is "" if no skills matched.
        """
        if not self.skill_registry:
            return "", prompt
        parts: list[str] = []
        remaining = prompt
        while True:
            m = self._SKILL_PATTERN.match(remaining)
            if not m:
                break
            skill_name = m.group(1)
            args = (m.group(2) or "").strip() or None
            skill = self.skill_registry.lookup(skill_name)
            if skill is None:
                break
            skill_prompt = await self.skill_registry.activate(skill_name, args)
            if not skill_prompt.startswith("Error:"):
                parts.append(skill_prompt)
            end = m.end()
            remaining = remaining[end:].strip()
        if parts:
            return "\n\n".join(parts) + "\n\n---\n\n", remaining
        return "", prompt

    def _workspace_info(self) -> tuple[str, str, str]:
        """Return (workspace_root, workspace_name, project_summary) for the prompt builder.

        Returns ("", "", "") when not running inside a workspace.
        """
        ws_path = getattr(self.config, "workspace", "") or ""
        if not ws_path or not os.path.isdir(ws_path):
            self._workspace_info_cache = None
            return "", "", ""
        cache_key = ws_path
        if (
            self._workspace_info_cache is not None
            and self._workspace_info_cache[0] == cache_key
            and self._cache_fresh(self._workspace_info_cache[1])
        ):
            return self._workspace_info_cache[2]

        ws_name = os.path.basename(ws_path)

        # Load workspace config overrides from .encre/config.json
        yim_dir = os.path.join(ws_path, ".encre")
        ws_config_path = os.path.join(yim_dir, "config.json")
        ws_config: dict[str, Any] = {}
        if os.path.isfile(ws_config_path):
            try:
                with open(ws_config_path, "r", encoding="utf-8") as f:
                    ws_config = json.load(f)
            except Exception:
                pass

        summary_lines: list[str] = []

        custom_prompt = ws_config.get("system_prompt", "")
        if custom_prompt:
            summary_lines.append("Project-specific instructions:")
            summary_lines.append(custom_prompt)
            summary_lines.append("")

        # Top-level directory contents
        try:
            visible: list[tuple[str, bool]] = []
            with os.scandir(ws_path) as entries:
                for entry in entries:
                    name = entry.name
                    if name.startswith(".") and name != ".encre":
                        continue
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        is_dir = False
                    visible.append((name, is_dir))
            visible.sort(key=lambda item: (not item[1], item[0]))
            if visible:
                summary_lines.append("Top-level entries:")
                for name, is_dir in visible[:40]:
                    prefix = "/" if is_dir else " "
                    summary_lines.append(f"  {prefix}{name}")
                if len(visible) > 40:
                    summary_lines.append(f"  ... and {len(visible) - 40} more entries")
        except Exception:
            pass

        # Git state
        try:
            git_repo = EncreGitRepo(ws_path)
            if git_repo.is_in_repo():
                state = git_repo.get_state()
                summary_lines.append("")
                summary_lines.append("Git status:")
                summary_lines.append(f"  branch: {state.branch}")
                summary_lines.append(f"  clean: {'yes' if state.is_clean else 'no'}")
                if state.changed_files:
                    summary_lines.append(f"  changed: {', '.join(state.changed_files[:20])}")
                if state.untracked_files:
                    summary_lines.append(f"  untracked: {', '.join(state.untracked_files[:10])}")
        except Exception:
            pass

        result = (ws_path, ws_name, "\n".join(summary_lines))
        self._workspace_info_cache = (cache_key, time.time(), result)
        return result

    def _build_workspace_context(self) -> str:
        """Deprecated — workspace context is now produced by _workspace_info()
        and consumed by EncrePromptBuilder. Kept for backward compatibility with
        external callers; returns an empty string in the new pipeline."""
        return ""

    async def _build_codebase_context(self) -> str:
        """Build codebase context from the workspace index when available.

        The workspace index is **always** built in a background subprocess
        (see :class:`encre.codebase.index_manager.IndexManager`). This method
        is intentionally read-only and never instantiates
        :class:`EncreCodeIndex` itself — that constructor can rebuild the
        BM25 inverted index, which is a CPU-bound operation that would
        stall the agent's main event loop if executed in the same thread
        (Python GIL prevents ThreadPoolExecutor from providing true
        parallelism for CPU work).

        If the externally injected ``self._code_index`` is missing,
        pointing at a different workspace, or not yet ready, this method
        returns an empty string immediately so the message pipeline is
        never blocked.  Callers that need richer context should subscribe
        to the index progress notifications exposed by ``IndexManager``.
        """
        ws_path = getattr(self.config, "workspace", "") or ""
        if not ws_path or not os.path.isdir(ws_path):
            return ""

        # Index not yet injected by an external owner (e.g. IndexManager).
        # Returning immediately is the whole point: the agent loop must
        # stay responsive while indexing is happening elsewhere.
        if self._code_index is None:
            return ""
        if getattr(self._code_index, "workspace", "") != ws_path:
            return ""
        if not getattr(self._code_index, "_indexed", False):
            return ""

        if self._code_index is None:
            return ""

        # Lazily build derived structures (dep graph + BM25) the first
        # time a query is performed.  This still runs synchronously on
        # the calling thread, but only when we actually need results —
        # the previous hot path that unconditionally rebuilt the BM25
        # index inside ``_build_codebase_context`` is gone.
        try:
            self._code_index._ensure_query_ready()
        except Exception:
            return ""

        modules = self._code_index.list_all_modules()
        total = len(modules)
        if total == 0:
            return ""

        by_lang: dict[str, int] = {}
        for mod in modules:
            lang = mod.language or "other"
            by_lang[lang] = by_lang.get(lang, 0) + 1
        lang_summary_items = tuple(sorted(by_lang.items(), key=lambda x: (-x[1], x[0])))
        cache_key = (ws_path, total, int(self._code_index._indexed), lang_summary_items)
        if (
            self._codebase_context_cache is not None
            and self._codebase_context_cache[0] == cache_key
            and self._cache_fresh(self._codebase_context_cache[1])
        ):
            return self._codebase_context_cache[2]

        lines: list[str] = []
        lines.append("## Codebase Index")
        lines.append(f"Indexed {total} source files in the workspace.")
        lines.append("Use `codebase_search` to find relevant code, or `codebase_context` to view a specific file's details.")

        # Quick top-level summary: count by language
        if lang_summary_items:
            lang_summary = ", ".join(f"{lang}: {count}" for lang, count in lang_summary_items)
            lines.append(f"Language breakdown: {lang_summary}")

        result = "\n".join(lines)
        self._codebase_context_cache = (cache_key, time.time(), result)
        return result

    def _build_document_context(self) -> str:
        from encre.config import get_data_dir

        try:
            data_dir = str(get_data_dir())
            index_path = os.path.join(data_dir, "documents", "index.json")
            try:
                st = os.stat(index_path)
                cache_key = f"{data_dir}:{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                cache_key = data_dir
            if (
                self._document_prompt_cache is not None
                and self._document_prompt_cache[0] == cache_key
                and self._cache_fresh(self._document_prompt_cache[1])
            ):
                return self._document_prompt_cache[2]

            if self._document_manager is None or self._document_manager_data_dir != data_dir:
                self._document_manager = EncreDocumentManager(data_dir)
                self._document_manager_data_dir = data_dir
            else:
                self._document_manager._load()
            prompt = self._document_manager.build_context()
            self._document_prompt_cache = (cache_key, time.time(), prompt)
            return prompt
        except Exception:
            return ""

    def _build_memory_prompt(self) -> str:
        if self.memory_system is None:
            return ""

        memory_dir = self.memory_system.get_memory_path()
        cache_key = memory_dir
        if (
            self._memory_prompt_cache is not None
            and self._memory_prompt_cache[0] == cache_key
            and self._cache_fresh(self._memory_prompt_cache[1])
        ):
            return self._memory_prompt_cache[2]

        prompt = self.memory_system.build_prompt()
        self._memory_prompt_cache = (cache_key, time.time(), prompt)
        return prompt

    def _build_soul_prompt(self) -> str:
        if self.soul_system is None:
            return ""

        soul_dir = self.soul_system.get_soul_dir()
        cache_key = soul_dir
        if (
            self._soul_prompt_cache is not None
            and self._soul_prompt_cache[0] == cache_key
            and self._cache_fresh(self._soul_prompt_cache[1])
        ):
            return self._soul_prompt_cache[2]

        prompt = self.soul_system.build_prompt()
        self._soul_prompt_cache = (cache_key, time.time(), prompt)
        return prompt

    def _refresh_profile_in_system(self) -> None:
        if self.profile_system is None:
            return
        if not self.session.messages or self.session.messages[0].get("role") != "system":
            return
        try:
            # Use the last user message as query for relevance matching
            query = ""
            for m in reversed(self.session.messages):
                if m.get("role") == "user":
                    query = m.get("content", "")
                    break
            fresh = self.profile_system.build_relevant_prompt(query=query, threshold=0.0)
            if not fresh:
                return
            content = self.session.messages[0].get("content", "")
            content = re.sub(
                r"\n+## User Profile.*?(?=\n+## |\Z)",
                "",
                content,
                count=1,
                flags=re.DOTALL,
            )
            content = content.rstrip() + "\n\n" + fresh
            self.session.messages[0]["content"] = content
            self.session.mark_messages_dirty()
        except Exception:
            pass

    def _build_profile_prompt(self, query: str) -> str:
        if self.profile_system is None:
            return ""
        cache_key = (getattr(self.profile_system, "_profile_path", ""), query)
        if (
            self._profile_prompt_cache is not None
            and self._profile_prompt_cache[0] == cache_key[0]
            and self._profile_prompt_cache[1] == cache_key[1]
            and self._cache_fresh(self._profile_prompt_cache[2])
        ):
            return self._profile_prompt_cache[3]
        prompt = self.profile_system.build_relevant_prompt(query=query, threshold=0.0)
        self._profile_prompt_cache = (cache_key[0], cache_key[1], time.time(), prompt)
        return prompt

    def _build_rules_prompt(self) -> str:
        ws_root = getattr(self.config, "workspace", "") or ""
        cache_key = (
            ws_root,
            bool(self.config.enable_project_rules),
            bool(self.config.enable_global_rules),
        )
        if (
            self._rules_prompt_cache is not None
            and self._rules_prompt_cache[0] == cache_key
            and self._cache_fresh(self._rules_prompt_cache[1])
        ):
            return self._rules_prompt_cache[2]
        prompt = self._rules_loader.build_rules_prompt(
            ws_root,
            enable_project=self.config.enable_project_rules,
            enable_global=self.config.enable_global_rules,
        )
        self._rules_prompt_cache = (cache_key, time.time(), prompt)
        return prompt

    async def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        custom_instructions: str = "",
    ) -> AsyncGenerator[AgentEvent, None]:
        if self.backend is None:
            logger.warning("Agent run requested but no backend configured")
            yield create_finish("error", error="No backend configured. Send a 'configure' message first.")
            return

        # Mark this loop as the active loop so context-aware tools (find_tool,
        # EncreAgentTool) see the correct discovery/registry/session even when
        # nested inside a sub-agent.
        from encre.tools.builtin.find_tool import set_active_loop, reset_active_loop
        from encre.tools.builtin.agent import set_active_loop as set_agent_active_loop
        from encre.tools.builtin.agent import reset_active_loop as reset_agent_active_loop
        _loop_token = set_active_loop(self)
        _agent_loop_token = set_agent_active_loop(self)
        try:
            async for ev in self._run_impl(prompt, system_prompt, custom_instructions):
                yield ev
        finally:
            reset_active_loop(_loop_token)
            reset_agent_active_loop(_agent_loop_token)

    async def _run_impl(
        self,
        prompt: str,
        system_prompt: str | None = None,
        custom_instructions: str = "",
    ) -> AsyncGenerator[AgentEvent, None]:
        # Clear any stale cancel/pause state from a previous run so new
        # messages are not immediately rejected after a user cancellation.
        self._cancel_event.clear()
        # Classify user intent for dynamic prompt assembly
        intents = classify_intents(prompt)

        # Activate any skills invoked via /skill-name syntax
        skill_prompt, prompt = await self._activate_skills(prompt)
        _t0 = time.time()
        tools = None
        if self.backend.supports_tool_calling():
            tools = self.discovery.get_active_tools_payload(self.session.id, fmt="openai")
        ws_root, ws_name, ws_summary = self._workspace_info()

        if system_prompt is None:
            # Cache the base system prompt by a content-hash key so we don't
            # rebuild it every turn when nothing changed.
            _cache_key = (
                self.config.permission_mode,
                self.session.id,
                tuple(t.get("function", {}).get("name", "") for t in tools) if tools else (),
                tuple(sorted(intents)),
                ws_root, ws_name, ws_summary,
                self.config.language_preference,
                self.config.language,
                custom_instructions,
            )
            if (
                hasattr(self, "_sys_prompt_cache")
                and self._sys_prompt_cache_key == _cache_key
            ):
                system_prompt = self._sys_prompt_cache
            else:
                system_prompt = self.prompt_builder.build_system_prompt(
                    self.config.permission_mode,
                    tools=tools,
                    intents=intents,
                    workspace_root=ws_root,
                    workspace_name=ws_name,
                    project_summary=ws_summary,
                    language_preference=self.config.language_preference,
                    app_language=self.config.language,
                    custom_instructions=custom_instructions,
                )
                self._sys_prompt_cache = system_prompt
                self._sys_prompt_cache_key = _cache_key
        elif self.config.permission_mode in ("plan", "spec"):
            # Custom system_prompt was provided (e.g., from an active agent).
            # Plan/spec mode requires mode-specific instructions — build the
            # full mode-aware prompt and prepend the custom content so both
            # the custom prompt and the mode instructions are in effect.
            built = self.prompt_builder.build_system_prompt(
                self.config.permission_mode,
                tools=tools,
                intents=intents,
                workspace_root=ws_root,
                workspace_name=ws_name,
                project_summary=ws_summary,
                language_preference=self.config.language_preference,
                app_language=self.config.language,
                custom_instructions=custom_instructions,
            )
            system_prompt = system_prompt + "\n\n" + built

        # When a custom system_prompt was provided by a parent agent (not
        # None, not plan/spec mode), skip workspace context enrichment.
        # However, when no custom prompt was given (system_prompt was None),
        # the agent runs as a full session — don't skip enrichments.
        _original_system_prompt_was_none = system_prompt is None
        _skip_enrichment = (
            system_prompt is not None
            and self.config.permission_mode not in ("plan", "spec")
            and not _original_system_prompt_was_none
        )

        # Inject codebase index context (multi-language code search + dependencies)
        if not _skip_enrichment:
            codebase_ctx = await self._build_codebase_context()
            if codebase_ctx:
                system_prompt = system_prompt + "\n\n" + codebase_ctx

        # Prepend skill prompt to system prompt
        if skill_prompt:
            system_prompt = skill_prompt + system_prompt

        if _skip_enrichment:
            # Sub-agent behavioral framework: essential blocks every agent
            # needs (tool protocol, safety, identity, output format) but
            # WITHOUT workspace-specific context that would distract from
            # the delegated task.
            from encre.prompts.loader import PromptLoader
            _loader = PromptLoader()
            _behavioral_parts: list[str] = []
            for _bname in ("identity", "safety", "tool_usage", "output_format"):
                try:
                    _bcontent = _loader.load(_bname)
                    if _bcontent:
                        _behavioral_parts.append(_bcontent)
                except Exception:
                    pass
            # Language preference
            _lang_pref = self.config.language_preference or ""
            _app_lang = self.config.language or ""
            _resolved = _lang_pref if _lang_pref and _lang_pref != "auto" else _app_lang
            if _resolved == "zh":
                _behavioral_parts.append(
                    "IMPORTANT: You must always respond in Chinese (中文) "
                    "throughout the entire conversation."
                )
            elif _resolved == "en":
                _behavioral_parts.append(
                    "IMPORTANT: You must always respond in English "
                    "throughout the entire conversation."
                )
            if _behavioral_parts:
                system_prompt = system_prompt + "\n\n" + "\n\n".join(_behavioral_parts)
        else:
            # Inject persistent memory context (encrypted memories from disk)
            if self.memory_system is not None:
                try:
                    memory_prompt = self._build_memory_prompt()
                    if memory_prompt:
                        system_prompt = system_prompt + "\n\n" + memory_prompt
                except Exception:
                    pass

            # Inject relevant profile context — only fields matching the user's query
            if self.profile_system is not None:
                try:
                    profile_prompt = self._build_profile_prompt(prompt)
                    if profile_prompt:
                        system_prompt = system_prompt + "\n\n" + profile_prompt
                except Exception:
                    pass

            # Inject agent soul / identity context (SOUL.md, IDENTITY.md, USER.md)
            if self.soul_system is not None:
                try:
                    soul_prompt = self._build_soul_prompt()
                    if soul_prompt:
                        system_prompt = system_prompt + "\n\n" + soul_prompt
                except Exception:
                    pass

            # Inject reference document context
            try:
                doc_prompt = self._build_document_context()
                if doc_prompt:
                    system_prompt = system_prompt + "\n\n" + doc_prompt
            except Exception:
                pass

            # Inject user rules (project-level + global)
            try:
                rules_prompt = self._build_rules_prompt()
                if rules_prompt:
                    from encre.prompts.loader import PromptLoader
                    _loader = PromptLoader()
                    rules_block = _loader.load_with_context("rules", rules_content=rules_prompt)
                    system_prompt = system_prompt + "\n\n" + rules_block
            except Exception:
                pass

        # Update system message on every run so prompt blocks match current intents
        has_system = any(
            m.get("role") == "system" and m.get("branch_id", self.session.active_branch_id) == self.session.active_branch_id
            for m in self.session.messages
        )
        if has_system:
            for i, m in enumerate(self.session.messages):
                if m.get("role") == "system" and m.get("branch_id", self.session.active_branch_id) == self.session.active_branch_id:
                    self.session.messages[i] = {"role": "system", "content": system_prompt, "branch_id": self.session.active_branch_id}
                    self.session.mark_messages_dirty()
                    break
        else:
            self.session.messages.insert(0, {"role": "system", "content": system_prompt, "branch_id": self.session.active_branch_id})
            self.session.mark_messages_dirty()

        # Add user prompt if not a duplicate of the last user message
        # in the active branch context (not just self.session.messages[-1],
        # which may be from a different branch during retry).
        ctx_msgs = self.session.get_context_messages()
        last_ctx_user = None
        for m in reversed(ctx_msgs):
            if m.get("role") == "user":
                last_ctx_user = m
                break
        if last_ctx_user is None or last_ctx_user.get("content") != prompt:
            if skill_prompt and last_ctx_user is not None:
                # Skill was activated — don't add a duplicate with the stripped text.
                # Keep the original message content so the user sees what they typed.
                pass
            else:
                logger.info("[sub_agent] adding user message to session | prompt_len=%d | last_ctx_user_exists=%s",
                            len(prompt), last_ctx_user is not None)
                self.session.add_message("user", prompt)

        if time.time() - _t0 > 0.1:
            logger.info("[perf] prompt build %.1fs", time.time() - _t0)
        _t_hook = time.time()
        await self.hook_system.emit_session_start()
        logger.info("[run] emit_session_start done (%.2fs)", time.time() - _t_hook)
        _last_backend_usage: dict[str, Any] | None = None

        # Sanitize session messages on every run — old sessions loaded from disk
        # may contain broken tool_call groups (from crashes) that cause 400 errors.
        # Only sanitize active branch context; other branches remain untouched.
        active_branch_id = self.session.active_branch_id
        if active_branch_id not in self._sanitized_branches:
            self.session.replace_branch_messages(active_branch_id, self.compact_engine.sanitize(ctx_msgs))
            self._sanitized_branches.add(active_branch_id)
            ctx_msgs = self.session.get_context_messages()

        while not self.session.is_max_turns_reached() and not self._cancelled():
            turn_start = time.time()
            turn_events = 0
            _t_ts = time.time()
            await self.hook_system.emit_turn_start(self.session.turn_count)
            logger.info("[run] emit_turn_start done turn=%d (%.2fs)", self.session.turn_count, time.time() - _t_ts)
            _t_ck = time.time()
            self.session.checkpoint(f"turn_{self.session.turn_count}")
            await self.hook_system.emit_checkpoint(f"turn_{self.session.turn_count}")
            logger.info("[run] emit_checkpoint done turn=%d (%.2fs)", self.session.turn_count, time.time() - _t_ck)
            # Refresh context at the start of every turn so the model
            # sees its own assistant messages and tool results from
            # previous turns — without this the context stays frozen on
            # the initial user message, causing repeated tool invocations.
            ctx_msgs = self.session.get_context_messages()
            context_msgs = ctx_msgs
            if await self.compact_engine.should_compact(
                context_msgs, self.backend.context_window_size()
            ):
                old_count = len(context_msgs)
                est_tokens = count_message_tokens(context_msgs)
                await self.hook_system.emit_pre_compact(old_count, est_tokens)
                compacted = await self.compact_engine.compact(
                    context_msgs, self.backend.context_window_size()
                )
                self.session.replace_branch_messages(self.session.active_branch_id, compacted)
                new_context = self.session.get_context_messages()
                ctx_msgs = new_context
                new_tokens = count_message_tokens(new_context)
                yield CompactNotification(
                    old_count=old_count,
                    new_count=len(new_context),
                    old_tokens=est_tokens,
                    new_tokens=new_tokens,
                )
                await self.hook_system.emit_post_compact(old_count, len(new_context))
                self._refresh_profile_in_system()

            tools = None
            if self.backend.supports_tool_calling():
                tools = self.discovery.get_active_tools_payload(self.session.id, fmt="openai")

            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_call_buffers: dict[int, dict[str, Any]] = {}
            # Secondary buffers for intra-turn split: when thinking/text appears
            # AFTER tool calls within the same backend.chat() stream, route to
            # these so we can yield them as a separate assistant message later.
            _extra_thinking: list[str] = []
            _extra_text: list[str] = []
            _extra_buffers: dict[int, dict[str, Any]] = {}
            _tool_seen = False
            _in_extra = False

            _t_pm = time.time()
            pre_model = await self.hook_system.emit_pre_model_request(
                self.session.messages, tools
            )
            logger.info("[run] emit_pre_model_request done turn=%d (%.2fs)", self.session.turn_count, time.time() - _t_pm)
            backend_messages = list(ctx_msgs)
            backend_tools = tools
            if pre_model and pre_model.get("modified_input"):
                mi = pre_model["modified_input"]
                backend_messages = mi.get("messages", backend_messages)

            # Inject evolution guidance and feedback into backend messages only
            # (not into self.session.messages) so they don't appear as user input in the UI,
            # don't cause tool duplication on subsequent turns, and — critically —
            # don't end the agent prematurely.  Guidance is merged into the LAST user
            # message rather than appended as a NEW user turn, because a separate turn
            # tricks the model into responding to the guidance as if it were a fresh
            # instruction, often producing a text-only summary that hits the `return`
            # at the text-only-exit points below, terminating the entire agent loop.
            if self.session.turn_count > 0:
                guidance_parts: list[str] = []
                learner_hint = self.learner.get_guidance("__any__", prompt[:300])
                if not learner_hint:
                    learner_hint = ""  # no guidance yet
                reflex_hint = self.reflex.get_improvement_context()
                meta_hint = self.meta.get_self_awareness_context()
                for hint in [learner_hint, reflex_hint, meta_hint]:
                    if hint:
                        guidance_parts.append(hint)

                def _merge_into_last_user(msgs: list[dict[str, Any]], suffix: str) -> None:
                    """Append `suffix` to the last user message content *in place*.
                    Creates a shallow copy of the target dict so the original session
                    messages are not mutated."""
                    for i in range(len(msgs) - 1, -1, -1):
                        if msgs[i].get("role") == "user":
                            msg = dict(msgs[i])
                            existing = (msg.get("content") or "")
                            msg["content"] = existing + "\n\n" + suffix
                            msgs[i] = msg
                            return

                if guidance_parts:
                    guidance_msg = "\n\n".join(guidance_parts)
                    _merge_into_last_user(backend_messages, f"[SYSTEM GUIDANCE]\n{guidance_msg}")

                if self.feedback is not None:
                    fb = self.feedback.get_relevant_feedback("__any__", prompt[:300])
                    if fb:
                        _merge_into_last_user(backend_messages, f"[PAST CORRECTIONS]\n{fb}")

            response_text = ""
            _backend_usage: dict[str, Any] | None = None
            _t_chat = time.time()
            logger.info("[run] calling backend.chat() turn=%d msgs=%d tools=%s",
                        self.session.turn_count, len(backend_messages),
                        bool(backend_tools))
            _chat_first_event = True
            try:
                # Wrap the chat generator with a 120s timeout on the first event,
                # so a hanging API call (wrong key, no network, etc.) surfaces an
                # error rather than freezing the UI indefinitely.
                _chat_gen = self.backend.chat(
                    messages=backend_messages,
                    tools=backend_tools,
                    max_tokens=self.config.max_tokens,
                    enable_caching=self.config.enable_prompt_caching and self.backend.supports_prompt_caching(),
                )
                async for event in self._chat_with_timeout(_chat_gen, timeout=120.0):
                    if _chat_first_event:
                        logger.info("[run] backend.chat() first event after %.1fs turn=%d",
                                    time.time() - _t_chat, self.session.turn_count)
                        _chat_first_event = False
                    if isinstance(event, BackendText):
                        if _in_extra:
                            _extra_text.append(event.text)
                            yield create_text_delta(event.text)
                        elif _tool_seen:
                            _in_extra = True
                            yield create_assistant_boundary()
                            _extra_text.append(event.text)
                            yield create_text_delta(event.text)
                        else:
                            text_parts.append(event.text)
                            yield create_text_delta(event.text)
                        turn_events += 1

                    elif isinstance(event, BackendThinking):
                        if _tool_seen:
                            if not _in_extra:
                                _in_extra = True
                                yield create_assistant_boundary()
                            _extra_thinking.append(event.text)
                            yield create_thinking_delta(event.text)
                        else:
                            thinking_parts.append(event.text)
                            yield create_thinking_delta(event.text)
                        turn_events += 1

                    elif isinstance(event, BackendToolCallDelta):
                        _tool_seen = True
                        if _in_extra:
                            idx = event.index
                            # If this tool index was already being accumulated
                            # in tool_call_buffers before _in_extra, keep
                            # appending there instead of creating a duplicate
                            # in _extra_buffers.
                            if idx in tool_call_buffers:
                                buf = tool_call_buffers[idx]
                                if event.key == "name":
                                    buf["name"] += event.value
                                elif event.key == "arguments":
                                    buf["arguments"] += event.value
                            else:
                                if idx not in _extra_buffers:
                                    _extra_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                                buf = _extra_buffers[idx]
                                if event.key == "name":
                                    buf["name"] += event.value
                                elif event.key == "arguments":
                                    buf["arguments"] += event.value
                        else:
                            idx = event.index
                            if idx not in tool_call_buffers:
                                tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                            buf = tool_call_buffers[idx]
                            if event.key == "name":
                                buf["name"] += event.value
                            elif event.key == "arguments":
                                buf["arguments"] += event.value
                            yield create_tool_call_delta(
                                id=f"call_{self.session.turn_count}_{idx}",
                                key=event.key,
                                value=event.value,
                            )
                        turn_events += 1

                    elif isinstance(event, BackendToolCall):
                        _tool_seen = True
                        if _in_extra:
                            # Check if this tool already exists in tool_call_buffers
                            # (accumulated from deltas before _in_extra) and update
                            # in-place to avoid duplicates.
                            found = False
                            for existing_idx, buf in tool_call_buffers.items():
                                if buf["id"] == event.id or (not buf["id"] and buf["name"] == event.name):
                                    buf["id"] = event.id or buf["id"]
                                    buf["name"] = event.name
                                    buf["arguments"] = event.arguments
                                    found = True
                                    break
                            if not found:
                                for existing_idx, buf in _extra_buffers.items():
                                    if buf["id"] == event.id or (not buf["id"] and buf["name"] == event.name):
                                        buf["id"] = event.id or buf["id"]
                                        buf["name"] = event.name
                                        buf["arguments"] = event.arguments
                                        found = True
                                        break
                            if not found:
                                idx = len(_extra_buffers)
                                _extra_buffers[idx] = {
                                    "id": event.id,
                                    "name": event.name,
                                    "arguments": event.arguments,
                                }
                        else:
                            # Update existing buffer entry (from deltas) if present;
                            # otherwise create a new one.
                            found = False
                            for existing_idx, buf in tool_call_buffers.items():
                                if buf["id"] == event.id or (not buf["id"] and buf["name"] == event.name):
                                    buf["id"] = event.id or buf["id"]
                                    buf["name"] = event.name
                                    buf["arguments"] = event.arguments
                                    found = True
                                    break
                            if not found:
                                idx = len(tool_call_buffers)
                                tool_call_buffers[idx] = {
                                    "id": event.id,
                                    "name": event.name,
                                    "arguments": event.arguments,
                                }

                    elif isinstance(event, BackendFinish):
                        # Capture token usage from the backend
                        if event.usage:
                            _backend_usage = event.usage
                            _last_backend_usage = event.usage

                    elif isinstance(event, BackendError):
                        await self.hook_system.emit_error(
                            Exception(event.error),
                            "backend_error"
                        )
                        await self.hook_system.emit_backend_error(
                            event.error, self.config.backend_type
                        )
                        await self.hook_system.emit_session_end()
                        yield create_finish("error")
                        return

            except Exception as exc:
                logger.error("[run] backend.chat() raised exception after %.1fs turn=%d: %s",
                            time.time() - _t_chat, self.session.turn_count, exc)
                await self.hook_system.emit_error(exc, "backend_chat_exception")
                await self.hook_system.emit_backend_error(str(exc), type(self.backend).__name__ if self.backend else "unknown")
                await self.hook_system.emit_session_end()
                yield create_finish("error", error=str(exc))
                return
            else:
                logger.info("[run] backend.chat() completed in %.1fs turn=%d events=%d",
                            time.time() - _t_chat, self.session.turn_count, turn_events)

            # Post-model hook
            response_text = "".join(text_parts)
            await self.hook_system.emit_post_model_response(
                response_text, len(tool_call_buffers)
            )

            if text_parts and not tool_call_buffers:
                full_text = "".join(text_parts)

                # Merge into the previous assistant that had tool_calls, so that
                # tool-calling turns don't create a second assistant message in
                # the session.  Scan backwards — if we find an assistant with
                # tool_calls before any user message, it belongs to the same
                # logical response from the user's perspective.
                merged = False
                for i in range(len(self.session.messages) - 1, -1, -1):
                    m = self.session.messages[i]
                    if m.get("role") == "user":
                        break
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        existing = m.get("content") or ""
                        m["content"] = (existing + "\n\n" + full_text) if existing else full_text
                        if thinking_parts:
                            existing_r = m.get("reasoning_content", "") or ""
                            m["reasoning_content"] = existing_r + "".join(thinking_parts)
                        if _backend_usage:
                            m["usage"] = _backend_usage
                        # Preserve segment ordering
                        new_segs = []
                        if thinking_parts:
                            new_segs.append({"kind": "thinking", "text": "".join(thinking_parts)})
                        if full_text:
                            new_segs.append({"kind": "text", "text": full_text})
                        if new_segs:
                            existing_segs = m.get("segments", [])
                            m["segments"] = existing_segs + new_segs
                        self.session.mark_messages_dirty()
                        merged = True
                        break

                if not merged:
                    txt_kwargs: dict[str, Any] = {}
                    if thinking_parts:
                        txt_kwargs["reasoning_content"] = "".join(thinking_parts)
                    if _backend_usage:
                        txt_kwargs["usage"] = _backend_usage
                    segs = []
                    if thinking_parts:
                        segs.append({"kind": "thinking", "text": "".join(thinking_parts)})
                    if full_text:
                        segs.append({"kind": "text", "text": full_text})
                    if segs:
                        txt_kwargs["segments"] = segs
                    self.session.add_message("assistant", full_text, **txt_kwargs)

                await self.hook_system.emit_session_end()
                logger.debug("Agent finished (text-only response, %d chars)", len(full_text))
                yield create_finish("stop", usage=_backend_usage)
                return

            if not tool_call_buffers:
                await self.hook_system.emit_session_end()
                logger.debug("Agent finished (empty response, no tool calls)")
                yield create_finish("stop", usage=_backend_usage)
                return

            assistant_content = "".join(text_parts) if text_parts else ""

            # Build OpenAI-format tool_calls from buffers
            assistant_tool_calls: list[dict[str, Any]] = []
            for idx in sorted(tool_call_buffers.keys()):
                tc = tool_call_buffers[idx]
                assistant_tool_calls.append({
                    "id": tc["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })

            msg_kwargs: dict[str, Any] = {}
            if assistant_tool_calls:
                msg_kwargs["tool_calls"] = assistant_tool_calls
            if _backend_usage:
                msg_kwargs["usage"] = _backend_usage
            if thinking_parts:
                msg_kwargs["reasoning_content"] = "".join(thinking_parts)
            # Build segments from streaming order
            segs = []
            if thinking_parts:
                segs.append({"kind": "thinking", "text": "".join(thinking_parts)})
            if assistant_content:
                segs.append({"kind": "text", "text": assistant_content})
            for tc in assistant_tool_calls:
                segs.append({"kind": "tool", "tool_id": tc["id"]})
            if segs:
                msg_kwargs["segments"] = segs
            self.session.add_message("assistant", assistant_content or None, **msg_kwargs)

            # ── Prepare tool calls: parse args, resolve tools, categorize ──
            # NOTE: client-facing events use a stable synthetic id ("call_{turn}_{idx}")
            # so they match the ids already emitted on tool_call_delta events.
            # Internal session/history/telemetry continues to use the real
            # backend id (tc["id"]). Without this split, the UI would create
            # one stub entry from the deltas (call_N) and a second entry from
            # tool_call_start (real id), rendering each tool call twice.
            prepared: list[dict[str, Any]] = []
            for idx in sorted(tool_call_buffers.keys()):
                tc = tool_call_buffers[idx]
                client_id = f"call_{self.session.turn_count}_{idx}"
                yield create_tool_call_start(name=tc["name"], id=client_id)
                turn_events += 1

                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                    err_msg = f"Error: Invalid JSON arguments: {tc['arguments']}"
                    yield create_tool_result(id=client_id, content=err_msg, is_error=True)
                    self.session.add_tool_result(tc["id"], err_msg, is_error=True)
                    turn_events += 1
                    self.telemetry.record_tool_call(
                        tool_name=tc["name"], latency_ms=0, success=False, error_message=err_msg,
                    )
                    yield create_tool_call_end(id=client_id)
                    turn_events += 1
                    prepared.append({"id": tc["id"], "client_id": client_id,
                                     "name": tc["name"], "args": args,
                                     "tool": None, "skip": True, "error": err_msg})
                    continue

                tool = self.tool_registry.get(tc["name"])
                if tool is None:
                    err_msg = f"Error: Unknown tool: {tc['name']}"
                    yield create_tool_result(id=client_id, content=err_msg, is_error=True)
                    self.session.add_tool_result(tc["id"], err_msg, is_error=True)
                    turn_events += 1
                    self.telemetry.record_tool_call(
                        tool_name=tc["name"], latency_ms=0, success=False, error_message=err_msg,
                    )
                    yield create_tool_call_end(id=client_id)
                    turn_events += 1
                    prepared.append({"id": tc["id"], "client_id": client_id,
                                     "name": tc["name"], "args": args,
                                     "tool": None, "skip": True, "error": err_msg})
                    continue

                is_safe = tool.is_concurrency_safe(args)
                prepared.append({
                    "id": tc["id"], "client_id": client_id,
                    "name": tc["name"], "args": args,
                    "tool": tool, "skip": False, "safe": is_safe,
                    "args_summary": _args_summary(args),
                })

            # ── Permission & hooks for all tools (sequential — these may need user input) ──
            if self._cancelled():
                break
            for p in prepared:
                if p["skip"]:
                    continue
                permission = await self.safety.check_tool_permission(p["name"], p["args"])
                if permission.behavior == "ask":
                    permission_reason = _permission_reason(p["name"])
                    await self.hook_system.emit_permission_request(
                        p["name"], permission_reason
                    )
                    yield create_permission_request(
                        tool_name=p["name"],
                        reason=permission_reason,
                    )
                    self._pending_tool_name = p["name"]
                    self._permission_event = asyncio.Event()
                    self._permission_decision = False
                    try:
                        await asyncio.wait_for(self._permission_event.wait(), timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Permission request timed out for tool '{p['name']}' after 120s",
                            extra={"tool_name": p["name"]},
                        )
                    self._permission_event = None
                    await self.hook_system.emit_permission_response(
                        p["name"], self._permission_decision
                    )
                    if not self._permission_decision:
                        err_msg = "Permission denied by user."
                        yield create_tool_result(id=p["client_id"], content=err_msg, is_error=True)
                        self.session.add_tool_result(p["id"], err_msg, is_error=True)
                        turn_events += 1
                        self.telemetry.record_tool_call(
                            tool_name=p["name"], latency_ms=0,
                            success=False, error_message=err_msg,
                        )
                        yield create_tool_call_end(id=p["client_id"])
                        turn_events += 1
                        p["skip"] = True
                        p["error"] = err_msg
                        continue

                pre_hook = await self.hook_system.emit_pre_tool(p["name"], p["args"])
                if pre_hook and pre_hook.get("block"):
                    block_reason = pre_hook.get("block_reason") or f"Blocked by hook: {p['name']}"
                    yield create_tool_progress(id=p["client_id"], tool_name=p["name"], status="blocked")
                    yield create_tool_result(id=p["client_id"], content=block_reason, is_error=True)
                    self.session.add_tool_result(p["id"], block_reason, is_error=True)
                    turn_events += 1
                    self.telemetry.record_tool_call(
                        tool_name=p["name"], latency_ms=0,
                        success=False, error_message=block_reason,
                    )
                    yield create_tool_call_end(id=p["client_id"])
                    turn_events += 1
                    p["skip"] = True
                    p["error"] = block_reason
                    continue
                if pre_hook and pre_hook.get("modified_input"):
                    p["args"] = pre_hook["modified_input"]

                # ── Question tool: block until user answers ──
                if p["name"] == "question":
                    args = p["args"]
                    questions_list: list[dict[str, Any]] = []
                    questions_raw = args.get("questions")
                    if isinstance(questions_raw, str):
                        try: questions_raw = json.loads(questions_raw)
                        except: pass
                    if questions_raw and isinstance(questions_raw, list):
                        for q in questions_raw:
                            if isinstance(q, dict):
                                text = (q.get("question") or "").strip()
                                if text:
                                    item: dict[str, Any] = {"question": text}
                                    if q.get("details"):
                                        item["details"] = str(q["details"]).strip()
                                    if q.get("options") and isinstance(q["options"], list):
                                        item["options"] = [str(o) for o in q["options"]]
                                    questions_list.append(item)
                    q_text = (args.get("question") or "").strip()
                    if q_text:
                        item: dict[str, Any] = {"question": q_text}
                        if args.get("details"):
                            item["details"] = str(args["details"]).strip()
                        if args.get("options") and isinstance(args["options"], list):
                            item["options"] = [str(o) for o in args["options"]]
                        questions_list.append(item)
                    yield create_question_request(
                        tool_call_id=p["client_id"], questions=questions_list,
                    )
                    yield create_tool_progress(id=p["client_id"], tool_name=p["name"], status="running")
                    self._question_event = asyncio.Event()
                    self._question_answers = ""
                    try:
                        await asyncio.wait_for(self._question_event.wait(), timeout=300.0)
                    except asyncio.TimeoutError:
                        self._question_answers = "Error: Question timed out."
                    self._question_event = None
                    result = self._question_answers
                    yield create_tool_result(id=p["client_id"], content=result)
                    self.session.add_tool_result(p["id"], result)
                    turn_events += 1
                    self.telemetry.record_tool_call(
                        tool_name=p["name"], latency_ms=0, success=True,
                    )
                    yield create_tool_call_end(id=p["client_id"])
                    turn_events += 1
                    p["skip"] = True
                    p["result"] = result
                    continue

            # ── Split into safe (concurrent) and unsafe (sequential) groups ──
            safe_tools = [p for p in prepared if not p.get("skip") and p.get("safe")]
            unsafe_tools = [p for p in prepared if not p.get("skip") and not p.get("safe")]

            # ── Execute safe tools in parallel ──
            if safe_tools:
                # Emit progress for all safe tools upfront
                for p in safe_tools:
                    yield create_tool_progress(id=p["client_id"], tool_name=p["name"], status="running")

                async def _execute_safe(p: dict[str, Any]) -> dict[str, Any]:
                    tool_start = time.time()
                    tool_error = False
                    executor = RetryableExecutor(self.recovery_engine)
                    state = await executor.execute(
                        tool_name=p["name"],
                        tool_args=p["args"],
                        execute_fn=lambda args: p["tool"].execute(**args),
                    )
                    if state.succeeded:
                        result = state.final_result
                        sub_agent_messages = None
                        if isinstance(result, dict):
                            sub_agent_messages = result.get("messages")
                            result = str(result.get("content", ""))
                        result = self.safety.validate_tool_output(p["name"], result)
                    else:
                        result = state.final_result
                        sub_agent_messages = None
                        if isinstance(result, dict):
                            sub_agent_messages = result.get("messages")
                            result = str(result.get("content", ""))
                        tool_error = True
                    extra = await self.hook_system.emit_post_tool(p["name"], p["args"], result)
                    if extra:
                        result = result + "\n" + extra
                    p["result"] = result
                    p["sub_agent_messages"] = sub_agent_messages
                    p["is_error"] = tool_error
                    p["recovery_history"] = list(state.recovery_history)
                    p["latency_ms"] = (time.time() - tool_start) * 1000
                    return p

                safe_tasks = [_execute_safe(p) for p in safe_tools]
                completed = await asyncio.gather(*safe_tasks, return_exceptions=True)
                for idx, item in enumerate(completed):
                    if isinstance(item, BaseException):
                        p = safe_tools[idx]
                        err_msg = f"Tool execution crashed: {type(item).__name__}: {item}"
                        yield create_tool_result(id=p["client_id"], content=err_msg, is_error=True)
                        self.session.add_tool_result(p["id"], err_msg, is_error=True)
                        turn_events += 1
                        self.telemetry.record_tool_call(
                            tool_name=p["name"], latency_ms=0.0,
                            success=False, error_message=err_msg,
                        )
                        self.learner.record_error(
                            tool_name=p["name"], error_type="unhandled_exception",
                            context=p["args_summary"], correction="",
                        )
                        yield create_tool_call_end(id=p["client_id"])
                        turn_events += 1
                        continue
                    p = item
                    p["result"] = _apply_result_budget(p["result"], p["tool"])
                    yield create_tool_result(
                        id=p["client_id"],
                        content=p["result"],
                        is_error=p["is_error"],
                        sub_agent_messages=p.get("sub_agent_messages"),
                    )
                    self.session.add_tool_result(
                        p["id"],
                        p["result"],
                        is_error=p["is_error"],
                        sub_agent_messages=p.get("sub_agent_messages"),
                    )
                    turn_events += 1
                    self.telemetry.record_tool_call(
                        tool_name=p["name"], latency_ms=p["latency_ms"],
                        success=not p["is_error"],
                        error_message=p["result"] if p["is_error"] else "",
                    )
                    if p["is_error"]:
                        self.learner.record_error(
                            tool_name=p["name"], error_type="execution_error",
                            context=p["args_summary"], correction="",
                        )
                        if self.feedback is not None:
                            self.feedback.record_correction(
                                tool_name=p["name"], error_type="execution_error",
                                error_context=p["args_summary"],
                                user_correction=p["result"][:400],
                            )
                    else:
                        self.learner.record_success(
                            tool_name=p["name"], intent=prompt[:300], params=p["args"],
                            outcome=p["result"][:500], latency_ms=p["latency_ms"],
                        )
                        if p.get("recovery_history"):
                            correction = ErrorRecoveryEngine.infer_correction_from_history(p["recovery_history"], p["name"])
                            self.learner.record_correction(
                                tool_name=p["name"],
                                error_context=p["args_summary"],
                                correction=correction,
                            )
                    self.optimizer.record_outcome(
                        tool_name=p["name"], params=p["args"],
                        success=not p["is_error"], latency_ms=p["latency_ms"],
                    )
                    yield create_tool_call_end(id=p["client_id"])
                    turn_events += 1
                    if not p["is_error"]:
                        fp = _extract_file_path(p["name"], p["result"])
                        if fp:
                            if p["name"] == "apply_patch":
                                for ap_path in _extract_apply_patch_paths(p["result"]):
                                    entry = self.session.add_artifact(ap_path, p["name"], diff_text="")
                                    yield Artifact(artifact=entry)
                            else:
                                diff_text = _extract_diff_text(p["name"], p["result"])
                                entry = self.session.add_artifact(fp, p["name"], diff_text=diff_text)
                                yield Artifact(artifact=entry)
                        plan_items = _ensure_plan_items(p["name"], p["args"])
                        if plan_items:
                            yield PlanUpdate(plan_items=plan_items)
                            self.session.plan_items = plan_items

            # ── Execute unsafe tools sequentially ──
            for p in unsafe_tools:
                tool_start = time.time()
                yield create_tool_progress(id=p["client_id"], tool_name=p["name"], status="running")

                tool_error = False
                sub_agent_messages = None
                sub_agent_session_id = None
                try:
                    if p["name"] == "agent":
                        progress_queue: asyncio.Queue[list[dict[str, Any]] | None] = asyncio.Queue()

                        async def _sub_agent_progress(messages: list[dict[str, Any]]) -> None:
                            nonlocal sub_agent_messages
                            sub_agent_messages = messages
                            await progress_queue.put(messages)

                        agent_args = dict(p["args"])
                        agent_args["progress_callback"] = _sub_agent_progress

                        async def _run_agent_tool() -> Any:
                            try:
                                return await p["tool"].execute(**agent_args)
                            finally:
                                await progress_queue.put(None)

                        agent_task = asyncio.create_task(_run_agent_tool())
                        while True:
                            live_messages = await progress_queue.get()
                            if live_messages is None:
                                break
                            yield create_tool_progress(
                                id=p["client_id"],
                                tool_name=p["name"],
                                status="running",
                                sub_agent_messages=live_messages,
                            )
                        result_obj = await agent_task
                        sub_agent_session_id = None
                        sub_agent_messages = None
                        if isinstance(result_obj, dict):
                            sub_agent_messages = result_obj.get("messages")
                            sub_agent_session_id = result_obj.get("session_id")
                            if sub_agent_messages:
                                yield create_tool_progress(
                                    id=p["client_id"],
                                    tool_name=p["name"],
                                    status="running",
                                    sub_agent_messages=sub_agent_messages,
                                )
                            result = str(result_obj.get("content", ""))
                        else:
                            result = str(result_obj)
                        result = self.safety.validate_tool_output(p["name"], result)
                    else:
                        executor = RetryableExecutor(self.recovery_engine)
                        state = await executor.execute(
                            tool_name=p["name"],
                            tool_args=p["args"],
                            execute_fn=lambda args: p["tool"].execute(**args),
                        )
                        if state.succeeded:
                            result = state.final_result
                            if isinstance(result, dict):
                                sub_agent_messages = result.get("messages")
                                result = str(result.get("content", ""))
                            result = self.safety.validate_tool_output(p["name"], result)
                            if state.recovery_history:
                                correction = ErrorRecoveryEngine.infer_correction(state)
                                self.learner.record_correction(
                                    tool_name=p["name"],
                                    error_context=p["args_summary"],
                                    correction=correction,
                                )
                        else:
                            result = state.final_result
                            if isinstance(result, dict):
                                sub_agent_messages = result.get("messages")
                                result = str(result.get("content", ""))
                            tool_error = True

                    extra = await self.hook_system.emit_post_tool(p["name"], p["args"], result)
                    if extra:
                        result = result + "\n" + extra
                except Exception as exc:
                    result = f"Tool execution crashed: {type(exc).__name__}: {exc}"
                    tool_error = True

                result = _apply_result_budget(result, p["tool"])
                yield create_tool_result(
                    id=p["client_id"],
                    content=result,
                    is_error=tool_error,
                    sub_agent_messages=sub_agent_messages,
                    sub_agent_session_id=sub_agent_session_id,
                )
                self.session.add_tool_result(
                    p["id"],
                    result,
                    is_error=tool_error,
                    sub_agent_messages=sub_agent_messages,
                    sub_agent_session_id=sub_agent_session_id,
                )
                turn_events += 1

                tool_latency = (time.time() - tool_start) * 1000
                self.telemetry.record_tool_call(
                    tool_name=p["name"], latency_ms=tool_latency,
                    success=not tool_error,
                    error_message=result if tool_error else "",
                )
                if tool_error:
                    self.learner.record_error(
                        tool_name=p["name"], error_type="execution_error",
                        context=p["args_summary"], correction="",
                    )
                    if self.feedback is not None:
                        self.feedback.record_correction(
                            tool_name=p["name"], error_type="execution_error",
                            error_context=p["args_summary"],
                            user_correction=result[:400],
                        )
                else:
                    self.learner.record_success(
                        tool_name=p["name"], intent=prompt[:300], params=p["args"],
                        outcome=result[:500], latency_ms=tool_latency,
                    )
                self.optimizer.record_outcome(
                    tool_name=p["name"], params=p["args"],
                    success=not tool_error, latency_ms=tool_latency,
                )
                yield create_tool_call_end(id=p["client_id"])
                turn_events += 1
                if not tool_error:
                    fp = _extract_file_path(p["name"], result)
                    if fp:
                        if p["name"] == "apply_patch":
                            for ap_path in _extract_apply_patch_paths(result):
                                entry = self.session.add_artifact(ap_path, p["name"], diff_text="")
                                yield Artifact(artifact=entry)
                        else:
                            diff_text = _extract_diff_text(p["name"], result)
                            entry = self.session.add_artifact(fp, p["name"], diff_text=diff_text)
                            yield Artifact(artifact=entry)
                    plan_items = _ensure_plan_items(p["name"], p["args"])
                    if plan_items:
                        yield PlanUpdate(plan_items=plan_items)

            # ── Intra-turn split: merge post-tool content into existing assistant message ──
            # When the model produces thinking/text → tool_calls → more thinking/text
            # within the same backend.chat() call, the post-tool content is
            # buffered in _extra_* variables. Merge it into the existing assistant
            # message so the session doesn't get split into two messages for a
            # single model response.
            if _in_extra and (_extra_text or _extra_thinking or _extra_buffers):
                for i in range(len(self.session.messages) - 1, -1, -1):
                    if self.session.messages[i].get("role") == "assistant":
                        msg = self.session.messages[i]
                        if _extra_text:
                            existing = msg.get("content") or ""
                            extra = "".join(_extra_text)
                            msg["content"] = (existing + "\n\n" + extra) if existing else extra
                        if _extra_buffers:
                            extra_tc = []
                            for idx in sorted(_extra_buffers.keys()):
                                tc = _extra_buffers[idx]
                                extra_tc.append({
                                    "id": tc["id"] or f"call_{idx}",
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                })
                            existing_tc = msg.get("tool_calls", [])
                            msg["tool_calls"] = existing_tc + extra_tc
                        if _extra_thinking:
                            existing_r = msg.get("reasoning_content", "") or ""
                            extra_r = "".join(_extra_thinking)
                            msg["reasoning_content"] = existing_r + extra_r
                        # Preserve segment ordering for intra-turn extra content
                        extra_segs = []
                        if _extra_thinking:
                            extra_segs.append({"kind": "thinking", "text": "".join(_extra_thinking)})
                        if _extra_text:
                            extra_segs.append({"kind": "text", "text": "".join(_extra_text)})
                        for etc in (extra_tc if _extra_buffers else []):
                            extra_segs.append({"kind": "tool", "tool_id": etc["id"]})
                        if extra_segs:
                            existing_segs = msg.get("segments", [])
                            msg["segments"] = existing_segs + extra_segs
                        self.session.mark_messages_dirty()
                        break

                # Prepare secondary tool calls
                extra_prepared: list[dict[str, Any]] = []
                for idx in sorted(_extra_buffers.keys()):
                    tc = _extra_buffers[idx]
                    client_id = f"call_{self.session.turn_count}_extra_{idx}"
                    yield create_tool_call_start(name=tc["name"], id=client_id)
                    turn_events += 1

                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                        err_msg = f"Error: Invalid JSON arguments: {tc['arguments']}"
                        yield create_tool_result(id=client_id, content=err_msg, is_error=True)
                        self.session.add_tool_result(tc["id"], err_msg, is_error=True)
                        turn_events += 1
                        yield create_tool_call_end(id=client_id)
                        turn_events += 1
                        continue

                    tool = self.tool_registry.get(tc["name"])
                    if tool is None:
                        err_msg = f"Error: Unknown tool: {tc['name']}"
                        yield create_tool_result(id=client_id, content=err_msg, is_error=True)
                        self.session.add_tool_result(tc["id"], err_msg, is_error=True)
                        turn_events += 1
                        yield create_tool_call_end(id=client_id)
                        turn_events += 1
                        continue

                    extra_prepared.append({
                        "id": tc["id"], "client_id": client_id,
                        "name": tc["name"], "args": args,
                        "tool": tool,
                        "args_summary": _args_summary(args),
                    })

                # Permission & hooks for secondary tools
                if not self._cancelled():
                    for p in extra_prepared:
                        permission = await self.safety.check_tool_permission(p["name"], p["args"])
                        if permission.behavior == "ask":
                            permission_reason = _permission_reason(p["name"])
                            await self.hook_system.emit_permission_request(
                                p["name"], permission_reason
                            )
                            yield create_permission_request(
                                tool_name=p["name"],
                                reason=permission_reason,
                            )
                            self._pending_tool_name = p["name"]
                            self._permission_event = asyncio.Event()
                            self._permission_decision = False
                            try:
                                await asyncio.wait_for(self._permission_event.wait(), timeout=120.0)
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"Permission request timed out for tool '{p['name']}' after 120s",
                                    extra={"tool_name": p["name"]},
                                )
                            self._permission_event = None
                            await self.hook_system.emit_permission_response(
                                p["name"], self._permission_decision
                            )
                            if not self._permission_decision:
                                err_msg = "Permission denied by user."
                                yield create_tool_result(id=p["client_id"], content=err_msg, is_error=True)
                                self.session.add_tool_result(p["id"], err_msg, is_error=True)
                                turn_events += 1
                                self.telemetry.record_tool_call(
                                    tool_name=p["name"], latency_ms=0,
                                    success=False, error_message=err_msg,
                                )
                                yield create_tool_call_end(id=p["client_id"])
                                turn_events += 1
                                continue

                        pre_hook = await self.hook_system.emit_pre_tool(p["name"], p["args"])
                        if pre_hook and pre_hook.get("block"):
                            block_reason = pre_hook.get("block_reason") or f"Blocked by hook: {p['name']}"
                            yield create_tool_result(id=p["client_id"], content=block_reason, is_error=True)
                            self.session.add_tool_result(p["id"], block_reason, is_error=True)
                            turn_events += 1
                            self.telemetry.record_tool_call(
                                tool_name=p["name"], latency_ms=0,
                                success=False, error_message=block_reason,
                            )
                            yield create_tool_call_end(id=p["client_id"])
                            turn_events += 1
                            continue
                        if pre_hook and pre_hook.get("modified_input"):
                            p["args"] = pre_hook["modified_input"]

                # Execute secondary tools sequentially
                for p in extra_prepared:
                    yield create_tool_progress(id=p["client_id"], tool_name=p["name"], status="running")
                    tool_start = time.time()
                    tool_error = False
                    try:
                        executor = RetryableExecutor(self.recovery_engine)
                        state = await executor.execute(
                            tool_name=p["name"],
                            tool_args=p["args"],
                            execute_fn=lambda args: p["tool"].execute(**args),
                        )
                        if state.succeeded:
                            result = state.final_result
                            result = self.safety.validate_tool_output(p["name"], result)
                            if state.recovery_history:
                                correction = ErrorRecoveryEngine.infer_correction(state)
                                self.learner.record_correction(
                                    tool_name=p["name"],
                                    error_context=p["args_summary"],
                                    correction=correction,
                                )
                        else:
                            result = state.final_result
                            tool_error = True
                        extra = await self.hook_system.emit_post_tool(p["name"], p["args"], result)
                        if extra:
                            result = result + "\n" + extra
                    except Exception as exc:
                        result = f"Tool execution crashed: {type(exc).__name__}: {exc}"
                        tool_error = True
                    result = _apply_result_budget(result, p["tool"])
                    yield create_tool_result(id=p["client_id"], content=result, is_error=tool_error)
                    self.session.add_tool_result(p["id"], result, is_error=tool_error)
                    turn_events += 1
                    tool_latency = (time.time() - tool_start) * 1000
                    self.telemetry.record_tool_call(
                        tool_name=p["name"], latency_ms=tool_latency,
                        success=not tool_error, error_message=result if tool_error else "",
                    )
                    if tool_error:
                        self.learner.record_error(
                            tool_name=p["name"], error_type="execution_error",
                            context=p["args_summary"], correction="",
                        )
                        if self.feedback is not None:
                            self.feedback.record_correction(
                                tool_name=p["name"], error_type="execution_error",
                                error_context=p["args_summary"],
                                user_correction=result[:400],
                            )
                    else:
                        self.learner.record_success(
                            tool_name=p["name"], intent=prompt[:300], params=p["args"],
                            outcome=result[:500], latency_ms=tool_latency,
                        )
                    self.optimizer.record_outcome(
                        tool_name=p["name"], params=p["args"],
                        success=not tool_error, latency_ms=tool_latency,
                    )
                    yield create_tool_call_end(id=p["client_id"])
                    turn_events += 1
                    if not tool_error:
                        fp = _extract_file_path(p["name"], result)
                        if fp:
                            if p["name"] == "apply_patch":
                                for ap_path in _extract_apply_patch_paths(result):
                                    entry = self.session.add_artifact(ap_path, p["name"], diff_text="")
                                    yield Artifact(artifact=entry)
                            else:
                                diff_text = _extract_diff_text(p["name"], result)
                                entry = self.session.add_artifact(fp, p["name"], diff_text=diff_text)
                                yield Artifact(artifact=entry)
                        plan_items = _ensure_plan_items(p["name"], p["args"])
                        if plan_items:
                            yield PlanUpdate(plan_items=plan_items)
                            self.session.plan_items = plan_items

            # Don't yield an assistant_boundary here — doing so makes the frontend
            # split the response into separate bubbles after every tool-calling
            # turn.  All model output within a single user turn (including post-tool
            # follow-ups) stays in one assistant message on both session and UI.

            self.session.turn_count += 1
            turn_latency = (time.time() - turn_start) * 1000
            self.telemetry.record_turn(
                turn_number=self.session.turn_count,
                event_count=turn_events,
                latency_ms=turn_latency,
                token_usage=_backend_usage or {},
                model=self.config.model,
            )

            # Evolution: reflex + meta-cognition
            tool_outcomes: list[dict[str, Any]] = [
                {"tool_name": tc["name"], "is_error": False}
                for tc in tool_call_buffers.values()
            ]
            reflection = self.reflex.reflect(
                turn_number=self.session.turn_count,
                tool_results=tool_outcomes,
                turn_latency_ms=turn_latency,
            )
            self.meta.assess_turn(
                prompt=prompt,
                tool_results=tool_outcomes,
            )

            await self.hook_system.emit_turn_end(self.session.turn_count)
            self.rollback.commit(self.session, f"turn_{self.session.turn_count}")

            # ── Proactive micro-compaction ──────────────────────────────
            # After each turn, strip old tool results to keep context lean.
            # This delays full compaction and saves API costs on summaries.
            if self.session.turn_count > _MICROCOMPACT_THRESHOLD:
                ctx = self.session.get_context_messages()
                trimmed = _microcompact_old_results(ctx, keep_recent_turns=_MICROCOMPACT_THRESHOLD)
                if len(trimmed) != len(ctx) or any(
                    a.get("content") != b.get("content")
                    for a, b in zip(trimmed, ctx)
                    if a.get("role") == "tool"
                ):
                    self.session.replace_branch_messages(self.session.active_branch_id, trimmed)
                    logger.info("[compact] micro-compacted %d old tool results turn=%d",
                                sum(1 for m in trimmed if m.get("role") == "tool"),
                                self.session.turn_count)

        await self.hook_system.emit_session_end()
        yield create_finish(
            "cancelled" if self._cancelled() else "max_tokens",
            usage=_last_backend_usage,
        )

    async def _run_sub_agent(self, prompt: str,
                              system_prompt: str = "", max_turns: int = 10,
                              model: str = "", api_key: str = "",
                              base_url: str = "",
                              progress_callback: Any = None) -> dict[str, Any]:
        logger.info("[sub_agent] _run_sub_agent | prompt_len=%d | sys_prompt_len=%d",
                    len(prompt), len(system_prompt))
        logger.info("[sub_agent] prompt_text=%.300s", prompt)

        # Create a full EncreAgent (same as SessionManager.create_session / normal user flow).
        # Lazy-import to avoid circular dependency (agent.py imports EncreLoop from this module).
        from encre.agent import EncreAgent
        from encre.config import EncreConfig
        from encre.tools.registry import ToolRegistry

        sub_config = EncreConfig(
            model=model or self.config.model,
            api_key=api_key or self.config.api_key,
            base_url=base_url or self.config.base_url,
            max_tokens=self.config.max_tokens,
            max_turns=max_turns,
            permission_mode="bypass",
            backend_type=self.config.backend_type,
            backend_kwargs=self.config.backend_kwargs,
        )
        # Clone tool registry (same as session_manager._clone_tool_registry)
        tool_registry = ToolRegistry()
        tool_registry._tools = dict(self.tool_registry._tools)

        sub_agent = EncreAgent(
            config=sub_config,
            tool_registry=tool_registry,
            memory_system=self.memory_system,
            profile_system=self.profile_system,
            soul_system=self.soul_system,
            skill_registry=self.skill_registry,
            hook_system=self.hook_system,
            safety=self.safety,
        )
        sub_agent.loop.sub_agent_depth = self.sub_agent_depth + 1
        # Add the prompt as a user message, exactly like ws.py does for normal input
        sub_agent.add_message("user", prompt)

        # Give the sub-agent a proper session ID
        import uuid
        sub_agent.session.id = sub_agent.session.id or str(uuid.uuid4())
        saved_session_id = sub_agent.session.id

        def _save():
            try:
                from encre.config import get_data_dir
                _dir = get_data_dir() / "sessions" / saved_session_id
                _dir.mkdir(parents=True, exist_ok=True)
                sub_agent.session.save_to_dir(str(_dir))
            except Exception:
                logger.warning("[sub_agent] failed to persist session", exc_info=True)

        result_parts: list[str] = []
        text_buffer = ""
        # "Draft" assistant state tracked from the streaming events. Once the
        # sub-agent's own loop commits a new assistant message into
        # sub_agent.session.messages, we drop the matching draft and rely on
        # the committed record. This guarantees the snapshot never contains
        # duplicate / out-of-order assistant turns.
        draft_content: list[str] = []
        draft_reasoning: list[str] = []
        draft_tool_calls: list[dict[str, Any]] = []
        draft_tool_id_to_idx: dict[str, int] = {}
        draft_segments: list[dict[str, Any]] = []
        last_seen_msg_count = 0
        last_seen_assistant_id: str | None = None

        def _has_uncommitted_draft() -> bool:
            return bool(
                draft_content
                or draft_reasoning
                or draft_tool_calls
                or draft_segments
            )

        def _reset_draft() -> None:
            draft_content.clear()
            draft_reasoning.clear()
            draft_tool_calls.clear()
            draft_tool_id_to_idx.clear()
            draft_segments.clear()

        def _draft_as_message() -> dict[str, Any]:
            return {
                "role": "assistant",
                "content": "".join(draft_content),
                "reasoning_content": "".join(draft_reasoning),
                "tool_calls": [dict(tc) for tc in draft_tool_calls],
                "segments": [dict(s) for s in draft_segments],
                "created_at": time.time(),
            }

        def _sync_draft_with_session() -> None:
            """Drop the draft when the sub-agent's loop has committed a
            matching (or superseding) assistant message into session.messages.
            """
            nonlocal last_seen_msg_count, last_seen_assistant_id
            msgs = sub_agent.session.messages
            current_count = len(msgs)
            current_assistant_id: str | None = None
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    current_assistant_id = str(m.get("id") or "")
                    break
            # Reset the draft whenever a new assistant message has been
            # committed since the last emit. The agent's loop appends the
            # assistant message to session.messages only AFTER the streaming
            # for that turn has finished, so a new id means "the previous
            # turn's draft is now committed; start fresh".
            if (
                current_assistant_id is not None
                and current_assistant_id != last_seen_assistant_id
            ):
                last_seen_msg_count = current_count
                last_seen_assistant_id = current_assistant_id
                _reset_draft()
            elif current_count != last_seen_msg_count:
                last_seen_msg_count = current_count

        def _build_snapshot() -> list[dict[str, Any]]:
            """Build the messages snapshot for progress callbacks.

            Prefers sub_agent.session.messages (canonical, committed history
            with full tool_call / tool_result structure). If a streaming turn
            is still in progress and has not yet been committed, appends the
            draft so the frontend sees live tokens.
            """
            _sync_draft_with_session()
            snapshot = [dict(m) for m in sub_agent.session.messages]
            if _has_uncommitted_draft():
                snapshot.append(_draft_as_message())
            return snapshot

        async def _emit_live() -> None:
            if progress_callback is not None:
                await progress_callback(_build_snapshot())

        def _flush_text_buffer() -> None:
            nonlocal text_buffer
            text = text_buffer.strip()
            if text:
                result_parts.append(f"### Assistant\n{text}\n")
            text_buffer = ""

        # Run exactly like ws.py does: agent.run(prompt, system_prompt=None) → full system prompt build
        async for event in sub_agent.run(prompt=prompt, system_prompt=system_prompt or None):
            if isinstance(event, TextDelta):
                text_buffer += event.text
                draft_content.append(event.text)
                if draft_segments and draft_segments[-1].get("kind") == "text":
                    draft_segments[-1]["text"] = (
                        str(draft_segments[-1].get("text") or "") + event.text
                    )
                else:
                    draft_segments.append({"kind": "text", "text": event.text})
                await _emit_live()
            elif isinstance(event, ThinkingDelta):
                _flush_text_buffer()
                thought = event.text.strip()
                if thought:
                    result_parts.append(f"### Thought\n{thought}\n")
                    draft_reasoning.append(event.text)
                    if draft_segments and draft_segments[-1].get("kind") == "thinking":
                        draft_segments[-1]["text"] = (
                            str(draft_segments[-1].get("text") or "") + event.text
                        )
                    else:
                        draft_segments.append({"kind": "thinking", "text": event.text})
                    await _emit_live()
            elif isinstance(event, ToolCallStart):
                _flush_text_buffer()
                result_parts.append(f"### Tool Start\n- id: `{event.id}`\n- name: `{event.name}`\n")
                tc_dict = {
                    "id": event.id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": "{}"},
                }
                draft_tool_calls.append(tc_dict)
                draft_tool_id_to_idx[event.id] = len(draft_tool_calls) - 1
                draft_segments.append({"kind": "tool", "tool_id": event.id})
                await _emit_live()
            elif isinstance(event, ToolProgress):
                _flush_text_buffer()
                result_parts.append(f"### Tool Progress\n- id: `{event.id}`\n- name: `{event.tool_name}`\n- status: `{event.status}`\n")
                await _emit_live()
            elif isinstance(event, ToolCallEnd):
                _flush_text_buffer()
                result_parts.append(f"### Tool End\n- id: `{event.id}`\n")
                await _emit_live()
            elif isinstance(event, ToolResult):
                _flush_text_buffer()
                content = event.content.strip()
                if len(content) > 2000:
                    content = f"{content[:2000]}\n... (truncated)"
                result_parts.append(
                    f"### Tool Result\n- id: `{event.id}`\n- error: `{'yes' if event.is_error else 'no'}`\n\n```text\n{content}\n```\n"
                )
                # Tool results are already persisted into sub_agent.session.messages
                # by the sub-agent's own loop. The snapshot builder picks them up
                # directly so we do NOT maintain a separate live_messages list.
                await _emit_live()
            elif isinstance(event, Finish):
                _flush_text_buffer()
                await _emit_live()
                if event.reason == "error":
                    _save()
                    return {
                        "content": "Error: Sub-agent failed",
                        "messages": sub_agent.session.messages,
                        "session_id": saved_session_id,
                    }

        _save()
        # Extract the sub-agent's final response: prefer text content, fall back
        # to reasoning content, then to "Tool calls executed" if only tools ran.
        final_text = ""
        for msg in reversed(sub_agent.session.messages):
            if msg.get("role") != "assistant":
                continue
            txt = str(msg.get("content") or "")
            if txt.strip():
                final_text = txt
                break
            # No text content — check reasoning
            rsn = str(msg.get("reasoning_content") or "")
            if rsn.strip():
                final_text = f"[Thinking]\n{rsn}"
                break
            # No text or reasoning — check tool calls
            tcs = msg.get("tool_calls") or []
            if tcs:
                names = [tc.get("function", {}).get("name", "?") for tc in tcs]
                final_text = f"[Tool calls executed: {', '.join(names)}]"
                break
        logger.info("[sub_agent] done session_id=%s final_len=%d msgs=%d",
                     saved_session_id, len(final_text), len(sub_agent.session.messages))
        logger.info("[sub_agent] final_text=%.200s", final_text)
        return {
            "content": final_text or "No output from sub-agent",
            "messages": sub_agent.session.messages,
            "session_id": saved_session_id,
        }
