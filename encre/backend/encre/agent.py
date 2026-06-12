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
from typing import Any, AsyncGenerator

from encre.config import EncreConfig
from encre.evolution.config import EvolutionConfig
from encre.hooks.system import EncreHookSystem
from encre.loop import EncreLoop
from encre.memdir.system import EncreMemorySystem
from encre.profile.system import EncreProfileSystem
from encre.soul.system import EncreSoulSystem
from encre.plugins.registry import PluginRegistry
from encre.feedback.learner import EncreFeedbackLearner
from encre.codebase.indexer import EncreCodeIndex
from encre.recovery import ErrorRecoveryEngine
from encre.safety import EncreSafetyEngine
from encre.session import EncreSession
from encre.skills.registry import EncreSkillRegistry
from encre.skills.bundled import create_bundled_skills
from encre.telemetry import EncreTelemetry
from encre.tools.defaults import register_default_tools
from encre.tools.registry import ToolRegistry
from encre.utils.types import AgentEvent, ToolCallStart, ToolProgress

_BUNDLED_SKILLS_LOADED = False
_USER_SKILLS_LOADED_DIRS: set[str] = set()
_PLUGINS_DISCOVERED = False
_SHARED_SKILL_REGISTRY: EncreSkillRegistry | None = None
_SHARED_PLUGIN_REGISTRY: PluginRegistry | None = None


def _ensure_bundled_skills_loaded(registry: EncreSkillRegistry) -> None:
    global _BUNDLED_SKILLS_LOADED
    if _BUNDLED_SKILLS_LOADED and registry.list_all():
        return
    create_bundled_skills(registry)
    _BUNDLED_SKILLS_LOADED = True


def _ensure_user_skills_loaded(registry: EncreSkillRegistry, skills_dir: str) -> None:
    global _USER_SKILLS_LOADED_DIRS
    if skills_dir in _USER_SKILLS_LOADED_DIRS:
        return
    from encre.skills.types import SkillSource
    registry.load_from_dir(skills_dir, source=SkillSource.USER)
    _USER_SKILLS_LOADED_DIRS.add(skills_dir)


def _get_shared_skill_registry() -> EncreSkillRegistry:
    global _SHARED_SKILL_REGISTRY
    if _SHARED_SKILL_REGISTRY is None:
        registry = EncreSkillRegistry()
        _ensure_bundled_skills_loaded(registry)
        from encre.config import get_data_dir
        _ensure_user_skills_loaded(registry, str(get_data_dir() / "skills"))
        _SHARED_SKILL_REGISTRY = registry
    return _SHARED_SKILL_REGISTRY


def _get_shared_plugin_registry() -> PluginRegistry:
    global _SHARED_PLUGIN_REGISTRY, _PLUGINS_DISCOVERED
    if _SHARED_PLUGIN_REGISTRY is None:
        _SHARED_PLUGIN_REGISTRY = PluginRegistry()
    if not _PLUGINS_DISCOVERED:
        _SHARED_PLUGIN_REGISTRY.discover_all()
        _PLUGINS_DISCOVERED = True
    return _SHARED_PLUGIN_REGISTRY


class EncreAgent:
    def __init__(
        self,
        config: EncreConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        hook_system: EncreHookSystem | None = None,
        memory_system: EncreMemorySystem | None = None,
        profile_system: EncreProfileSystem | None = None,
        soul_system: EncreSoulSystem | None = None,
        skill_registry: EncreSkillRegistry | None = None,
        safety: EncreSafetyEngine | None = None,
        recovery: ErrorRecoveryEngine | None = None,
        plugin_registry: PluginRegistry | None = None,
        feedback: EncreFeedbackLearner | None = None,
        code_index: EncreCodeIndex | None = None,
    ) -> None:
        self.config = config or EncreConfig()
        self.tool_registry = tool_registry or ToolRegistry()
        if not self.tool_registry.list_tools():
            register_default_tools(self.tool_registry)
        self.hook_system = hook_system or EncreHookSystem()
        if memory_system is not None:
            self.memory_system = memory_system
        else:
            from encre.config import get_data_dir
            self.memory_system = EncreMemorySystem(str(get_data_dir() / "memory"))
        if profile_system is not None:
            self.profile_system = profile_system
        else:
            mem_dir = str(get_data_dir() / "memory")
            self.profile_system = EncreProfileSystem(mem_dir)
            self.profile_system.load()
        if soul_system is not None:
            self.soul_system = soul_system
        else:
            self.soul_system = EncreSoulSystem()
            self.soul_system.ensure_defaults()
            self.soul_system.load()
        self.safety = safety or EncreSafetyEngine(self.config)
        self.recovery = recovery or ErrorRecoveryEngine()
        self.feedback = feedback or EncreFeedbackLearner()
        self.code_index = code_index
        self.session = EncreSession(self.config)
        # Inject built-in sub-agents (hidden from settings UI)
        from encre.agents.builtin import get_builtin_sub_agents
        existing_names = {sa.name for sa in self.config.sub_agents}
        for builtin in get_builtin_sub_agents():
            if builtin.name not in existing_names:
                self.config.sub_agents.append(builtin)
        self.telemetry = EncreTelemetry(enabled=self.config.telemetry_enabled)
        self.evolution = EvolutionConfig.create_default()
        self.plugin_registry = plugin_registry or _get_shared_plugin_registry()
        self.skill_registry = skill_registry or _get_shared_skill_registry()
        self.loop = EncreLoop(
            self.config, self.session, self.tool_registry,
            self.hook_system, self.safety,
            self.memory_system,
            profile_system=self.profile_system,
            soul_system=self.soul_system,
            skill_registry=self.skill_registry,
            telemetry=self.telemetry,
            evolution=self.evolution,
            recovery=self.recovery,
            feedback=self.feedback,
            code_index=self.code_index,
        )
        self._wire_tools()
        self._load_plugins()
        # MCP lifecycle (lazy init on first run)
        self._mcp_tools: list[Any] = []
        self._mcp_initialized = False

    async def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        custom_instructions: str = "",
    ) -> AsyncGenerator[AgentEvent, None]:
        # Lazy-init MCP connections on first run
        if not self._mcp_initialized:
            await self._init_mcp()
            self._mcp_initialized = True
        tool_names: list[str] = []
        async for event in self.loop.run(prompt, system_prompt, custom_instructions=custom_instructions):
            if isinstance(event, ToolCallStart):
                tool_names.append(event.name)
            elif isinstance(event, ToolProgress):
                tool_names.append(event.tool_name)
            yield event
        # Trigger async profile inference after session completes
        if hasattr(self, "profile_system") and self.profile_system is not None:
            try:
                import asyncio
                asyncio.create_task(self.profile_system.infer_from_session(
                    self.session.messages, self.loop.backend
                ))
            except Exception:
                pass

        # Feed tool usage pattern to learning engine for skill crystallization
        if tool_names and hasattr(self, "_learning_engine") and self._learning_engine is not None:
            try:
                await self._learning_engine.analyze_run(tool_names, prompt)
            except Exception:
                pass

    async def run_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        for tool in tools:
            self.tool_registry.register(tool)
        async for event in self.run(prompt, system_prompt):
            yield event

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.session.add_message(role, content, **kwargs)

    def rebuild_backend(self) -> None:
        """Recreate the loop's backend from current config.

        Call this after changing config.backend_type, config.api_key,
        config.base_url, or config.model so the backend instance matches
        the updated settings.
        """
        from encre.backend import create_backend as _cb
        self.loop.backend = _cb(
            self.config.backend_type,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            **self.config.backend_kwargs,
        )

    def load_plugins(self, discover: bool = True) -> int:
        """Load and activate all plugins. Returns count of activated plugins.

        If discover=True, scans entry points and plugin directories first.
        """
        global _PLUGINS_DISCOVERED
        if discover:
            if self.plugin_registry is _SHARED_PLUGIN_REGISTRY:
                if not _PLUGINS_DISCOVERED:
                    self.plugin_registry.discover_all()
                    _PLUGINS_DISCOVERED = True
            elif not _PLUGINS_DISCOVERED:
                self.plugin_registry.discover_all()
                _PLUGINS_DISCOVERED = True
        self.plugin_registry.activate_all()

        # Inject plugin tools
        for tool in self.plugin_registry.get_all_tools():
            self.tool_registry.register(tool)

        # Inject plugin skills
        for skill in self.plugin_registry.get_all_skills():
            self.skill_registry.register(skill)

        # Inject plugin hooks
        for event_type, handlers in self.plugin_registry.get_all_hooks().items():
            for handler in handlers:
                self.hook_system.register_handler(event_type, handler)

        # Register plugin backends
        for name, backend_cls in self.plugin_registry.get_all_backends().items():
            from encre.backend import create_backend as _cb
            # Plugin backends are registered by name for later use

        return self.plugin_registry.active_count

    def _load_plugins(self) -> None:
        """Auto-load plugins during agent initialization."""
        if getattr(self, "_plugins_loaded", False):
            return
        self.load_plugins(discover=True)
        self._plugins_loaded = True

    def _wire_tools(self) -> None:
        """Wire parent loop reference to tools that need it."""
        from encre.tools.builtin.agent import set_parent_loop as _agent_set_parent
        from encre.tools.builtin.find_tool import set_parent_loop as _find_set_parent
        _agent_set_parent(self.loop)
        _find_set_parent(self.loop)

    # ------------------------------------------------------------------
    # MCP lifecycle
    # ------------------------------------------------------------------

    async def _init_mcp(self) -> None:
        """Connect configured MCP servers and register their tools."""
        from encre.tools.mcp import EncreMCPTool

        for server in self.config.mcp_servers:
            # Support both old (enabled) and new (disabled) field names
            enabled_old = server.get("enabled", True)
            disabled_new = server.get("disabled", False)
            if isinstance(enabled_old, bool):
                if not enabled_old:
                    continue
            elif disabled_new:
                continue

            # Support both old (transport) and new (type) field names
            transport = server.get("type") or server.get("transport", "stdio")
            command = server.get("command", "")
            args = server.get("args", [])
            # Support both old (server_url) and new (url) field names
            server_url = server.get("url") or server.get("server_url", "")
            env = server.get("env")
            cwd = server.get("cwd")
            # Support both old (http_timeout) and new (timeout) field names
            http_timeout = server.get("timeout") or server.get("http_timeout", 60.0)

            # Skip if no command for stdio or no URL for http
            if transport == "http":
                if not server_url:
                    continue
            elif not command:
                continue

            # Build full command with args
            if args:
                full_command = command + " " + " ".join(str(a) for a in args)
            else:
                full_command = command

            try:
                mcp_tool = EncreMCPTool(
                    command=full_command if transport == "stdio" else "",
                    server_url=server_url if transport == "http" else "",
                    env=env if env else None,
                    cwd=cwd or None,
                    http_timeout=float(http_timeout) if http_timeout else 60.0,
                )
                await mcp_tool.register_with(self.tool_registry, prefix="mcp__")
                self._mcp_tools.append(mcp_tool)
            except Exception:
                import logging
                logging.getLogger("encre.agent").exception(
                    "Failed to connect MCP server: %s", server.get("name", command)
                )

    async def _disconnect_mcp(self) -> None:
        """Disconnect all MCP servers and remove their tools from the registry."""
        # Remove discovered MCP tool entries from the registry
        mcp_keys = [k for k in self.tool_registry.list_tools() if k.startswith("mcp__")]
        for key in mcp_keys:
            self.tool_registry._tools.pop(key, None)

        # Disconnect MCP clients
        for mcp in self._mcp_tools:
            try:
                await mcp._disconnect()
            except Exception:
                pass
        self._mcp_tools.clear()

    async def reconnect_mcp(self) -> None:
        """Disconnect old MCP connections and reconnect with current config."""
        await self._disconnect_mcp()
        await self._init_mcp()

    def set_scheduler(self, scheduler: Any) -> None:
        """Wire a scheduler instance to cron tools."""
        from encre.tools.builtin.cron_create import EncreCronCreateTool
        from encre.tools.builtin.cron_delete import EncreCronDeleteTool
        from encre.tools.builtin.cron_list import EncreCronListTool
        EncreCronCreateTool.set_scheduler(scheduler)
        EncreCronDeleteTool.set_scheduler(scheduler)
        EncreCronListTool.set_scheduler(scheduler)

    def reset(self) -> None:
        self.session = EncreSession(self.config)
        self.telemetry.reset()
        self.evolution = EvolutionConfig.create_default()
        self.loop = EncreLoop(
            self.config, self.session, self.tool_registry,
            self.hook_system, self.safety,
            self.memory_system,
            profile_system=self.profile_system,
            soul_system=self.soul_system,
            skill_registry=self.skill_registry,
            telemetry=self.telemetry,
            evolution=self.evolution,
            recovery=self.recovery,
            feedback=self.feedback,
            code_index=self.code_index,
        )
        self._wire_tools()
        self._plugins_loaded = False
        self._load_plugins()
        self._mcp_initialized = False

    async def aclose(self) -> None:
        """Release all resources held by this agent.

        Closes the backend (httpx clients, model memory), clears session state,
        flushes telemetry, and disconnects MCP servers.
        """
        # Disconnect MCP servers
        try:
            await self._disconnect_mcp()
        except Exception:
            pass

        # Close the agent loop (which closes the backend)
        try:
            await self.loop.aclose()
        except Exception:
            pass

        # Flush telemetry
        if self.telemetry is not None:
            try:
                self.telemetry.flush()
            except Exception:
                pass

        # Clear session
        if self.session is not None:
            self.session.messages.clear()
            self.session.rebuild_runtime_caches()

    def activate_skill(self, name: str, args: str | None = None) -> str:
        return self.skill_registry.activate(name, args)

    def respond_permission(self, decision: bool) -> None:
        """Approve or deny the pending permission request from the current turn."""
        self.loop.resolve_permission(decision)

    def goal(
        self,
        description: str,
        success_criteria: str,
        max_attempts: int = 20,
        timeout_seconds: int = 3600,
    ) -> "EncreGoalLoop":
        """Create a goal-driven autonomous loop for this agent.

        Usage:
            result = await agent.goal(
                "Implement login", "JWT tokens work, tests pass"
            ).execute()
        """
        from encre.goal import EncreGoalLoop
        return EncreGoalLoop(
            self,
            description=description,
            success_criteria=success_criteria,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
        )

    def swarm(
        self,
        goal: str,
        max_concurrent: int = 5,
        enable_reviewer: bool = True,
        timeout_seconds: float = 3600.0,
    ) -> "EncreSwarmSession":
        """Create a multi-agent swarm session for this agent.

        Decomposes the goal, assigns roles, and executes in parallel with
        shared blackboard and optional reviewer gates.

        Usage:
            result = await agent.swarm(
                "Build a full-stack TODO app with auth"
            ).execute()
        """
        from encre.swarm.session import EncreSwarmSession
        return EncreSwarmSession(
            self,
            goal=goal,
            max_concurrent=max_concurrent,
            enable_reviewer=enable_reviewer,
            timeout_seconds=timeout_seconds,
        )
