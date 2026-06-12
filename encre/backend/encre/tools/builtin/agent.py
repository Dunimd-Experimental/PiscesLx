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

import types
from contextvars import ContextVar
from typing import Any

from encre.logging_config import get_logger
from encre.tools.base import build_tool

logger = get_logger(__name__)

MAX_SUB_AGENT_DEPTH = 3

_current_loop: ContextVar[Any] = ContextVar("encre_agent_current_loop", default=None)
_parent_loop: Any = None  # Set by agent during initialization


def set_parent_loop(loop: Any) -> None:
    global _parent_loop
    _parent_loop = loop


def set_active_loop(loop: Any) -> Any:
    return _current_loop.set(loop)


def reset_active_loop(token: Any) -> None:
    _current_loop.reset(token)


def _resolve_loop() -> Any:
    ctx_loop = _current_loop.get()
    if ctx_loop is not None:
        return ctx_loop
    return _parent_loop


def _build_agents_list() -> str:
    """Build a formatted string listing available sub-agents for the tool description."""
    loop = _resolve_loop()
    if loop is None:
        return ""
    sub_agents = getattr(loop.config, "sub_agents", [])
    if not sub_agents:
        return ""
    lines = ["Available sub-agents:"]
    for sa in sub_agents:
        desc = sa.description or "No description"
        lines.append(f"  - {sa.name}: {desc}")
    return "\n".join(lines)


async def _agent_execute(**kwargs: Any) -> Any:
    prompt = kwargs.get("prompt", "")
    agent_name = kwargs.get("agent_name", "")
    progress_callback = kwargs.get("progress_callback")
    parent_loop = _resolve_loop()

    logger.info("[agent] agent_name=%s | prompt_len=%d", agent_name, len(prompt))
    logger.info("[agent] prompt=%.300s", prompt)

    if parent_loop is not None and parent_loop.sub_agent_depth >= MAX_SUB_AGENT_DEPTH:
        return {"content": "Error: Maximum sub-agent recursion depth reached", "messages": []}

    if parent_loop is not None:
        sub_result = await parent_loop._run_sub_agent(
            prompt=prompt,
            max_turns=50,
            progress_callback=progress_callback,
        )
        if isinstance(sub_result, dict):
            return sub_result
        return {"content": str(sub_result), "messages": []}

    from encre.loop import EncreLoop
    from encre.session import EncreSession
    from encre.config import EncreConfig

    config = EncreConfig()
    session = EncreSession(config)
    loop = EncreLoop(config, session)
    sub_result = await loop._run_sub_agent(prompt, progress_callback=progress_callback)
    if isinstance(sub_result, dict):
        return sub_result
    return {"content": str(sub_result), "messages": []}


EncreAgentTool = build_tool(
    name="agent",
    description="Spawn a sub-agent to perform a specific task and return the result. "
                "The sub-agent runs as a fully-capable session, just like the main process. "
                "Be specific in the prompt — it is the complete instruction for the sub-agent.",
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The full task instruction for the sub-agent. "
                               "The sub-agent runs as an independent session with "
                               "all tools and capabilities available.",
            },
            "agent_name": {
                "type": "string",
                "description": "Optional sub-agent name (reserved for future use)",
            },
        },
        "required": ["prompt"],
    },
    execute=_agent_execute,
    intents=["general", "coding", "system"],
)


def _agent_to_openai_format(self) -> dict[str, Any]:
    agents_block = _build_agents_list()
    description = self.description
    if agents_block:
        description += f"\n\n{agents_block}"
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": description,
            "parameters": self.input_schema,
        },
    }


def _agent_to_anthropic_format(self) -> dict[str, Any]:
    agents_block = _build_agents_list()
    description = self.description
    if agents_block:
        description += f"\n\n{agents_block}"
    return {
        "name": self.name,
        "description": description,
        "input_schema": self.input_schema,
    }


# Monkey-patch the format methods to include dynamic agents list
EncreAgentTool.to_openai_format = types.MethodType(_agent_to_openai_format, EncreAgentTool)
EncreAgentTool.to_anthropic_format = types.MethodType(_agent_to_anthropic_format, EncreAgentTool)
# Backward-compat: keep ``.set_parent_loop()`` callable on the tool object.
EncreAgentTool.set_parent_loop = set_parent_loop
