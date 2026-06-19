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



"""``agent`` tool -- spawn sub-agent sessions.

The training pipeline runs the model in a single, self-contained process
where there is no parent agent loop to dispatch sub-agents into.  This
tool therefore returns a structured error so the model learns that
``agent`` is not a usable tool inside training rollouts.  In a real
agent runtime (outside the slimmed EnCRE core) the same tool name would
be re-implemented to fan out sub-agent invocations.
"""

import logging
from typing import Any

from enta.tools.base import build_tool

logger = logging.getLogger(__name__)


def _set_parent_loop(loop: Any) -> None:
    """Optional hook a runtime agent loop can call to enable dispatch."""
    logger.debug("[agent] parent loop attached: %r", loop)


async def _agent_execute(**kwargs: Any) -> str:
    prompt = kwargs.get("prompt", "")
    agent_name = kwargs.get("agent_name", "")
    tasks = kwargs.get("tasks")
    logger.info(
        "[agent] dispatch attempt: agent_name=%s tasks=%d prompt_len=%d",
        agent_name,
        len(tasks) if isinstance(tasks, list) else 0,
        len(prompt),
    )
    return (
        "Error: the 'agent' tool requires a parent agent loop, which is "
        "not available inside the slimmed EnCRE training core.  Use the "
        "specialised tools (bash, file_*, glob, grep, git_tool, apply_patch, "
        "task_*, web_*, etc.) to complete the work directly."
    )


EncreAgentTool = build_tool(
    name="agent",
    description=(
        "Spawn a sub-agent session.  In the slimmed EnCRE training core "
        "this tool is intentionally disabled: there is no parent loop to "
        "host a sub-agent.  Use specialised tools (bash, file_*, glob, "
        "grep, git_tool, apply_patch, task_*, web_*, etc.) to complete "
        "the work directly inside the current session."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The full task instruction (unused when dispatch is disabled).",
            },
            "agent_name": {
                "type": "string",
                "description": "Optional sub-agent name (unused when dispatch is disabled).",
            },
        },
    },
    execute=_agent_execute,
    intents=["general", "coding", "system"],
)

EncreAgentTool.set_parent_loop = _set_parent_loop


__all__ = ["EncreAgentTool"]
