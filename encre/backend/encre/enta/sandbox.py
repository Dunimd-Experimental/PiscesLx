#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTASandbox — Containerised execution environment for training tasks.

Provides:
    - Isolated filesystem for file operations
    - Python / shell execution environment
    - Web fetch / search capabilities
    - Desktop / browser automation (via Encre computer/)
    - Student model tool call execution via EncreAgent
    - Objective reward computation from task outcomes

Architecture:
    The sandbox receives a task and teacher data, invokes the PiscesL1 7B
    student model, captures its tool calls, executes them against the
    EncreContainerSandbox, and computes an objective reward signal.

    The reward drives:
    - 7B core SFT (via bridge → opss.train)
    - 0.5B subconscious head RL (via bridge → SubconsciousTrainer)
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("encre.enta")


@dataclass
class SandboxResult:
    """Result from a sandbox execution.

    Args:
        success: Task completed without fatal errors.
        reward: Computed reward signal (0.0–1.0).
        output: Full text output from execution.
        tool_calls: Number of tool calls made by the student.
        tool_trace: Detailed per-call trace for analysis.
        errors: Any errors encountered.
        hidden_states: 7B hidden states for subconscious training.
        metadata: Additional result metadata.
    """
    success: bool = False
    reward: float = 0.0
    output: str = ""
    tool_calls: int = 0
    tool_trace: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    hidden_states: Optional[Any] = None
    metadata: Dict = field(default_factory=dict)


# Regex patterns for parsing tool calls from student output
# The student outputs <TOOL_CALL>json</TOOL_CALL> blocks
_TOOL_CALL_RE = re.compile(
    r'<TOOL_CALL>\s*(\{.*?\})\s*</TOOL_CALL>',
    re.DOTALL,
)


class EnTASandbox:
    """Training sandbox — invokes student model, executes tool calls, computes reward.

    Wraps Encre's container sandbox and tool system to provide a controlled
    training environment with objective rewards.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        container_sandbox: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None,
    ):
        self.tools = tools
        self.container = container_sandbox
        self._reward_fn = reward_fn or self._default_reward
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="enta_sbx_")
        self.exec_log: List[Dict] = []

        logger.info(
            f"Sandbox: {len(self.tools)} tools, container={'yes' if container_sandbox else 'no'}, "
            f"work_dir={self.work_dir}"
        )

    async def execute(
        self,
        task: Any,
        teacher_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a training task: invoke student → run tool calls → compute reward.

        Args:
            task: TaskTemplate from curriculum.
            teacher_data: Raw teacher output (used as reference for reward).

        Returns:
            Dict with reward, success, tool_calls, hidden_states.
        """
        self.exec_log.clear()
        result = SandboxResult()

        try:
            # Build the full prompt: task description + teacher demonstration
            teacher_output = ""
            if teacher_data:
                teacher_output = teacher_data.get("raw_response", "")

            # Invoke student model — this is where the 7B core generates
            # a response that may include <TOOL_CALL>...</TOOL_CALL> blocks
            student_response = await self._invoke_student(task, teacher_output)

            # Parse and execute tool calls from student output
            tool_trace = await self._execute_tool_calls(
                student_response.get("text", ""),
                task,
            )

            # Build result
            result.tool_trace = tool_trace
            result.tool_calls = len(tool_trace)
            result.output = student_response.get("text", "")
            result.hidden_states = student_response.get("hidden_states")
            result.success = all(
                t.get("success", False) for t in tool_trace
            ) if tool_trace else bool(result.output.strip())

            # Collect errors
            for t in tool_trace:
                if t.get("error"):
                    result.errors.append(t["error"])

            # Compute reward
            result.reward = self._compute_reward(result, task)

            self.exec_log.append(dict(
                task=task.description,
                success=result.success,
                reward=result.reward,
                tool_calls=result.tool_calls,
                errors=len(result.errors),
            ))

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            result.errors.append(str(e))
            result.reward = 0.0

        return dict(
            reward=result.reward,
            success=result.success,
            output=result.output,
            tool_calls=result.tool_calls,
            tool_trace=result.tool_trace,
            errors=result.errors,
            hidden_states=result.hidden_states,
            metadata=dict(work_dir=self.work_dir),
        )

    async def _invoke_student(
        self,
        task: Any,
        teacher_output: str = "",
    ) -> Dict[str, Any]:
        """Invoke the PiscesL1 7B student model on a training task.

        Constructs the prompt with available tools and optional teacher
        demonstration, then calls model.generate().

        Args:
            task: TaskTemplate to execute.
            teacher_output: Optional teacher demonstration to include.

        Returns:
            Dict with 'text' (generated output), 'hidden_states' (for
            subconscious training), and optionally 'tool_calls'.
        """
        # Build tools description
        tools_lines = []
        for tname in task.tools_required:
            tool = self.tools.get(tname)
            if tool is None:
                continue
            doc = (tool.__doc__ or tool.__class__.__doc__ or "").strip()[:200]
            tools_lines.append(f"  {tname}: {doc}")

        tools_str = "\n".join(tools_lines) if tools_lines else "  (no tools needed)"

        # Optional teacher demonstration
        teacher_section = (
            f"\nTeacher demonstration:\n{teacher_output[:500]}\n"
            if teacher_output else ""
        )

        prompt = (
            f"Complete the following task. "
            f"If you need to use a tool, output:\n"
            f"<TOOL_CALL>{{'tool': 'tool_name', 'args': {{...}}}}</TOOL_CALL>\n\n"
            f"Task: {task.description}\n"
            f"Available tools:\n{tools_str}\n"
            f"{teacher_section}\n"
            "Plan and execute step by step."
        )

        try:
            # Check if we have an actual model to call
            from encre.agent import EncreAgent
            agent = getattr(self, '_training_agent', None)

            if agent is not None:
                # Use EncreAgent to run the task with tool support
                response = await agent.run(prompt)
                text = response.get("text", "")
            else:
                # Without a running model, return the prompt itself
                # This works for testing the pipeline structure
                text = prompt

            return dict(text=text, hidden_states=None)

        except Exception as e:
            logger.error(f"Student invocation failed: {e}")
            return dict(text="", hidden_states=None, errors=[str(e)])

    async def _execute_tool_calls(
        self,
        student_text: str,
        task: Any,
    ) -> List[Dict]:
        """Parse and execute tool calls from student output.

        Args:
            student_text: The full text output from the student model.
            task: Original task for context.

        Returns:
            List of tool execution traces, each containing tool name,
            args, result, success status, and any errors.
        """
        matches = _TOOL_CALL_RE.findall(student_text)
        trace = []

        for idx, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                tool_name = call_data.get("tool", "")
                tool_args = call_data.get("args", {})

                tool = self.tools.get(tool_name)
                if tool is None:
                    trace.append(dict(
                        index=idx,
                        tool=tool_name,
                        success=False,
                        error=f"Unknown tool: {tool_name}",
                    ))
                    continue

                # Execute the tool call
                if hasattr(tool, "execute") and callable(tool.execute):
                    result = await tool.execute(**tool_args) if hasattr(result := None, '__await__') else tool.execute(**tool_args)
                    trace.append(dict(
                        index=idx,
                        tool=tool_name,
                        args=tool_args,
                        success=True,
                        result=str(result)[:500],
                    ))
                else:
                    trace.append(dict(
                        index=idx,
                        tool=tool_name,
                        success=False,
                        error="Tool has no execute method",
                    ))

            except json.JSONDecodeError as e:
                trace.append(dict(
                    index=idx,
                    tool="parse_error",
                    success=False,
                    error=f"JSON parse error: {e}",
                    raw=match[:100],
                ))
            except Exception as e:
                trace.append(dict(
                    index=idx,
                    tool="execution_error",
                    success=False,
                    error=str(e),
                ))

        return trace

    def set_agent(self, agent: Any) -> None:
        """Set an EncreAgent for student model invocation.

        Args:
            agent: EncreAgent instance connected to the PiscesL1 model.
        """
        self._training_agent = agent

    def _compute_reward(self, result: SandboxResult, task: Any) -> float:
        """Compute reward from execution results.

        Delegates to the configured reward function.

        Args:
            result: Execution result.
            task: Original task.

        Returns:
            Reward from 0.0 to 1.0.
        """
        return self._reward_fn(result, task)

    @staticmethod
    def _default_reward(result: SandboxResult, task: Any) -> float:
        """Default reward function.

        Factors:
            - Base 1.0 for success
            - Penalty for excessive tool calls (> 2× expected)
            - Penalty for errors

        Args:
            result: Execution result.
            task: Original task.

        Returns:
            Computed reward.
        """
        if not result.success:
            return 0.0

        reward = 1.0

        # Penalize excessive tool calls (expected ≈ 2 per required tool)
        expected_calls = len(task.tools_required) * 2
        if result.tool_calls > expected_calls:
            excess = result.tool_calls - expected_calls
            reward *= max(0.3, 1.0 - excess * 0.15)

        # Penalize errors
        if result.errors:
            reward *= max(0.2, 1.0 - len(result.errors) * 0.4)

        return reward
