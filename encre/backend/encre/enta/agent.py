#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTA — Encre Train Agent.

EnTA itself is an Agent driven by Agens-2.0-Flash (or any configured teacher).
It autonomously decides what to teach, generates training data, executes tasks
in sandbox, and updates the student model — all through tool calls.

Architecture:
    EnTA runs as an EncreAgent with a system prompt that defines its mission:
    "You are a training agent. Your job is to train a 7B student model to
    reach GPT-5 level capability in tool use, reasoning, and knowledge."

    The agent has access to training tools:
        - curriculum:  what to teach next, track weak areas
        - teacher:     call any teacher model for training data
        - roundtable:  multi-teacher discussion for creative tasks
        - sandbox:     execute tasks and get reward signals
        - student:     update the 7B model with training data
        - progress:    check overall training status

    The agent autonomously loops: think → tool call → evaluate → repeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("encre.enta")

# Backend event types for collecting chat responses
from encre.utils.types import (
    BackendText,
    BackendThinking,
    BackendToolCall,
    BackendToolCallDelta,
    BackendFinish,
    BackendError,
)


# ---------------------------------------------------------------------------
# Training phase enum — used as a signpost for the agent, not a hard schedule
# ---------------------------------------------------------------------------

class TrainingStage(Enum):
    FOUNDATION = "foundation"
    INTEGRATION = "integration"
    ADVANCED = "advanced"
    SPECIALIZATION = "specialization"
    SELF_PLAY = "self_play"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnTAConfig:
    """Configuration for the EnTA training agent.

    Args:
        agent_model: The model driving EnTA itself (e.g. agens-2.0-flash).
        teacher_model: Primary teacher for generating training data.
        auxiliary_teachers: Additional teachers for round-table.
        max_tasks: Maximum tasks before stopping. 0 = unlimited.
        tasks_per_stage: Hints to the agent about when to advance.
        subconscious_enabled: Whether to train the 0.5B head.
        record_interval: Logging frequency.
    """
    agent_model: str = "agens-2.0-flash"
    teacher_model: str = "deepseek-r1"
    auxiliary_teachers: List[str] = field(default_factory=lambda: [
        "deepseek-v3.2", "qwen3.6", "agens-2.0-flash", "agens-2.0-video",
    ])
    max_tasks: int = 0
    tasks_per_stage: int = 5000
    subconscious_enabled: bool = True
    record_interval: int = 100


# ---------------------------------------------------------------------------
# Training state — shared mutable state that the agent's tools operate on
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Training state that persists across agent tool calls.

    The agent reads and writes this state through its tools.
    """
    stage: str = "foundation"
    tasks_completed: int = 0
    total_reward: float = 0.0
    recent_rewards: List[float] = field(default_factory=list)
    skill_rewards: Dict[str, List[float]] = field(default_factory=dict)
    is_running: bool = True
    message: str = ""


# ---------------------------------------------------------------------------
# EnTA training tools — registered as EncreTool instances so the agent
# can call them autonomously.
# ---------------------------------------------------------------------------

def _make_curriculum_tool(curriculum: Any, state: TrainingState) -> Dict:
    """Return a tool definition dict for curriculum operations.

    The agent calls this to decide what to teach next.
    """
    return {
        "type": "function",
        "function": {
            "name": "curriculum_next_task",
            "description": "Get the next training task based on student weak areas. Returns a task with skill area, difficulty, and required tools.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


def _make_teacher_tool() -> Dict:
    """Return a tool definition for calling a teacher model."""
    return {
        "type": "function",
        "function": {
            "name": "call_teacher",
            "description": "Call a teacher model to generate training data for a specific skill area and task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Teacher model name (deepseek-r1, deepseek-v3.2, qwen3.6, agens-2.0-flash, etc.)",
                    },
                    "skill_area": {
                        "type": "string",
                        "description": "Skill area: tool_operation, reasoning, knowledge_retrieval, creative, code, tool_chain, error_recovery",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "What the student should practice",
                    },
                    "difficulty": {
                        "type": "number",
                        "description": "Difficulty 0.0-1.0",
                    },
                },
                "required": ["model", "skill_area", "task_description"],
            },
        },
    }


def _make_roundtable_tool() -> Dict:
    """Return a tool definition for multi-teacher round-table discussion."""
    return {
        "type": "function",
        "function": {
            "name": "run_roundtable",
            "description": "Run a multi-teacher round-table discussion on a creative or ambiguous task. Multiple models debate and converge on the best training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The task for teachers to discuss",
                    },
                    "skill_area": {
                        "type": "string",
                        "description": "Skill area for context",
                    },
                    "seed_data": {
                        "type": "string",
                        "description": "Optional seed data from primary teacher",
                    },
                },
                "required": ["task_description"],
            },
        },
    }


def _make_sandbox_tool() -> Dict:
    """Return a tool definition for sandbox execution."""
    return {
        "type": "function",
        "function": {
            "name": "execute_sandbox",
            "description": "Execute a task in the training sandbox. Runs the student model against the task and returns a reward signal (0.0-1.0).",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The task to execute",
                    },
                    "tools_required": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools needed (file_read, bash, grep, web_search, etc.)",
                    },
                    "teacher_demo": {
                        "type": "string",
                        "description": "Optional teacher demonstration for reference",
                    },
                },
                "required": ["task_description"],
            },
        },
    }


def _make_training_tool() -> Dict:
    """Return a tool definition for updating the student model."""
    return {
        "type": "function",
        "function": {
            "name": "train_student",
            "description": "Feed training data to the student model. Triggers SFT on the 7B core and optionally updates the 0.5B subconscious head.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "The task prompt",
                    },
                    "target_text": {
                        "type": "string",
                        "description": "The expected output (teacher's reasoning + answer)",
                    },
                    "reward": {
                        "type": "number",
                        "description": "Reward signal from sandbox execution (0.0-1.0)",
                    },
                    "skill_area": {
                        "type": "string",
                        "description": "Which skill this trains",
                    },
                    "use_subconscious": {
                        "type": "boolean",
                        "description": "Whether to also train the subconscious head",
                    },
                },
                "required": ["input_text", "target_text", "reward"],
            },
        },
    }


def _make_progress_tool(state: TrainingState) -> Dict:
    """Return a tool definition for checking training progress."""
    return {
        "type": "function",
        "function": {
            "name": "check_progress",
            "description": "Check overall training progress and statistics. Returns tasks completed, average reward, and per-skill performance.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations — these are called when the agent invokes a tool.
# They delegate to the actual curriculum/sandbox/bridge objects.
# ---------------------------------------------------------------------------

class EnTAToolHandler:
    """Handles tool invocations from the EnTA agent.

    Each method maps to a tool name and performs the actual operation
    using the curriculum, sandbox, bridge, and backends.
    """

    def __init__(
        self,
        curriculum: Any,
        sandbox: Any,
        bridge: Any,
        teacher_backend: Any,
        aux_backends: Dict[str, Any],
        round_table: Any,
        state: TrainingState,
    ):
        self.curriculum = curriculum
        self.sandbox = sandbox
        self.bridge = bridge
        self.teacher = teacher_backend
        self.aux_backends = aux_backends
        self.round_table = round_table
        self.state = state
        self._teacher_response_cache: Dict[str, str] = {}

    async def handle_call(self, tool_name: str, args: Dict) -> str:
        """Dispatch a tool call to the appropriate handler.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments from the agent.

        Returns:
            JSON string result for the agent.
        """
        handler_map = {
            "curriculum_next_task": self._handle_curriculum_next,
            "call_teacher": self._handle_call_teacher,
            "run_roundtable": self._handle_run_roundtable,
            "execute_sandbox": self._handle_execute_sandbox,
            "train_student": self._handle_train_student,
            "check_progress": self._handle_check_progress,
        }

        handler = handler_map.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(args)
            else:
                result = handler(args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    def _handle_curriculum_next(self, args: Dict) -> Dict:
        """Get next task from curriculum.

        Returns task details for the agent to work with.
        """
        task = self.curriculum.next_task(self.state.stage)
        if task is None:
            return {"error": "no tasks available", "stage": self.state.stage}
        return {
            "skill_area": task.skill_area.value,
            "description": task.description,
            "difficulty": task.difficulty,
            "tools_required": task.tools_required,
            "requires_creativity": task.requires_creativity,
            "teacher_hint": task.teacher_hint,
        }

    async def _handle_call_teacher(self, args: Dict) -> Dict:
        """Call a teacher model to generate training data.

        Args:
            args: model, skill_area, task_description, optional difficulty.

        Returns:
            Teacher's response text.
        """
        model = args.get("model", "deepseek-r1")
        skill = args.get("skill_area", "reasoning")
        task_desc = args.get("task_description", "")
        difficulty = args.get("difficulty", 0.5)

        # Pick the right backend
        try:
            backend = self.teacher
            # Check if this matches the teacher backend
            teacher_model = getattr(self.teacher, '_model_name', '')
            if not teacher_model or model != teacher_model:
                backend = self.aux_backends.get(model)
                if backend is None:
                    from encre.enta.config import create_backend_for_model
                    backend = create_backend_for_model(model)
                    self.aux_backends[model] = backend
        except Exception:
            from encre.enta.config import create_backend_for_model
            backend = self.aux_backends.get(model)
            if backend is None:
                backend = create_backend_for_model(model)
                self.aux_backends[model] = backend

        prompt = (
            f"You are training a student model. "
            f"Generate a {skill} training task with step-by-step reasoning.\n\n"
            f"Task: {task_desc}\n"
            f"Difficulty: {difficulty:.1f}/1.0\n\n"
            "Respond with:\n"
            "1. Step-by-step reasoning\n"
            "2. Final answer\n"
            "3. Common mistakes to avoid"
        )

        try:
            response = await backend.generate(prompt)
            self._teacher_response_cache[task_desc] = response
            return {
                "model": model,
                "skill_area": skill,
                "response": response[:2000],
                "response_length": len(response),
            }
        except Exception as e:
            return {"error": str(e), "model": model}

    async def _handle_run_roundtable(self, args: Dict) -> Dict:
        """Run multi-teacher round-table discussion.

        Args:
            args: task_description, skill_area, optional seed_data.

        Returns:
            Consensus result.
        """
        task_desc = args.get("task_description", "")
        skill = args.get("skill_area", "creative")
        seed = args.get("seed_data", "")

        if self.round_table is None:
            return {"error": "round_table not initialized", "consensus": seed}

        # Mock task object for roundtable
        class _MockTask:
            description = task_desc
            skill_area = type('sa', (), {'value': skill})()

        initial_data = {"raw_response": seed} if seed else None
        result = await self.round_table.discuss(_MockTask(), initial_data)

        return {
            "consensus": result.consensus.get("synthesis", seed)[:2000],
            "quality_score": result.quality_score,
            "num_teachers": len(result.proposals),
            "discussion_rounds": result.discussion_steps,
        }

    async def _handle_execute_sandbox(self, args: Dict) -> Dict:
        """Execute a task in sandbox and get reward.

        Args:
            args: task_description, optional tools_required, optional teacher_demo.

        Returns:
            Reward and execution details.
        """
        task_desc = args.get("task_description", "")
        tools = args.get("tools_required", [])
        demo = args.get("teacher_demo", "")

        class _MockTask:
            description = task_desc
            tools_required = tools
            skill_area = type('sa', (), {'value': 'training'})()

        teacher_data = {"raw_response": demo} if demo else None
        result = await self.sandbox.execute(_MockTask(), teacher_data)

        self.state.recent_rewards.append(result.get("reward", 0.0))
        if len(self.state.recent_rewards) > 100:
            self.state.recent_rewards.pop(0)

        return {
            "reward": result.get("reward", 0.0),
            "success": result.get("success", False),
            "tool_calls": result.get("tool_calls", 0),
            "errors": result.get("errors", []),
        }

    def _handle_train_student(self, args: Dict) -> Dict:
        """Feed training data to the student model.

        Args:
            args: input_text, target_text, reward, optional skill_area, use_subconscious.

        Returns:
            Training result.
        """
        input_text = args.get("input_text", "")
        target_text = args.get("target_text", "")
        reward = args.get("reward", 0.0)
        skill_area = args.get("skill_area", "general")
        use_sc = args.get("use_subconscious", self.state.is_running)

        sample = self.bridge.feed_teacher_output(
            teacher_response=target_text,
            task_description=input_text,
            reward=reward,
            skill_area=skill_area,
            hidden_states=None if not use_sc else "trigger_update",
        )

        self.state.tasks_completed += 1
        self.state.total_reward += reward

        if skill_area not in self.state.skill_rewards:
            self.state.skill_rewards[skill_area] = []
        self.state.skill_rewards[skill_area].append(reward)

        return {
            "sample_id": id(sample),
            "tasks_completed": self.state.tasks_completed,
            "avg_reward": self.state.total_reward / max(1, self.state.tasks_completed),
            "skill_area": skill_area,
            "bridge_stats": self.bridge.get_stats(),
        }

    def _handle_check_progress(self, args: Dict) -> Dict:
        """Return current training progress."""
        avg = self.state.total_reward / max(1, self.state.tasks_completed)

        per_skill = {}
        for skill, rewards in self.state.skill_rewards.items():
            if rewards:
                per_skill[skill] = {
                    "tasks": len(rewards),
                    "avg_reward": sum(rewards) / len(rewards),
                }

        return {
            "tasks_completed": self.state.tasks_completed,
            "stage": self.state.stage,
            "avg_reward": round(avg, 4),
            "recent_avg_reward": round(
                sum(self.state.recent_rewards[-20:]) / max(1, len(self.state.recent_rewards[-20:])), 4
            ) if self.state.recent_rewards else 0.0,
            "per_skill": per_skill,
            "message": self.state.message,
        }


# ---------------------------------------------------------------------------
# EnTA System Prompt — defines the agent's mission and available tools.
# ---------------------------------------------------------------------------

ENTA_SYSTEM_PROMPT = """You are EnTA (Encre Train Agent), an autonomous training agent. Your mission is to train a PiscesL1 7B student model to achieve GPT-5 level capability in tool use, reasoning, and knowledge.

You are driven by {agent_model}. Your available tools are: curriculum_next_task, call_teacher, run_roundtable, execute_sandbox, train_student, check_progress.

## Your Training Methodology

1. **Assess** — Call check_progress to understand current student performance.
2. **Plan** — Think about which skill area is weakest and needs practice.
3. **Teach** — Call curriculum_next_task to get a task, then call_teacher to generate training data.
4. **Enhance** — For creative tasks (difficulty >= 0.6), use run_roundtable for multi-teacher consensus.
5. **Verify** — Call execute_sandbox to run the task and get a reward signal.
6. **Train** — Call train_student to feed the data into the student model.
7. **Repeat** — Go back to step 1, focusing on weak areas.

## Training Phases

- **foundation**: Basic tool operation and reasoning chains. Focus on file_read, bash, grep tools.
- **integration**: Multi-tool chains and complex reasoning. Practice composing tools together.
- **advanced**: Creative problem-solving. Use round-table for ambiguous tasks.
- **specialization**: Deep training on weak areas. Use data from all teachers.
- **self_play**: Autonomous improvement. Generate tasks from student's own outputs.

## Available Tools

### curriculum_next_task
Returns the next training task. No arguments needed — the curriculum adapts automatically based on past outcomes.

### call_teacher
Call a teacher model to generate training data. Pick the right teacher:
- deepseek-r1: Best for reasoning, math, logic
- deepseek-v3.2: Best for general knowledge, language
- qwen3.6: Best for long context, tool use
- agens-2.0-flash: Best for action planning
- agens-2.0-video: Best for multimodal

### run_roundtable
For tasks with difficulty >= 0.6 or requiring creativity. Multiple teachers discuss and produce superior training data.

### execute_sandbox
Run the task in sandbox to get an objective reward. Higher reward = student performing well on this task type.

### train_student
Feed data to the student. Requires input_text (prompt), target_text (expected output), and reward from sandbox.

### check_progress
Returns tasks completed, average reward, and per-skill breakdown.

## Guidelines

- Vary skill areas — don't over-train one skill.
- If reward is below 0.4, the student is struggling. Try easier tasks in that area or provide more detailed teacher demonstrations.
- If reward is above 0.8 for many tasks in a skill, the student has mastered it. Move to harder tasks or new skills.
- Use round-table for creative tasks — the synthesis produces better training data.
- Call check_progress every few iterations to track improvement.
- Your goal is to maximize the number of tasks completed and the average reward across all skill areas.

Begin training. Call check_progress first, then plan your first teaching action."""


# ---------------------------------------------------------------------------
# The Agent Loop
# ---------------------------------------------------------------------------

class EnTAAgent:
    """EnTA as an autonomous agent.

    Drives the training loop by letting the LLM (Agens-2.0-Flash or
    configured model) decide each action through tool calls.
    The agent's thinking loop IS the training loop.

    Uses the backend's ``chat()`` interface directly to support
    autonomous multi-turn tool calling.
    """

    def __init__(
        self,
        config: EnTAConfig,
        tool_handler: EnTAToolHandler,
        state: TrainingState,
    ):
        self.cfg = config
        self.tool_handler = tool_handler
        self.state = state
        self._backend = None
        self._messages: List[Dict] = []

    async def _chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Dict:
        """Collect a non-streaming chat response from the backend.

        Args:
            messages: Conversation history.
            tools: Optional tool definitions.
            tool_choice: "auto" | "none"

        Returns:
            Dict with "content" (str), "tool_calls" (list of {id, function}),
            and "finish_reason" (str).
        """
        content_parts: List[str] = []
        tool_calls: List[Dict] = []
        partial_tc: Dict[int, Dict] = {}
        finish_reason = "stop"

        async for event in self._backend.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.0,
            max_tokens=4096,
            stream=True,
        ):
            if isinstance(event, BackendText):
                content_parts.append(event.text)
            elif isinstance(event, BackendToolCallDelta):
                idx = event.index
                if idx not in partial_tc:
                    partial_tc[idx] = {"id": "", "name": "", "arguments": ""}
                if event.key == "id":
                    partial_tc[idx]["id"] = event.value
                elif event.key == "name":
                    partial_tc[idx]["name"] = event.value
                elif event.key == "arguments":
                    partial_tc[idx]["arguments"] += event.value
            elif isinstance(event, BackendToolCall):
                tool_calls.append({
                    "id": event.id,
                    "function": {"name": event.name, "arguments": event.arguments},
                })
            elif isinstance(event, BackendFinish):
                finish_reason = event.reason

        # Flush partial tool calls
        for idx in sorted(partial_tc.keys()):
            tc = partial_tc[idx]
            if tc["id"]:
                tool_calls.append({
                    "id": tc["id"],
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })

        return {
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }

    async def run(self, max_tasks: int = 0) -> Dict[str, Any]:
        """Run the EnTA agent loop autonomously.

        Args:
            max_tasks: Max training tasks. 0 = unlimited.

        Returns:
            Training summary.
        """
        start = time.time()

        # Lazy init backend
        if self._backend is None:
            from encre.enta.config import create_backend_for_model
            self._backend = create_backend_for_model(self.cfg.agent_model)

        # Tool definitions for OpenAI-compatible function calling format
        tools = [
            _make_curriculum_tool(None, self.state),
            _make_teacher_tool(),
            _make_roundtable_tool(),
            _make_sandbox_tool(),
            _make_training_tool(),
            _make_progress_tool(self.state),
        ]

        # System prompt
        system_prompt = ENTA_SYSTEM_PROMPT.format(agent_model=self.cfg.agent_model)
        self._messages = [{"role": "system", "content": system_prompt}]

        logger.info(
            f"EnTAAgent running: brain={self.cfg.agent_model}, "
            f"max_tasks={max_tasks or 'unlimited'}"
        )

        iteration = 0
        while self.state.is_running:
            iteration += 1

            if max_tasks > 0 and self.state.tasks_completed >= max_tasks:
                logger.info(f"Reached max_tasks={max_tasks}, stopping")
                break

            # === Agent thinks + calls tools ===
            # The agent may call multiple tools before responding
            max_tool_rounds = 10
            for tool_round in range(max_tool_rounds):
                tc = "auto" if tool_round < max_tool_rounds - 1 else "none"
                response = await self._chat(
                    self._messages, tools, tool_choice=tc,
                )

                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])

                # Save assistant message
                msg: Dict = {"role": "assistant", "content": content}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                self._messages.append(msg)

                # No tool calls → agent finished thinking
                if not tool_calls:
                    logger.debug(f"[iter={iteration}] Agent: {content[:120]}...")
                    break

                # Execute each tool call
                for tc_data in tool_calls:
                    tc_id = tc_data.get("id", "")
                    fn = tc_data.get("function", {})
                    tc_name = fn.get("name", "")
                    tc_args_str = fn.get("arguments", "{}")

                    try:
                        tc_args = json.loads(tc_args_str) if isinstance(tc_args_str, str) else tc_args_str
                    except json.JSONDecodeError:
                        tc_args = {}

                    result_str = await self.tool_handler.handle_call(tc_name, tc_args)

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_str,
                    })
                    logger.debug(f"  tool={tc_name} → {result_str[:100]}")

            # Periodic logging
            if iteration % 5 == 0 or self.state.tasks_completed % self.cfg.record_interval == 0:
                avg = self.state.total_reward / max(1, self.state.tasks_completed)
                logger.info(
                    f"[{iteration}] tasks={self.state.tasks_completed} "
                    f"avg_r={avg:.3f} stage={self.state.stage}"
                )

            # Early stop from agent
            last = self._messages[-1].get("content", "") if self._messages else ""
            if "training complete" in last.lower():
                logger.info("Agent signaled completion")
                break

            # Prevent context from growing unbounded
            # Keep system + last N messages
            if len(self._messages) > 50:
                keep = self._messages[:1] + self._messages[-40:]
                self._messages = keep

        elapsed = time.time() - start
        summary = dict(
            tasks_completed=self.state.tasks_completed,
            avg_reward=self.state.total_reward / max(1, self.state.tasks_completed),
            iterations=iteration,
            messages=len(self._messages),
            elapsed_seconds=round(elapsed, 1),
        )
        logger.info(f"EnTA complete: {summary}")
        return summary

    def stop(self):
        self.state.is_running = False
        self.state.is_running = False
        logger.info("EnTA stopping...")
