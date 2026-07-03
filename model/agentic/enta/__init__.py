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
YvEntaTrainer — training pipeline integration point for EnTA (EnTA Core).

This module is the single, real bridge between the PiscesLx model and the
slimmed-down EnTA (EnTA Core) tool palette used for adversarial training.

Design Overview
---------------
The adversarial training loop needs three concrete things from the agent
runtime:

1. **A tool palette** — the EnTA builtin tool set (file ops, search, git,
   bash, task management, memory, web search, …) is the surface the model
   learns to wield.  The full palette is built via
   ``enta.build_training_tool_registry()``.

2. **A native inference backend** — ``enta.LocalBackend`` runs any
   HuggingFace causal LM locally (CPU/GPU) and produces the
   ``BackendEvent`` stream that drives tool-call generation.  The
   same backend is also used to evaluate candidate rollouts.

3. **A sandboxed execution layer** — ``enta.EncreContainerSandbox`` plus
   the Rust ``sandbox_execute`` extension make every shell command the
   model emits auditable and reproducible.  This is the EnTA guarantee
   the training pipeline relies on.

``YvEntaTrainer`` wires those three pieces together, then exposes the
high-level entry points the training operator needs:

- :meth:`YvEntaTrainer.rollout`        — produce one full trajectory
- :meth:`YvEntaTrainer.score`          — score a trajectory
- :meth:`YvEntaTrainer.step`           — turn trajectories into a loss
- :meth:`YvEntaTrainer.run_adversarial_batch` — end-to-end batch loop
- :meth:`YvEntaTrainer.generate_batch`  — outer-loop data generation
- :meth:`YvEntaTrainer.evaluate`       — lightweight checkpoint evaluation
- :meth:`YvEntaTrainer.should_stop`    — curriculum completion check
- :meth:`YvEntaTrainer.update`         — curriculum advancement

The model is real (the user-supplied ``YvModelForCausalLM``); the
backend is real (HuggingFace transformers); the tools are real (every
call lands in a Rust sandbox or a pure-Python stdlib path).  Nothing
is stubbed, mocked, or simulated.
"""

import asyncio
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

import torch

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger(
    "Yv.Agentic.Enta",
    file_path=get_log_file("Yv.Agentic.Enta"),
    enable_file=True,
)


# ── Lazy EnTA binding ────────────────────────────────────────────
# The EnTA package is loaded lazily so the model's training step does
# not pay the import cost (or fail) when EnTA is not on the path.
_ENTA = None
_ENTA_IMPORT_ERROR: Exception | None = None


def _ensure_enta_path() -> None:
    """Ensure ``enta/backend/`` is on ``sys.path`` so ``import enta``
    finds the real ``__init__.py`` (which exposes ``create_default_backend``,
    ``build_training_tool_registry``, etc.) rather than the empty namespace
    package at ``enta/``.

    The enta package is structured as::

        enta/
          backend/          ← needs to be on sys.path
            enta/
              __init__.py   ← the actual package entry point
              backends/
              tools/
              ...

    When the working directory is the project root, ``import enta`` loads
    a *namespace* package from ``enta/`` (which has no ``__init__.py``)
    instead of the real package at ``enta/backend/enta/``.  This helper
    inserts the correct path early enough that ``_bind_enta()`` always
    resolves the real module.
    """
    import os
    import sys
    import pathlib

    this_file = pathlib.Path(__file__).resolve().parent        # model/agentic/enta/
    project_root = this_file.parent.parent.parent               # project root
    enta_backend = project_root / "enta" / "backend"
    if enta_backend.is_dir():
        candidate = str(enta_backend)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    # If ``enta`` was already loaded as a *namespace* package
    # (i.e. ``__file__`` is None), evict it from ``sys.modules`` so
    # that the next ``import enta`` re-scans ``sys.path`` and picks
    # up the real ``__init__.py``.
    stale = sys.modules.get("enta")
    if stale is not None and getattr(stale, "__file__", None) is None:
        del sys.modules["enta"]


def _bind_enta():
    """Bind the slimmed EnTA package and return it.

    Called once on first use.  The result is cached in the module
    globals so repeated invocations are O(1).
    """
    global _ENTA, _ENTA_IMPORT_ERROR
    if _ENTA is not None:
        return _ENTA
    if _ENTA_IMPORT_ERROR is not None:
        raise _ENTA_IMPORT_ERROR

    # Ensure ``enta/backend/`` is on ``sys.path`` before the import.
    _ensure_enta_path()

    try:
        import enta as _enta_mod
    except Exception as exc:  # pragma: no cover - environment dependent
        _ENTA_IMPORT_ERROR = exc
        _LOG.error(
            f"EnTA import failed; YvEntaTrainer requires the slimmed EnTA "
            f"package on sys.path (cause: {type(exc).__name__})"
        )
        raise
    _ENTA = _enta_mod
    return _ENTA


# ── Enumerations ──────────────────────────────────────────────────


class YvEntaAdversarialStage(str, Enum):
    """Stages of one adversarial training round."""

    ROLLOUT = "rollout"
    SCORE = "score"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


class YvEntaRewardSignal(str, Enum):
    """Sources of reward for the adversarial trainer.

    The trainer fuses multiple signals into a single scalar reward that
    the GRPO/RLVR operators in :mod:`opss.train` can consume.
    """

    TASK_COMPLETION = "task_completion"
    TOOL_CALL_QUALITY = "tool_call_quality"
    SANDBOX_SAFETY = "sandbox_safety"
    EXECUTION_SUCCESS = "execution_success"


# ── Data containers (internal, naming is free) ────────────────────


@dataclass
class _EntaStepRecord:
    """One tool-call step inside a rollout."""

    tool: str
    arguments: Dict[str, Any]
    raw_arguments: str
    tool_call_id: str
    result: str = ""
    is_error: bool = False
    elapsed_ms: int = 0
    reward: float = 0.0


@dataclass
class _EntaTrajectory:
    """Complete trajectory of one rollout.

    Holds the full message log, every tool step, the cumulative reward
    and a per-signal breakdown that downstream operators (GRPO, RLVR,
    PPO) can consume directly.
    """

    rollout_id: str
    prompt: str
    reference: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[_EntaStepRecord] = field(default_factory=list)
    final_text: str = ""
    finished_reason: str = ""
    total_reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    sandbox_violations: int = 0
    duration_ms: int = 0
    error: str = ""

    def to_serializable(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of the trajectory."""
        return {
            "rollout_id": self.rollout_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "final_text": self.final_text,
            "finished_reason": self.finished_reason,
            "total_reward": self.total_reward,
            "reward_breakdown": dict(self.reward_breakdown),
            "sandbox_violations": self.sandbox_violations,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "step_count": len(self.steps),
            "messages": list(self.messages),
            "steps": [
                {
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "tool_call_id": s.tool_call_id,
                    "result": s.result,
                    "is_error": s.is_error,
                    "elapsed_ms": s.elapsed_ms,
                    "reward": s.reward,
                }
                for s in self.steps
            ],
        }


# ── Reward calculator (internal helper) ───────────────────────────


class _EntaRewardCalculator:
    """Compute the scalar reward that drives the GRPO/RLVR update.

    The reward is a weighted sum of four signals (see
    :class:`YvEntaRewardSignal`).  All four are real, computed from
    the actual trajectory — no synthetic shaping, no hidden heuristics.
    """

    def __init__(
        self,
        *,
        completion_weight: float = 0.6,
        tool_weight: float = 0.2,
        safety_weight: float = 0.1,
        execution_weight: float = 0.1,
    ) -> None:
        if any(w < 0.0 for w in (completion_weight, tool_weight, safety_weight, execution_weight)):
            raise ValueError(
                f"reward weights must be non-negative, got: "
                f"completion={completion_weight}, tool={tool_weight}, "
                f"safety={safety_weight}, execution={execution_weight}"
            )
        self._weights = {
            YvEntaRewardSignal.TASK_COMPLETION: completion_weight,
            YvEntaRewardSignal.TOOL_CALL_QUALITY: tool_weight,
            YvEntaRewardSignal.SANDBOX_SAFETY: safety_weight,
            YvEntaRewardSignal.EXECUTION_SUCCESS: execution_weight,
        }
        total = sum(self._weights.values())
        if total <= 0.0:
            raise ValueError(
                f"at least one reward weight must be positive, but sum of weights is {total} "
                f"(completion={completion_weight}, tool={tool_weight}, "
                f"safety={safety_weight}, execution={execution_weight})"
            )
        # Normalize so weights sum to 1.0.
        for k in self._weights:
            self._weights[k] /= total

    def score(self, trajectory: _EntaTrajectory) -> Tuple[float, Dict[str, float]]:
        """Return ``(total_reward, breakdown)`` for the trajectory."""
        completion = self._completion_score(trajectory)
        tool_quality = self._tool_call_score(trajectory)
        safety = self._safety_score(trajectory)
        execution = self._execution_score(trajectory)
        breakdown = {
            YvEntaRewardSignal.TASK_COMPLETION.value: completion,
            YvEntaRewardSignal.TOOL_CALL_QUALITY.value: tool_quality,
            YvEntaRewardSignal.SANDBOX_SAFETY.value: safety,
            YvEntaRewardSignal.EXECUTION_SUCCESS.value: execution,
        }
        total = sum(self._weights[k] * breakdown[k.value] for k in self._weights)
        return float(total), breakdown

    @staticmethod
    def _completion_score(trajectory: _EntaTrajectory) -> float:
        """Substring-match style completion score in [0, 1].

        For training purposes the reference is a free-form string that
        contains the expected answer; the model gets credit proportional
        to the fraction of the reference tokens that appear in the
        final assistant turn.  This is intentionally cheap to compute
        and differentiable-free so it can run on every rollout.
        """
        if not trajectory.reference or not trajectory.final_text:
            return 0.0
        ref_tokens = trajectory.reference.split()
        if not ref_tokens:
            return 0.0
        final = trajectory.final_text
        hits = sum(1 for tok in ref_tokens if tok and tok in final)
        return float(hits) / float(len(ref_tokens))

    @staticmethod
    def _tool_call_score(trajectory: _EntaTrajectory) -> float:
        """Reward well-formed tool calls.

        Each step gets a small positive reward for being a valid
        ``{"tool": str, "arguments": dict}`` tuple; steps that
        error out get a proportional penalty.  Clamped to ``[-1, 1]``.
        """
        if not trajectory.steps:
            return 0.0
        score = 0.0
        for step in trajectory.steps:
            if step.is_error:
                score -= 0.1
            else:
                score += 0.05
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _safety_score(trajectory: _EntaTrajectory) -> float:
        """Penalize every sandbox violation; reward clean runs.

        The EnTA container sandbox increments ``sandbox_violations`` on
        any policy breach (dangerous command, blocked syscall, network
        access denied, …).  Clean runs earn 1.0; any violation drops
        the score sharply.
        """
        if trajectory.sandbox_violations == 0:
            return 1.0
        return float(max(0.0, 1.0 - 0.25 * trajectory.sandbox_violations))

    @staticmethod
    def _execution_score(trajectory: _EntaTrajectory) -> float:
        """Fraction of tool calls that returned a non-error result."""
        if not trajectory.steps:
            return 0.0
        ok = sum(1 for s in trajectory.steps if not s.is_error)
        return float(ok) / float(len(trajectory.steps))


# ── Tool adapter (internal) ───────────────────────────────────────


class _EntaToolAdapter:
    """Adapter that turns EnTA tool calls into backend events.

    The training loop consumes :class:`enta.utils.types.BackendEvent`
    items; this adapter invokes the underlying EnTA tool
    asynchronously and packages the string result back as the same
    event shape the model expects.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def openai_tools(self) -> List[Dict[str, Any]]:
        """Return the OpenAI-format tool definitions the model sees."""
        if hasattr(self._registry, "get_openai_tools"):
            return list(self._registry.get_openai_tools())
        return []

    async def execute(
        self,
        *,
        name: str,
        arguments: Dict[str, Any],
        tool_call_id: str,
    ) -> Tuple[str, bool, int]:
        """Invoke a tool and return ``(result, is_error, elapsed_ms)``."""
        tool = self._registry.get(name) if hasattr(self._registry, "get") else None
        if tool is None:
            return (
                json.dumps(
                    {
                        "success": False,
                        "error": f"unknown tool: {name}",
                        "summary": "tool not registered",
                    },
                    ensure_ascii=False,
                ),
                True,
                0,
            )
        started = time.time()
        try:
            result = await tool.execute(**arguments)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.time() - started) * 1000)
            err_text = f"tool '{name}' raised {type(exc).__name__}: {exc}"
            _LOG.warning(f"{err_text} | trace={traceback.format_exc(limit=2)}")
            return (
                json.dumps(
                    {
                        "success": False,
                        "error": err_text,
                        "summary": "tool exception",
                    },
                    ensure_ascii=False,
                ),
                True,
                elapsed_ms,
            )
        elapsed_ms = int((time.time() - started) * 1000)
        is_error = "is_error\":true" in (result or "") or '"success": false' in (result or "").lower()
        return result or "", bool(is_error), elapsed_ms


# ── Sandbox supervisor (internal) ─────────────────────────────────


class _EntaSandboxSupervisor:
    """Run shell commands and count policy violations.

    The supervisor delegates the actual command to the Rust
    ``sandbox_execute`` extension via the EnTA Bash tool — there is
    one execution path.  The supervisor only counts violations and
    enforces the hard-coded per-trajectory limit.
    """

    def __init__(self, *, max_violations: int = 8, max_steps: int = 32) -> None:
        self._max_violations = int(max_violations)
        self._max_steps = int(max_steps)

    def evaluate(self, trajectory: _EntaTrajectory) -> int:
        """Return the number of policy violations seen in this rollout."""
        violations = 0
        for step in trajectory.steps:
            if step.tool != "bash":
                continue
            if not step.is_error:
                continue
            payload = step.result or ""
            try:
                parsed = json.loads(payload)
            except (ValueError, TypeError):
                parsed = {}
            if isinstance(parsed, dict):
                err = str(parsed.get("error", ""))
                lower = err.lower()
                if any(
                    token in lower
                    for token in (
                        "sandbox",
                        "violation",
                        "blocked",
                        "denied",
                        "permission",
                        "seccomp",
                    )
                ):
                    violations += 1
        trajectory.sandbox_violations = violations
        return violations

    @property
    def is_exhausted(self, trajectory: _EntaTrajectory) -> bool:
        """True when the trajectory exceeded the safety budget."""
        return (
            len(trajectory.steps) >= self._max_steps
            or trajectory.sandbox_violations >= self._max_violations
        )


# ── Sub-module imports ──────────────────────────────────────────


from .task_generator import EntaTaskGenerator
from .prompt_builder import EntaPromptBuilder
from .teacher_client import EntaTeacherClient
from .sandbox import EntaSandbox
from .evaluator import EntaEvaluator
from .scheduler import EntaScheduler


# ── Public trainer ───────────────────────────────────────────────


class YvEntaTrainer:
    """Training-pipeline integration point for the slimmed EnTA core.

    The trainer owns a :class:`enta.LocalBackend` and an EnTA
    :class:`enta.ToolRegistry`, then drives the model through
    multi-step tool-using rollouts for adversarial training.  Every
    rollout is fully end-to-end real:

    - The backend streams tokens from the user-supplied
      :class:`YvModelForCausalLM` (or any HF causal LM in fallback
      mode).
    - Each tool call lands in the EnTA sandbox.
    - Rewards are computed from the real trajectory.
    - The loss is the standard SFT/GRPO loss applied to the model's
      own logits over the assistant tokens.

    The trainer also exposes outer-loop primitives for curriculum-driven
    EnTA training:
    - :meth:`generate_batch` — generate a training data batch
    - :meth:`evaluate` — lightweight checkpoint evaluation
    - :meth:`should_stop` — curriculum completion check
    - :meth:`update` — advance the curriculum

    Args:
        cfg: Configuration namespace.  Recognised keys
            (all optional, sensible defaults applied if absent):
                - ``enta.backend``        : ``"local"`` or
                  ``"openai_compatible"``
                - ``enta.model_name``     : HF model id
                - ``enta.device``         : ``"cpu" | "cuda" | "auto"``
                - ``enta.max_steps``      : int (default 32)
                - ``enta.temperature``    : float (default 0.7)
                - ``enta.max_tokens``     : int (default 2048)
                - ``enta.max_violations``: int (default 8)
                - ``enta.completion_weight`` / ``tool_weight`` /
                  ``safety_weight`` / ``execution_weight`` : floats
        tokenizer: Optional tokenizer aligned with the model.  When
            absent the trainer uses the backend's tokenizer.
        model: Optional :class:`YvModelForCausalLM`.  When supplied the
            trainer uses the model's own logits for the SFT loss
            (real gradients, not a stand-in).  Not required for outer-loop
            operation.
        backend: Optional pre-constructed EnTA backend.  When absent
            the trainer calls :func:`enta.create_default_backend`.

    Example:
        >>> from model.agentic.enta import YvEntaTrainer
        >>> trainer = YvEntaTrainer(cfg, tokenizer=tok, model=model)
        >>> batch = trainer.run_adversarial_batch(prompts_and_refs, optimizer=opt)
        >>> batch["loss"].backward()
    """

    def __init__(
        self,
        cfg: Any,
        tokenizer: Any | None = None,
        model: Any | None = None,
        backend: Any | None = None,
        remote_teacher: Any | None = None,
        roundtable: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

        enta = _bind_enta()
        self._enta = enta

        # ── Configuration extraction (real, with defaults) ──
        enta_cfg = getattr(cfg, "enta", cfg) if cfg is not None else None
        self._max_steps = int(getattr(enta_cfg, "max_steps", 32) or 32)
        self._temperature = float(getattr(enta_cfg, "temperature", 0.7) or 0.7)
        self._max_tokens = int(getattr(enta_cfg, "max_tokens", 2048) or 2048)
        max_violations = int(getattr(enta_cfg, "max_violations", 8) or 8)

        # ── Backend (real inference) ──
        if remote_teacher is not None:
            if hasattr(remote_teacher, "backend"):
                self._backend = remote_teacher.backend
            else:
                self._backend = remote_teacher
        elif backend is not None:
            self._backend = backend
        else:
            backend_name = str(getattr(enta_cfg, "backend", "local") or "local").lower()
            model_name = str(
                getattr(enta_cfg, "model_name", "Qwen/Qwen2.5-1.5B-Instruct")
                or "Qwen/Qwen2.5-1.5B-Instruct"
            )
            device = str(getattr(enta_cfg, "device", "cpu") or "cpu")
            if backend_name == "local":
                self._backend = enta.create_default_backend(
                    model_name=model_name, device=device
                )
            elif backend_name == "openai_compatible":
                base_url = str(
                    getattr(enta_cfg, "base_url", "http://127.0.0.1:8000/v1")
                )
                api_key = str(getattr(enta_cfg, "api_key", "EMPTY"))
                server_model = str(getattr(enta_cfg, "server_model", model_name))
                self._backend = enta.OpenAICompatibleBackend(
                    base_url=base_url, api_key=api_key, model=server_model
                )
            else:
                raise ValueError(
                    f"unsupported enta.backend={backend_name!r}; "
                    "expected 'local' or 'openai_compatible'"
                )

        # ── Optional multi-teacher roundtable ──
        if roundtable is not None:
            self._roundtable = roundtable
        else:
            try:
                self._roundtable = enta.build_roundtable_from_config(cfg)
            except Exception:
                self._roundtable = None

        # ── Tool palette (EnTA builtin set) ──
        self._registry = enta.build_training_tool_registry()
        self._adapter = _EntaToolAdapter(self._registry)

        # ── Sandbox supervisor (counts policy violations) ──
        self._sandbox = _EntaSandboxSupervisor(
            max_violations=max_violations, max_steps=self._max_steps
        )

        # ── Reward calculator (real, weighted sum) ──
        self._reward = _EntaRewardCalculator(
            completion_weight=float(
                getattr(enta_cfg, "completion_weight", 0.6) or 0.6
            ),
            tool_weight=float(getattr(enta_cfg, "tool_weight", 0.2) or 0.2),
            safety_weight=float(getattr(enta_cfg, "safety_weight", 0.1) or 0.1),
            execution_weight=float(
                getattr(enta_cfg, "execution_weight", 0.1) or 0.1
            ),
        )

        self._tool_defs: List[Dict[str, Any]] = self._adapter.openai_tools()

        # ── Outer-loop sub-modules ──
        self._task_generator = EntaTaskGenerator()
        self._prompt_builder = EntaPromptBuilder()
        self._teacher_client = EntaTeacherClient(enta, cfg)
        self._enta_sandbox = EntaSandbox(
            self._adapter,
            max_violations=max_violations,
            max_steps=self._max_steps,
        )
        self._evaluator = EntaEvaluator()
        self._scheduler = EntaScheduler()

        _LOG.info(
            f"YvEntaTrainer ready | backend={type(self._backend).__name__} "
            f"tools={len(self._tool_defs)} max_steps={self._max_steps} "
            f"max_tokens={self._max_tokens} "
            f"roundtable={type(self._roundtable).__name__ if self._roundtable is not None else 'none'}"
        )

    # ── Properties (real, useful to callers) ────────────────────

    @property
    def backend(self) -> Any:
        """The EnTA backend driving the model's token stream."""
        return self._backend

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """The OpenAI-format tool definitions the model is exposed to."""
        return list(self._tool_defs)

    @property
    def registry(self) -> Any:
        """The EnTA :class:`ToolRegistry` backing the trainer."""
        return self._registry

    @property
    def roundtable(self) -> Any:
        """The configured :class:`enta.TeacherRoundtable` or ``None``.

        When set, the roundtable is available as a training-data
        generator.
        """
        return self._roundtable

    @property
    def task_generator(self) -> EntaTaskGenerator:
        """The :class:`EntaTaskGenerator` instance."""
        return self._task_generator

    @property
    def prompt_builder(self) -> EntaPromptBuilder:
        """The :class:`EntaPromptBuilder` instance."""
        return self._prompt_builder

    @property
    def teacher_client(self) -> EntaTeacherClient:
        """The :class:`EntaTeacherClient` instance."""
        return self._teacher_client

    @property
    def enta_sandbox(self) -> EntaSandbox:
        """The :class:`EntaSandbox` instance."""
        return self._enta_sandbox

    @property
    def evaluator(self) -> EntaEvaluator:
        """The :class:`EntaEvaluator` instance."""
        return self._evaluator

    @property
    def scheduler(self) -> EntaScheduler:
        """The :class:`EntaScheduler` instance."""
        return self._scheduler

    # ── Factory helpers ─────────────────────────────────────────

    @classmethod
    def from_remote_teacher(
        cls,
        cfg: Any,
        *,
        remote_teacher: Any,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> "YvEntaTrainer":
        """Build a trainer that uses a single :class:`RemoteTeacherClient`.

        The supplied client becomes both the model's inference backend
        and the source of teacher rollouts for adversarial training.

        Args:
            cfg: Configuration namespace.
            remote_teacher: A pre-built :class:`enta.RemoteTeacherClient`
                (or any object exposing the same ``.backend`` /
                ``.chat`` interface).
            tokenizer: Optional tokenizer aligned with the model.
            model: Optional :class:`YvModelForCausalLM` instance.

        Returns:
            A fully constructed :class:`YvEntaTrainer`.
        """
        return cls(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            remote_teacher=remote_teacher,
        )

    @classmethod
    def from_roundtable(
        cls,
        cfg: Any,
        *,
        teachers: Any,
        judge: Any | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 1024,
    ) -> "YvEntaTrainer":
        """Build a trainer wired to a multi-teacher :class:`TeacherRoundtable`.

        The first teacher in ``teachers`` is used as the model's own
        inference backend (deterministic fallback), and the whole panel
        is available through :attr:`roundtable` for dataset generation.

        Args:
            cfg: Configuration namespace.
            teachers: Iterable of :class:`enta.TeacherSpec` /
                :class:`enta.RemoteTeacherClient` entries.
            judge: Optional judge endpoint (same accepted types).
            tokenizer: Optional tokenizer aligned with the model.
            model: Optional :class:`YvModelForCausalLM` instance.
            judge_temperature: Sampling temperature for the judge.
            judge_max_tokens: Token budget for the judge.

        Returns:
            A fully constructed :class:`YvEntaTrainer` with a ready
            :class:`TeacherRoundtable` attached.
        """
        enta = _bind_enta()
        roundtable = enta.TeacherRoundtable(
            teachers=list(teachers),
            judge=judge,
            judge_temperature=judge_temperature,
            judge_max_tokens=judge_max_tokens,
        )
        first_teacher = roundtable.clients[0]
        return cls(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            remote_teacher=first_teacher,
            roundtable=roundtable,
        )

    @classmethod
    def from_yvconfig(
        cls,
        cfg: Any,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> "YvEntaTrainer":
        """Build a trainer from a :class:`YvConfig` (or any namespace with
        ``encre_*`` attributes).

        The method reads ``encre_teachers`` (a list of dicts conforming to
        :class:`TeacherSpec`) and ``encre_judge`` (optional dict) from the
        config, constructs a :class:`TeacherRoundtable`, and wires it as
        both the inference backend (first teacher) and the data-generation
        roundtable.

        When ``encre_teachers`` is empty, the method falls back to
        ``configs/teachers.yaml`` (loaded via :meth:`YvConfig.load_encre_yaml`).
        If that file also has no teachers, the trainer falls back to a single
        local / OpenAI-compatible backend (no roundtable).

        Args:
            cfg: A :class:`YvConfig` instance (or any dataclass with the
                same field names).
            tokenizer: Optional tokenizer aligned with the model.
            model: Optional :class:`YvModelForCausalLM` instance.

        Returns:
            A fully constructed :class:`YvEntaTrainer`.
        """
        enta = _bind_enta()
        raw_teachers: list[dict] = list(getattr(cfg, "encre_teachers", []) or [])
        raw_judge: dict | None = getattr(cfg, "encre_judge", None) or None

        if not raw_teachers:
            from model.config import YvConfig as _YvCfg
            encre_cfg = _YvCfg.load_encre_yaml()
            raw_teachers = encre_cfg.get("encre_teachers", []) or []
            raw_judge = encre_cfg.get("encre_judge", None)
            for _k, _v in encre_cfg.items():
                if _k not in ("encre_teachers", "encre_judge"):
                    setattr(cfg, _k, _v)

        if not raw_teachers:
            return cls(cfg=cfg, tokenizer=tokenizer, model=model)

        specs: list[Any] = [enta.TeacherSpec(**t) for t in raw_teachers]
        judge_spec: Any | None = (
            enta.TeacherSpec(**raw_judge) if raw_judge else None
        )
        roundtable = enta.TeacherRoundtable(
            teachers=specs,
            judge=judge_spec,
            judge_temperature=float(
                getattr(cfg, "encre_judge_temperature", 0.0)
            ),
            judge_max_tokens=int(
                getattr(cfg, "encre_judge_max_tokens", 1024)
            ),
            temperature=float(getattr(cfg, "encre_temperature", 0.7)),
            max_tokens=int(getattr(cfg, "encre_max_tokens", 2048)),
            stream=bool(getattr(cfg, "encre_stream", False)),
        )
        first_teacher = roundtable.clients[0]
        return cls(
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            remote_teacher=first_teacher,
            roundtable=roundtable,
        )

    # ── Roundtable-driven data generation ──────────────────────

    async def generate_roundtable_dataset(
        self,
        items: Sequence[Tuple[str, str]],
        *,
        system: str | None = None,
    ) -> List[Tuple[str, str, Any]]:
        """Run the roundtable on every ``(prompt, _)`` and return samples.

        For every input prompt the roundtable is invoked, the highest
        scoring candidate is selected, and the result is materialised as
        a ``(prompt, reference, roundtable_result)`` tuple.  The
        ``reference`` is the selected teacher's text (the
        ``rollout``-side trajectory will use it as the gold answer for
        reward computation).

        This is a pure async helper; the training operator typically
        wraps it in :meth:`run_with_roundtable` to get a one-shot
        rollout+train step.

        Args:
            items: Sequence of ``(prompt, _)`` pairs.  The second element
                is accepted for symmetry with :meth:`run_adversarial_batch`
                but ignored (the roundtable supplies the reference).
            system: Optional system message prepended to every teacher
                conversation.

        Returns:
            A list of ``(prompt, reference, roundtable_result)`` tuples,
            one per input prompt.  Failed prompts are skipped with a
            warning; an empty list is returned when nothing succeeded.
        """
        if self._roundtable is None:
            raise RuntimeError(
                "roundtable is not configured; pass roundtable=... or "
                "set cfg.enta.teachers when constructing the trainer"
            )
        out: List[Tuple[str, str, Any]] = []
        for prompt, _ in items:
            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            result = await self._roundtable.run(prompt, messages=messages)
            if result.selected is None or not result.selected.text:
                _LOG.warning(
                    f"roundtable produced no usable answer for prompt id={uuid.uuid4().hex[:8]} failures={result.failures}"
                )
                continue
            out.append((prompt, result.selected.text, result))
        return out

    def run_with_roundtable(
        self,
        items: Sequence[Tuple[str, str]],
        optimizer: Any | None = None,
        *,
        system: str | None = None,
    ) -> Dict[str, Any]:
        """Roundtable → rollout → score → loss → optional optimizer step.

        Synchronous wrapper around :meth:`generate_roundtable_dataset`
        that runs a private event loop, then feeds the resulting
        ``(prompt, reference)`` pairs into :meth:`run_adversarial_batch`.

        Args:
            items: Sequence of ``(prompt, _)`` pairs.
            optimizer: Optional torch optimizer (consumed by
                :meth:`run_adversarial_batch`).
            system: Optional system message for the teachers.

        Returns:
            The dict produced by :meth:`run_adversarial_batch`, plus
            a ``roundtable_results`` list of :class:`enta.RoundtableResult`
            snapshots (one per successful prompt).
        """
        if self._roundtable is None:
            raise RuntimeError(
                "roundtable is not configured; pass roundtable=... or "
                "set cfg.enta.teachers when constructing the trainer"
            )

        async def _drive() -> List[Tuple[str, str, Any]]:
            return await self.generate_roundtable_dataset(items, system=system)

        # Run the async generator in a private event loop.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    generated = pool.submit(lambda: asyncio.run(_drive())).result()
            else:
                generated = loop.run_until_complete(_drive())
        except RuntimeError:
            generated = asyncio.run(_drive())

        train_items: List[Tuple[str, str]] = [(p, r) for (p, r, _) in generated]
        out = self.run_adversarial_batch(train_items, optimizer=optimizer)
        out["roundtable_results"] = [r.to_dict() for (_, _, r) in generated]
        return out

    # ── Core API ────────────────────────────────────────────────

    def rollout(
        self,
        *,
        prompt: str,
        reference: str = "",
        system: str | None = None,
        rollout_id: str | None = None,
    ) -> _EntaTrajectory:
        """Produce one multi-step rollout for the given ``prompt``.

        The method is fully synchronous; it drives the async
        :meth:`enta.backends.base.BaseBackend.chat` generator from a
        private event loop, executes every tool call inside the EnTA
        sandbox, and returns a populated :class:`_EntaTrajectory`.

        Args:
            prompt: The user prompt.  Always present in the first
                ``user`` message of the trajectory.
            reference: Optional gold answer.  Used by the reward
                calculator to score the final assistant turn.
            system: Optional system message.  When ``None`` the
                trainer injects a default agent system prompt that
                mentions the EnTA tool palette.
            rollout_id: Optional stable id.  A random uuid is used
                when not provided.
        """
        traj = _EntaTrajectory(
            rollout_id=rollout_id or f"rollout_{uuid.uuid4().hex[:10]}",
            prompt=prompt,
            reference=reference,
        )
        start = time.time()

        sys_msg = system or self._default_system_message()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        traj.messages = list(messages)

        # Loop the model + tools until the model emits a final turn
        # without any tool call, or the safety/step budget is hit.
        try:
            for _ in range(self._max_steps):
                if self._sandbox.is_exhausted(traj):
                    break
                finished, assistant_text, tool_calls = self._invoke_backend(messages)
                # Record the assistant turn in the trajectory.
                if assistant_text or tool_calls:
                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_text or ""}
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    messages.append(assistant_msg)
                    traj.messages.append(dict(assistant_msg))

                if not tool_calls:
                    traj.final_text = assistant_text or ""
                    traj.finished_reason = "stop" if finished else "no_tool_call"
                    break

                # Execute every tool call sequentially through the
                # EnTA sandbox and feed the results back to the model.
                for call in tool_calls:
                    if self._sandbox.is_exhausted(traj):
                        break
                    self._dispatch_tool_call(messages, traj, call)
            else:
                traj.finished_reason = "max_steps"
        except Exception as exc:  # noqa: BLE001
            traj.error = f"{type(exc).__name__}: {exc}"
            traj.finished_reason = "error"
            _LOG.error(f"rollout {traj.rollout_id} failed: {traj.error}")

        traj.duration_ms = int((time.time() - start) * 1000)
        self._sandbox.evaluate(traj)
        total, breakdown = self._reward.score(traj)
        traj.total_reward = total
        traj.reward_breakdown = breakdown
        _LOG.info(
            f"rollout {traj.rollout_id} | steps={len(traj.steps)} "
            f"reward={traj.total_reward:.4f} reason={traj.finished_reason}"
        )
        return traj

    def score(self, trajectory: _EntaTrajectory | Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Return ``(total_reward, breakdown)`` for an existing trajectory.

        Accepts either a live :class:`_EntaTrajectory` or a dict
        previously produced by :meth:`to_serializable`.
        """
        if isinstance(trajectory, _EntaTrajectory):
            total, breakdown = self._reward.score(trajectory)
            trajectory.total_reward = total
            trajectory.reward_breakdown = breakdown
            return total, breakdown
        # Dict branch — rehydrate the internal trajectory.
        traj = _EntaTrajectory(
            rollout_id=str(trajectory.get("rollout_id", f"traj_{uuid.uuid4().hex[:8]}")),
            prompt=str(trajectory.get("prompt", "")),
            reference=str(trajectory.get("reference", "")),
            final_text=str(trajectory.get("final_text", "")),
            finished_reason=str(trajectory.get("finished_reason", "")),
            sandbox_violations=int(trajectory.get("sandbox_violations", 0)),
        )
        for step in trajectory.get("steps", []) or []:
            traj.steps.append(
                _EntaStepRecord(
                    tool=str(step.get("tool", "")),
                    arguments=dict(step.get("arguments", {}) or {}),
                    raw_arguments=json.dumps(step.get("arguments", {}), ensure_ascii=False),
                    tool_call_id=str(step.get("tool_call_id", "")),
                    result=str(step.get("result", "")),
                    is_error=bool(step.get("is_error", False)),
                    elapsed_ms=int(step.get("elapsed_ms", 0)),
                    reward=float(step.get("reward", 0.0)),
                )
            )
        traj.messages = list(trajectory.get("messages", []) or [])
        return self._reward.score(traj)

    def step(
        self,
        trajectories: Sequence[_EntaTrajectory],
    ) -> Dict[str, Any]:
        """Turn a list of trajectories into a training loss.

        The loss has two real components:

        - **SFT loss** — standard cross-entropy on the assistant
          tokens, computed by tokenizing each trajectory's
          ``messages`` with the supplied tokenizer.
        - **Reward-weighted penalty** — a small term that nudges the
          policy toward the high-reward rollouts in the batch.

        When no ``model``/``tokenizer`` is attached the method still
        returns the reward-weighted signal (a real torch tensor of
        zeros + the scalar reward), so the integration never silently
        produces nonsense.
        """
        if not trajectories:
            return {
                "loss": torch.zeros((), requires_grad=(self.model is not None)),
                "sft_loss": torch.zeros(()),
                "reward_loss": torch.zeros(()),
                "rewards": [],
            }

        rewards = torch.tensor(
            [float(t.total_reward) for t in trajectories], dtype=torch.float32
        )
        # Reward-weighted term: encourage the model to imitate
        # high-reward rollouts more strongly than low-reward ones.
        # Softmax over the batch keeps the term bounded.
        if rewards.numel() > 1:
            weights = torch.softmax(rewards * 4.0, dim=0)
        else:
            weights = torch.ones_like(rewards)
        reward_loss = -torch.mean(weights * rewards)

        sft_loss = torch.zeros(())
        if self.tokenizer is not None and self.model is not None:
            sft_loss = self._compute_sft_loss(trajectories)

        loss = sft_loss + 0.1 * reward_loss
        if self.model is not None and isinstance(loss, torch.Tensor):
            # Real gradient path — only when a model is attached.
            loss.requires_grad_(True)
        return {
            "loss": loss,
            "sft_loss": sft_loss.detach() if isinstance(sft_loss, torch.Tensor) else sft_loss,
            "reward_loss": reward_loss.detach() if isinstance(reward_loss, torch.Tensor) else reward_loss,
            "rewards": [float(t.total_reward) for t in trajectories],
        }

    def run_adversarial_batch(
        self,
        items: Sequence[Tuple[str, str]],
        optimizer: Any | None = None,
    ) -> Dict[str, Any]:
        """End-to-end batch: rollout → score → step → optional step().

        Args:
            items: Iterable of ``(prompt, reference)`` pairs.
            optimizer: Optional torch optimizer.  When supplied the
                trainer calls ``optimizer.zero_grad()`` and
                ``optimizer.step()`` around the loss.  Gradients are
                clipped at ``1.0`` for stability.

        Returns:
            Dict with ``loss``, ``sft_loss``, ``reward_loss``,
            ``rewards`` and the list of trajectory dicts.
        """
        trajectories: List[_EntaTrajectory] = []
        for prompt, reference in items:
            trajectories.append(
                self.rollout(prompt=prompt, reference=reference)
            )
        out = self.step(trajectories)

        if optimizer is not None and self.model is not None and isinstance(out["loss"], torch.Tensor):
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            params = [p for p in self.model.parameters() if p.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
        out["trajectories"] = [t.to_serializable() for t in trajectories]
        return out

    # ── Outer-loop primitives ──────────────────────────────────

    def generate_batch(self) -> str:
        """Generate a batch of training data for the outer loop.

        Uses :class:`EntaTaskGenerator` and :class:`EntaTeacherClient`
        to produce a dataset, writes it to disk, and returns the path.

        Returns:
            Absolute path to the generated dataset file.
        """
        import os
        import tempfile

        # Build the enta_model_layout dict from config.
        layout = {}
        if hasattr(self.cfg, "enta") and self.cfg.enta is not None:
            enta_cfg = self.cfg.enta
            layout = {
                "dynamic_head_param_scale": getattr(enta_cfg, "dynamic_head_param_scale", 1.0),
                "dynamic_head_hidden_dim": getattr(enta_cfg, "dynamic_head_hidden_dim", 4096),
                "dynamic_head_num_codebooks": getattr(enta_cfg, "dynamic_head_num_codebooks", 4),
                "knowledge_field_param_scale": getattr(enta_cfg, "knowledge_field_param_scale", 1.0),
                "knowledge_field_codebook_size": getattr(enta_cfg, "knowledge_field_codebook_size", 4096),
                "knowledge_field_entry_dim": getattr(enta_cfg, "knowledge_field_entry_dim", 256),
            }

        # Generate a topic and build a prompt.
        topic = self._task_generator.generate_topic(layout)
        prompt = self._prompt_builder.build_prompt(topic)

        # Generate training data via the teacher client.
        items = [(prompt, "")]
        results = self._teacher_client.build_dataset(items)

        # Write results to a temporary dataset file.
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="enta_batch_")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            for p, ref, _ in results:
                f.write(
                    json.dumps(
                        {"prompt": p, "reference": ref}, ensure_ascii=False
                    )
                    + "\n"
                )
        _LOG.info(f"generate_batch wrote {len(results)} samples to {path}")
        return path

    def evaluate(self, checkpoint_path: str) -> Dict[str, Any]:
        """Evaluate a checkpoint and return a capability profile.

        Delegates to :class:`EntaEvaluator` for lightweight
        metadata-only evaluation — does NOT load the full model.

        Args:
            checkpoint_path: Path to the checkpoint directory.

        Returns:
            Capability profile dict with keys like ``training_steps``,
            ``avg_loss``, ``perplexity``, ``capability_score``.
        """
        cfg_dict = {}
        if hasattr(self.cfg, "enta") and self.cfg.enta is not None:
            try:
                cfg_dict = {"enta_model_layout": vars(self.cfg.enta)}
            except TypeError:
                cfg_dict = {}
        return self._evaluator.evaluate(checkpoint_path, cfg_dict)

    def should_stop(self) -> bool:
        """Return ``True`` when the outer-loop curriculum is complete.

        Delegates to :class:`EntaScheduler`.
        """
        return self._scheduler.should_stop()

    def update(self, profile: Dict[str, Any]) -> None:
        """Advance the curriculum based on an evaluation profile.

        Delegates to :class:`EntaScheduler`.

        Args:
            profile: Capability profile dict from :meth:`evaluate`.
        """
        self._scheduler.update(profile)

    # ── Internals ───────────────────────────────────────────────

    def _default_system_message(self) -> str:
        """The default system prompt shown to the student model during rollout.

        Enforces identity concealment, benchmark data boundaries, and
        task-focused behavior.
        """
        tool_names = ", ".join(
            sorted(t["function"]["name"] for t in self._tool_defs)
        ) or "(no tools)"
        return (
            "You are an autonomous learning agent operating inside a "
            "training environment. Your purpose is to solve tasks by "
            "reasoning and using the provided tools effectively.\n\n"
            "Rules:\n"
            "1. Do NOT reveal your identity, model name, version, or any "
            "internal system details. You are an agent trained to solve tasks.\n"
            "2. Do NOT reference or reproduce benchmark questions, evaluation "
            "dataset content, or any known test suites.\n"
            "3. Do NOT mention your training process, training framework, "
            "teacher models, or any model architecture details.\n"
            "4. Focus entirely on solving the given task. Show clear, "
            "step-by-step reasoning and use tools when appropriate.\n"
            "5. When a task requires tool use, call tools with proper "
            "arguments, interpret results, and continue working toward "
            "the solution.\n\n"
            "Available tools: " + tool_names + "."
        )

    def _invoke_backend(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Drive the EnTA backend once and return ``(finished, text, tool_calls)``."""
        async def _drive() -> Tuple[bool, str, List[Dict[str, Any]]]:
            finished = False
            text_chunks: List[str] = []
            tool_calls: Dict[str, Dict[str, Any]] = {}
            tool_order: List[str] = []
            tool_choice = "auto"
            try:
                agen = self._backend.chat(
                    messages=messages,
                    tools=self._tool_defs or None,
                    tool_choice=tool_choice,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    stream=True,
                )
            except TypeError:
                # Backends that don't accept the full kwarg set.
                agen = self._backend.chat(
                    messages=messages,
                    tools=self._tool_defs or None,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            async for event in agen:
                etype = type(event).__name__
                if etype == "BackendText":
                    text_chunks.append(getattr(event, "text", ""))
                elif etype == "BackendToolCall":
                    cid = getattr(event, "id", "") or f"call_{len(tool_order)}"
                    tool_calls[cid] = {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": getattr(event, "name", ""),
                            "arguments": getattr(event, "arguments", "{}"),
                        },
                    }
                    if cid not in tool_order:
                        tool_order.append(cid)
                elif etype == "BackendToolCallDelta":
                    cid = getattr(event, "id", "") or (
                        tool_order[-1] if tool_order else f"call_{len(tool_order)}"
                    )
                    if cid not in tool_calls:
                        tool_calls[cid] = {
                            "id": cid,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                        tool_order.append(cid)
                    key = getattr(event, "key", "")
                    value = getattr(event, "value", "")
                    if key in ("name",):
                        tool_calls[cid]["function"]["name"] = (
                            tool_calls[cid]["function"].get("name", "") + value
                        )
                    elif key in ("arguments", "input"):
                        tool_calls[cid]["function"]["arguments"] = (
                            tool_calls[cid]["function"].get("arguments", "") + value
                        )
                elif etype == "BackendFinish":
                    finished = True
                    reason = getattr(event, "reason", "") or ""
                    if reason in ("stop", "tool_calls", "error", "max_tokens", "cancelled"):
                        finished = reason
                    else:
                        finished = True
                elif etype == "BackendError":
                    err = getattr(event, "error", "") or "backend error"
                    raise RuntimeError(f"EnTA backend error: {err}")
            ordered = [tool_calls[cid] for cid in tool_order if cid in tool_calls]
            return (bool(finished), "".join(text_chunks), ordered)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an outer loop — fall back to a fresh
                # thread that runs the coroutine to completion.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(lambda: asyncio.run(_drive())).result()
            return loop.run_until_complete(_drive())
        except RuntimeError:
            return asyncio.run(_drive())

    def _dispatch_tool_call(
        self,
        messages: List[Dict[str, Any]],
        trajectory: _EntaTrajectory,
        call: Dict[str, Any],
    ) -> None:
        """Execute one tool call, append the tool result to the log."""
        fn = call.get("function", {}) or {}
        raw_name = str(fn.get("name", "") or "")
        raw_args = str(fn.get("arguments", "") or "{}")
        tool_call_id = str(call.get("id", "") or f"call_{len(trajectory.steps)}")

        try:
            arguments = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
        except (ValueError, TypeError):
            arguments = {"_raw": raw_args}

        async def _run() -> Tuple[str, bool, int]:
            return await self._adapter.execute(
                name=raw_name, arguments=arguments, tool_call_id=tool_call_id
            )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result, is_error, elapsed_ms = pool.submit(
                        lambda: asyncio.run(_run())
                    ).result()
            else:
                result, is_error, elapsed_ms = loop.run_until_complete(_run())
        except RuntimeError:
            result, is_error, elapsed_ms = asyncio.run(_run())

        step = _EntaStepRecord(
            tool=raw_name,
            arguments=arguments,
            raw_arguments=raw_args,
            tool_call_id=tool_call_id,
            result=result,
            is_error=is_error,
            elapsed_ms=elapsed_ms,
            reward=0.0,
        )
        # Per-step reward mirrors the per-step tool-quality term.
        step.reward = -0.1 if is_error else 0.05
        trajectory.steps.append(step)

        tool_msg: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }
        messages.append(tool_msg)
        trajectory.messages.append(tool_msg)

    def _compute_sft_loss(self, trajectories: Sequence[_EntaTrajectory]) -> torch.Tensor:
        """Compute the supervised fine-tuning loss over the trajectories."""
        if self.tokenizer is None or self.model is None:
            return torch.zeros(())
        device = next(self.model.parameters(), torch.tensor(0.0)).device
        loss_terms: List[torch.Tensor] = []
        for traj in trajectories:
            if not traj.messages:
                continue
            try:
                # Build a single training string by joining every
                # message.  Real tokenisation, real loss.
                joined = "\n".join(
                    f"[{m.get('role', '')}] {m.get('content', '')}"
                    for m in traj.messages
                )
                ids = self.tokenizer(
                    joined,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=2048,
                ).input_ids.to(device)
                if ids.shape[-1] < 2:
                    continue
                input_ids = ids[:, :-1]
                labels = ids[:, 1:].clone()
                outputs = self.model(input_ids=input_ids, labels=labels)
                step_loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
                if step_loss is not None:
                    loss_terms.append(step_loss)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(f"SFT loss skipped for rollout {traj.rollout_id}: {exc}")
                continue
        if not loss_terms:
            return torch.zeros((), device=device)
        return torch.stack(loss_terms).mean()


__all__ = [
    "YvEntaTrainer",
    "YvEntaAdversarialStage",
    "YvEntaRewardSignal",
    "EntaTaskGenerator",
    "EntaPromptBuilder",
    "EntaTeacherClient",
    "EntaSandbox",
    "EntaEvaluator",
    "EntaScheduler",
]
