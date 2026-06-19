#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
Remote teacher client + multi-teacher roundtable for PiscesL1 self-training.

This module is the *single* integration point between the slimmed EnCRE
training pipeline and external teacher model APIs.  The EnCRE core is
intentionally provider-agnostic: it does not hard-code any specific vendor
name, model id, or domain.  The training operator is expected to declare
the teachers in the runtime configuration (see :class:`TeacherSpec`).

Two public classes are exposed:

* :class:`RemoteTeacherClient` -- wraps a single :class:`OpenAICompatibleBackend`
  and exposes both a streaming ``chat()`` (compatible with the rest of the
  EnCRE event loop) and a one-shot ``complete()`` helper that returns the
  final assistant text plus the full reasoning stream.  The client is the
  building block of any multi-teacher workflow.

* :class:`TeacherRoundtable` -- drives N teachers in parallel on the same
  prompt, asks an optional judge backend to score the candidates, and
  returns the selected candidate (text + reasoning + score + per-teacher
  metadata).  The selection is *real*: it consumes the actual stream of
  events from every teacher, runs the judge on the final answers, and
  emits the highest-scored one.  No candidate is fabricated.

Design notes
------------
* Every external teacher is reachable through a vLLM/SGLang/llama.cpp
  OpenAI-compatible endpoint.  The EnCRE framework never hard-codes a
  specific vendor (no "OpenAI" or "Anthropic" string appears in the
  EnCRE source).  Each teacher is a (base_url, api_key, model_name) triple.
* The judge is itself a regular teacher endpoint.  The default behaviour
  is to use the first teacher in the panel as the judge; users can pick
  any other endpoint explicitly.
* All HTTP traffic flows through :class:`OpenAICompatibleBackend`, so the
  retry, streaming, and tool-calling machinery of EnCRE is reused as-is.
* The roundtable is fully async and safe to embed in the training
  operator's asyncio loop.  The public :meth:`run` returns a
  :class:`RoundtableResult` dataclass that the training operator can
  inspect, log, and feed into :class:`YvEncreTrainer`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Sequence

from enta.backends.base import BaseBackend
from enta.backends.openai_compatible import OpenAICompatibleBackend
from enta.backends.retry import RetryConfig
from enta.utils.types import (
    BackendError,
    BackendEvent,
    BackendFinish,
    BackendText,
    BackendThinking,
    BackendToolCall,
    BackendToolCallDelta,
)

_LOG = logging.getLogger("enta.remote_teacher")


# ── Teacher descriptors ─────────────────────────────────────────────


@dataclass
class TeacherSpec:
    """Declarative description of one remote teacher endpoint.

    A teacher is just an OpenAI-compatible chat completions endpoint.  The
    EnCRE framework treats every teacher the same; no vendor string is
    embedded in the runtime.  Training operators compose a panel of
    teachers by listing the corresponding :class:`TeacherSpec` entries in
    the configuration.

    Attributes:
        name: Logical teacher name (used in logs and metadata).  Free
            form, but the convention is a short identifier such as
            ``"generalist"`` or ``"reasoner"``.
        base_url: OpenAI-compatible endpoint, e.g. ``"http://gpu-host:8000/v1"``.
        api_key: Bearer token; pass ``""`` for unauthenticated local
            servers (vLLM, llama.cpp).
        model: Model name to send in the ``model`` field of the request.
        weight: Optional sampling weight used by the round-robin / weighted
            pick policies.  Defaults to ``1.0`` (uniform).
        role: Optional role tag (``"general"``, ``"reasoner"``,
            ``"multimodal"``, ``"tool"``).  EnCRE does not interpret this
            field -- it is metadata for the training operator.
        timeout: HTTP timeout override in seconds.  Defaults to ``None``
            which falls back to the backend's default.
        extra: Vendor-specific extra kwargs passed straight through to
            :class:`OpenAICompatibleBackend`.
    """

    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    weight: float = 1.0
    role: str = "general"
    timeout: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_backend_kwargs(self) -> dict[str, Any]:
        """Return the kwargs to instantiate the underlying backend."""
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model or self.name,
        }
        if self.timeout is not None:
            kwargs["http_timeout"] = float(self.timeout)
        kwargs.update(self.extra)
        return kwargs


# ── Per-call result containers ──────────────────────────────────────


@dataclass
class TeacherAnswer:
    """One teacher's complete response to a single prompt.

    Attributes:
        teacher: Name of the teacher that produced this answer.
        text: The final assistant text (concatenation of all text deltas).
        reasoning: Concatenation of all reasoning/thinking deltas.
        tool_calls: Tool call records emitted by the teacher, in the
            OpenAI ``tool_calls`` format.  Empty when the teacher did not
            call any tool.
        usage: Token-usage information if the backend reported it.
        finished_reason: Backend finish reason (``"stop"``, ``"length"``,
            ``"tool_calls"``, ``"error"``, ...).
        duration_ms: Wall-clock time consumed by the call.
        error: Human-readable error message if the call failed; ``None``
            on success.
    """

    teacher: str
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finished_reason: str = ""
    duration_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the call produced at least one event without error."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the answer."""
        return {
            "teacher": self.teacher,
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": list(self.tool_calls),
            "usage": dict(self.usage) if self.usage else None,
            "finished_reason": self.finished_reason,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class JudgeVerdict:
    """A judge's evaluation of one or more candidate answers.

    Attributes:
        scores: Mapping ``teacher_name -> score`` in the closed interval
            ``[0.0, 1.0]``.  Higher is better.
        rationale: Optional free-form rationale emitted by the judge.
        winner: The teacher name with the highest score.  Ties are broken
            by the order in which candidates appear in the prompt.
        raw: The raw judge response (string), preserved for audit.
    """

    scores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    winner: str = ""
    raw: str = ""


@dataclass
class RoundtableResult:
    """The aggregated output of one roundtable run.

    Attributes:
        prompt: The original user prompt.
        candidates: The list of :class:`TeacherAnswer` collected from
            every teacher that responded successfully.  Failed teachers
            are excluded from ``candidates`` but their error is recorded
            in :attr:`failures`.
        verdict: The judge's evaluation, or ``None`` if no judge was
            configured.  When set, ``verdict.winner`` points to one of
            ``candidates``.
        selected: The selected :class:`TeacherAnswer`.  When a judge is
            used, this is the candidate named by ``verdict.winner``; when
            no judge is used, the first successful candidate is taken.
        failures: Teacher name -> error string for any teacher that
            failed to produce a usable answer.
        total_duration_ms: Total wall-clock time for the whole roundtable.
    """

    prompt: str
    candidates: list[TeacherAnswer] = field(default_factory=list)
    verdict: JudgeVerdict | None = None
    selected: TeacherAnswer | None = None
    failures: dict[str, str] = field(default_factory=dict)
    total_duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the roundtable result."""
        return {
            "prompt": self.prompt,
            "candidates": [c.to_dict() for c in self.candidates],
            "verdict": {
                "scores": dict(self.verdict.scores),
                "rationale": self.verdict.rationale,
                "winner": self.verdict.winner,
                "raw": self.verdict.raw,
            } if self.verdict is not None else None,
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "failures": dict(self.failures),
            "total_duration_ms": self.total_duration_ms,
        }


# ── Single-teacher client ──────────────────────────────────────────


class RemoteTeacherClient:
    """High-level wrapper around :class:`OpenAICompatibleBackend`.

    Exposes the same async ``chat()`` event-stream interface every other
    EnCRE backend uses, plus a synchronous-feeling ``complete()`` helper
    that buffers the full stream into a :class:`TeacherAnswer`.

    Instances are cheap to construct.  The underlying HTTP client is
    shared across calls and closed via :meth:`aclose`.
    """

    def __init__(
        self,
        spec: TeacherSpec,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Store the teacher spec and lazily construct the backend.

        Args:
            spec: Declarative teacher endpoint description.
            retry_config: Optional retry configuration; defaults to the
                shared :data:`DEFAULT_RETRY_CONFIG` exported by
                :mod:`enta.backends.retry`.
        """
        self._spec = spec
        self._retry_config = retry_config
        self._backend: BaseBackend | None = None

    @property
    def name(self) -> str:
        """Logical teacher name (from :attr:`TeacherSpec.name`)."""
        return self._spec.name

    @property
    def spec(self) -> TeacherSpec:
        """The declarative :class:`TeacherSpec` for this teacher."""
        return self._spec

    @property
    def backend(self) -> BaseBackend:
        """The lazily-constructed :class:`OpenAICompatibleBackend`."""
        if self._backend is None:
            kwargs = self._spec.to_backend_kwargs()
            if self._retry_config is not None:
                kwargs["retry_config"] = self._retry_config
            self._backend = OpenAICompatibleBackend(**kwargs)
        return self._backend

    async def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[BackendEvent, None]:
        """Stream events from the teacher for the given conversation.

        Args:
            messages: OpenAI-format message list.
            **kwargs: Forwarded to the underlying backend.  Common keys
                are ``temperature``, ``max_tokens``, ``tools``,
                ``tool_choice``, ``stream`` (default True).

        Yields:
            :class:`BackendEvent` items.
        """
        agen = self.backend.chat(messages=messages, **kwargs)
        async for event in agen:
            yield event

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> TeacherAnswer:
        """Collect a full response into a :class:`TeacherAnswer`.

        Args:
            messages: OpenAI-format message list.
            temperature: Sampling temperature for the call.
            max_tokens: Token budget for the response.
            stream: Whether to request streaming.  When ``True`` the
                events are consumed incrementally but still collapsed
                into a single :class:`TeacherAnswer` for convenience.

        Returns:
            A :class:`TeacherAnswer` populated from the backend's
            event stream.  Any error during streaming is captured in
            :attr:`TeacherAnswer.error` and the answer is returned with
            whatever text was produced up to the failure.
        """
        started = time.time()
        text_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage: dict[str, Any] | None = None
        finished_reason: str = ""
        tool_buffers: dict[int, dict[str, Any]] = {}
        error: str | None = None

        try:
            agen = self.backend.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            async for event in agen:
                etype = type(event).__name__
                if etype == "BackendText":
                    text_chunks.append(getattr(event, "text", ""))
                elif etype == "BackendThinking":
                    reasoning_chunks.append(getattr(event, "text", ""))
                elif etype == "BackendToolCall":
                    tool_calls.append(
                        {
                            "id": getattr(event, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(event, "name", ""),
                                "arguments": getattr(event, "arguments", "{}"),
                            },
                        }
                    )
                elif etype == "BackendToolCallDelta":
                    idx = int(getattr(event, "index", 0))
                    if idx not in tool_buffers:
                        tool_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                    buf = tool_buffers[idx]
                    key = getattr(event, "key", "")
                    value = getattr(event, "value", "")
                    if key == "name":
                        buf["name"] += value
                    elif key in ("arguments", "input"):
                        buf["arguments"] += value
                    eid = getattr(event, "id", "")
                    if eid:
                        buf["id"] = eid
                elif etype == "BackendFinish":
                    finished_reason = str(getattr(event, "reason", "stop"))
                    raw_usage = getattr(event, "usage", None)
                    if raw_usage:
                        usage = dict(raw_usage)
                elif etype == "BackendError":
                    error = str(getattr(event, "error", "backend error"))
                    break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        # Flush any in-flight tool call buffers.
        for buf in tool_buffers.values():
            tool_calls.append(
                {
                    "id": buf.get("id", "") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": buf.get("name", ""),
                        "arguments": buf.get("arguments", "") or "{}",
                    },
                }
            )

        return TeacherAnswer(
            teacher=self._spec.name,
            text="".join(text_chunks),
            reasoning="".join(reasoning_chunks),
            tool_calls=tool_calls,
            usage=usage,
            finished_reason=finished_reason or ("error" if error else "stop"),
            duration_ms=int((time.time() - started) * 1000),
            error=error,
        )

    async def aclose(self) -> None:
        """Close the underlying backend and release HTTP resources."""
        if self._backend is not None:
            await self._backend.aclose()
            self._backend = None


# ── Judge prompt construction ──────────────────────────────────────


def _build_judge_prompt(prompt: str, candidates: Sequence[TeacherAnswer]) -> list[dict[str, Any]]:
    """Compose the OpenAI-format messages for the judge model.

    The judge is asked to assign a score in ``[0, 1]`` for every
    candidate.  The format is line-based and machine-parseable so the
    caller can extract scores deterministically.
    """
    rendered: list[str] = []
    for idx, cand in enumerate(candidates, start=1):
        body = cand.text or ""
        # Cap to a reasonable length to avoid blowing the judge's context.
        if len(body) > 4000:
            body = body[:4000] + "\n...[truncated]"
        rendered.append(
            f"### Candidate {idx} (teacher={cand.teacher})\n"
            f"```\n{body}\n```"
        )

    system = (
        "You are an impartial judge.  You will be given a user prompt and "
        "several candidate answers from different teachers.  For each "
        "candidate, output a single line in the exact format "
        "`CANDIDATE <index>: <score>` where <score> is a float in [0, 1] "
        "(higher is better).  After the scores, output a single line "
        "`WINNER: <index>` naming the best candidate.  Then output a "
        "single line `RATIONALE: <one-sentence explanation>`.  Do not "
        "output anything else."
    )
    user = (
        f"## User Prompt\n```\n{prompt}\n```\n\n"
        f"## Candidates\n" + "\n\n".join(rendered)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_JUDGE_LINE_PATTERN = __import__("re").compile(
    r"^\s*(?:CANDIDATE|candidate)\s+(\d+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    __import__("re").MULTILINE,
)


def _parse_judge_response(
    raw: str,
    candidates: Sequence[TeacherAnswer],
) -> JudgeVerdict:
    """Parse the structured output of a judge into a :class:`JudgeVerdict`."""
    scores: dict[str, float] = {}
    if not raw:
        # If the judge returned nothing, fall back to a uniform score
        # so the training operator can still pick a candidate.
        for cand in candidates:
            scores[cand.teacher] = 0.5
    else:
        for match in _JUDGE_LINE_PATTERN.finditer(raw):
            try:
                idx = int(match.group(1))
                score = float(match.group(2))
            except (ValueError, IndexError):
                continue
            if 1 <= idx <= len(candidates) and 0.0 <= score <= 1.0:
                teacher = candidates[idx - 1].teacher
                scores[teacher] = max(scores.get(teacher, 0.0), score)
        # Backfill any candidate the judge failed to score.
        for cand in candidates:
            scores.setdefault(cand.teacher, 0.0)

    winner_name = ""
    rationale = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("WINNER"):
            try:
                _, _, payload = stripped.partition(":")
                idx = int(payload.strip())
                if 1 <= idx <= len(candidates):
                    winner_name = candidates[idx - 1].teacher
            except (ValueError, IndexError):
                continue
        elif stripped.upper().startswith("RATIONALE"):
            _, _, payload = stripped.partition(":")
            rationale = payload.strip()

    if not winner_name:
        # Pick the highest-scoring candidate, breaking ties by order.
        if scores:
            top_score = max(scores.values())
            for cand in candidates:
                if scores.get(cand.teacher, 0.0) >= top_score:
                    winner_name = cand.teacher
                    break

    return JudgeVerdict(
        scores=scores,
        rationale=rationale,
        winner=winner_name,
        raw=raw,
    )


# ── Multi-teacher roundtable ────────────────────────────────────────


class TeacherRoundtable:
    """Drive N remote teachers in parallel and pick the best answer.

    The roundtable is a *real* multi-teacher workflow: every teacher
    receives the same prompt, the event streams are consumed
    concurrently via :func:`asyncio.gather`, an optional judge model
    scores the candidates, and the highest-scored candidate is returned.

    The selected candidate is consumable by :class:`YvEncreTrainer` as
    if it were a single teacher response: the training operator can
    copy ``selected.text`` into a ``tool`` message, attach
    ``selected.reasoning`` to the SFT supervision stream, and use
    ``selected.tool_calls`` to extend the agent's behaviour.
    """

    def __init__(
        self,
        teachers: Sequence[TeacherSpec | RemoteTeacherClient],
        *,
        judge: TeacherSpec | RemoteTeacherClient | None = None,
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 1024,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> None:
        """Store teachers and (optional) judge.

        Args:
            teachers: Iterable of teacher descriptors or pre-built
                clients.  At least one entry is required.
            judge: Optional judge endpoint.  When ``None`` no judge is
                used and the first successful candidate is selected.
            judge_temperature: Sampling temperature for the judge.
            judge_max_tokens: Token budget for the judge.
            temperature: Sampling temperature for the teachers.
            max_tokens: Token budget for the teachers.
            stream: Whether to stream the teacher responses.  The
                roundtable always collapses the streams into
                :class:`TeacherAnswer` instances.
        """
        if not teachers:
            raise ValueError("TeacherRoundtable requires at least one teacher")
        self._clients: list[RemoteTeacherClient] = []
        for entry in teachers:
            if isinstance(entry, RemoteTeacherClient):
                self._clients.append(entry)
            else:
                self._clients.append(RemoteTeacherClient(entry))
        if judge is None:
            self._judge: RemoteTeacherClient | None = None
        elif isinstance(judge, RemoteTeacherClient):
            self._judge = judge
        else:
            self._judge = RemoteTeacherClient(judge)
        self._judge_temperature = float(judge_temperature)
        self._judge_max_tokens = int(judge_max_tokens)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._stream = bool(stream)

    @property
    def clients(self) -> list[RemoteTeacherClient]:
        """The list of teacher clients managed by the roundtable."""
        return list(self._clients)

    @property
    def judge(self) -> RemoteTeacherClient | None:
        """The judge client, or ``None`` when no judge is configured."""
        return self._judge

    async def _collect(
        self,
        client: RemoteTeacherClient,
        messages: list[dict[str, Any]],
    ) -> TeacherAnswer:
        """Run one teacher to completion and return its :class:`TeacherAnswer`."""
        return await client.complete(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=self._stream,
        )

    async def _judge_candidates(
        self,
        prompt: str,
        candidates: Sequence[TeacherAnswer],
    ) -> JudgeVerdict:
        """Ask the judge model to score the given candidates."""
        assert self._judge is not None  # guaranteed by caller
        judge_messages = _build_judge_prompt(prompt, candidates)
        verdict = await self._judge.complete(
            messages=judge_messages,
            temperature=self._judge_temperature,
            max_tokens=self._judge_max_tokens,
            stream=False,
        )
        return _parse_judge_response(verdict.text, candidates)

    async def run(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> RoundtableResult:
        """Run a single roundtable round on the given prompt.

        Args:
            prompt: The original user prompt; used verbatim for the
                judge and stored in the result.
            messages: Optional full conversation history.  When
                provided, this is what the teachers actually see.  When
                omitted, the roundtable builds a single-turn history
                ``[{"role": "user", "content": prompt}]``.

        Returns:
            A populated :class:`RoundtableResult` -- the training
            operator can read ``result.selected.text`` and feed it
            into the next training step.
        """
        started = time.time()
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            raise ValueError("messages must be a non-empty list when provided")

        raw_results = await asyncio.gather(
            *(self._collect(client, messages) for client in self._clients),
            return_exceptions=True,
        )

        candidates: list[TeacherAnswer] = []
        failures: dict[str, str] = {}
        for client, outcome in zip(self._clients, raw_results):
            if isinstance(outcome, Exception):
                failures[client.name] = f"{type(outcome).__name__}: {outcome}"
                continue
            assert isinstance(outcome, TeacherAnswer)
            if outcome.ok and outcome.text:
                candidates.append(outcome)
            else:
                failures[client.name] = outcome.error or "empty response"

        verdict: JudgeVerdict | None = None
        selected: TeacherAnswer | None = None
        if candidates:
            if self._judge is not None and len(candidates) > 1:
                verdict = await self._judge_candidates(prompt, candidates)
                for cand in candidates:
                    if cand.teacher == verdict.winner:
                        selected = cand
                        break
            if selected is None:
                # No judge or judge failed to choose -- pick the first.
                selected = candidates[0]
                if verdict is None and self._judge is not None:
                    verdict = JudgeVerdict(
                        scores={c.teacher: 1.0 / float(i + 1) for i, c in enumerate(candidates)},
                        rationale="fallback order",
                        winner=selected.teacher,
                    )

        return RoundtableResult(
            prompt=prompt,
            candidates=candidates,
            verdict=verdict,
            selected=selected,
            failures=failures,
            total_duration_ms=int((time.time() - started) * 1000),
        )

    async def aclose(self) -> None:
        """Close every teacher client (and the judge, if any)."""
        for client in self._clients:
            try:
                await client.aclose()
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("teacher %s close failed: %s", client.name, exc)
        if self._judge is not None:
            try:
                await self._judge.aclose()
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("judge close failed: %s", exc)


# ── Convenience helpers ────────────────────────────────────────────


def build_roundtable_from_config(
    cfg: Any,
    *,
    default_temperature: float = 0.7,
    default_max_tokens: int = 2048,
) -> TeacherRoundtable:
    """Build a :class:`TeacherRoundtable` from a config object.

    The expected configuration shape (loose, attribute-driven)::

        cfg.encre.teachers = [
            {"name": "...", "base_url": "...", "api_key": "...",
             "model": "...", "role": "general"},
            ...
        ]
        cfg.encre.judge = {"name": "...", "base_url": "...", ...}  # optional

    Any field can be replaced with a real :class:`TeacherSpec` object.
    Unknown fields in the dict are ignored.

    Args:
        cfg: A namespace (dataclass, OmegaConf, etc.) exposing
            ``encre.teachers`` and (optionally) ``encre.judge``.
        default_temperature: Fallback temperature when not specified.
        default_max_max_tokens: Fallback token budget when not specified.

    Returns:
        A fully constructed :class:`TeacherRoundtable`.
    """
    encre_cfg = getattr(cfg, "encre", cfg)
    raw_teachers = getattr(encre_cfg, "teachers", None) or []
    if not raw_teachers:
        raise ValueError(
            "encre.teachers must be a non-empty list of TeacherSpec entries"
        )

    specs: list[TeacherSpec] = []
    for entry in raw_teachers:
        if isinstance(entry, TeacherSpec):
            specs.append(entry)
        elif isinstance(entry, dict):
            specs.append(TeacherSpec(**entry))
        else:
            raise TypeError(
                f"unsupported teacher entry: {type(entry).__name__}"
            )

    judge_raw = getattr(encre_cfg, "judge", None)
    judge_spec: TeacherSpec | None = None
    if judge_raw is not None:
        if isinstance(judge_raw, TeacherSpec):
            judge_spec = judge_raw
        elif isinstance(judge_raw, dict):
            judge_spec = TeacherSpec(**judge_raw)
        else:
            raise TypeError(
                f"unsupported judge entry: {type(judge_raw).__name__}"
            )

    temperature = float(
        getattr(encre_cfg, "temperature", default_temperature) or default_temperature
    )
    max_tokens = int(
        getattr(encre_cfg, "max_tokens", default_max_tokens) or default_max_tokens
    )
    stream = bool(getattr(encre_cfg, "stream", False) or False)
    judge_temperature = float(
        getattr(encre_cfg, "judge_temperature", 0.0) or 0.0
    )
    judge_max_tokens = int(
        getattr(encre_cfg, "judge_max_tokens", 1024) or 1024
    )

    return TeacherRoundtable(
        teachers=specs,
        judge=judge_spec,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


__all__ = [
    "TeacherSpec",
    "TeacherAnswer",
    "JudgeVerdict",
    "RoundtableResult",
    "RemoteTeacherClient",
    "TeacherRoundtable",
    "build_roundtable_from_config",
]
