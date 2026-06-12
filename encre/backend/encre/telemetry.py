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
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from encre.logging_config import get_logger
from encre.crypto import encrypt, decrypt

logger = get_logger("encre.telemetry")


@dataclass
class ToolCallRecord:
    tool_name: str
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    error_message: str = ""


@dataclass
class TurnRecord:
    turn_number: int
    event_count: int
    latency_ms: float
    compact_triggered: bool = False
    token_usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetryRecord:
    """Record of a backend retry event."""
    attempt: int
    error_type: str  # "http_status", "exception"
    error_detail: str
    delay_s: float
    timestamp: float = field(default_factory=time.time)


class EncreTelemetry:
    def __init__(self, enabled: bool = True, session_id: str = "") -> None:
        self.enabled = enabled
        self.session_id = session_id or str(int(time.time() * 1000))
        self.tool_calls: list[ToolCallRecord] = []
        self.turns: list[TurnRecord] = []
        self.retries: list[RetryRecord] = []
        self._session_started_at: float = time.time()
        self._output_dir: str = ""

    def _ensure_output(self) -> None:
        if self._output_dir:
            return
        from encre.config import get_data_dir
        _dir = get_data_dir() / "telemetry"
        _dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = str(_dir)

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self._output_dir:
            self._ensure_output()
        try:
            _path = os.path.join(self._output_dir, f"{self.session_id}.jsonl")
            line = json.dumps(record, ensure_ascii=False)
            try:
                encrypted_line = encrypt(line)
            except Exception:
                encrypted_line = line
            with open(_path, "a", encoding="utf-8") as f:
                f.write(encrypted_line + "\n")
        except Exception:
            pass  # never crash on telemetry write failure

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool = True,
        tokens_used: int = 0,
        error_message: str = "",
    ) -> None:
        if not self.enabled:
            return
        record = ToolCallRecord(
            tool_name=tool_name,
            latency_ms=latency_ms,
            success=success,
            tokens_used=tokens_used,
            error_message=error_message,
        )
        self.tool_calls.append(record)
        entry = {
            "event": "tool_call",
            "session_id": self.session_id,
            "timestamp": record.timestamp,
            "tool_name": record.tool_name,
            "latency_ms": record.latency_ms,
            "success": record.success,
            "tokens_used": record.tokens_used,
            "error": record.error_message or None,
        }
        logger.debug(json.dumps(entry, ensure_ascii=False))
        self._write_jsonl(entry)
        self._update_cumulative_from_tool_call(tool_name)

    def record_turn(
        self,
        turn_number: int,
        event_count: int,
        latency_ms: float,
        compact_triggered: bool = False,
        token_usage: dict[str, int] | None = None,
        model: str = "",
    ) -> None:
        if not self.enabled:
            return
        record = TurnRecord(
            turn_number=turn_number,
            event_count=event_count,
            latency_ms=latency_ms,
            compact_triggered=compact_triggered,
            token_usage=token_usage or {},
            model=model,
        )
        self.turns.append(record)
        entry = {
            "event": "turn",
            "session_id": self.session_id,
            "timestamp": record.timestamp,
            "turn_number": record.turn_number,
            "event_count": record.event_count,
            "latency_ms": record.latency_ms,
            "compact_triggered": record.compact_triggered,
            "token_usage": record.token_usage,
            "model": record.model,
        }
        logger.debug(json.dumps(entry, ensure_ascii=False))
        self._write_jsonl(entry)
        self._update_cumulative_from_turn(token_usage)

    def record_retry(
        self,
        attempt: int,
        error_type: str,
        error_detail: str,
        delay_s: float,
    ) -> None:
        """Record a backend retry event."""
        if not self.enabled:
            return
        record = RetryRecord(
            attempt=attempt,
            error_type=error_type,
            error_detail=error_detail,
            delay_s=delay_s,
        )
        self.retries.append(record)
        entry = {
            "event": "retry",
            "timestamp": record.timestamp,
            "attempt": record.attempt,
            "error_type": record.error_type,
            "error_detail": record.error_detail,
            "delay_s": record.delay_s,
        }
        logger.warning(json.dumps(entry, ensure_ascii=False))
        self._write_jsonl(entry)

    def get_summary(self) -> dict[str, Any]:
        total_tool_calls = len(self.tool_calls)
        successful_tool_calls = sum(1 for t in self.tool_calls if t.success)
        failed_tool_calls = total_tool_calls - successful_tool_calls
        tool_latencies = [t.latency_ms for t in self.tool_calls]
        avg_tool_latency = sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0.0

        total_turns = len(self.turns)
        turn_latencies = [t.latency_ms for t in self.turns]
        avg_turn_latency = sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0.0
        total_events = sum(t.event_count for t in self.turns)
        compactions = sum(1 for t in self.turns if t.compact_triggered)
        session_duration_s = time.time() - self._session_started_at

        tool_usage: dict[str, int] = {}
        for t in self.tool_calls:
            tool_usage[t.tool_name] = tool_usage.get(t.tool_name, 0) + 1

        total_retries = len(self.retries)
        retry_by_type: dict[str, int] = {}
        for r in self.retries:
            retry_by_type[r.error_detail] = retry_by_type.get(r.error_detail, 0) + 1

        return {
            "session_duration_s": session_duration_s,
            "total_turns": total_turns,
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "failed_tool_calls": failed_tool_calls,
            "avg_tool_latency_ms": round(avg_tool_latency, 2),
            "avg_turn_latency_ms": round(avg_turn_latency, 2),
            "total_events": total_events,
            "compactions": compactions,
            "tool_usage": tool_usage,
            "total_retries": total_retries,
            "retry_by_error": retry_by_type,
        }

    def flush(self) -> dict[str, Any]:
        summary = self.get_summary()
        entry = {"event": "session_summary", "session_id": self.session_id, **summary}
        logger.debug(json.dumps(entry, ensure_ascii=False))
        self._write_jsonl(entry)
        return summary

    def reset(self) -> None:
        self.tool_calls.clear()
        self.turns.clear()
        self.retries.clear()
        self._session_started_at = time.time()

    # ── Persistent cumulative counters (survive session deletion) ─────

    @staticmethod
    def _cumulative_path() -> str:
        from encre.config import get_data_dir
        return str(get_data_dir() / "telemetry_cumulative.json")

    @staticmethod
    def _load_cumulative() -> dict[str, Any]:
        path = EncreTelemetry._cumulative_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            defaults = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tool_calls": 0,
                "tool_call_breakdown": {},
            }
            for k, v in defaults.items():
                data.setdefault(k, v)
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tool_calls": 0,
                "tool_call_breakdown": {},
            }

    @staticmethod
    def _save_cumulative(data: dict[str, Any]) -> None:
        try:
            path = EncreTelemetry._cumulative_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # never crash on telemetry write failure

    def _update_cumulative_from_turn(self, token_usage: dict[str, int] | None) -> None:
        if not token_usage:
            return
        cu = self._load_cumulative()
        cu["total_input_tokens"] += token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
        cu["total_output_tokens"] += token_usage.get("output_tokens", token_usage.get("completion_tokens", 0))
        self._save_cumulative(cu)

    def _update_cumulative_from_tool_call(self, tool_name: str) -> None:
        cu = self._load_cumulative()
        cu["total_tool_calls"] += 1
        cu["tool_call_breakdown"][tool_name] = cu["tool_call_breakdown"].get(tool_name, 0) + 1
        self._save_cumulative(cu)

    @staticmethod
    def get_all_sessions_usage() -> dict[str, Any]:
        """Aggregate telemetry data across all sessions.

        Core totals (tokens, tool calls) come from persistent cumulative
        counters that survive session deletion. Model breakdown and
        per-session detail come from scanning JSONL files.
        """
        # Load persistent cumulative counters
        cumulative = EncreTelemetry._load_cumulative()

        total_input = cumulative["total_input_tokens"]
        total_output = cumulative["total_output_tokens"]
        total_tokens = total_input + total_output
        total_tool_calls = cumulative["total_tool_calls"]
        tool_call_breakdown = dict(cumulative["tool_call_breakdown"])

        # Scan JSONL files for model breakdown and session list
        from encre.config import get_data_dir
        telemetry_dir = get_data_dir() / "telemetry"
        model_breakdown: dict[str, dict[str, int]] = {}
        session_summaries: dict[str, dict[str, Any]] = {}

        if telemetry_dir.is_dir():
            for fpath in sorted(telemetry_dir.glob("*.jsonl")):
                session_id = fpath.stem
                try:
                    lines = fpath.read_text("utf-8").strip().split("\n")
                except Exception:
                    continue

                session_input = 0
                session_output = 0
                session_tokens = 0
                session_tool_calls = 0
                session_tool_breakdown: dict[str, int] = {}
                session_model = ""
                turn_count = 0
                session_first_active = 0.0

                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        try:
                            decrypted = decrypt(line.strip())
                        except Exception:
                            decrypted = line.strip()
                        data = json.loads(decrypted)
                    except Exception:
                        continue

                    evt = data.get("event", "")
                    ts = data.get("timestamp", 0) or 0

                    # Capture the first event timestamp as session start time
                    if ts and session_first_active == 0:
                        session_first_active = ts

                    if evt == "turn":
                        tu = data.get("token_usage", {}) or {}
                        inp = tu.get("input_tokens", tu.get("prompt_tokens", 0)) or 0
                        out = tu.get("output_tokens", tu.get("completion_tokens", 0)) or 0
                        total = tu.get("total_tokens", inp + out) or 0
                        session_input += inp
                        session_output += out
                        session_tokens += total
                        turn_count += 1
                        model = data.get("model", "") or "unknown"
                        session_model = model
                        md = model_breakdown.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "turns": 0})
                        md["input_tokens"] += inp
                        md["output_tokens"] += out
                        md["total_tokens"] += total
                        md["turns"] += 1

                    elif evt == "tool_call":
                        tn = data.get("tool_name", "unknown")
                        session_tool_calls += 1
                        session_tool_breakdown[tn] = session_tool_breakdown.get(tn, 0) + 1

                    elif evt == "session_summary":
                        pass

                if session_tokens > 0 or session_tool_calls > 0:
                    # Fallback: use file mtime if no event timestamp found
                    if session_first_active == 0:
                        try:
                            session_first_active = fpath.stat().st_mtime
                        except Exception:
                            session_first_active = 0.0
                    session_summaries[session_id] = {
                        "session_id": session_id,
                        "model": session_model,
                        "input_tokens": session_input,
                        "output_tokens": session_output,
                        "total_tokens": session_tokens,
                        "turns": turn_count,
                        "tool_calls": session_tool_calls,
                        "tool_call_breakdown": session_tool_breakdown,
                        "first_active": session_first_active,
                    }

        # Build sorted session list (by first_active timestamp ascending) for time-series charts
        sorted_sessions = sorted(
            session_summaries.values(),
            key=lambda s: s.get("first_active", 0) or 0,
        )

        return {
            "total_sessions": len(session_summaries),
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tool_calls": total_tool_calls,
            "tool_call_breakdown": tool_call_breakdown,
            "model_breakdown": {
                m: v for m, v in sorted(
                    model_breakdown.items(),
                    key=lambda x: -x[1]["total_tokens"],
                )
            },
            "sessions": sorted_sessions,
        }
