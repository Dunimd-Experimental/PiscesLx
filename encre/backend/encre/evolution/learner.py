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
import re
import time
from dataclasses import dataclass, field
from typing import Any

from encre.crypto import encrypt, decrypt


@dataclass
class SuccessRecord:
    tool_name: str
    intent_signature: str
    param_pattern: str
    outcome: str
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    reuse_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "intent_signature": self.intent_signature,
            "param_pattern": self.param_pattern,
            "outcome": self.outcome,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "reuse_count": self.reuse_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SuccessRecord":
        return cls(
            tool_name=d["tool_name"],
            intent_signature=d.get("intent_signature", ""),
            param_pattern=d.get("param_pattern", ""),
            outcome=d.get("outcome", ""),
            latency_ms=d.get("latency_ms", 0.0),
            timestamp=d.get("timestamp", 0.0),
            reuse_count=d.get("reuse_count", 0),
        )


@dataclass
class ErrorRecord:
    tool_name: str
    error_type: str
    error_context: str
    correction: str
    timestamp: float = field(default_factory=time.time)
    trigger_count: int = 0
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "error_type": self.error_type,
            "error_context": self.error_context,
            "correction": self.correction,
            "timestamp": self.timestamp,
            "trigger_count": self.trigger_count,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ErrorRecord":
        return cls(
            tool_name=d["tool_name"],
            error_type=d.get("error_type", ""),
            error_context=d.get("error_context", ""),
            correction=d.get("correction", ""),
            timestamp=d.get("timestamp", 0.0),
            trigger_count=d.get("trigger_count", 0),
            resolved=d.get("resolved", False),
        )


class EncreEvolutionLearner:
    MAX_SUCCESS = 200
    MAX_ERRORS = 200
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, storage_path: str | None = None) -> None:
        self._successes: list[SuccessRecord] = []
        self._errors: list[ErrorRecord] = []
        if storage_path is None:
            from encre.config import get_data_dir
            _dir = get_data_dir() / "evolution"
            _dir.mkdir(parents=True, exist_ok=True)
            storage_path = str(_dir / "state.json")
        self._storage_path = storage_path
        self._tool_success_index: dict[str, list[int]] = {}
        self._tool_error_index: dict[str, list[int]] = {}

    # ── Recording ──────────────────────────────────────────────

    def record_success(
        self,
        tool_name: str,
        intent: str,
        params: dict[str, Any],
        outcome: str,
        latency_ms: float = 0.0,
    ) -> None:
        sig = _extract_signature(intent)
        param_str = _serialize_params(params)
        existing = self._find_similar_success(tool_name, sig, param_str)
        if existing >= 0:
            rec = self._successes[existing]
            rec.reuse_count += 1
            rec.latency_ms = (rec.latency_ms + latency_ms) / 2.0
            rec.timestamp = time.time()
        else:
            rec = SuccessRecord(
                tool_name=tool_name,
                intent_signature=sig,
                param_pattern=param_str,
                outcome=_truncate(outcome, 500),
                latency_ms=latency_ms,
            )
            idx = len(self._successes)
            self._successes.append(rec)
            self._tool_success_index.setdefault(tool_name, []).append(idx)
            self._prune_successes()

    def record_error(
        self,
        tool_name: str,
        error_type: str,
        context: str,
        correction: str,
    ) -> None:
        existing = self._find_similar_error(tool_name, error_type, context)
        if existing >= 0:
            rec = self._errors[existing]
            rec.trigger_count += 1
            rec.correction = correction
            rec.timestamp = time.time()
            rec.resolved = False
        else:
            rec = ErrorRecord(
                tool_name=tool_name,
                error_type=error_type,
                error_context=_truncate(context, 600),
                correction=_truncate(correction, 600),
            )
            idx = len(self._errors)
            self._errors.append(rec)
            self._tool_error_index.setdefault(tool_name, []).append(idx)
            self._prune_errors()

    def mark_error_resolved(self, tool_name: str, context: str) -> None:
        idx = self._find_similar_error(tool_name, "", context)
        if idx >= 0:
            self._errors[idx].resolved = True

    def record_correction(self, tool_name: str, error_context: str, correction: str) -> None:
        """Update the most recent matching open error with the fix that worked."""
        indices = self._tool_error_index.get(tool_name, [])
        best_idx = -1
        best_sim = 0.0
        for idx in indices:
            rec = self._errors[idx]
            if rec.resolved:
                continue
            sim = _token_similarity(rec.error_context, error_context)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim > self.SIMILARITY_THRESHOLD:
            rec = self._errors[best_idx]
            rec.correction = _truncate(correction, 600)
            rec.resolved = True
            rec.timestamp = time.time()

    # ── Retrieval ──────────────────────────────────────────────

    def get_guidance(self, tool_name: str, context: str) -> str:
        parts: list[str] = []
        # Errors first (critical)
        error_msgs = self._get_relevant_errors(tool_name, context)
        if error_msgs:
            parts.append("**Past mistakes to avoid:**")
            parts.extend(f"  - {m}" for m in error_msgs)
        # Success patterns
        success_msgs = self._get_relevant_successes(tool_name, context)
        if success_msgs:
            parts.append("**Proven approaches:**")
            parts.extend(f"  - {m}" for m in success_msgs)
        return "\n".join(parts)

    def get_tool_best_params(self, tool_name: str, intent: str) -> dict[str, Any] | None:
        sig = _extract_signature(intent)
        indices = self._tool_success_index.get(tool_name, [])
        best: SuccessRecord | None = None
        best_score = -1.0
        for idx in indices:
            rec = self._successes[idx]
            sim = _token_similarity(rec.intent_signature, sig)
            score = sim * (1.0 + rec.reuse_count * 0.3)
            if score > best_score and score > self.SIMILARITY_THRESHOLD:
                best_score = score
                best = rec
        if best is not None and best.param_pattern:
            try:
                return json.loads(best.param_pattern)
            except json.JSONDecodeError:
                return None
        return None

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_successes": len(self._successes),
            "total_errors": len(self._errors),
            "active_errors": sum(1 for e in self._errors if not e.resolved),
            "resolved_errors": sum(1 for e in self._errors if e.resolved),
            "top_tools_by_success": self._top_tools(self._tool_success_index, 5),
            "top_tools_by_error": self._top_tools(self._tool_error_index, 5),
        }

    # ── Internal ───────────────────────────────────────────────

    def _get_relevant_errors(self, tool_name: str, context: str) -> list[str]:
        indices = self._tool_error_index.get(tool_name, [])
        candidates: list[tuple[float, ErrorRecord]] = []
        for idx in indices:
            rec = self._errors[idx]
            if rec.resolved:
                continue
            sim = _token_similarity(rec.error_context, context)
            weight = sim * (1.0 + rec.trigger_count * 0.25)
            if weight > 0.3:
                candidates.append((weight, rec))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            f"[{r.error_type}] {_truncate(r.error_context, 150)} → {_truncate(r.correction, 150)}"
            for _, r in candidates[:4]
        ]

    def _get_relevant_successes(self, tool_name: str, context: str) -> list[str]:
        indices = self._tool_success_index.get(tool_name, [])
        candidates: list[tuple[float, SuccessRecord]] = []
        for idx in indices:
            rec = self._successes[idx]
            sim = _token_similarity(rec.intent_signature, context)
            weight = sim * (1.0 + rec.reuse_count * 0.2)
            if weight > 0.35:
                candidates.append((weight, rec))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            f"{_truncate(r.outcome, 200)}"
            for _, r in candidates[:3]
        ]

    def _find_similar_success(self, tool_name: str, sig: str, params: str) -> int:
        indices = self._tool_success_index.get(tool_name, [])
        for idx in indices:
            rec = self._successes[idx]
            if _token_similarity(rec.intent_signature, sig) > 0.7:
                if _token_similarity(rec.param_pattern, params) > 0.5:
                    return idx
        return -1

    def _find_similar_error(self, tool_name: str, error_type: str, context: str) -> int:
        indices = self._tool_error_index.get(tool_name, [])
        for idx in indices:
            rec = self._errors[idx]
            if error_type and rec.error_type != error_type:
                continue
            if _token_similarity(rec.error_context, context) > self.SIMILARITY_THRESHOLD:
                return idx
        return -1

    def _prune_successes(self) -> None:
        if len(self._successes) > self.MAX_SUCCESS:
            sorted_idx = sorted(
                range(len(self._successes)),
                key=lambda i: (self._successes[i].reuse_count, self._successes[i].timestamp),
            )
            to_remove = set(sorted_idx[:len(self._successes) - self.MAX_SUCCESS])
            self._successes = [r for i, r in enumerate(self._successes) if i not in to_remove]
            self._rebuild_success_index()

    def _prune_errors(self) -> None:
        if len(self._errors) > self.MAX_ERRORS:
            sorted_idx = sorted(
                range(len(self._errors)),
                key=lambda i: (self._errors[i].trigger_count, self._errors[i].timestamp),
            )
            to_remove = set(sorted_idx[:len(self._errors) - self.MAX_ERRORS])
            self._errors = [r for i, r in enumerate(self._errors) if i not in to_remove]
            self._rebuild_error_index()

    def _rebuild_success_index(self) -> None:
        self._tool_success_index.clear()
        for i, rec in enumerate(self._successes):
            self._tool_success_index.setdefault(rec.tool_name, []).append(i)

    def _rebuild_error_index(self) -> None:
        self._tool_error_index.clear()
        for i, rec in enumerate(self._errors):
            self._tool_error_index.setdefault(rec.tool_name, []).append(i)

    @staticmethod
    def _top_tools(index: dict[str, list[int]], n: int) -> list[dict[str, Any]]:
        counts = [(name, len(indices)) for name, indices in index.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        return [{"tool_name": name, "count": c} for name, c in counts[:n]]

    # ── Persistence ────────────────────────────────────────────

    def save(self) -> None:
        data = {
            "successes": [r.to_dict() for r in self._successes],
            "errors": [r.to_dict() for r in self._errors],
        }
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        try:
            payload = encrypt(payload)
        except Exception:
            pass
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as f:
            f.write(payload)

    def load(self) -> bool:
        if not os.path.exists(self._storage_path):
            return False
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw and not raw.startswith("{"):
                try:
                    raw = decrypt(raw)
                except Exception:
                    pass
            data = json.loads(raw)
            self._successes = [SuccessRecord.from_dict(d) for d in data.get("successes", [])]
            self._errors = [ErrorRecord.from_dict(d) for d in data.get("errors", [])]
            self._rebuild_success_index()
            self._rebuild_error_index()
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self._successes.clear()
        self._errors.clear()
        self._tool_success_index.clear()
        self._tool_error_index.clear()


# ── Helpers ─────────────────────────────────────────────────────

def _extract_signature(text: str) -> str:
    tokens = re.findall(r'[a-zA-Z_]\w*', text.lower())
    return " ".join(tokens[:60])


def _serialize_params(params: dict[str, Any]) -> str:
    safe: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 300:
            safe[k] = v[:300] + "..."
        else:
            safe[k] = v
    return json.dumps(safe, ensure_ascii=False, sort_keys=True)


def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(re.findall(r'[a-zA-Z_]\w*', a.lower()))
    tokens_b = set(re.findall(r'[a-zA-Z_]\w*', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
