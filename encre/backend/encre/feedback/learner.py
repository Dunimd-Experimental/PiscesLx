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

import json
import os
import re
import time
from dataclasses import dataclass, field


@dataclass
class CorrectionRecord:
    tool_name: str
    error_type: str
    error_context: str
    user_correction: str
    timestamp: float = field(default_factory=time.time)
    trigger_count: int = 0
    missed_count: int = 0
    stale: bool = False

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "error_type": self.error_type,
            "error_context": self.error_context,
            "user_correction": self.user_correction,
            "timestamp": self.timestamp,
            "trigger_count": self.trigger_count,
            "missed_count": self.missed_count,
            "stale": self.stale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CorrectionRecord:
        return cls(
            tool_name=d["tool_name"],
            error_type=d["error_type"],
            error_context=d["error_context"],
            user_correction=d["user_correction"],
            timestamp=d.get("timestamp", 0.0),
            trigger_count=d.get("trigger_count", 0),
            missed_count=d.get("missed_count", 0),
            stale=d.get("stale", False),
        )


class EncreFeedbackLearner:
    MAX_RECORDS: int = 100
    STALE_THRESHOLD: int = 5
    REMOVE_THRESHOLD: int = 10

    def __init__(self, storage_path: str | None = None) -> None:
        self._records: list[CorrectionRecord] = []
        if storage_path is None:
            from encre.config import get_data_dir
            _dir = get_data_dir() / "feedback"
            _dir.mkdir(parents=True, exist_ok=True)
            storage_path = str(_dir / "corrections.json")
        self._storage_path: str = storage_path
        self._tool_index: dict[str, list[int]] = {}

    def record_correction(
        self,
        tool_name: str,
        error_type: str,
        error_context: str,
        user_correction: str,
    ) -> None:
        existing_idx = self._find_similar(tool_name, error_type, error_context)
        if existing_idx >= 0:
            rec = self._records[existing_idx]
            rec.trigger_count += 1
            rec.missed_count = max(0, rec.missed_count - 1)
            rec.stale = False
            rec.user_correction = user_correction
            rec.timestamp = time.time()
        else:
            rec = CorrectionRecord(
                tool_name=tool_name,
                error_type=error_type,
                error_context=error_context,
                user_correction=user_correction,
            )
            idx = len(self._records)
            self._records.append(rec)
            if tool_name not in self._tool_index:
                self._tool_index[tool_name] = []
            self._tool_index[tool_name].append(idx)
            self._prune()

    def _find_similar(self, tool_name: str, error_type: str, context: str) -> int:
        indices = self._tool_index.get(tool_name, [])
        for idx in indices:
            rec = self._records[idx]
            if rec.error_type != error_type:
                continue
            if rec.stale:
                continue
            if self._context_similarity(rec.error_context, context) > 0.6:
                return idx
        return -1

    def _context_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_tokens = set(re.findall(r'[a-zA-Z_]\w*', a.lower()))
        b_tokens = set(re.findall(r'[a-zA-Z_]\w*', b.lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens
        jaccard = len(intersection) / len(union)
        if len(a_tokens) > 5:
            longest = 0
            for token_a in a_tokens:
                for token_b in b_tokens:
                    if token_a == token_b:
                        longest = max(longest, len(token_a))
            length_bonus = min(longest / 8.0, 0.3)
            jaccard += length_bonus
        return min(jaccard, 1.0)

    def get_relevant_feedback(self, tool_name: str, context: str) -> str:
        indices = self._tool_index.get(tool_name, [])
        if not indices:
            return ""
        candidates: list[tuple[float, CorrectionRecord]] = []
        for idx in indices:
            rec = self._records[idx]
            if rec.stale:
                continue
            sim = self._context_similarity(rec.error_context, context)
            weight = sim * (1.0 + rec.trigger_count * 0.2)
            if weight > 0.3:
                candidates.append((weight, rec))
        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:3]
        lines: list[str] = ["Previous errors and corrections to keep in mind:"]
        for _, rec in top:
            lines.append(
                f"- When using [{rec.tool_name}], avoid: {cut_str(rec.error_context, 200)}. "
                f"Instead: {cut_str(rec.user_correction, 200)}"
            )
        return "\n".join(lines)

    def _apply_decay(self) -> None:
        now = time.time()
        for rec in self._records:
            if rec.stale:
                continue
            age_days = (now - rec.timestamp) / 86400.0
            if age_days > 30.0:
                rec.missed_count += 1
                if rec.missed_count >= self.STALE_THRESHOLD:
                    rec.stale = True
        self._cleanup_stale()

    def _cleanup_stale(self) -> None:
        stale_indices = {i for i, rec in enumerate(self._records) if rec.stale}
        if not stale_indices:
            return
        if len(stale_indices) >= self.REMOVE_THRESHOLD or len(self._records) > self.MAX_RECORDS:
            self._records = [rec for rec in self._records if not rec.stale]
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._tool_index.clear()
        for idx, rec in enumerate(self._records):
            if rec.tool_name not in self._tool_index:
                self._tool_index[rec.tool_name] = []
            self._tool_index[rec.tool_name].append(idx)

    def _prune(self) -> None:
        if len(self._records) > self.MAX_RECORDS:
            self._apply_decay()
        if len(self._records) > self.MAX_RECORDS:
            sorted_records = sorted(
                [(i, rec) for i, rec in enumerate(self._records)],
                key=lambda x: (x[1].trigger_count, -x[1].timestamp),
            )
            to_remove = set(i for i, _ in sorted_records[:len(self._records) - self.MAX_RECORDS])
            self._records = [rec for i, rec in enumerate(self._records) if i not in to_remove]
            self._rebuild_index()

    def save(self) -> None:
        self._apply_decay()
        data = [rec.to_dict() for rec in self._records]
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        if not os.path.exists(self._storage_path):
            return False
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._records = [CorrectionRecord.from_dict(d) for d in data]
            self._rebuild_index()
            self._apply_decay()
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self._records.clear()
        self._tool_index.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def active_count(self) -> int:
        return sum(1 for rec in self._records if not rec.stale)


def cut_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
