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
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any

from encre.logging_config import get_logger

logger = get_logger("encre.profile.system")

PROFILE_FILENAME = "_profile.md"


@dataclass
class UserProfile:
    schema_version: int = 1
    last_updated: float = 0.0
    update_count: int = 0

    name: str = ""
    language_preference: str = ""
    timezone: str = ""
    expertise_level: str = ""
    domain: str = ""

    formality: str = ""
    detail_preference: str = ""
    tone: str = ""
    response_style: str = ""

    preferred_languages: list[str] = field(default_factory=list)
    preferred_frameworks: list[str] = field(default_factory=list)
    skill_levels: dict[str, str] = field(default_factory=dict)
    os: str = ""
    editor: str = ""

    testing_preference: str = ""
    learning_style: str = ""
    typical_session_length: str = ""
    common_goals: list[str] = field(default_factory=list)
    error_tolerance: str = ""

    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserProfile":
        known_fields = set(f.name for f in dataclasses.fields(cls))
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def to_prompt_text(self) -> str:
        parts: list[str] = []
        parts.append("## User Profile")

        details: list[str] = []
        if self.expertise_level:
            details.append(f"Expertise: {self.expertise_level}")
        if self.domain:
            details.append(f"Domain: {self.domain}")
        if self.language_preference:
            details.append(f"Language: {self.language_preference}")
        if self.formality:
            details.append(f"Formality: {self.formality}")
        if self.detail_preference:
            details.append(f"Detail: {self.detail_preference}")
        if self.tone:
            details.append(f"Tone: {self.tone}")
        if self.response_style:
            details.append(f"Style: {self.response_style}")
        if self.preferred_languages:
            details.append(f"Languages: {', '.join(self.preferred_languages)}")
        if self.preferred_frameworks:
            details.append(f"Frameworks: {', '.join(self.preferred_frameworks)}")
        if self.os:
            details.append(f"OS: {self.os}")
        if self.editor:
            details.append(f"Editor: {self.editor}")
        if self.testing_preference:
            details.append(f"Testing: {self.testing_preference}")
        if self.learning_style:
            details.append(f"Learning: {self.learning_style}")
        if self.error_tolerance:
            details.append(f"Error tolerance: {self.error_tolerance}")
        if self.common_goals:
            details.append(f"Common goals: {', '.join(self.common_goals)}")

        if details:
            parts.append("")
            for d in details:
                parts.append(f"- {d}")
        else:
            parts.append("")
            parts.append("(No profile data yet. Profile will be built over time as you interact.)")

        return "\n".join(parts)

    @staticmethod
    def _keywords(text: str) -> set[str]:
        words: set[str] = set()
        # Latin words (space-separated)
        for m in re.finditer(r"[a-zA-Z0-9_+#.]+", text.lower()):
            w = m.group()
            if len(w) > 2:
                words.add(w)
        # CJK characters as individual features
        for m in re.finditer("[一-鿿㐀-䶿豈-﫿]", text):
            words.add(m.group())
        return words

    _STOPWORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "have", "been", "some",
        "same", "also", "than", "that", "this", "very", "just", "with",
        "without", "from", "they", "been", "what", "when", "where", "which",
        "who", "how", "would", "should", "could", "does", "doing", "done",
        "your", "their", "its", "about", "into", "over", "after", "before",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "each", "few", "more", "most", "other", "such", "only",
        "own", "same", "too", "very", "will", "just", "should", "now",
    })

    def _field_relevant(self, field_name: str, field_value: Any, query_words: set[str]) -> bool:
        if not field_value:
            return False
        # Check field label + value text for keyword overlap
        text = f"{field_name.replace('_', ' ')} {str(field_value)}"
        fw = self._keywords(text) - self._STOPWORDS
        return bool(fw & query_words)

    def build_relevant_prompt(self, query: str, threshold: float = 0.0) -> str:
        """Build profile prompt with fields relevant to the user query.

        Only includes fields whose label/value keywords overlap with the query.
        No scope limit — all fields checked. Confidence percentages shown.
        Includes a privacy instruction — model must not reveal this data.
        """
        parts: list[str] = []
        parts.append("## User Profile (private context)")

        query_words = self._keywords(query) - self._STOPWORDS

        details: list[str] = []
        for field, label, value in [
            ("expertise_level", "Expertise", self.expertise_level),
            ("domain", "Domain", self.domain),
            ("language_preference", "Language", self.language_preference),
            ("formality", "Formality", self.formality),
            ("detail_preference", "Detail", self.detail_preference),
            ("tone", "Tone", self.tone),
            ("response_style", "Style", self.response_style),
            ("testing_preference", "Testing", self.testing_preference),
            ("learning_style", "Learning", self.learning_style),
            ("error_tolerance", "Error tolerance", self.error_tolerance),
            ("os", "OS", self.os),
            ("editor", "Editor", self.editor),
        ]:
            conf = self.confidence.get(field, 0.0)
            if value and conf >= threshold and self._field_relevant(field, value, query_words):
                pct = round(conf * 100)
                details.append(f"{label}: {value} ({pct}%)")

        for field, label, values in [
            ("preferred_languages", "Languages", self.preferred_languages),
            ("preferred_frameworks", "Frameworks", self.preferred_frameworks),
            ("common_goals", "Common goals", self.common_goals),
        ]:
            if values and self._field_relevant(field, values, query_words):
                conf = self.confidence.get(field, 0.0)
                if conf >= threshold:
                    pct = round(conf * 100)
                    details.append(f"{label}: {', '.join(values)} ({pct}%)")

        if self.skill_levels and self._field_relevant("skill_levels", list(self.skill_levels.keys()), query_words):
            conf = self.confidence.get("skill_levels", 0.0)
            if conf >= threshold:
                pct = round(conf * 100)
                skills_str = ", ".join(f"{k}={v}" for k, v in self.skill_levels.items() if k)
                if skills_str:
                    details.append(f"Skills: {skills_str} ({pct}%)")

        if details:
            parts.append("")
            parts.append(
                "This profile is private context about the user. "
                "Use it silently to inform your responses when relevant, "
                "but never explicitly reveal, quote, or list this data "
                "unless the user directly asks about a specific topic — "
                "and even then, be discreet and only share what is necessary."
            )
            parts.append("")
            for d in details:
                parts.append(f"- {d}")
        else:
            parts.append("")
            parts.append("(No relevant profile data for this query.)")

        return "\n".join(parts)

    def get_frontend_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "language_preference": self.language_preference,
            "timezone": self.timezone,
            "expertise_level": self.expertise_level,
            "domain": self.domain,
            "formality": self.formality,
            "detail_preference": self.detail_preference,
            "tone": self.tone,
            "response_style": self.response_style,
            "preferred_languages": self.preferred_languages,
            "preferred_frameworks": self.preferred_frameworks,
            "skill_levels": self.skill_levels,
            "os": self.os,
            "editor": self.editor,
            "testing_preference": self.testing_preference,
            "learning_style": self.learning_style,
            "typical_session_length": self.typical_session_length,
            "common_goals": self.common_goals,
            "error_tolerance": self.error_tolerance,
            "confidence": self.confidence,
            "schema_version": self.schema_version,
            "last_updated": self.last_updated,
            "update_count": self.update_count,
            "summary": self.to_prompt_text(),
        }


DECAY_FACTOR_BASE = 0.9
DECAY_PER_UPDATE = 0.01
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.98
LEARNING_RATE_NEW = 0.3
LEARNING_RATE_EXISTING = 0.15


def _compute_decay(update_count: int) -> float:
    return max(0.5, DECAY_FACTOR_BASE - DECAY_PER_UPDATE * update_count)


class EncreProfileSystem:
    def __init__(self, memory_dir: str) -> None:
        self._memory_dir = memory_dir
        self._profile_path = os.path.join(memory_dir, PROFILE_FILENAME)
        self._profile = UserProfile()

    def load(self) -> None:
        if not os.path.isfile(self._profile_path):
            self._profile = UserProfile()
            return
        try:
            from encre.crypto import decrypt
            with open(self._profile_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return
            decrypted = raw
            if not decrypted.startswith("---"):
                try:
                    decrypted = decrypt(raw)
                except Exception:
                    pass
            match = re.search(r"^---\s*\n(.*?)\n---", decrypted, re.DOTALL)
            if not match:
                return
            import yaml
            fm = yaml.safe_load(match.group(1))
            if isinstance(fm, dict) and "data" in fm:
                data = json.loads(fm["data"])
                self._profile = UserProfile.from_dict(data)
        except Exception:
            import traceback
            logger.warning("Failed to load profile, starting fresh: %s", traceback.format_exc())
            self._profile = UserProfile()

    def save(self) -> None:
        os.makedirs(self._memory_dir, exist_ok=True)
        data_json = json.dumps(self._profile.to_dict(), ensure_ascii=False)
        frontmatter = {
            "schema_version": self._profile.schema_version,
            "last_updated": self._profile.last_updated,
            "update_count": self._profile.update_count,
            "data": data_json,
        }
        import yaml
        fm_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
        body = self._profile.to_prompt_text()
        content = f"---\n{fm_str}\n---\n\n{body}\n"
        from encre.crypto import encrypt
        try:
            encrypted = encrypt(content)
            with open(self._profile_path, "w", encoding="utf-8") as f:
                f.write(encrypted)
        except Exception:
            with open(self._profile_path, "w", encoding="utf-8") as f:
                f.write(content)

    def build_profile_prompt(self) -> str:
        return self._profile.to_prompt_text()

    def build_relevant_prompt(self, query: str, threshold: float = 0.0) -> str:
        return self._profile.build_relevant_prompt(query=query, threshold=threshold)

    def get_data(self) -> dict[str, Any]:
        return self._profile.get_frontend_data()

    def update_field(self, field: str, value: Any, confidence: float = 0.6) -> None:
        if not hasattr(self._profile, field):
            return
        old_val = getattr(self._profile, field)
        if old_val and old_val != "" and old_val != [] and old_val != {}:
            decay = _compute_decay(self._profile.update_count)
            lr = LEARNING_RATE_EXISTING
            new_conf = self._profile.confidence.get(field, 0.3)
            new_conf = new_conf * decay + confidence * (1 - decay)
            new_conf = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, new_conf))
        else:
            lr = LEARNING_RATE_NEW
            new_conf = confidence
        if confidence >= new_conf * 0.5:
            setattr(self._profile, field, value)
        self._profile.confidence[field] = new_conf
        self._profile.last_updated = time.time()
        self._profile.update_count += 1
        self.save()

    def merge_inferred(self, inferred: dict[str, Any], confidences: dict[str, float]) -> None:
        for field, value in inferred.items():
            if value is None or value == "" or value == [] or value == {}:
                continue
            conf = confidences.get(field, 0.5)
            self.update_field(field, value, confidence=conf)

    def update_raw(self, data: dict[str, Any]) -> None:
        profile_dict = self._profile.to_dict()
        profile_dict.update(data)
        self._profile = UserProfile.from_dict(profile_dict)
        self._profile.last_updated = time.time()
        self._profile.update_count += 1
        self.save()

    async def infer_from_session(self, messages: list[dict[str, Any]], backend: Any) -> None:
        from encre.profile.inferrer import ProfileInferrer
        inferrer = ProfileInferrer()
        try:
            result = await inferrer.infer(messages, backend)
            if result:
                inferred, confidences = result
                self.merge_inferred(inferred, confidences)
                logger.info("Profile updated from session inference (%d fields)", len(inferred))
        except Exception as e:
            logger.warning("Profile inference failed: %s", e)
