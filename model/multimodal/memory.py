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

"""Internal Agentic runtime memory for Yv multimodal agents.

This module intentionally contains no external semantic encoder, FAISS backend,
vector database, model download path, or small-model dependency. It is the
process-local memory surface for the Agentic modality: observations, actions,
reflections, lightweight recency/importance retrieval, and tensor lifetime
tracking for multimodal fusion.
"""

import json
import time
import uuid
import weakref
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .types import YvAgenticAction, YvAgenticObservation

_LOG = PiscesLxLogger("Yv.Multimodal", file_path=get_log_file("Yv.Multimodal"), enable_file=True)


@dataclass
class YvMemoryConfig:
    """Configuration for internal Agentic runtime memory.

    The memory is designed to be cheap during training/inference: retrieval is
    bounded to recent and important entries, tensor tracking is weak-reference
    based, and persistence is opt-in JSON metadata only.
    """

    storage_dir: str = ".pisceslx/memory"
    max_memories: int = 4096
    retrieval_top_k: int = 8
    recent_window: int = 256
    tensor_track_limit: int = 512
    enable_persistence: bool = False
    importance_decay: float = 0.995


class YvMemoryType(Enum):
    """Runtime memory entry categories."""

    OBSERVATION = auto()
    ACTION = auto()
    REFLECTION = auto()
    EXPERIENCE = auto()
    TOOL_RESULT = auto()
    TENSOR = auto()


@dataclass
class YvMemoryEntry:
    """Single Agentic memory record."""

    id: str
    memory_type: YvMemoryType
    content: Any
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_access: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[torch.Tensor] = None

    def touch(self) -> None:
        self.access_count += 1
        self.last_access = datetime.now().isoformat()


class YvMemory:
    """Pure internal memory for the Agentic modality.

    This class preserves the runtime interfaces used by ``YvAgentic`` and
    ``YvDynamicModalFusion`` while removing external semantic retrieval. Query
    matching is intentionally lexical/metadata based and bounded; tensor payloads
    are stored as weak references only, so memory tracking does not keep GPU
    activations alive.
    """

    def __init__(self, config: Optional[YvMemoryConfig] = None):
        self.config = config or YvMemoryConfig()
        self.observations: List[YvAgenticObservation] = []
        self.actions: List[YvAgenticAction] = []
        self.reflections: List[str] = []
        self._entries: Dict[str, YvMemoryEntry] = {}
        self._order: List[str] = []
        self._tensor_refs: Dict[str, weakref.ReferenceType[torch.Tensor]] = {}
        self._tensor_meta: Dict[str, Dict[str, Any]] = {}

    @property
    def memories(self) -> List[YvMemoryEntry]:
        return [self._entries[mid] for mid in self._order if mid in self._entries]

    def add_observation(self, observation: YvAgenticObservation) -> str:
        self.observations.append(observation)
        return self._add_entry(
            YvMemoryType.OBSERVATION,
            observation,
            importance=self._estimate_importance(observation),
            metadata={
                "modality": getattr(observation, "modality", "unknown"),
                "kind": "observation",
            },
        )

    def add_action(self, action: YvAgenticAction) -> str:
        self.actions.append(action)
        return self._add_entry(
            YvMemoryType.ACTION,
            action,
            importance=self._estimate_importance(action),
            metadata={
                "action_type": getattr(action, "action_type", "unknown"),
                "confidence": float(getattr(action, "confidence", 0.0) or 0.0),
                "kind": "action",
            },
        )

    def add_reflection(self, reflection: str) -> str:
        self.reflections.append(reflection)
        return self._add_entry(
            YvMemoryType.REFLECTION,
            reflection,
            importance=0.75,
            metadata={"kind": "reflection"},
        )

    def add_experience(
        self,
        observation: Optional[YvAgenticObservation] = None,
        action: Optional[YvAgenticAction] = None,
        result: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        content = {"observation": observation, "action": action, "result": result}
        return self._add_entry(
            YvMemoryType.EXPERIENCE,
            content,
            importance=max(self._estimate_importance(observation), self._estimate_importance(action), 0.6),
            metadata={"kind": "experience", **(metadata or {})},
        )

    def get_context_with_retrieval(
        self,
        query: Optional[str] = None,
        k: Optional[int] = None,
        include_compressed: bool = True,
    ) -> Dict[str, Any]:
        """Return bounded Agentic memory context without external retrieval."""

        top_k = int(k or self.config.retrieval_top_k)
        results = self.semantic_search(query=query, k=top_k)
        recent_entries = self.memories[-min(len(self._order), self.config.recent_window) :]
        recent = [self._entry_to_dict(entry) for entry in recent_entries[-top_k:]]

        return {
            "memories": results,
            "recent": recent,
            "observations": self.observations[-top_k:],
            "actions": self.actions[-top_k:],
            "reflections": self.reflections[-top_k:],
            "memory_summary": self.summary(),
            "embeddings": self._collect_internal_embeddings(results),
            "compressed": self._compressed_summary(results) if include_compressed else {},
        }

    def semantic_search(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[torch.Tensor] = None,
        k: int = 5,
        threshold: float = 0.0,
        memory_type: Optional[YvMemoryType] = None,
    ) -> List[Dict[str, Any]]:
        """Compatibility search using internal lexical, recency, and importance scores.

        ``query_embedding`` is accepted because the Agentic planner already has an
        internal base-model representation. It is never produced by, nor compared
        with, an external encoder.
        """

        entries = self._candidate_entries(memory_type=memory_type)
        if not entries:
            return []

        query_terms = self._tokenize(query or "")
        now_rank_base = max(1, len(entries))
        scored: List[Tuple[float, YvMemoryEntry]] = []
        for rank, entry in enumerate(entries):
            lexical = self._lexical_score(query_terms, entry)
            recency = (rank + 1) / now_rank_base
            importance = max(0.0, min(1.0, float(entry.importance)))
            access = min(0.2, entry.access_count * 0.02)
            score = 0.45 * lexical + 0.35 * importance + 0.20 * recency + access
            if query_embedding is not None and entry.embedding is not None:
                score += 0.05 * self._internal_embedding_affinity(query_embedding, entry.embedding)
            if score >= threshold:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, entry in scored[: max(0, int(k))]:
            entry.touch()
            item = self._entry_to_dict(entry)
            item["score"] = float(score)
            results.append(item)
        return results

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.semantic_search(query=query, k=top_k or self.config.retrieval_top_k)

    def get_recent_context(self, k: Optional[int] = None) -> Dict[str, Any]:
        top_k = int(k or self.config.retrieval_top_k)
        return {
            "observations": self.observations[-top_k:],
            "actions": self.actions[-top_k:],
            "reflections": self.reflections[-top_k:],
            "memory_summary": self.summary(),
        }

    def register_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Track tensor lifetime without retaining GPU memory."""

        if not torch.is_tensor(tensor):
            return

        key = f"{name}:{uuid.uuid4().hex[:10]}"
        try:
            self._tensor_refs[key] = weakref.ref(tensor)
        except TypeError:
            return

        self._tensor_meta[key] = {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "timestamp": time.time(),
        }
        self._trim_tensor_refs()

    def cleanup_tensors(self) -> int:
        dead = [key for key, ref in self._tensor_refs.items() if ref() is None]
        for key in dead:
            self._tensor_refs.pop(key, None)
            self._tensor_meta.pop(key, None)
        return len(dead)

    def summary(self) -> Dict[str, int]:
        self.cleanup_tensors()
        return {
            "total_count": len(self._entries),
            "observations": len(self.observations),
            "actions": len(self.actions),
            "reflections": len(self.reflections),
            "tracked_tensors": len(self._tensor_refs),
        }

    def persist(self, storage_dir: Optional[str] = None) -> None:
        if not self.config.enable_persistence and storage_dir is None:
            return
        path = Path(storage_dir or self.config.storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [self._entry_to_json(entry) for entry in self.memories],
            "summary": self.summary(),
        }
        (path / "agentic_memory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, storage_dir: Optional[str] = None) -> None:
        path = Path(storage_dir or self.config.storage_dir) / "agentic_memory.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._entries.clear()
        self._order.clear()
        self.observations.clear()
        self.actions.clear()
        self.reflections.clear()

        for raw in payload.get("entries", []):
            memory_type = YvMemoryType[raw["memory_type"]]
            entry = YvMemoryEntry(
                id=raw["id"],
                memory_type=memory_type,
                content=raw.get("content"),
                importance=float(raw.get("importance", 0.5)),
                timestamp=raw.get("timestamp", datetime.now().isoformat()),
                access_count=int(raw.get("access_count", 0)),
                last_access=raw.get("last_access", datetime.now().isoformat()),
                metadata=raw.get("metadata", {}),
            )
            self._entries[entry.id] = entry
            self._order.append(entry.id)
            if memory_type == YvMemoryType.REFLECTION:
                self.reflections.append(str(entry.content))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.reflections.clear()
        self._entries.clear()
        self._order.clear()
        self._tensor_refs.clear()
        self._tensor_meta.clear()

    def _add_entry(
        self,
        memory_type: YvMemoryType,
        content: Any,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        memory_id = uuid.uuid4().hex
        entry = YvMemoryEntry(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            importance=max(0.0, min(1.0, float(importance))),
            metadata=metadata or {},
            embedding=self._internal_tensor(content),
        )
        self._entries[memory_id] = entry
        self._order.append(memory_id)
        self._trim_entries()
        return memory_id

    def _trim_entries(self) -> None:
        overflow = len(self._order) - int(self.config.max_memories)
        if overflow <= 0:
            return
        ranked = sorted(
            (self._entries[mid] for mid in self._order if mid in self._entries),
            key=lambda entry: (entry.importance, entry.access_count, entry.timestamp),
        )
        remove = {entry.id for entry in ranked[:overflow]}
        self._order = [mid for mid in self._order if mid not in remove]
        for mid in remove:
            self._entries.pop(mid, None)

    def _trim_tensor_refs(self) -> None:
        self.cleanup_tensors()
        limit = int(self.config.tensor_track_limit)
        if len(self._tensor_refs) <= limit:
            return
        oldest = sorted(self._tensor_meta.items(), key=lambda item: item[1].get("timestamp", 0.0))
        for key, _ in oldest[: len(self._tensor_refs) - limit]:
            self._tensor_refs.pop(key, None)
            self._tensor_meta.pop(key, None)

    def _candidate_entries(self, memory_type: Optional[YvMemoryType] = None) -> List[YvMemoryEntry]:
        entries = [self._entries[mid] for mid in self._order if mid in self._entries]
        if memory_type is not None:
            entries = [entry for entry in entries if entry.memory_type == memory_type]
        window = int(self.config.recent_window)
        if window > 0 and len(entries) > window:
            important = sorted(entries, key=lambda entry: entry.importance, reverse=True)[: window // 4]
            recent = entries[-window:]
            seen = set()
            merged = []
            for entry in recent + important:
                if entry.id not in seen:
                    merged.append(entry)
                    seen.add(entry.id)
            return merged
        return entries

    def _estimate_importance(self, item: Any) -> float:
        if item is None:
            return 0.3
        confidence = getattr(item, "confidence", None)
        if confidence is not None:
            return max(0.35, min(1.0, float(confidence)))
        metadata = getattr(item, "metadata", {}) or {}
        if isinstance(metadata, dict):
            if "importance" in metadata:
                return max(0.0, min(1.0, float(metadata["importance"])))
            if metadata.get("error") or metadata.get("critical"):
                return 0.9
        content = getattr(item, "content", item)
        text_len = len(str(content))
        return max(0.35, min(0.85, 0.35 + text_len / 4096.0))

    def _lexical_score(self, query_terms: Counter, entry: YvMemoryEntry) -> float:
        if not query_terms:
            return 0.0
        entry_terms = self._tokenize(self._stringify(entry.content))
        if not entry_terms:
            return 0.0
        overlap = sum(min(count, entry_terms.get(term, 0)) for term, count in query_terms.items())
        denom = max(1, sum(query_terms.values()))
        return max(0.0, min(1.0, overlap / denom))

    def _tokenize(self, text: str) -> Counter:
        current = []
        tokens: List[str] = []
        for ch in text.lower():
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
                current.append(ch)
            elif current:
                token = "".join(current)
                tokens.extend(self._split_token(token))
                current = []
        if current:
            tokens.extend(self._split_token("".join(current)))
        return Counter(tokens)

    def _split_token(self, token: str) -> Iterable[str]:
        if any("\u4e00" <= ch <= "\u9fff" for ch in token):
            return list(token) + [token]
        return [token]

    def _internal_tensor(self, content: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(content):
            return content.detach().flatten()[:128].float().cpu()
        if isinstance(content, dict):
            for value in content.values():
                if torch.is_tensor(value):
                    return value.detach().flatten()[:128].float().cpu()
        return None

    def _internal_embedding_affinity(self, query: torch.Tensor, memory: torch.Tensor) -> float:
        try:
            q = query.detach().flatten().float().cpu()
            m = memory.detach().flatten().float().cpu()
            size = min(q.numel(), m.numel())
            if size == 0:
                return 0.0
            q = q[:size]
            m = m[:size]
            denom = torch.linalg.vector_norm(q) * torch.linalg.vector_norm(m)
            if denom.item() == 0.0:
                return 0.0
            return float(torch.dot(q, m) / denom)
        except Exception:
            return 0.0

    def _collect_internal_embeddings(self, memories: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        tensors = []
        for item in memories:
            entry = self._entries.get(item.get("id", ""))
            if entry is not None and entry.embedding is not None:
                tensors.append(entry.embedding)
        if not tensors:
            return None
        max_len = max(t.numel() for t in tensors)
        padded = []
        for tensor in tensors:
            flat = tensor.flatten().float()
            if flat.numel() < max_len:
                flat = torch.nn.functional.pad(flat, (0, max_len - flat.numel()))
            padded.append(flat)
        return torch.stack(padded, dim=0)

    def _compressed_summary(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        types = Counter(item.get("type", "unknown") for item in memories)
        return {
            "count": len(memories),
            "types": dict(types),
            "top_ids": [item.get("id") for item in memories[:3]],
        }

    def _entry_to_dict(self, entry: YvMemoryEntry) -> Dict[str, Any]:
        return {
            "id": entry.id,
            "type": entry.memory_type.name.lower(),
            "content": self._serializable(entry.content),
            "importance": float(entry.importance),
            "timestamp": entry.timestamp,
            "access_count": int(entry.access_count),
            "metadata": dict(entry.metadata),
        }

    def _entry_to_json(self, entry: YvMemoryEntry) -> Dict[str, Any]:
        item = self._entry_to_dict(entry)
        item["memory_type"] = entry.memory_type.name
        item["last_access"] = entry.last_access
        return item

    def _serializable(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return {"tensor_shape": tuple(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if hasattr(value, "__dataclass_fields__"):
            return self._serializable(asdict(value))
        if isinstance(value, dict):
            return {str(k): self._serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serializable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _stringify(self, value: Any) -> str:
        serializable = self._serializable(value)
        try:
            return json.dumps(serializable, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(serializable)


__all__ = ["YvMemory", "YvMemoryConfig", "YvMemoryEntry", "YvMemoryType"]
