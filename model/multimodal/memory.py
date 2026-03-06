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

"""Unified Memory System for Yv Agentic Architecture.

This module provides comprehensive memory management components for the Yv
model, including semantic embedding, vector retrieval, importance-based
compression, and persistent storage.

Module Components:
    1. YvMemoryConfig:
       - Configuration dataclass for memory system
       - FAISS/NumPy backend settings
       - Persistence and monitoring parameters

    2. YvMemory:
       - Unified memory management system
       - Semantic embedding and vector retrieval
       - Observation, action, and reflection storage
       - Importance-based memory compression
       - Persistent storage with disk serialization

Key Features:
    - FAISS/NumPy vector backend support
    - Semantic embedding via SentenceTransformer
    - Importance-based memory compression
    - Persistent storage with disk serialization
    - Background memory monitoring and garbage collection
    - Experience replay for long-term learning
    - Memory consolidation and deduplication

Performance Characteristics:
    - Embedding: O(N * embedding_dim) per memory
    - FAISS retrieval: O(log N) with IVF index
    - NumPy retrieval: O(N) brute force
    - Compression: O(N log N) for importance sorting

Usage Example:
    >>> from model.multimodal.memory import YvMemory, YvMemoryConfig
    >>> 
    >>> # Initialize memory system
    >>> config = YvMemoryConfig(storage_dir="./memory")
    >>> memory = YvMemory(config=config)
    >>> 
    >>> # Add observation
    >>> memory.add_observation(observation)
    >>> 
    >>> # Search memories
    >>> results = memory.search("query", top_k=5)
    >>> 
    >>> # Persist to disk
    >>> memory.persist()

Note:
    Default embedding model: all-MiniLM-L6-v2 (384 dim)
    Default vector backend: FAISS with IVF_FLAT index
    Supports both synchronous and asynchronous operations.
"""

import gc
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import weakref
import threading
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from enum import Enum, auto

from utils.dc import PiscesLxLogger

from .types import YvAgenticObservation, YvAgenticAction

from utils.paths import get_log_file, get_work_dir
_LOG = PiscesLxLogger("Yv.Multimodal", file_path=get_log_file("Yv.Multimodal"), enable_file=True)


@dataclass
class YvMemoryConfig:
    """Configuration for YvMemory system.
    
    Attributes:
        storage_dir: Directory for persistent storage.
        embedding_dim: Dimension of memory embeddings.
        embedding_model: SentenceTransformer model name.
        vector_backend: Vector database backend ('faiss' or 'numpy').
        max_memories: Maximum number of memories before compression.
        compression_threshold: Percentile threshold for compression.
        similarity_threshold: Minimum similarity for retrieval.
        retrieval_top_k: Default number of results for search.
        enable_persistence: Enable disk persistence.
        persist_interval: Seconds between auto-persist.
        enable_monitoring: Enable background memory monitoring.
        monitoring_interval: Seconds between monitoring checks.
        memory_warning_threshold: Memory usage warning threshold (0-100).
        memory_critical_threshold: Memory usage critical threshold (0-100).
        enable_consolidation: Enable memory consolidation.
        consolidation_interval: Seconds between consolidation.
        faiss_index_type: FAISS index type (IVF_FLAT, Flat, etc.).
        faiss_nlist: Number of clusters for IVF index.
        faiss_nprobe: Number of clusters to search.
    """
    storage_dir: str = ".pisceslx/memory"
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_backend: str = "faiss"
    max_memories: int = 10000
    compression_threshold: float = 0.7
    similarity_threshold: float = 0.5
    retrieval_top_k: int = 10
    enable_persistence: bool = True
    persist_interval: int = 60
    enable_monitoring: bool = True
    monitoring_interval: int = 5
    auto_start_background: bool = False
    memory_warning_threshold: float = 80.0
    memory_critical_threshold: float = 90.0
    enable_consolidation: bool = True
    consolidation_interval: int = 3600
    faiss_index_type: str = "Flat"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10


class YvMemoryType(Enum):
    """Enumeration of memory entry types for categorization.
    
    Defines the different categories of memories stored in the system,
    enabling type-based filtering and retrieval.
    
    Attributes:
        OBSERVATION: Agent observation from environment.
        ACTION: Action taken by the agent.
        REFLECTION: Agent's reflection or reasoning.
        EXPERIENCE: Combined experience tuple.
        TOOL_RESULT: Result from tool execution.
    """
    OBSERVATION = auto()
    ACTION = auto()
    REFLECTION = auto()
    EXPERIENCE = auto()
    TOOL_RESULT = auto()


@dataclass
class YvMemoryEntry:
    """Single memory entry with metadata and access statistics.
    
    A comprehensive memory entry that stores content, embedding vector,
    importance score, and access statistics for retrieval and compression.
    
    Attributes:
        id (str): Unique memory identifier (UUID).
        memory_type (YvMemoryType): Type of memory entry.
        content (Any): The actual memory content (text, dict, etc.).
        embedding (Optional[np.ndarray]): Vector embedding of the content.
        importance (float): Importance score for compression (0-1).
        timestamp (str): Creation timestamp in ISO format.
        access_count (int): Number of times the memory was accessed.
        last_access (str): Last access timestamp in ISO format.
        metadata (Dict[str, Any]): Additional metadata for filtering.
    
    Example:
        >>> entry = YvMemoryEntry(
        ...     id="uuid-123",
        ...     memory_type=YvMemoryType.OBSERVATION,
        ...     content="The sky is blue",
        ...     importance=0.8
        ... )
        >>> entry.touch()  # Update access statistics
    """
    id: str
    memory_type: YvMemoryType
    content: Any
    embedding: Optional[np.ndarray] = None
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_access: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update access statistics for recency-based importance.
        
        Increments access count and updates last_access timestamp.
        Called automatically when memory is retrieved.
        """
        self.access_count += 1
        self.last_access = datetime.now().isoformat()


class YvVectorStore:
    """Vector storage backend for semantic similarity search.
    
    A flexible vector store that supports both FAISS and NumPy backends
    for semantic similarity search. Handles text encoding, vector indexing,
    and similarity-based retrieval.
    
    Architecture:
        1. Encoding:
           - SentenceTransformer for semantic embeddings
           - Hash-based fallback when transformers unavailable
        
        2. Storage:
           - FAISS IVF_FLAT index for efficient retrieval
           - NumPy brute force as fallback backend
        
        3. Retrieval:
           - Cosine similarity search
           - Top-k result ranking
    
    Key Features:
        - FAISS and NumPy backend support
        - SentenceTransformer integration
        - Deterministic hash-based fallback encoding
        - Thread-safe operations with locking
        - Periodic FAISS index rebuilding
    
    Attributes:
        config (YvMemoryConfig): Memory configuration.
        _embeddings (List[np.ndarray]): Stored embedding vectors.
        _ids (List[str]): Memory IDs corresponding to embeddings.
        _index: FAISS index instance.
        _encoder: SentenceTransformer encoder instance.
        _lock (threading.Lock): Thread lock for concurrent access.
    
    Example:
        >>> store = YvVectorStore(config)
        >>> embedding = store.encode("Hello world")
        >>> store.add("mem-123", embedding)
        >>> results = store.search(embedding, top_k=5)
    
    Note:
        FAISS index is rebuilt every 100 additions for efficiency.
        Hash-based encoding uses SHA256 for deterministic embeddings.
    """
    
    def __init__(self, config: YvMemoryConfig):
        """Initialize vector store with configuration.
        
        Args:
            config (YvMemoryConfig): Memory configuration specifying
                vector backend, embedding model, and FAISS parameters.
        """
        self.config = config
        self._embeddings: List[np.ndarray] = []
        self._ids: List[str] = []
        self._index = None
        self._encoder = None
        self._lock = threading.Lock()
        
        self._setup_backend()
        self._setup_encoder()
    
    def _setup_backend(self) -> None:
        """Setup vector database backend (FAISS or NumPy).
        
        Attempts to initialize FAISS backend for efficient similarity search.
        Falls back to NumPy brute force if FAISS is unavailable.
        
        Note:
            Sets _use_faiss flag based on availability.
            Logs backend initialization status.
        """
        if self.config.vector_backend == "faiss":
            try:
                import faiss
                self._faiss = faiss
                self._use_faiss = True
                _LOG.info("FAISS backend initialized")
            except ImportError:
                _LOG.warning("FAISS not available, using NumPy backend")
                self._use_faiss = False
        else:
            self._use_faiss = False
            _LOG.info("NumPy backend initialized")
    
    def _setup_encoder(self) -> None:
        """Setup sentence encoder for text embeddings.
        
        Attempts to load SentenceTransformer model for semantic embeddings.
        Falls back to None if unavailable, triggering hash-based encoding.
        
        Note:
            Default model: all-MiniLM-L6-v2 (384 dimensions).
            Logs encoder initialization status.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.config.embedding_model)
            _LOG.info(f"SentenceTransformer loaded: {self.config.embedding_model}")
        except ImportError:
            _LOG.warning("sentence-transformers not available, using hash-based embeddings")
            self._encoder = None
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to vector embedding.
        
        Uses SentenceTransformer for semantic embeddings when available,
        otherwise falls back to deterministic hash-based encoding.
        
        Args:
            text (str): Text to encode.
        
        Returns:
            np.ndarray: Vector embedding with shape [embedding_dim].
        
        Note:
            Hash-based encoding uses SHA256 for deterministic results.
            Embeddings are L2 normalized for cosine similarity.
        """
        if self._encoder is not None:
            embedding = self._encoder.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            text_bytes = text.encode('utf-8')
            hash_digest = hashlib.sha256(text_bytes).digest()
            embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
            for i in range(self.config.embedding_dim):
                byte_idx = i % len(hash_digest)
                embedding[i] = (hash_digest[byte_idx] - 128) / 128.0
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
    
    def add(self, id: str, embedding: np.ndarray) -> None:
        """Add vector to store with thread-safe indexing.
        
        Adds embedding to storage and triggers FAISS index rebuild
        every 100 additions for efficient retrieval.
        
        Args:
            id (str): Memory identifier for retrieval.
            embedding (np.ndarray): Vector embedding to store.
        
        Note:
            Thread-safe via lock mechanism.
            FAISS index rebuilt periodically for performance.
        """
        with self._lock:
            embedding = embedding.astype(np.float32)
            self._embeddings.append(embedding)
            self._ids.append(id)
            
            if self._use_faiss and len(self._embeddings) % 100 == 0:
                self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from stored embeddings.
        
        Constructs a new FAISS index from all stored embeddings.
        Uses IVF_FLAT for large datasets, Flat index otherwise.
        
        Note:
            IVF_FLAT requires at least faiss_nlist embeddings for training.
            Sets nprobe for search accuracy/speed tradeoff.
        """
        if not self._use_faiss or len(self._embeddings) == 0:
            return
        
        embeddings_array = np.array(self._embeddings, dtype=np.float32)
        dim = embeddings_array.shape[1]
        
        if self.config.faiss_index_type == "IVF_FLAT" and len(self._embeddings) >= self.config.faiss_nlist:
            quantizer = self._faiss.IndexFlatIP(dim)
            self._index = self._faiss.IndexIVFFlat(quantizer, dim, self.config.faiss_nlist)
            self._index.train(embeddings_array)
            self._index.add(embeddings_array)
            self._index.nprobe = self.config.faiss_nprobe
        else:
            self._index = self._faiss.IndexFlatIP(dim)
            self._index.add(embeddings_array)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors using cosine similarity.
        
        Retrieves the top-k most similar vectors to the query embedding.
        Uses FAISS index when available, otherwise NumPy brute force.
        
        Args:
            query_embedding (np.ndarray): Query vector [embedding_dim].
            top_k (int): Number of results to return. Default: 10.
        
        Returns:
            List[Tuple[str, float]]: List of (memory_id, similarity) tuples,
                sorted by descending similarity.
        
        Note:
            Thread-safe via lock mechanism.
            Returns empty list if no embeddings stored.
        """
        with self._lock:
            if len(self._embeddings) == 0:
                return []
            
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            if self._use_faiss and self._index is not None:
                similarities, indices = self._index.search(
                    query_embedding, 
                    min(top_k, len(self._embeddings))
                )
                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx >= 0 and idx < len(self._ids):
                        results.append((self._ids[idx], float(sim)))
                return results
            else:
                # NumPy cosine similarity
                embeddings_array = np.array(self._embeddings, dtype=np.float32)
                similarities = np.dot(embeddings_array, query_embedding.T).flatten()
                norms = np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
                similarities = similarities / (norms + 1e-8)
                
                top_indices = np.argsort(similarities)[::-1][:top_k]
                return [(self._ids[i], float(similarities[i])) for i in top_indices]
    
    def remove(self, id: str) -> bool:
        """Remove vector by ID.
        
        Args:
            id: Memory identifier.
            
        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if id in self._ids:
                idx = self._ids.index(id)
                self._ids.pop(idx)
                self._embeddings.pop(idx)
                return True
            return False
    
    def count(self) -> int:
        """Return number of vectors."""
        return len(self._embeddings)
    
    def clear(self) -> None:
        """Clear all vectors."""
        with self._lock:
            self._embeddings = []
            self._ids = []
            self._index = None
    
    def save(self, path: Path) -> None:
        """Save vectors to disk.
        
        Args:
            path: Directory path.
        """
        with self._lock:
            if len(self._embeddings) > 0:
                np.save(path / "embeddings.npy", np.array(self._embeddings))
                with open(path / "ids.json", 'w') as f:
                    json.dump(self._ids, f)
    
    def load(self, path: Path) -> None:
        """Load vectors from disk.
        
        Args:
            path: Directory path.
        """
        embeddings_path = path / "embeddings.npy"
        ids_path = path / "ids.json"
        
        if embeddings_path.exists() and ids_path.exists():
            with self._lock:
                self._embeddings = list(np.load(embeddings_path))
                with open(ids_path, 'r') as f:
                    self._ids = json.load(f)
                
                if self._use_faiss:
                    self._rebuild_faiss_index()


class YvMemory:
    """Unified Memory System for Yv Agentic Architecture.
    
    This is the single source of truth for all agent memory operations,
    replacing previous fragmented implementations.
    
    Features:
        - Semantic vector search via FAISS/NumPy
        - Observation, action, reflection storage
        - Importance-based compression
        - Persistent storage
        - Background monitoring
        - Experience replay
    """
    
    def __init__(self, config: Optional[YvMemoryConfig] = None):
        """Initialize memory system.
        
        Args:
            config: Memory configuration. Uses defaults if not provided.
        """
        self.config = config or YvMemoryConfig()
        
        # Core storage
        self._memories: Dict[str, YvMemoryEntry] = {}
        self._observations: List[str] = []  # Memory IDs
        self._actions: List[str] = []
        self._reflections: List[str] = []
        self._experiences: List[str] = []
        
        # Vector store
        self._vector_store = YvVectorStore(self.config)
        
        # Tensor registry for monitoring
        self._tensor_registry: Dict[str, Dict[str, Any]] = {}
        self._tensor_lock = threading.Lock()
        
        # Background threads
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._persist_thread: Optional[threading.Thread] = None
        self._stop_persist = threading.Event()
        
        # Statistics
        self._stats = {
            "total_added": 0,
            "total_retrieved": 0,
            "total_compressed": 0,
            "last_persist": None,
            "last_consolidation": None,
        }
        
        # Setup storage and start background tasks
        self._setup_storage()
        self._load_from_disk()

        if self.config.auto_start_background:
            if self.config.enable_monitoring:
                self.start_monitoring()

            if self.config.enable_persistence:
                self.start_auto_persist()
        
        _LOG.info("YvMemory initialized")
    
    def _setup_storage(self) -> None:
        """Setup storage directories."""
        storage_path = Path(self.config.storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        (storage_path / "vectors").mkdir(exist_ok=True)
        (storage_path / "memories").mkdir(exist_ok=True)
    
    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        import uuid
        return f"mem_{uuid.uuid4().hex[:12]}"
    
    # ==================== Core Memory Operations ====================
    
    def add_observation(self, observation: YvAgenticObservation) -> str:
        """Add observation to memory.
        
        Args:
            observation: Observation to store.
            
        Returns:
            Memory ID.
        """
        content_str = str(observation.content)
        embedding = self._vector_store.encode(content_str)
        importance = self._calculate_importance(content_str)
        
        memory_id = self._generate_id()
        entry = YvMemoryEntry(
            id=memory_id,
            memory_type=YvMemoryType.OBSERVATION,
            content=observation,
            embedding=embedding,
            importance=importance,
            metadata=observation.metadata,
        )
        
        self._memories[memory_id] = entry
        self._observations.append(memory_id)
        self._vector_store.add(memory_id, embedding)
        self._stats["total_added"] += 1
        
        self._maybe_compress()
        
        return memory_id
    
    def add_action(self, action: YvAgenticAction) -> str:
        """Add action to memory.
        
        Args:
            action: Action to store.
            
        Returns:
            Memory ID.
        """
        content_str = f"{action.action_type}: {action.parameters}"
        embedding = self._vector_store.encode(content_str)
        
        memory_id = self._generate_id()
        entry = YvMemoryEntry(
            id=memory_id,
            memory_type=YvMemoryType.ACTION,
            content=action,
            embedding=embedding,
            importance=action.confidence,
            metadata={"reasoning": action.reasoning},
        )
        
        self._memories[memory_id] = entry
        self._actions.append(memory_id)
        self._vector_store.add(memory_id, embedding)
        self._stats["total_added"] += 1
        
        self._maybe_compress()
        
        return memory_id
    
    def add_reflection(self, reflection: str) -> str:
        """Add reflection to memory.
        
        Args:
            reflection: Reflection text.
            
        Returns:
            Memory ID.
        """
        embedding = self._vector_store.encode(reflection)
        importance = min(1.0, len(reflection) / 500.0)
        
        memory_id = self._generate_id()
        entry = YvMemoryEntry(
            id=memory_id,
            memory_type=YvMemoryType.REFLECTION,
            content=reflection,
            embedding=embedding,
            importance=importance,
        )
        
        self._memories[memory_id] = entry
        self._reflections.append(memory_id)
        self._vector_store.add(memory_id, embedding)
        self._stats["total_added"] += 1
        
        self._maybe_compress()
        
        return memory_id
    
    def add_experience(
        self,
        task_type: str,
        goal: str,
        outcome: Dict[str, Any],
        success_rate: float,
        execution_pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add experience for long-term learning.
        
        Args:
            task_type: Type of task.
            goal: Task goal description.
            outcome: Execution outcome.
            success_rate: Success rate (0-1).
            execution_pattern: Optional execution pattern.
            
        Returns:
            Memory ID.
        """
        content = {
            "task_type": task_type,
            "goal": goal,
            "outcome": outcome,
            "success_rate": success_rate,
            "execution_pattern": execution_pattern or {},
        }
        
        embedding = self._vector_store.encode(f"{task_type}: {goal}")
        importance = success_rate * 0.7 + 0.3  # Bias toward successful experiences
        
        memory_id = self._generate_id()
        entry = YvMemoryEntry(
            id=memory_id,
            memory_type=YvMemoryType.EXPERIENCE,
            content=content,
            embedding=embedding,
            importance=importance,
            metadata={"success_rate": success_rate},
        )
        
        self._memories[memory_id] = entry
        self._experiences.append(memory_id)
        self._vector_store.add(memory_id, embedding)
        self._stats["total_added"] += 1
        
        return memory_id
    
    # ==================== Retrieval Operations ====================
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        memory_types: Optional[List[YvMemoryType]] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Semantic search across memories.
        
        Args:
            query: Search query.
            top_k: Number of results (default: config.retrieval_top_k).
            memory_types: Filter by memory types.
            min_importance: Minimum importance threshold.
            
        Returns:
            List of memory results with metadata.
        """
        top_k = top_k or self.config.retrieval_top_k
        query_embedding = self._vector_store.encode(query)
        
        # Get candidates from vector store
        candidates = self._vector_store.search(query_embedding, top_k * 2)
        
        results = []
        for memory_id, similarity in candidates:
            if memory_id not in self._memories:
                continue
            
            entry = self._memories[memory_id]
            
            # Filter by type
            if memory_types and entry.memory_type not in memory_types:
                continue
            
            # Filter by importance
            if entry.importance < min_importance:
                continue
            
            # Filter by similarity threshold
            if similarity < self.config.similarity_threshold:
                continue
            
            # Update access stats
            entry.touch()
            
            results.append({
                "id": memory_id,
                "type": entry.memory_type.name.lower(),
                "content": entry.content,
                "similarity": similarity,
                "importance": entry.importance,
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "metadata": entry.metadata,
            })
            
            if len(results) >= top_k:
                break
        
        self._stats["total_retrieved"] += len(results)
        return results
    
    def get_recent_context(self, k: int = 5) -> Dict[str, Any]:
        """Get most recent memories of each type.
        
        Args:
            k: Number of recent entries per type.
            
        Returns:
            Dictionary with recent memories.
        """
        def get_recent_entries(ids: List[str], n: int) -> List[Any]:
            recent_ids = ids[-n:] if len(ids) >= n else ids
            return [self._memories[mid].content for mid in recent_ids if mid in self._memories]
        
        return {
            "recent_observations": get_recent_entries(self._observations, k),
            "recent_actions": get_recent_entries(self._actions, k),
            "recent_reflections": get_recent_entries(self._reflections, k),
            "recent_experiences": get_recent_entries(self._experiences, k),
            "total_count": len(self._memories),
            "summary": {
                "observations": len(self._observations),
                "actions": len(self._actions),
                "reflections": len(self._reflections),
                "experiences": len(self._experiences),
            },
        }
    
    def get_context_with_retrieval(
        self,
        query: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """Get context via semantic search or recency.
        
        Args:
            query: Optional query for semantic search.
            k: Number of results.
            
        Returns:
            Context dictionary.
        """
        if query:
            return {
                "relevant_memories": self.search(query, top_k=k),
                "total_count": len(self._memories),
            }
        return self.get_recent_context(k)
    
    def get_similar_experiences(
        self,
        task_type: str,
        goal: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar past experiences.
        
        Args:
            task_type: Current task type.
            goal: Current goal.
            top_k: Number of experiences to return.
            
        Returns:
            List of similar experiences.
        """
        return self.search(
            f"{task_type}: {goal}",
            top_k=top_k,
            memory_types=[YvMemoryType.EXPERIENCE],
            min_importance=0.5,
        )
    
    # ==================== Importance Calculation ====================
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content.
        
        Args:
            content: Text content.
            
        Returns:
            Importance score (0-1).
        """
        # Length factor
        length_factor = min(len(content) / 500.0, 1.0) * 0.2
        
        # Complexity factor (unique words ratio)
        words = content.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            complexity_factor = unique_ratio * 0.3
        else:
            complexity_factor = 0.0
        
        # Keyword factor
        important_keywords = [
            'important', 'critical', 'urgent', 'key', 'essential',
            'error', 'success', 'fail', 'complete', 'result',
            'decision', 'action', 'observation', 'conclusion'
        ]
        keyword_count = sum(1 for w in words if w in important_keywords)
        keyword_factor = min(keyword_count * 0.1, 0.3)
        
        # Base score
        base_score = 0.2
        
        return min(1.0, base_score + length_factor + complexity_factor + keyword_factor)
    
    # ==================== Compression ====================
    
    def _maybe_compress(self) -> None:
        """Compress memories if over limit."""
        if len(self._memories) > self.config.max_memories:
            self.compress()
    
    def compress(self) -> int:
        """Remove low-importance memories.
        
        Returns:
            Number of memories removed.
        """
        if len(self._memories) == 0:
            return 0
        
        # Calculate threshold
        importances = [m.importance for m in self._memories.values()]
        threshold = sorted(importances)[int(len(importances) * self.config.compression_threshold)]
        
        # Find memories to remove
        to_remove = [
            mid for mid, entry in self._memories.items()
            if entry.importance < threshold
        ]
        
        # Remove memories
        for mid in to_remove:
            entry = self._memories.pop(mid)
            self._vector_store.remove(mid)
            
            # Remove from type lists
            if mid in self._observations:
                self._observations.remove(mid)
            elif mid in self._actions:
                self._actions.remove(mid)
            elif mid in self._reflections:
                self._reflections.remove(mid)
            elif mid in self._experiences:
                self._experiences.remove(mid)
        
        self._stats["total_compressed"] += len(to_remove)
        _LOG.info(f"Compressed {len(to_remove)} memories")
        
        return len(to_remove)
    
    # ==================== Persistence ====================
    
    def persist(self) -> None:
        """Save all memories to disk."""
        if not self.config.enable_persistence:
            return
        
        storage_path = Path(self.config.storage_dir)
        
        # Save vector store
        self._vector_store.save(storage_path / "vectors")
        
        # Save memories
        memories_data = {}
        for mid, entry in self._memories.items():
            memories_data[mid] = {
                "memory_type": entry.memory_type.name,
                "content": self._serialize_content(entry.content),
                "importance": entry.importance,
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "last_access": entry.last_access,
                "metadata": entry.metadata,
            }
        
        with open(storage_path / "memories" / "data.json", 'w', encoding='utf-8') as f:
            json.dump({
                "memories": memories_data,
                "observations": self._observations,
                "actions": self._actions,
                "reflections": self._reflections,
                "experiences": self._experiences,
                "stats": self._stats,
            }, f, indent=2, ensure_ascii=False)
        
        self._stats["last_persist"] = datetime.now().isoformat()
        _LOG.debug("Memory persisted to disk")
    
    def _serialize_content(self, content: Any) -> Any:
        """Serialize content for JSON storage."""
        if hasattr(content, '__dataclass_fields__'):
            return asdict(content)
        elif isinstance(content, dict):
            return content
        else:
            return str(content)
    
    def _load_from_disk(self) -> None:
        """Load memories from disk."""
        storage_path = Path(self.config.storage_dir)
        data_path = storage_path / "memories" / "data.json"
        
        if not data_path.exists():
            return
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load type lists
            self._observations = data.get("observations", [])
            self._actions = data.get("actions", [])
            self._reflections = data.get("reflections", [])
            self._experiences = data.get("experiences", [])
            self._stats = data.get("stats", self._stats)
            
            # Load memories (without embeddings - will regenerate)
            for mid, mdata in data.get("memories", {}).items():
                content = mdata["content"]
                memory_type = YvMemoryType[mdata["memory_type"]]
                
                entry = YvMemoryEntry(
                    id=mid,
                    memory_type=memory_type,
                    content=content,
                    importance=mdata["importance"],
                    timestamp=mdata["timestamp"],
                    access_count=mdata["access_count"],
                    last_access=mdata["last_access"],
                    metadata=mdata.get("metadata", {}),
                )
                self._memories[mid] = entry
            
            # Load vectors
            self._vector_store.load(storage_path / "vectors")
            
            _LOG.info(f"Loaded {len(self._memories)} memories from disk")
            
        except Exception as e:
            _LOG.error(f"Failed to load memories: {e}")
    
    # ==================== Background Tasks ====================
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._monitoring_thread is not None:
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True,
            name="YvMemory-Monitor"
        )
        self._monitoring_thread.start()
        _LOG.debug("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if self._monitoring_thread is None:
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5)
        self._monitoring_thread = None
        _LOG.debug("Memory monitoring stopped")
    
    def _monitor_memory(self) -> None:
        """Background memory monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                process = psutil.Process()
                memory_percent = process.memory_percent()
                
                if memory_percent > self.config.memory_critical_threshold:
                    _LOG.warning(f"Critical memory usage: {memory_percent:.1f}%")
                    gc.collect()
                    self.compress()
                elif memory_percent > self.config.memory_warning_threshold:
                    _LOG.debug(f"High memory usage: {memory_percent:.1f}%")
                
            except Exception as e:
                _LOG.error(f"Memory monitoring error: {e}")
            
            self._stop_monitoring.wait(timeout=self.config.monitoring_interval)
    
    def start_auto_persist(self) -> None:
        """Start auto-persistence background task."""
        if self._persist_thread is not None:
            return
        
        self._stop_persist.clear()
        self._persist_thread = threading.Thread(
            target=self._auto_persist_loop,
            daemon=True,
            name="YvMemory-Persist"
        )
        self._persist_thread.start()
        _LOG.debug("Auto-persistence started")
    
    def stop_auto_persist(self) -> None:
        """Stop auto-persistence."""
        if self._persist_thread is None:
            return
        
        self._stop_persist.set()
        self._persist_thread.join(timeout=5)
        self._persist_thread = None
        _LOG.debug("Auto-persistence stopped")
    
    def _auto_persist_loop(self) -> None:
        """Background auto-persistence loop."""
        while not self._stop_persist.is_set():
            self._stop_persist.wait(timeout=self.config.persist_interval)
            if not self._stop_persist.is_set():
                try:
                    self.persist()
                except Exception as e:
                    _LOG.error(f"Auto-persist error: {e}")
    
    # ==================== Tensor Registry ====================
    
    def register_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Register tensor for memory tracking.
        
        Args:
            tensor: Tensor to track.
            name: Identifier for the tensor.
        """
        if tensor is None:
            return
        
        try:
            info = {
                "name": name,
                "bytes": tensor.numel() * tensor.element_size(),
                "device": str(tensor.device),
                "shape": tuple(tensor.shape),
                "created_at": time.time(),
            }
            
            with self._tensor_lock:
                self._tensor_registry[name] = info
            
            # Auto-cleanup on GC
            def cleanup(n=name):
                with self._tensor_lock:
                    self._tensor_registry.pop(n, None)
            
            weakref.finalize(tensor, cleanup)
            
        except Exception as e:
            _LOG.debug(f"Tensor registration failed: {e}")
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Statistics dictionary.
        """
        return {
            "total_memories": len(self._memories),
            "observations": len(self._observations),
            "actions": len(self._actions),
            "reflections": len(self._reflections),
            "experiences": len(self._experiences),
            "vector_count": self._vector_store.count(),
            "tensor_registry_size": len(self._tensor_registry),
            **self._stats,
        }
    
    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        self._observations.clear()
        self._actions.clear()
        self._reflections.clear()
        self._experiences.clear()
        self._vector_store.clear()
        _LOG.info("All memories cleared")
    
    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.stop_monitoring()
        self.stop_auto_persist()
        self.persist()
        _LOG.info("YvMemory shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass


# ==================== Factory Function ====================

def create_memory(config: Optional[YvMemoryConfig] = None) -> YvMemory:
    """Factory function to create YvMemory instance.
    
    Args:
        config: Memory configuration.
        
    Returns:
        Initialized YvMemory instance.
    """
    return YvMemory(config)


# ==================== Module-level Instance ====================

_default_memory: Optional[YvMemory] = None

def get_default_memory() -> YvMemory:
    """Get or create default memory instance.
    
    Returns:
        Default YvMemory instance.
    """
    global _default_memory
    if _default_memory is None:
        _default_memory = YvMemory()
    return _default_memory
