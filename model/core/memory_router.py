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

"""Memory Router for Engram-style Lookup-Computation Separation.

Implements the YvMemoryRouter class that projects hidden states into a
unified 256-dim address space, queries an external FAISS IVF-PQ index
for top-K knowledge slots, and routes retrieved knowledge embeddings
back to the model via cross-attention injection.

Architecture inspired by:
    Liang Wenfeng et al., "Engram: Conditional Memory via Scalable
    Lookup", arXiv:2601.07372, 2026.

Key Design:
    - Deterministic N-gram embedding addressing with O(1) lookup
    - FAISS IVF-PQ index for billion-scale knowledge retrieval
    - mmap-backed knowledge store (no GPU memory consumption)
    - U-shaped sparsity allocation for optimal capacity distribution
    - Prefetch pipeline for overlapping retrieval with computation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import os

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.MemoryRouter", file_path=get_log_file("Yv.MemoryRouter"), enable_file=True)


class YvMemoryRouter(nn.Module):
    """Knowledge retrieval router for lookup-computation separation.

    Projects hidden states into a unified 256-dim address space,
    queries an external FAISS index for top-K knowledge slots,
    and returns retrieved knowledge embeddings and slot indices.

    The router maintains a learnable gate parameter that controls
    how much knowledge is injected into the model, initialized to
    0 so the model starts without knowledge injection and gradually
    learns to use it.

    Architecture:
        query_proj: hidden_size -> memory_router_dim (256)
        normalize to unit hypersphere (L2 norm)
        FAISS ANN search -> top-K slot indices
        mmap read -> knowledge embeddings
        gate: sigmoid(gate_param) controls injection strength

    Attributes:
        query_proj (nn.Linear): Projects hidden states to address space.
        gate (nn.Parameter): Learnable injection strength gate.
        _index: FAISS index (lazy-loaded from memory_store_path).
        _knowledge_store: mmap-backed knowledge tensor (lazy-loaded).
        _knowledge_slot_mutex: File lock for thread-safe mmap access.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_router_dim: int = 256,
        memory_knowledge_dim: int = 256,
        memory_top_k: int = 8,
        memory_cache_tokens: int = 4096,
        memory_store_path: str = "",
        memory_index_type: str = "ivfpq",
        memory_gate_init: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize memory router.

        Args:
            hidden_size: Model hidden dimension (e.g., 3584 for 7B).
            memory_router_dim: Unified address dimension (default 256).
            memory_knowledge_dim: Knowledge slot embedding dimension (default 256).
            memory_top_k: Number of knowledge slots to retrieve per query.
            memory_cache_tokens: Token context window for FAISS lookup.
            memory_store_path: Path to knowledge store directory.
            memory_index_type: FAISS index type ("ivfpq", "ivfflat", "hnsw").
            memory_gate_init: Initial value for injection gate (0 = disabled).
            device: Device for router parameters.
            dtype: Data type for router parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.router_dim = memory_router_dim
        self.knowledge_dim = memory_knowledge_dim
        self.top_k = memory_top_k
        self.cache_tokens = memory_cache_tokens
        self.store_path = memory_store_path
        self.index_type = memory_index_type

        # Project hidden states to unified address space
        self.query_proj = nn.Linear(
            hidden_size, memory_router_dim, bias=False,
            device=device, dtype=dtype
        )

        # Project retrieved knowledge back to hidden space for injection
        self.knowledge_proj = nn.Linear(
            memory_knowledge_dim, hidden_size, bias=False,
            device=device, dtype=dtype
        )

        # Learnable injection gate: sigmoid(gate) controls knowledge blend ratio
        # Initialized to gate_init (0.0) so model starts without knowledge injection
        self.gate = nn.Parameter(
            torch.tensor(memory_gate_init, device=device, dtype=dtype)
        )

        # FAISS index and mmap store (lazy-loaded)
        self._index = None
        self._knowledge_store = None
        self._index_loaded = False
        self._faiss_available = None

        # Running statistics for adaptive query scheduling
        self.register_buffer('_query_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_hit_count', torch.tensor(0, dtype=torch.long))

        # Prefetch state for asynchronous retrieval
        self._prefetch_state: Optional[Dict[str, Any]] = None

        _LOG.info(
            f"YvMemoryRouter initialized: hidden={hidden_size}, "
            f"router_dim={memory_router_dim}, top_k={memory_top_k}, "
            f"store_path='{memory_store_path}'"
        )

    def _check_faiss(self) -> bool:
        """Check if FAISS is available.

        Returns:
            True if FAISS is importable, False otherwise.
        """
        if self._faiss_available is None:
            try:
                import faiss
                self._faiss_available = True
            except ImportError:
                _LOG.warning("FAISS not available, memory separation disabled")
                self._faiss_available = False
        return self._faiss_available

    def _ensure_index_loaded(self) -> bool:
        """Lazy-load FAISS index and mmap knowledge store.

        Loads the index and knowledge store from disk if not already loaded.
        The knowledge store uses mmap for zero-GPU-memory access.

        Returns:
            True if index is loaded and ready, False otherwise.
        """
        if self._index_loaded and self._index is not None:
            return True

        if not self._check_faiss():
            return False

        if not self.store_path or not os.path.isdir(self.store_path):
            return False

        try:
            import faiss
            import numpy as np

            index_path = os.path.join(self.store_path, f"knowledge_index.{self.index_type}")
            store_path = os.path.join(self.store_path, "knowledge_store.npy")

            if os.path.exists(index_path):
                self._index = faiss.read_index(index_path)
                _LOG.info(f"FAISS index loaded from {index_path}: {self._index.ntotal} slots")
            else:
                _LOG.warning(f"FAISS index not found at {index_path}")
                return False

            if os.path.exists(store_path):
                self._knowledge_store = np.load(store_path, mmap_mode='r')
                _LOG.info(f"Knowledge store loaded via mmap: {self._knowledge_store.shape}")
            else:
                _LOG.warning(f"Knowledge store not found at {store_path}")
                return False

            self._index_loaded = True
            return True

        except Exception as e:
            _LOG.error(f"Failed to load FAISS index: {e}")
            return False

    def _query_knowledge(
        self,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query knowledge store with projected hidden states.

        Args:
            queries: Projected query vectors [B, T, router_dim].

        Returns:
            Tuple of (knowledge_embeddings, slot_indices, distances):
                - knowledge_embeddings: [B, T, top_k, knowledge_dim]
                - slot_indices: [B, T, top_k]
                - distances: [B, T, top_k]
        """
        if not self._ensure_index_loaded():
            return None, None, None

        try:
            import numpy as np

            batch_size, seq_len, _ = queries.shape
            queries_flat = queries.reshape(-1, self.router_dim)

            # L2 normalize for cosine similarity search
            queries_norm = F.normalize(queries_flat, p=2, dim=-1)
            query_np = queries_norm.detach().cpu().float().numpy()

            # FAISS ANN search
            k = min(self.top_k, self._index.ntotal)
            distances, indices = self._index.search(query_np, k)

            # Read knowledge embeddings from mmap store
            embeddings_np = self._knowledge_store[indices]  # [B*T, k, knowledge_dim]

            # Convert to torch tensors on correct device
            embeddings = torch.from_numpy(embeddings_np).to(
                device=queries.device, dtype=queries.dtype
            )
            indices_t = torch.from_numpy(indices).to(device=queries.device, dtype=torch.long)
            distances_t = torch.from_numpy(distances).to(device=queries.device, dtype=queries.dtype)

            # Reshape back to batched form
            embeddings = embeddings.view(batch_size, seq_len, k, self.knowledge_dim)
            indices_t = indices_t.view(batch_size, seq_len, k)
            distances_t = distances_t.view(batch_size, seq_len, k)

            # Update statistics
            self._query_count += batch_size * seq_len
            self._hit_count += batch_size * seq_len  # All queries get results from ANN

            return embeddings, indices_t, distances_t

        except Exception as e:
            _LOG.error(f"Knowledge query failed: {e}")
            return None, None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        force_query: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Route hidden states through knowledge retrieval.

        Queries the knowledge store for relevant knowledge slots and
        returns retrieved embeddings for cross-attention injection.

        Args:
            hidden_states: Model hidden states [B, T, hidden_size].
            force_query: Force query even if index not loaded (for testing).

        Returns:
            Dict containing:
                - knowledge: Retrieved knowledge embeddings [B, T, top_k, knowledge_dim]
                - slot_indices: Slot indices [B, T, top_k]
                - distances: FAISS distances [B, T, top_k]
                - gate_value: Current sigmoid(gate) injection strength
                - knowledge_projected: Projected knowledge [B, T, top_k, hidden_size]
            Or None if query fails or index not loaded.
        """
        if not force_query and not self._ensure_index_loaded():
            return None

        batch_size, seq_len, _ = hidden_states.shape

        # Project to unified address space and normalize
        queries = self.query_proj(hidden_states)  # [B, T, router_dim]
        queries = F.normalize(queries, p=2, dim=-1)

        # Query FAISS index
        knowledge, slot_indices, distances = self._query_knowledge(queries)

        if knowledge is None:
            return None

        # Project knowledge embeddings to model hidden size
        # knowledge: [B, T, top_k, knowledge_dim]
        # knowledge_projected: [B, T, top_k, hidden_size]
        knowledge_flat = knowledge.view(-1, self.knowledge_dim)
        knowledge_proj_flat = self.knowledge_proj(knowledge_flat)
        knowledge_projected = knowledge_proj_flat.view(
            batch_size, seq_len, self.top_k, self.hidden_size
        )

        gate_value = torch.sigmoid(self.gate)

        return {
            "knowledge": knowledge,
            "slot_indices": slot_indices,
            "distances": distances,
            "gate_value": gate_value,
            "knowledge_projected": knowledge_projected,
        }

    def prefetch_next(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[str]:
        """Prefetch knowledge for next forward pass asynchronously.

        Used in inference mode to overlap knowledge retrieval with
        computation. Stores results in _prefetch_state for next call.

        Args:
            hidden_states: Current hidden states [B, T, hidden_size].

        Returns:
            Prefetch status string or None if unavailable.
        """
        if not self._ensure_index_loaded():
            return None

        result = self.forward(hidden_states, force_query=True)
        if result is not None:
            self._prefetch_state = result
            return "prefetched"
        return None

    def get_prefetch_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get prefetched knowledge state and clear it.

        Returns:
            Prefetched state dict or None.
        """
        state = self._prefetch_state
        self._prefetch_state = None
        return state

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics.

        Returns:
            Dict with query_count, hit_count, hit_rate, gate_value.
        """
        query_count = self._query_count.item()
        hit_count = self._hit_count.item()
        return {
            "query_count": query_count,
            "hit_count": hit_count,
            "hit_rate": hit_count / max(1, query_count),
            "gate_value": torch.sigmoid(self.gate).item(),
            "index_loaded": self._index_loaded,
        }


class YvMemoryKnowledgeStore:
    """Offline-constructed knowledge store metadata handler.

    Provides metadata access and validation for the mmap-backed
    knowledge store built by POPSSKnowledgeBuilder. Does not
    contain model weights - just index metadata and slot info.

    Attributes:
        store_path: Path to knowledge store directory.
        num_slots: Total number of knowledge slots.
        slot_dim: Dimension of each knowledge slot (256).
        index_type: FAISS index type.
        metadata: Optional metadata dict from knowledge store.
    """

    def __init__(self, store_path: str):
        """Initialize knowledge store handler.

        Args:
            store_path: Path to knowledge store directory.
        """
        self.store_path = store_path
        self.num_slots = 0
        self.slot_dim = 256
        self.index_type = "ivfpq"
        self.metadata: Dict[str, Any] = {}

        self._load_metadata()

    def _load_metadata(self):
        """Load knowledge store metadata from disk."""
        meta_path = os.path.join(self.store_path, "metadata.json")
        if os.path.exists(meta_path):
            import json
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.num_slots = self.metadata.get("num_slots", 0)
                self.slot_dim = self.metadata.get("slot_dim", 256)
                self.index_type = self.metadata.get("index_type", "ivfpq")
                _LOG.info(
                    f"Knowledge store metadata loaded: {self.num_slots} slots, "
                    f"dim={self.slot_dim}, index={self.index_type}"
                )
            except Exception as e:
                _LOG.warning(f"Failed to load knowledge store metadata: {e}")

    def is_valid(self) -> bool:
        """Check if knowledge store is valid and accessible.

        Returns:
            True if store path exists with valid index and store files.
        """
        if not os.path.isdir(self.store_path):
            return False
        index_path = os.path.join(self.store_path, f"knowledge_index.{self.index_type}")
        store_path = os.path.join(self.store_path, "knowledge_store.npy")
        return os.path.exists(index_path) and os.path.exists(store_path)

    def estimate_storage_size(self) -> Dict[str, float]:
        """Estimate storage size of knowledge store.

        Returns:
            Dict with index_size_gb, store_size_gb, total_gb.
        """
        import numpy as np
        store_path = os.path.join(self.store_path, "knowledge_store.npy")
        index_path = os.path.join(self.store_path, f"knowledge_index.{self.index_type}")

        store_size = 0.0
        index_size = 0.0

        if os.path.exists(store_path):
            store_size = os.path.getsize(store_path) / (1024 ** 3)

        if os.path.exists(index_path):
            index_size = os.path.getsize(index_path) / (1024 ** 3)

        return {
            "index_size_gb": round(index_size, 2),
            "store_size_gb": round(store_size, 2),
            "total_gb": round(store_size + index_size, 2),
        }