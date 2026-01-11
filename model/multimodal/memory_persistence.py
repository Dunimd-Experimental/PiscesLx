#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Long-term memory persistence for Ruchbah agentic system.

This module provides persistent memory storage with:
- Vector database integration for semantic search
- Memory compression and summarization
- Experience replay and retrieval
- Memory consolidation
"""

import os
import sys
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib

import torch
import torch.nn as nn
import numpy as np

import dms_core
PiscesLxCoreLog = dms_core.log.get_logger

logger = PiscesLxCoreLog("pisceslx.tools.agentic.memory_persistence")


@dataclass
class RuchbahMemoryConfig:
    """Memory persistence configuration."""
    
    storage_dir: str = "./memory"
    
    embedding_dim: int = 1024
    embedding_model: str = "all-MiniLM-L6-v2"
    
    vector_db_type: str = "faiss"
    
    max_memories: int = 10000
    max_memory_size: int = 1000000
    
    similarity_threshold: float = 0.7
    retrieval_top_k: int = 10
    
    compression_interval: int = 100
    consolidation_interval: int = 3600
    
    enable_disk_cache: bool = True
    enable_compression: bool = True
    enable_consolidation: bool = True
    
    persist_frequency: int = 60
    
    index_type: str = "IVF_FLAT"
    nlist: int = 1024
    nprobe: int = 32


class RuchbahVectorDatabase:
    """Vector database for memory storage and retrieval."""
    
    def __init__(self, config: RuchbahMemoryConfig):
        """Initialize vector database.
        
        Args:
            config: Memory configuration.
        """
        self.config = config
        self.embeddings = []
        self.metadata = []
        self.ids = []
        
        self._index = None
        self._embedding_model = None
        
        self._setup_vector_db()
    
    def _setup_vector_db(self) -> None:
        """Setup vector database backend."""
        storage_dir = Path(self.config.storage_dir) / "vectors"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.vector_db_type == "faiss":
            self._setup_faiss()
        elif self.config.vector_db_type == "numpy":
            self._setup_numpy()
        else:
            self._setup_numpy()
    
    def _setup_faiss(self) -> None:
        """Setup FAISS-based vector database."""
        try:
            import faiss
            
            self._index_file = Path(self.config.storage_dir) / "vectors" / "index.faiss"
            
            if self._index_file.exists():
                self._index = faiss.read_index(str(self._index_file))
                self._load_metadata()
            
            logger.info("FAISS vector database initialized")
            
        except ImportError:
            logger.warning("FAISS not available, falling back to numpy")
            self.config.vector_db_type = "numpy"
            self._setup_numpy()
    
    def _setup_numpy(self) -> None:
        """Setup numpy-based vector database."""
        self._embeddings = []
        self._metadata = []
        self._ids = []
        self._next_id = 0
        
        self._load_from_disk()
        
        logger.info("NumPy vector database initialized")
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        metadata_file = Path(self.config.storage_dir) / "vectors" / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._metadata = data.get("metadata", [])
                self._ids = data.get("ids", [])
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        metadata_file = Path(self.config.storage_dir) / "vectors" / "metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": self._metadata,
                "ids": self._ids,
            }, f, indent=2, ensure_ascii=False)
    
    def _load_from_disk(self) -> None:
        """Load vector database from disk."""
        embeddings_file = Path(self.config.storage_dir) / "vectors" / "embeddings.npy"
        metadata_file = Path(self.config.storage_dir) / "vectors" / "metadata.json"
        
        if embeddings_file.exists():
            self._embeddings = np.load(str(embeddings_file))
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._metadata = data.get("metadata", [])
                self._ids = data.get("ids", [])
                self._next_id = data.get("next_id", 0)
    
    def _save_to_disk(self) -> None:
        """Save vector database to disk."""
        embeddings_file = Path(self.config.storage_dir) / "vectors" / "embeddings.npy"
        metadata_file = Path(self.config.storage_dir) / "vectors" / "metadata.json"
        
        if len(self._embeddings) > 0:
            np.save(str(embeddings_file), np.array(self._embeddings))
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": self._metadata,
                "ids": self._ids,
                "next_id": self._next_id,
            }, f, indent=2, ensure_ascii=False)
    
    def add(
        self,
        embedding: Union[torch.Tensor, np.ndarray, List[float]],
        metadata: Dict[str, Any],
    ) -> int:
        """Add a vector to the database.
        
        Args:
            embedding: Vector to add.
            metadata: Associated metadata.
            
        Returns:
            ID of the added vector.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        elif isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        embedding = embedding.astype(np.float32)
        
        vector_id = self._next_id
        self._next_id += 1
        
        self._embeddings.append(embedding)
        self._metadata.append(metadata)
        self._ids.append(vector_id)
        
        if self.config.vector_db_type == "faiss" and self._index is not None:
            if len(self._embeddings) == 1:
                dimension = embedding.shape[0]
                self._index = faiss.IndexFlatIP(dimension)
            
            self._index.add(embedding.reshape(1, -1))
        
        return vector_id
    
    def search(
        self,
        query: Union[torch.Tensor, np.ndarray, List[float]],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar vectors.
        
        Args:
            query: Query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.
            
        Returns:
            Tuple of (similarities, metadata_list).
        """
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        elif isinstance(query, list):
            query = np.array(query, dtype=np.float32)
        
        query = query.astype(np.float32)
        
        if len(self._embeddings) == 0:
            return np.array([]), []
        
        if self.config.vector_db_type == "faiss" and self._index is not None:
            return self._faiss_search(query, top_k, filter_metadata)
        else:
            return self._numpy_search(query, top_k, filter_metadata)
    
    def _numpy_search(
        self,
        query: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """NumPy-based similarity search."""
        embeddings_array = np.array(self._embeddings)
        
        similarities = np.dot(embeddings_array, query) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query) + 1e-8
        )
        
        indices = np.argsort(similarities)[::-1][:top_k]
        
        filtered_indices = []
        for idx in indices:
            if filter_metadata is None:
                filtered_indices.append(idx)
            else:
                matches = True
                for key, value in filter_metadata.items():
                    if self._metadata[idx].get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_indices.append(idx)
        
        result_similarities = similarities[filtered_indices]
        result_metadata = [self._metadata[i] for i in filtered_indices]
        
        return result_similarities, result_metadata
    
    def _faiss_search(
        self,
        query: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """FAISS-based similarity search."""
        similarities, indices = self._index.search(
            query.reshape(1, -1).astype(np.float32),
            min(top_k, len(self._embeddings))
        )
        
        result_similarities = similarities[0]
        result_metadata = [self._metadata[i] for i in indices[0] if i < len(self._metadata)]
        
        return result_similarities, result_metadata
    
    def persist(self) -> None:
        """Persist vector database to disk."""
        if self.config.vector_db_type == "faiss":
            if self._index is not None and self._index_file:
                faiss.write_index(self._index, str(self._index_file))
        
        self._save_metadata()
        
        logger.debug("Vector database persisted")
    
    def clear(self) -> None:
        """Clear all vectors."""
        self._embeddings = []
        self._metadata = []
        self._ids = []
        self._next_id = 0
        
        if self._index is not None:
            self._index.reset()
        
        logger.info("Vector database cleared")
    
    def count(self) -> int:
        """Return number of vectors."""
        return len(self._embeddings)


class RuchbahMemoryPersistence:
    """Long-term memory persistence manager for agentic system."""
    
    def __init__(self, config: Optional[RuchbahMemoryConfig] = None):
        """Initialize memory persistence manager.
        
        Args:
            config: Memory configuration.
        """
        self.config = config or RuchbahMemoryConfig()
        
        self.vector_db = RuchbahVectorDatabase(self.config)
        
        self._compression_count = 0
        self._last_consolidation = time.time()
        self._last_persist = time.time()
        
        self._lock = threading.Lock()
        
        self._setup_storage()
        
        logger.info("RuchbahMemoryPersistence initialized")
    
    def _setup_storage(self) -> None:
        """Setup storage directories."""
        Path(self.config.storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.storage_dir, "vectors").mkdir(exist_ok=True)
        Path(self.config.storage_dir, "compressed").mkdir(exist_ok=True)
        Path(self.config.storage_dir, "consolidated").mkdir(exist_ok=True)
    
    def store_memory(
        self,
        content: str,
        embedding: Optional[Union[torch.Tensor, np.ndarray]] = None,
        memory_type: str = "observation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a memory.
        
        Args:
            content: Memory content text.
            embedding: Pre-computed embedding (optional).
            memory_type: Type of memory (observation, action, reflection).
            metadata: Additional metadata.
            
        Returns:
            Memory ID.
        """
        if embedding is None:
            embedding = self._get_embedding(content)
        
        memory_metadata = {
            "type": memory_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0,
            "last_access": datetime.now().isoformat(),
            **(metadata or {}),
        }
        
        memory_id = self.vector_db.add(embedding, memory_metadata)
        
        self._maybe_persist()
        
        return memory_id
    
    def retrieve_memories(
        self,
        query: Union[str, torch.Tensor, np.ndarray],
        memory_types: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories.
        
        Args:
            query: Query text or embedding.
            memory_types: Filter by memory types.
            top_k: Number of results.
            
        Returns:
            List of retrieved memories with metadata.
        """
        if isinstance(query, str):
            embedding = self._get_embedding(query)
        else:
            embedding = query
        
        filter_metadata = None
        if memory_types:
            filter_metadata = {"type": lambda t: t in memory_types}
        
        similarities, metadata_list = self.vector_db.search(
            embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        with self._lock:
            for i, meta in enumerate(metadata_list):
                meta["similarity"] = float(similarities[i]) if i < len(similarities) else 0.0
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_access"] = datetime.now().isoformat()
        
        return metadata_list
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.vector_db._embedding_model is None:
                self.vector_db._embedding_model = SentenceTransformer(
                    self.config.embedding_model
                )
            
            embedding = self.vector_db._embedding_model.encode(
                text,
                convert_to_numpy=True,
            )
            
            return embedding
            
        except ImportError:
            logger.warning("sentence-transformers not available, using random embedding")
            return np.random.randn(self.config.embedding_dim).astype(np.float32)
    
    def _maybe_persist(self) -> None:
        """Persist if enough time has passed."""
        current_time = time.time()
        
        if current_time - self._last_persist >= self.config.persist_frequency:
            self.persist()
            self._last_persist = current_time
        
        self._compression_count += 1
        
        if self._compression_count >= self.config.compression_interval:
            self.compress_memories()
            self._compression_count = 0
        
        if self.config.enable_consolidation:
            if current_time - self._last_consolidation >= self.config.consolidation_interval:
                self.consolidate_memories()
                self._last_consolidation = current_time
    
    def compress_memories(self) -> None:
        """Compress old memories."""
        logger.info("Compressing memories...")
        
        if not self.config.enable_compression:
            return
        
        if self.vector_db.count() < self.config.max_memories // 2:
            return
        
        with self._lock:
            memories = []
            
            for i, meta in enumerate(self.vector_db._metadata):
                if meta.get("type") == "observation":
                    age = (datetime.now() - datetime.fromisoformat(meta["timestamp"])).total_seconds()
                    
                    if age > 86400:
                        memories.append({
                            "id": i,
                            "content": meta["content"],
                            "timestamp": meta["timestamp"],
                            "type": "compressed",
                        })
            
            if memories:
                compressed_file = Path(self.config.storage_dir) / "compressed" / f"{int(time.time())}.json"
                
                with open(compressed_file, 'w', encoding='utf-8') as f:
                    json.dump(memories, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Compressed {len(memories)} memories to {compressed_file}")
    
    def consolidate_memories(self) -> None:
        """Consolidate memories for long-term storage."""
        logger.info("Consolidating memories...")
        
        if not self.config.enable_consolidation:
            return
        
        with self._lock:
            high_value_memories = []
            
            for i, meta in enumerate(self.vector_db._metadata):
                access_count = meta.get("access_count", 0)
                age = (datetime.now() - datetime.fromisoformat(meta["timestamp"])).total_seconds()
                
                if access_count >= 5 or age < 3600:
                    high_value_memories.append({
                        "id": i,
                        "content": meta["content"],
                        "metadata": {
                            "type": meta["type"],
                            "importance_score": access_count * (1.0 / (age + 1)),
                            "consolidated_at": datetime.now().isoformat(),
                        }
                    })
            
            if high_value_memories:
                consolidated_file = Path(self.config.storage_dir) / "consolidated" / f"{int(time.time())}.json"
                
                with open(consolidated_file, 'w', encoding='utf-8') as f:
                    json.dump(high_value_memories, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Consolidated {len(high_value_memories)} memories")
    
    def persist(self) -> None:
        """Persist all data to disk."""
        with self._lock:
            self.vector_db.persist()
        
        logger.debug("Memory persistence completed")
    
    def clear(self) -> None:
        """Clear all memories."""
        with self._lock:
            self.vector_db.clear()
        
        logger.info("All memories cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            return {
                "total_memories": self.vector_db.count(),
                "vector_dim": self.config.embedding_dim,
                "storage_dir": self.config.storage_dir,
                "last_persist": datetime.fromtimestamp(self._last_persist).isoformat(),
                "compression_count": self._compression_count,
                "last_consolidation": datetime.fromtimestamp(self._last_consolidation).isoformat(),
            }


def create_memory_persistence(
    config: Optional[RuchbahMemoryConfig] = None,
) -> RuchbahMemoryPersistence:
    """Factory function to create memory persistence manager.
    
    Args:
        config: Memory configuration.
        
    Returns:
        Initialized memory persistence manager.
    """
    return RuchbahMemoryPersistence(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory persistence for PiscesL1")
    
    parser.add_argument("--storage_dir", type=str, default="./memory")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--test", action="store_true", default=False)
    
    args = parser.parse_args()
    
    config = RuchbahMemoryConfig(
        storage_dir=args.storage_dir,
        embedding_model=args.embedding_model,
    )
    
    memory = create_memory_persistence(config)
    
    if args.test:
        memory.store_memory(
            content="Test memory: The user asked about the architecture",
            memory_type="observation",
        )
        
        results = memory.retrieve_memories("architecture", top_k=5)
        print(f"Retrieved {len(results)} memories")
        
        print(memory.get_stats())
    
    memory.persist()
    memory.clear()
    
    logger.success("Memory persistence test completed")
