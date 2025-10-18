#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import time
import torch
import psutil
import weakref
import threading
from collections import defaultdict
from utils.log.core import PiscesLxCoreLog
from typing import List, Dict, Any, Optional
from .types import ArcticAgentObservation, ArcticAgentAction

logger = PiscesLxCoreLog("Arctic.Core.Memory")

class ArcticMemoryManager:
    def __init__(self, enable_background: bool = True):
        """
        Initialize the MemoryManager instance.

        Args:
            enable_background (bool, optional): Whether to enable background memory monitoring. Defaults to True.
        """
        self.observations: List[ArcticAgentObservation] = []  # List to store agent observations
        self.actions: List[ArcticAgentAction] = []  # List to store agent actions
        self.reflections: List[str] = []  # List to store reflections

        # Attributes for enhanced memory management
        self.embeddings: List[torch.Tensor] = []  # Semantic embeddings for memory retrieval
        self.importance_scores: List[float] = []  # Importance scores for memory compression
        self.max_memory_size = 1000  # Maximum number of memories that can be stored
        self.compression_threshold = 0.7  # Threshold for memory compression

        # Background monitoring attributes
        self.enable_background = enable_background
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # Registry for tracking tensors
        self._tensor_registry: Dict[str, Dict[str, Any]] = {}
        self._tensor_lock = threading.Lock()

    def start_monitoring(self):
        """
        Start the background thread for memory monitoring if background monitoring is enabled and the thread is not already running.
        """
        if self.enable_background and self.monitoring_thread is None:
            self.monitoring_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """
        Stop the background memory monitoring thread if it is running.
        """
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join()
            self.monitoring_thread = None

    def _monitor_memory(self):
        """
        Background thread function to monitor memory usage periodically.
        Logs high memory usage and triggers garbage collection if necessary.
        """
        while not self.stop_monitoring.is_set():
            try:
                # Get the current process and monitor memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()

                # Log if memory usage exceeds 80%
                if memory_percent > 80.0:
                    logger.debug(f"High memory usage: {memory_percent:.2f}%")

                # Trigger garbage collection if memory usage exceeds 90%
                if memory_percent > 90.0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

            # Check memory usage every 5 seconds
            time.sleep(5)

    def register_tensor(self, tensor: torch.Tensor, name: str):
        """
        Register a tensor for memory monitoring.

        Args:
            tensor (torch.Tensor): The tensor to be registered.
            name (str): The name of the tensor.
        """
        if tensor is None:
            return
        try:
            numel = tensor.numel()
            elem_size = tensor.element_size()
            nbytes = int(numel * elem_size)
            device = str(tensor.device)
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else None
        except Exception:
            nbytes = None
            device = "unknown"
            shape = None

        info = {
            "name": name,
            "bytes": nbytes,
            "device": device,
            "shape": shape,
            "created_at": time.time(),
        }
        try:
            with self._tensor_lock:
                self._tensor_registry[name] = info
        except Exception:
            # If lock or registry is unavailable, perform a soft fail
            self._tensor_registry[name] = info

        # Auto-cleanup registry entry when tensor is garbage-collected
        self_ref = weakref.ref(self)

        def _on_finalize(n=name, self_ref=self_ref):
            self_obj = self_ref()
            if self_obj is not None:
                try:
                    with self_obj._tensor_lock:
                        self_obj._tensor_registry.pop(n, None)
                except Exception:
                    self_obj._tensor_registry.pop(n, None)

        try:
            weakref.finalize(tensor, _on_finalize)
        except Exception as e:
            # If finalizer cannot be attached, continue execution
            logger.debug(f"Finalize not attached for tensor '{name}': {e}")

        logger.debug(f"Registered tensor '{name}' shape={shape} bytes={nbytes} device={device}")

    def add_observation(self, observation: ArcticAgentObservation):
        """
        Add an observation to the memory and generate its corresponding embedding and importance score.

        Args:
            observation (ArcticAgentObservation): The observation to be added.
        """
        self.observations.append(observation)
        try:
            # Generate semantic embedding using SentenceTransformer
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(str(observation.content), convert_to_tensor=True)

            # Calculate importance score based on content analysis
            content_str = str(observation.content)
            length_factor = min(len(content_str) / 100.0, 1.0)
            unique_words = len(set(content_str.lower().split()))
            total_words = max(len(content_str.split()), 1)
            complexity_factor = unique_words / total_words
            keyword_factor = sum(1 for word in content_str.lower().split() 
                               if word in ['important', 'critical', 'urgent', 'key', 'essential']) * 0.1

            importance = min(1.0, (length_factor * 0.3 + complexity_factor * 0.5 + keyword_factor * 0.2))

        except Exception:
            # Fallback to structured random embedding if SentenceTransformer fails
            import hashlib
            content_hash = int(hashlib.md5(str(observation.content).encode()).hexdigest(), 16)
            torch.manual_seed(content_hash % 2147483647)
            embedding = torch.randn(768)
            importance = min(1.0, len(str(observation.content)) / 100.0)

        self.embeddings.append(embedding)
        self.importance_scores.append(importance)

        # Trigger memory compression if the number of observations exceeds the maximum capacity
        if len(self.observations) > self.max_memory_size:
            self.compress_memory()

    def add_action(self, action: ArcticAgentAction):
        """
        Add an action to the memory and generate its corresponding embedding and importance score.

        Args:
            action (ArcticAgentAction): The action to be added.
        """
        self.actions.append(action)
        # Generate a random embedding for the action
        embedding = torch.randn(768)
        self.embeddings.append(embedding)
        # Use action confidence as the importance score
        importance = action.confidence
        self.importance_scores.append(importance)

    def add_reflection(self, reflection: str):
        """
        Add a reflection to the memory and generate its corresponding embedding and importance score.

        Args:
            reflection (str): The reflection to be added.
        """
        self.reflections.append(reflection)
        # Generate a random embedding for the reflection
        embedding = torch.randn(768)
        self.embeddings.append(embedding)
        importance = min(1.0, len(reflection) / 200.0)
        self.importance_scores.append(importance)

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search through memory using cosine similarity with enhanced relevance scoring.

        Args:
            query (str): The query string for semantic search.
            k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search results with memory type, content, similarity score, index, and importance.
        """
        if not self.embeddings:
            return []

        try:
            # Generate query embedding using SentenceTransformer
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query, convert_to_tensor=True)
        except Exception:
            # Fallback to structured random query embedding if SentenceTransformer fails
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            torch.manual_seed(query_hash % 2147483647)
            query_embedding = torch.randn(768)

        # Calculate enhanced similarity scores
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            semantic_similarity = torch.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()

            # Boost relevance based on importance score
            importance_boost = self.importance_scores[i] * 0.2

            # Apply time-based decay (more recent memories have higher relevance)
            time_decay = 1.0 - (i / max(len(self.embeddings), 1)) * 0.1

            final_score = semantic_similarity + importance_boost + time_decay
            similarities.append((i, final_score))

        # Sort results by enhanced similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []

        for idx, similarity in similarities[:k]:
            if idx < len(self.observations):
                results.append({
                    "type": "observation",
                    "content": self.observations[idx],
                    "similarity": similarity,
                    "index": idx,
                    "importance": self.importance_scores[idx]
                })
            elif idx < len(self.observations) + len(self.actions):
                action_idx = idx - len(self.observations)
                results.append({
                    "type": "action",
                    "content": self.actions[action_idx],
                    "similarity": similarity,
                    "index": idx,
                    "importance": self.importance_scores[idx]
                })
            else:
                reflection_idx = idx - len(self.observations) - len(self.actions)
                results.append({
                    "type": "reflection",
                    "content": self.reflections[reflection_idx],
                    "similarity": similarity,
                    "index": idx,
                    "importance": self.importance_scores[idx]
                })

        return results

    def compress_memory(self):
        """
        Perform intelligent memory compression based on importance scores.
        Remove low-importance memories to reduce memory usage.
        """
        if not self.importance_scores:
            return

        # Calculate the threshold for low-importance memories
        threshold = sorted(self.importance_scores)[int(len(self.importance_scores) * self.compression_threshold)]

        # Identify indices of memories to keep
        keep_indices = [i for i, score in enumerate(self.importance_scores) 
                       if score >= threshold]

        # Compress each type of memory
        self.observations = [self.observations[i] for i in keep_indices if i < len(self.observations)]
        self.actions = [self.actions[i] for i in keep_indices 
                       if len(self.observations) <= i < len(self.observations) + len(self.actions)]
        self.reflections = [self.reflections[i] for i in keep_indices 
                         if i >= len(self.observations) + len(self.actions)]
        self.embeddings = [self.embeddings[i] for i in keep_indices]
        self.importance_scores = [self.importance_scores[i] for i in keep_indices]

    def get_context_with_retrieval(self, query: str = None, k: int = 5) -> Dict[str, Any]:
        """
        Get context with semantic retrieval if a query is provided, otherwise get recent context.

        Args:
            query (str, optional): The query string for semantic retrieval. Defaults to None.
            k (int, optional): The number of top results to return for semantic search or recent context. Defaults to 5.

        Returns:
            Dict[str, Any]: A dictionary containing relevant memories and total memory count if query is provided, or recent context otherwise.
        """
        if query:
            relevant_memories = self.semantic_search(query, k)
            return {
                "relevant_memories": relevant_memories,
                "total_count": len(self.observations) + len(self.actions) + len(self.reflections)
            }
        else:
            return self.get_recent_context(k)

    def get_recent_context(self, k: int = 5) -> Dict[str, List]:
        """
        Get the most recent observations, actions, and reflections.

        Args:
            k (int, optional): The number of recent items to return for each memory type. Defaults to 5.

        Returns:
            Dict[str, List]: A dictionary containing recent observations, actions, reflections, total memory count, and a memory summary.
        """
        return {
            "recent_observations": self.observations[-k:],
            "recent_actions": self.actions[-k:],
            "recent_reflections": self.reflections[-k:],
            "total_count": len(self.observations) + len(self.actions) + len(self.reflections),
            "memory_summary": {
                "observations": len(self.observations),
                "actions": len(self.actions),
                "reflections": len(self.reflections)
            }
        }