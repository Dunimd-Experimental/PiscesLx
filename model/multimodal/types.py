#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import uuid
import torch
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

class ArcticAgenticState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"

class ArcticMCPMessageType(Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    CAPABILITY_REGISTER = "capability_register"
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

@dataclass
class ArcticGenerationCondition:
    text_prompt: str = ""
    emotion_vector: Optional[torch.Tensor] = None
    style_params: Optional[Dict[str, float]] = None
    generation_params: Optional[Dict[str, Any]] = None

@dataclass
class ArcticMCPMessage:
    message_type: str
    agentic_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"

@dataclass
class ArcticAgenticAction:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class ArcticAgenticObservation:
    modality: str  # "text", "image", "audio", "tool_result"
    content: Any
    metadata: Dict[str, Any]

@dataclass
class ArcticAgenticMemory:
    observations: List[ArcticAgenticObservation]
    actions: List[ArcticAgenticAction]
    reflections: List[str]
    
    def __post_init__(self):
        self.embeddings: List[torch.Tensor] = []
        self.importance_scores: List[float] = []
        self.max_memory_size = 1000
        self.compression_threshold = 0.7
    
    def add_observation(self, observation: ArcticAgenticObservation):
        self.observations.append(observation)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(str(observation.content), convert_to_tensor=True)
            content_str = str(observation.content)
            length_factor = min(len(content_str) / 100.0, 1.0)
            unique_words = len(set(content_str.lower().split()))
            total_words = max(len(content_str.split()), 1)
            complexity_factor = unique_words / total_words
            keyword_factor = sum(1 for word in content_str.lower().split() 
                               if word in ['important', 'critical', 'urgent', 'key', 'essential']) * 0.1
            importance = min(1.0, (length_factor * 0.3 + complexity_factor * 0.5 + keyword_factor * 0.2))
        except Exception:
            import hashlib
            content_hash = int(hashlib.md5(str(observation.content).encode()).hexdigest(), 16)
            torch.manual_seed(content_hash % 2147483647)
            embedding = torch.randn(768)
            importance = min(1.0, len(str(observation.content)) / 100.0)
        self.embeddings.append(embedding)
        self.importance_scores.append(importance)
        if len(self.observations) > self.max_memory_size:
            self.compress_memory()
    
    def add_action(self, action: ArcticAgenticAction):
        self.actions.append(action)
        embedding = torch.randn(768)
        self.embeddings.append(embedding)
        self.importance_scores.append(action.confidence)
    
    def add_reflection(self, reflection: str):
        self.reflections.append(reflection)
        embedding = torch.randn(768)
        self.embeddings.append(embedding)
        importance = min(1.0, len(reflection) / 200.0)
        self.importance_scores.append(importance)
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query, convert_to_tensor=True)
        except Exception:
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            torch.manual_seed(query_hash % 2147483647)
            query_embedding = torch.randn(768)
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            semantic_similarity = torch.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            importance_boost = self.importance_scores[i] * 0.2
            time_decay = 1.0 - (i / max(len(self.embeddings), 1)) * 0.1
            final_score = semantic_similarity + importance_boost + time_decay
            similarities.append((i, final_score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, similarity in similarities[:k]:
            if idx < len(self.observations):
                results.append({"type": "observation","content": self.observations[idx],"similarity": similarity,"index": idx,"importance": self.importance_scores[idx]})
            elif idx < len(self.observations) + len(self.actions):
                action_idx = idx - len(self.observations)
                results.append({"type": "action","content": self.actions[action_idx],"similarity": similarity,"index": idx,"importance": self.importance_scores[idx]})
            else:
                reflection_idx = idx - len(self.observations) - len(self.actions)
                results.append({"type": "reflection","content": self.reflections[reflection_idx],"similarity": similarity,"index": idx,"importance": self.importance_scores[idx]})
        return results
    
    def compress_memory(self):
        if not self.importance_scores:
            return
        threshold = sorted(self.importance_scores)[int(len(self.importance_scores) * self.compression_threshold)]
        keep_indices = [i for i, score in enumerate(self.importance_scores) if score >= threshold]
        self.observations = [self.observations[i] for i in keep_indices if i < len(self.observations)]
        self.actions = [self.actions[i] for i in keep_indices if len(self.observations) <= i < len(self.observations) + len(self.actions)]
        self.reflections = [self.reflections[i] for i in keep_indices if i >= len(self.observations) + len(self.actions)]
        self.embeddings = [self.embeddings[i] for i in keep_indices]
        self.importance_scores = [self.importance_scores[i] for i in keep_indices]
    
    def get_context_with_retrieval(self, query: str = None, k: int = 5) -> Dict[str, Any]:
        if query:
            relevant_memories = self.semantic_search(query, k)
            return {"relevant_memories": relevant_memories,"total_count": len(self.observations) + len(self.actions) + len(self.reflections)}
        else:
            return self.get_recent_context(k)
    
    def get_recent_context(self, k: int = 5) -> Dict[str, List]:
        return {"recent_observations": self.observations[-k:],"recent_actions": self.actions[-k:],"recent_reflections": self.reflections[-k:],"total_count": len(self.observations) + len(self.actions) + len(self.reflections),"memory_summary": {"observations": len(self.observations),"actions": len(self.actions),"reflections": len(self.reflections)}}