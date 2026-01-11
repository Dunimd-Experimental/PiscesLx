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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import hashlib
import re

class RuchbahSemanticEncoder(nn.Module):
    
    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 71164,
        embedding_dim: int = 512,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.feature_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.context_encoder = nn.GRU(
            embedding_dim, hidden_size // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        self.special_token_embedding = nn.Parameter(torch.randn(4, embedding_dim))
        
        self.register_buffer("_freqs_cis", self._precompute_freqs_cis(embedding_dim))
    
    def _precompute_freqs_cis(self, dim: int, max_len: int = 2048) -> torch.Tensor:
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def _encode_text_simple(self, text: str) -> torch.Tensor:
        tokens = []
        text_lower = text.lower().strip()
        
        special_patterns = [
            (r'<\|goal\|>', 0),
            (r'<\|context\|>', 1),
            (r'<\|action\|>', 2),
            (r'<\|observation\|>', 3),
        ]
        
        pos = 0
        for pattern, special_id in special_patterns:
            match = re.search(pattern, text_lower[pos:], re.IGNORECASE)
            if match:
                if match.start() > 0:
                    words = text_lower[pos:pos + match.start()].split()
                    for word in words[:self.max_seq_len - 4]:
                        tokens.append(hash(word) % (self.vocab_size - 4) + 4)
                tokens.append(special_id + 4)
                pos += match.end()
        
        if pos < len(text_lower) and len(tokens) < self.max_seq_len - 4:
            remaining = text_lower[pos:].split()
            for word in remaining[:self.max_seq_len - 4 - len(tokens)]:
                tokens.append(hash(word) % (self.vocab_size - 4) + 4)
        
        if not tokens:
            tokens = [hash(text_lower) % (self.vocab_size - 4) + 4]
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    def forward(
        self,
        text: str,
        encode_type: str = "semantic",
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        
        if encode_type == "simple":
            tokens = self._encode_text_simple(text)
            embeddings = self.token_embedding(tokens)
        else:
            tokens = self._encode_text_simple(text)
            seq_len = tokens.size(1)
            embeddings = self.token_embedding(tokens)
            
            positions = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device)
            pos_emb = self.position_embedding(positions)
            embeddings = embeddings + pos_emb.unsqueeze(0)
            
            if add_special_tokens:
                batch_size = embeddings.size(0)
                special_emb = self.special_token_embedding.unsqueeze(0).expand(batch_size, -1, -1)
                embeddings = torch.cat([special_emb, embeddings], dim=1)
        
        output = self.feature_projection(embeddings)
        
        context_out, hidden = self.context_encoder(output)
        
        pooled = context_out.mean(dim=1)
        
        return {
            "embeddings": output,
            "pooled": pooled,
            "tokens": tokens if encode_type == "simple" else None,
            "hidden_state": hidden,
            "context": context_out
        }
    
    def encode_goal(self, goal: str) -> torch.Tensor:
        result = self(goal, encode_type="semantic", add_special_tokens=True)
        return result["pooled"]
    
    def encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        context_text = str(context)[:1000]
        result = self(context_text, encode_type="semantic", add_special_tokens=False)
        return result["pooled"]
    
    def encode_with_history(
        self,
        current: str,
        history: List[Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        
        combined = current
        for h in history[-5:]:
            if "observation" in h:
                combined = f"<|observation|> {h['observation']} {combined}"
            if "action" in h:
                combined = f"<|action|> {h['action']} {combined}"
            if "thought" in h:
                combined = f"<|goal|> {h['thought']} {combined}"
        
        result = self(combined, encode_type="semantic", add_special_tokens=True)
        
        return {
            "embeddings": result["embeddings"],
            "pooled": result["pooled"],
            "context": result["context"],
            "hidden_state": result["hidden_state"]
        }


class RuchbahHybridEncoder(nn.Module):
    
    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 71164,
        embedding_dim: int = 512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.semantic_encoder = RuchbahSemanticEncoder(
            hidden_size, vocab_size, embedding_dim
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        text: str,
        additional_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        semantic_result = self.semantic_encoder(text)
        semantic_features = semantic_result["pooled"]
        
        if additional_features is not None:
            additional_features = additional_features.to(semantic_features.device)
            if additional_features.dim() == 1:
                additional_features = additional_features.unsqueeze(0)
            if additional_features.size(-1) != self.hidden_size:
                additional_features = F.adaptive_avg_pool1d(
                    additional_features.unsqueeze(0), self.hidden_size
                ).squeeze(0)
            
            gate_value = self.gate(semantic_features)
            fused = self.feature_fusion(
                torch.cat([semantic_features, additional_features], dim=-1)
            )
            final_features = gate_value * fused + (1 - gate_value) * semantic_features
        else:
            final_features = semantic_features
        
        return {
            "semantic_embeddings": semantic_result["embeddings"],
            "fused_embeddings": final_features.unsqueeze(0) if final_features.dim() == 1 else final_features,
            "pooled": final_features,
            "context": semantic_result.get("context"),
            "hidden_state": semantic_result.get("hidden_state")
        }
