#!/usr/bin/env/python3
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

"""Semantic text encoding utilities for Yv multimodal agents.

This module provides semantic text encoding components for the Yv model,
including text tokenization, embedding, context encoding, and feature fusion
for agent goal and context representation.

Module Components:
    1. YvSemanticEncoder:
       - Text tokenization with special token handling
       - Token and positional embeddings
       - Bidirectional GRU context encoding
       - Goal, context, and history encoding

    2. YvHybridEncoder:
       - Semantic and additional feature fusion
       - Gated feature combination
       - Multi-modal feature integration

Key Features:
    - Hash-based word tokenization
    - Special token patterns (goal, context, action, observation)
    - Precomputed rotary frequency cache
    - Bidirectional GRU for context modeling
    - Gated feature fusion for hybrid encoding

Performance Characteristics:
    - Tokenization: O(L) where L = text length
    - Embedding: O(L * embedding_dim)
    - GRU encoding: O(L * hidden_size^2)
    - Total complexity: O(L * hidden_size^2)

Usage Example:
    >>> from model.multimodal.semantic_encoder import YvSemanticEncoder
    >>> 
    >>> # Initialize encoder
    >>> encoder = YvSemanticEncoder(hidden_size=2048)
    >>> 
    >>> # Encode text
    >>> result = encoder("Hello world", encode_type="semantic")
    >>> embeddings = result["embeddings"]  # [B, seq_len, hidden_size]
    >>> pooled = result["pooled"]  # [B, hidden_size]
    >>> 
    >>> # Encode goal
    >>> goal_embedding = encoder.encode_goal("Complete the task")

Note:
    Default vocabulary size: 151,646 tokens
    Default embedding dimension: 512
    Default max sequence length: 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import hashlib
import re

class YvSemanticEncoder(nn.Module):
    """Semantic text encoder for agent goal and context representation.
    
    A comprehensive text encoder that tokenizes input text, applies token and
    positional embeddings, encodes context through bidirectional GRU, and
    produces semantic representations for agent workflows.
    
    Architecture:
        1. Tokenization:
           - Hash-based word tokenization
           - Special token pattern recognition
           - Vocabulary mapping with reserved tokens
        
        2. Embedding:
           - Token embedding (151,646 vocab, 512 dim)
           - Positional embedding (512 max length)
           - Special token parameters (4 tokens)
        
        3. Projection:
           - Linear projection to hidden_size
           - LayerNorm and GELU activation
        
        4. Context Encoding:
           - Bidirectional GRU (2 layers)
           - Dropout regularization (0.1)
    
    Key Features:
        - Hash-based word tokenization for vocabulary flexibility
        - Special token patterns for agent workflows
        - Precomputed rotary frequency cache
        - Bidirectional GRU for context modeling
        - Goal, context, and history encoding methods
    
    Attributes:
        hidden_size (int): Output embedding dimension.
        vocab_size (int): Vocabulary size for token embedding.
        embedding_dim (int): Intermediate embedding dimension.
        max_seq_len (int): Maximum sequence length for positional encoding.
        token_embedding (nn.Embedding): Token embedding layer.
        position_embedding (nn.Embedding): Positional embedding layer.
        feature_projection (nn.Sequential): Projection to hidden_size.
        context_encoder (nn.GRU): Bidirectional GRU for context modeling.
        special_token_embedding (nn.Parameter): Special token embeddings.
        _freqs_cis (torch.Tensor): Precomputed rotary frequencies.
    
    Example:
        >>> encoder = YvSemanticEncoder(hidden_size=2048)
        >>> result = encoder("Hello world", encode_type="semantic")
        >>> embeddings = result["embeddings"]  # [B, seq_len, hidden_size]
    
    Note:
        Default vocabulary size: 151,646 tokens
        Special tokens: goal, context, action, observation
        Uses hash-based tokenization for out-of-vocabulary words.
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 151646,
        embedding_dim: int = 512,
        max_seq_len: int = 512
    ):
        """Initialize the semantic encoder with configuration.
        
        Args:
            hidden_size (int): Output embedding dimension. Default: 2048.
            vocab_size (int): Vocabulary size for token embedding. Default: 151646.
            embedding_dim (int): Intermediate embedding dimension. Default: 512.
            max_seq_len (int): Maximum sequence length. Default: 512.
        """
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
        """Precompute rotary position embedding frequencies.
        
        Computes complex exponentials for rotary position embeddings
        following the standard RoPE formulation.
        
        Args:
            dim (int): Dimension for frequency computation.
            max_len (int): Maximum sequence length. Default: 2048.
        
        Returns:
            torch.Tensor: Precomputed complex frequencies [max_len, dim//2].
        
        Note:
            Uses base frequency of 10000 for geometric progression.
            Returns complex tensor for efficient rotation computation.
        """
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def _encode_text_simple(self, text: str) -> torch.Tensor:
        """Tokenize text using hash-based word tokenization.
        
        Converts input text to token indices using hash-based tokenization
        with special token pattern recognition for agent workflows.
        
        Args:
            text (str): Input text to tokenize.
        
        Returns:
            torch.Tensor: Token indices [1, seq_len].
        
        Note:
            Special patterns: <|goal|>, <|context|>, <|action|>, <|observation|>
            Uses hash(word) % vocab_size for out-of-vocabulary words.
            Truncates to max_seq_len - 4 to reserve space for special tokens.
        """
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
        """Encode text into semantic representations.
        
        Main entry point for text encoding. Tokenizes input, applies embeddings,
        projects to hidden size, and encodes context through bidirectional GRU.
        
        Args:
            text (str): Input text to encode.
            encode_type (str): Encoding mode:
                - "simple": Token embedding only, no positional encoding
                - "semantic": Full encoding with positional and special tokens
                Default: "semantic".
            add_special_tokens (bool): Whether to prepend special token embeddings.
                Default: True.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - embeddings: [B, seq_len, hidden_size] token embeddings
                - pooled: [B, hidden_size] pooled representation
                - tokens: Token indices (only for "simple" mode)
                - hidden_state: GRU hidden state
                - context: [B, seq_len, hidden_size] context output
        
        Note:
            Semantic mode adds positional embeddings and optional special tokens.
            Pooled output is computed as mean over context dimension.
        """
        
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
        """Encode an agent goal string into a pooled representation.
        
        Convenience method for encoding agent goals with semantic
        encoding and special tokens enabled.
        
        Args:
            goal (str): Goal text to encode.
        
        Returns:
            torch.Tensor: Pooled goal representation [B, hidden_size].
        """
        result = self(goal, encode_type="semantic", add_special_tokens=True)
        return result["pooled"]
    
    def encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode a context dictionary into a pooled representation.
        
        Converts context dictionary to string and encodes with semantic
        encoding. Truncates to 1000 characters for efficiency.
        
        Args:
            context (Dict[str, Any]): Context dictionary to encode.
        
        Returns:
            torch.Tensor: Pooled context representation [B, hidden_size].
        
        Note:
            Context is converted to string using str() and truncated to 1000 chars.
        """
        context_text = str(context)[:1000]
        result = self(context_text, encode_type="semantic", add_special_tokens=False)
        return result["pooled"]
    
    def encode_with_history(
        self,
        current: str,
        history: List[Dict[str, str]]
    ) -> Dict[str, torch.Tensor]:
        """Encode current text with conversation history.
        
        Combines current text with up to 5 most recent history entries,
        wrapping each with appropriate special tokens (observation, action, goal).
        
        Args:
            current (str): Current text to encode.
            history (List[Dict[str, str]]): List of history entries, each containing
                optional 'observation', 'action', and 'thought' keys.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - embeddings: [B, seq_len, hidden_size] token embeddings
                - pooled: [B, hidden_size] pooled representation
                - context: [B, seq_len, hidden_size] context output
                - hidden_state: GRU hidden state
        
        Note:
            Uses last 5 history entries maximum.
            History is prepended to current text with special token markers.
        """
        
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


class YvHybridEncoder(nn.Module):
    """Hybrid encoder combining semantic text with additional features.
    
    A comprehensive encoder that combines semantic text encoding with
    additional feature inputs through gated fusion, enabling multi-modal
    representation learning for agent workflows.
    
    Architecture:
        1. Semantic Encoding:
           - YvSemanticEncoder for text processing
           - Pooled semantic representation
        
        2. Feature Fusion:
           - Concatenation of semantic and additional features
           - Linear projection with LayerNorm and GELU
        
        3. Gated Combination:
           - Sigmoid gate for feature weighting
           - Weighted sum of fused and semantic features
    
    Key Features:
        - Semantic text encoding via YvSemanticEncoder
        - Additional feature integration with adaptive pooling
        - Gated fusion for balanced feature combination
        - Multi-modal representation output
    
    Attributes:
        hidden_size (int): Output embedding dimension.
        semantic_encoder (YvSemanticEncoder): Text encoder module.
        feature_fusion (nn.Sequential): Feature fusion network.
        gate (nn.Sequential): Gating network for feature combination.
    
    Example:
        >>> encoder = YvHybridEncoder(hidden_size=2048)
        >>> result = encoder("Hello world", additional_features=vision_features)
        >>> fused = result["fused_embeddings"]  # [B, hidden_size]
    
    Note:
        Additional features are adaptively pooled to hidden_size if needed.
        Gate controls the balance between fused and semantic features.
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 151646,
        embedding_dim: int = 512
    ):
        """Initialize the hybrid encoder with configuration.
        
        Args:
            hidden_size (int): Output embedding dimension. Default: 2048.
            vocab_size (int): Vocabulary size for semantic encoder. Default: 151646.
            embedding_dim (int): Embedding dimension for semantic encoder. Default: 512.
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.semantic_encoder = YvSemanticEncoder(
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
        """Encode text with optional additional features through gated fusion.
        
        Main entry point for hybrid encoding. Encodes text semantically,
        optionally combines with additional features through gated fusion.
        
        Args:
            text (str): Input text to encode.
            additional_features (Optional[torch.Tensor]): Optional additional
                features to combine with semantic encoding.
                Shape: [B, hidden_size] or [hidden_size].
                Will be adaptively pooled if dimension doesn't match.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - semantic_embeddings: [B, seq_len, hidden_size] text embeddings
                - fused_embeddings: [B, hidden_size] fused representation
                - pooled: [B, hidden_size] pooled representation
                - context: [B, seq_len, hidden_size] context output
                - hidden_state: GRU hidden state
        
        Note:
            If additional_features is None, returns semantic features only.
            Gate controls balance: gate * fused + (1 - gate) * semantic.
        """
        
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
