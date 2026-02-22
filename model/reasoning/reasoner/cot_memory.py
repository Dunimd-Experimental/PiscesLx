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

"""Chain-of-thought reasoning module with adaptive memory integration.

This module provides the YvCoTMemoryReasoner class which implements
chain-of-thought reasoning with adaptive depth control, memory retrieval,
and auxiliary diagnostic heads.

Architecture:
    1. Reasoning Layers:
       - Stack of TransformerEncoderLayer for multi-step reasoning
       - Adaptive depth based on problem complexity
       - Early stopping via abstraction gain monitoring
    
    2. Memory Integration:
       - Key-value projection for memory entries
       - Multi-head attention for memory retrieval
       - Residual connection with memory output
    
    3. Auxiliary Heads:
       - thinking_head: Next token prediction logits
       - difficulty_head: Problem difficulty classification (5 levels)
       - reflection_head: Reflection type prediction (4 types)
       - confidence_head: Confidence score estimation
       - error_analyzer: Error type, severity, and confidence
       - correction_head: Correction proposal generation

Key Features:
    - Adaptive reasoning depth (1-3 layers based on complexity)
    - Memory-augmented reasoning with attention
    - Early stopping via abstraction gain threshold
    - Comprehensive diagnostic outputs
    - Error analysis and correction proposals

Performance Characteristics:
    - Complexity Estimation: O(L * H) where L = seq_len, H = hidden_size
    - Memory Attention: O(L * M * H) where M = memory entries
    - Reasoning Layers: O(T * L * H^2) where T = active layers

Usage Example:
    >>> from model.reasoning.reasoner import YvCoTMemoryReasoner
    >>> 
    >>> # Initialize reasoner
    >>> reasoner = YvCoTMemoryReasoner(config)
    >>> 
    >>> # Forward pass with memory context
    >>> output = reasoner.forward(
    ...     input_ids=hidden_states,
    ...     memory_context=[mem1, mem2, mem3]
    >>> )
    >>> 
    >>> # Access outputs
    >>> thinking = output["thinking_logits"]
    >>> difficulty = output["difficulty_logits"]
    >>> confidence = output["confidence_score"]

Dependencies:
    - torch: Tensor operations and neural network modules
    - torch.nn.functional: Activation functions and normalization

Note:
    The reasoner uses TransformerEncoderLayer which implements
    pre-norm architecture with residual connections.
"""

import torch
from torch import nn
import torch.nn.functional as F


class YvCoTMemoryReasoner(nn.Module):
    """Execute multi-step CoT reasoning with adaptive depth and memory fusion.
    
    This class implements chain-of-thought reasoning with adaptive depth
    control based on problem complexity, memory integration via attention,
    and multiple auxiliary heads for diagnostics.
    
    Architecture:
        1. Input Processing:
           - Accepts pre-computed hidden states or generates random fallback
           - Integrates memory context via key-value attention
        
        2. Reasoning Pipeline:
           - LayerNorm for input stabilization
           - Adaptive depth selection (1-3 layers)
           - Early stopping via abstraction gain monitoring
           - Residual connections throughout
        
        3. Output Heads:
           - thinking_head: [B, H] -> [B, V] for token prediction
           - difficulty_head: [B, H] -> [B, 5] for difficulty levels
           - reflection_head: [B, H] -> [B, 4] for reflection types
           - confidence_head: [B, H] -> [B, 1] for confidence score
           - error_analyzer: [B, H] -> [B, 3] for error analysis
           - correction_head: [B, H+3] -> [B, H] for corrections
    
    Complexity Estimation:
        - Length complexity: min(seq_len / 256, 1.0)
        - Diversity complexity: sigmoid(semantic_variance * 15)
        - Reasoning complexity: sigmoid(energy * 0.01)
        - Final: 0.4 * length + 0.4 * diversity + 0.2 * reasoning
    
    Abstraction Gain:
        - Semantic advance: 1 - cosine_similarity(prev, curr)
        - Consolidation ratio: (prev_info - curr_info) / prev_info
        - Coherence score: sigmoid(1 / state_diff)
        - Final: 0.5 * advance + 0.3 * consolidation + 0.2 * coherence
    
    Attributes:
        cfg (Any): Configuration namespace with hyperparameters.
        hidden_size (int): Hidden dimension size.
        vocab_size (int): Vocabulary size for output predictions.
        reasoning_layers (nn.ModuleList): Stack of transformer encoder layers.
        depth_controller (nn.Sequential): Network for depth prediction.
        thinking_head (nn.Linear): Token prediction head.
        difficulty_head (nn.Linear): Difficulty classification head.
        reflection_head (nn.Linear): Reflection type head.
        confidence_head (nn.Linear): Confidence estimation head.
        memory_key_proj (nn.Linear): Memory key projection.
        memory_value_proj (nn.Linear): Memory value projection.
        memory_attention (nn.MultiheadAttention): Memory attention module.
        error_analyzer (nn.Sequential): Error analysis network.
        correction_head (nn.Sequential): Correction generation network.
        layer_norm (nn.LayerNorm): Input normalization layer.
    
    Example:
        >>> reasoner = YvCoTMemoryReasoner(config)
        >>> output = reasoner.forward(input_ids=hidden_states)
        >>> print(output["thinking_logits"].shape)
        torch.Size([batch_size, vocab_size])
    
    Note:
        The reasoner automatically adjusts depth based on complexity.
        Early stopping occurs when abstraction gain falls below 0.1.
    """
    
    def __init__(self, cfg):
        """Create the reasoning module from a configuration namespace.
        
        Initializes all components including reasoning layers, memory
        attention, and auxiliary heads.
        
        Args:
            cfg: Object containing hyperparameters such as:
                - hidden_size (int): Hidden dimension size
                - vocab_size (int): Vocabulary size
                - n_head (int): Number of attention heads
        
        Note:
            Uses TransformerEncoderLayer with:
            - d_model = hidden_size
            - nhead = n_head
            - dim_feedforward = hidden_size * 4
            - dropout = 0.1
            - batch_first = True
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        
        # Initialize a list of Transformer encoder layers for multi-step reasoning.
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.n_head,
                dim_feedforward=cfg.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Initialize a depth controller to adaptively adjust the reasoning depth.
        self.depth_controller = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # Initialize output heads for different prediction tasks.
        self.thinking_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.difficulty_head = nn.Linear(cfg.hidden_size, 5)
        self.reflection_head = nn.Linear(cfg.hidden_size, 4)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)

        # Initialize modules related to memory mechanism.
        self.memory_key_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_value_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_head,
            dropout=0.1,
            batch_first=True
        )

        # Initialize an error analysis module to predict error type, severity, and confidence.
        self.error_analyzer = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 3)  # error_type, severity, confidence
        )

        # Initialize a correction head to generate correction logits.
        # Input is concatenation of final_state (hidden_size) and error_analysis (3).
        self.correction_head = nn.Sequential(
            nn.Linear(cfg.hidden_size + 3, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )

        # Initialize layer normalization to stabilize the training process.
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)

    def _calculate_problem_complexity(self, hidden_states):
        """Estimate problem complexity using sequence statistics and energy proxies.
        
        Computes a normalized complexity score in [0, 1] based on three components:
        sequence length, semantic diversity, and activation energy.
        
        Args:
            hidden_states (torch.Tensor): Tensor shaped [batch, seq_len, hidden]
                representing current reasoning states.
        
        Returns:
            float: Scalar complexity score normalized to [0, 1].
                - < 0.3: Simple problems (1 reasoning layer)
                - 0.3-0.6: Medium problems (2 reasoning layers)
                - > 0.6: Complex problems (3 reasoning layers)
        
        Complexity Components:
            1. Length Complexity (weight 0.4):
               - Normalized by dividing by 256
               - Capped at 1.0
            
            2. Diversity Complexity (weight 0.4):
               - Semantic variance across sequence
               - Scaled by factor 15 and passed through sigmoid
            
            3. Reasoning Complexity (weight 0.2):
               - Energy proxy from squared activations
               - Scaled by factor 0.01 and passed through sigmoid
        
        Note:
            Uses torch.no_grad() for energy calculation to avoid
            accessing non-leaf .grad attributes which trigger warnings.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Calculate complexity based on the sequence length of the input.
        length_complexity = min(seq_len / 256, 1.0)
        # Calculate the mean state for semantic variance calculation.
        mean_state = hidden_states.mean(dim=1, keepdim=True)
        # Calculate the semantic variance of the input states.
        semantic_variance = torch.var(hidden_states - mean_state, dim=[1,2]).mean()
        # Calculate diversity complexity using the sigmoid function.
        diversity_complexity = torch.sigmoid(semantic_variance * 15)
        # Use a grad-free proxy to avoid accessing non-leaf .grad (which triggers warnings).
        with torch.no_grad():
            energy = (hidden_states.detach() ** 2).mean()
            reasoning_complexity = torch.sigmoid(energy * 0.01)
        # Combine three complexity components to get the final complexity score.
        complexity = (length_complexity * 0.4 + 
                      diversity_complexity * 0.4 + 
                      reasoning_complexity * 0.2)
        return complexity.item()

    def _calculate_abstraction_gain(self, prev_states, curr_states):
        """Assess semantic advancement between consecutive reasoning states.
        
        Computes an abstraction gain score to determine if reasoning should
        continue or stop early due to convergence.
        
        Args:
            prev_states (torch.Tensor): Tensor shaped [batch, seq_len, hidden]
                representing reasoning states before an iteration.
            curr_states (torch.Tensor): Tensor of the same shape representing
                states after an iteration.
        
        Returns:
            float: Scalar abstraction gain used to decide early stopping.
                - < 0.1: Low gain, suggests early stopping
                - > 0.1: Sufficient gain, continue reasoning
        
        Gain Components:
            1. Semantic Advance (weight 0.5):
               - 1 - cosine_similarity(prev_pooled, curr_pooled)
               - Measures directional change in representation
            
            2. Consolidation Ratio (weight 0.3):
               - (prev_info - curr_info) / prev_info
               - Measures information compression
               - Positive when information is consolidated
            
            3. Coherence Score (weight 0.2):
               - sigmoid(1 / state_diff)
               - Higher when states change smoothly
               - Lower when states change abruptly
        
        Note:
            Early stopping occurs when gain < 0.1, indicating the
            reasoning process has converged to a stable representation.
        """
        # Normalize the mean of the previous states.
        prev_pooled = F.normalize(prev_states.mean(dim=1), p=2, dim=-1)
        # Normalize the mean of the current states.
        curr_pooled = F.normalize(curr_states.mean(dim=1), p=2, dim=-1)
        # Calculate the semantic advance between previous and current states.
        semantic_advance = 1 - F.cosine_similarity(prev_pooled, curr_pooled, dim=-1).mean()
        # Calculate the information norm of the previous states.
        prev_info = torch.norm(torch.var(prev_states, dim=1), p=2)
        # Calculate the information norm of the current states.
        curr_info = torch.norm(torch.var(curr_states, dim=1), p=2)
        # Calculate the consolidation ratio of information.
        consolidation_ratio = max(0, (prev_info - curr_info) / (prev_info + 1e-8))
        # Calculate the mean difference between previous and current states.
        state_diff = torch.norm(curr_states - prev_states, p=2, dim=-1).mean()
        # Calculate the coherence score based on the state difference.
        coherence_score = torch.sigmoid(1.0 / (state_diff + 0.1))
        # Combine three components to get the abstraction gain.
        abstraction_gain = semantic_advance * 0.5 + consolidation_ratio * 0.3 + coherence_score * 0.2
        return abstraction_gain.item()

    def forward(self, input_ids=None, attention_mask=None, memory_context=None, **kwargs):
        """Run adaptive CoT reasoning over inputs with optional memory context.
        
        Performs multi-step chain-of-thought reasoning with adaptive depth
        control and optional memory integration.
        
        Args:
            input_ids (torch.Tensor | None): Pre-computed hidden states or token
                embeddings. Random noise is used when None is provided.
            attention_mask (torch.Tensor | None): Unused placeholder included for
                API compatibility. Default: None.
            memory_context (Sequence[torch.Tensor] | torch.Tensor | None): Optional
                memory entries incorporated via attention. Default: None.
            **kwargs: Additional keyword arguments reserved for future features.
        
        Returns:
            dict: Structured outputs containing:
                - thinking_logits (torch.Tensor): [B, V] token prediction logits
                - difficulty_logits (torch.Tensor): [B, 5] difficulty classification
                - reflection_logits (torch.Tensor): [B, 4] reflection type logits
                - confidence_score (torch.Tensor): [B, 1] confidence in [0, 1]
                - reasoning_states (torch.Tensor): [B, L, H] final hidden states
                - reasoning_steps (list): List of intermediate state tensors
                - correction_logits (torch.Tensor): [B, H] correction proposals
                - attention_weights (torch.Tensor | None): Memory attention weights
                - final_state (torch.Tensor): [B, H] pooled final representation
        
        Processing Pipeline:
            1. Input handling with random fallback
            2. Memory integration via key-value attention
            3. Complexity-based depth selection
            4. Multi-step reasoning with early stopping
            5. Auxiliary head predictions
        
        Note:
            Depth is selected based on complexity:
            - complexity < 0.3: 1 layer
            - 0.3 <= complexity < 0.6: 2 layers
            - complexity >= 0.6: 3 layers
            
            Early stopping occurs when abstraction gain < 0.1.
        """
        # Handle the input and initialize hidden states.
        if torch.is_tensor(input_ids):
            hidden_states = input_ids
        else:
            hidden_states = torch.randn(1, 1, self.cfg.hidden_size)

        batch_size, seq_len, _ = hidden_states.shape

        # Integrate memory context if it is provided.
        if memory_context is not None and len(memory_context) > 0:
            memory_tensor = torch.stack(memory_context).unsqueeze(0) if isinstance(memory_context[0], torch.Tensor) else torch.tensor(memory_context).unsqueeze(0)
            memory_keys = self.memory_key_proj(memory_tensor)
            memory_values = self.memory_value_proj(memory_tensor)
            memory_out, attention_weights = self.memory_attention(hidden_states, memory_keys, memory_values)
            hidden_states = hidden_states + memory_out
        else:
            attention_weights = None

        # Perform multi-step CoT reasoning with adaptive depth.
        reasoning_states = self.layer_norm(hidden_states)
        reasoning_steps = []
        complexity_score = self._calculate_problem_complexity(reasoning_states)
        if complexity_score < 0.3:
            num_layers = 1
        elif complexity_score < 0.6:
            num_layers = 2
        else:
            num_layers = len(self.reasoning_layers)
        
        for i, layer in enumerate(self.reasoning_layers[:num_layers]):
            prev_states = reasoning_states.clone()
            reasoning_states = layer(reasoning_states)
            reasoning_steps.append(reasoning_states.clone())
            if i > 0:
                abstraction_gain = self._calculate_abstraction_gain(prev_states, reasoning_states)
                if abstraction_gain < 0.1:
                    break
        
        final_state = reasoning_states[:, -1, :]
        thinking_logits = self.thinking_head(final_state)
        difficulty_logits = self.difficulty_head(final_state)
        reflection_logits = self.reflection_head(final_state)
        confidence_score = torch.sigmoid(self.confidence_head(final_state))
        error_analysis = self.error_analyzer(final_state)
        correction_input = torch.cat([final_state, error_analysis], dim=-1)
        correction_logits = self.correction_head(correction_input)
        
        return {
            "thinking_logits": thinking_logits,
            "difficulty_logits": difficulty_logits,
            "reflection_logits": reflection_logits,
            "confidence_score": confidence_score,
            "reasoning_states": reasoning_states,
            "reasoning_steps": reasoning_steps,
            "correction_logits": correction_logits,
            "attention_weights": attention_weights,
            "final_state": final_state
        }
