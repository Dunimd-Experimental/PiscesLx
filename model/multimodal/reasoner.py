#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import torch
from torch import nn
import torch.nn.functional as F

class ArcticReasoner(nn.Module):
    """Enhanced Pisces L1 reasoning module with multi-step CoT and memory integration"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        
        # Optimized multi-step CoT reasoning layers (reduced redundancy)
        # Merged similar reasoning layers and added dynamic depth control
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.n_head,
                dim_feedforward=cfg.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)  # Optimized: reduced from 4 to 3 layers
        ])
        
        # Dynamic reasoning depth controller
        self.depth_controller = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Enhanced reasoning components
        self.thinking_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.difficulty_head = nn.Linear(cfg.hidden_size, 5)  # Enhanced: 5 difficulty levels
        self.reflection_head = nn.Linear(cfg.hidden_size, 4)  # Enhanced: 4 reflection types
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # Memory integration for multi-turn context
        self.memory_key_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_value_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_head,
            dropout=0.1,
            batch_first=True
        )
        
        # Hierarchical reflection components
        self.error_analyzer = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 3)  # error_type, severity, confidence
        )
        
        self.correction_head = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)
    
    def _calculate_problem_complexity(self, hidden_states):
        """
        Calculate problem complexity score for PiscesReasoner.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            
        Returns:
            float: Complexity score between 0 and 1
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Factor 1: Sequence length complexity
        length_complexity = min(seq_len / 256, 1.0)  # Lower threshold for reasoning
        
        # Factor 2: Semantic diversity across the reasoning sequence
        mean_state = hidden_states.mean(dim=1, keepdim=True)
        semantic_variance = torch.var(hidden_states - mean_state, dim=[1,2]).mean()
        diversity_complexity = torch.sigmoid(semantic_variance * 15)  # Higher sensitivity
        
        # Factor 3: Reasoning pattern complexity (gradient-based)
        if hidden_states.requires_grad:
            grad_magnitude = torch.norm(hidden_states.grad) if hidden_states.grad is not None else torch.tensor(0.0)
            reasoning_complexity = torch.sigmoid(grad_magnitude * 5)
        else:
            reasoning_complexity = torch.tensor(0.5)
        
        # Combine factors with reasoning-specific weights
        complexity = (length_complexity * 0.4 + 
                     diversity_complexity * 0.4 + 
                     reasoning_complexity * 0.2)
        
        return complexity.item()
    
    def _calculate_abstraction_gain(self, prev_states, curr_states):
        """
        Calculate abstraction gain for reasoning layers.
        
        Args:
            prev_states: Previous layer hidden states
            curr_states: Current layer hidden states
            
        Returns:
            float: Abstraction gain score
        """
        # Measure reasoning improvement using multiple metrics
        prev_pooled = F.normalize(prev_states.mean(dim=1), p=2, dim=-1)
        curr_pooled = F.normalize(curr_states.mean(dim=1), p=2, dim=-1)
        
        # Semantic advancement
        semantic_advance = 1 - F.cosine_similarity(prev_pooled, curr_pooled, dim=-1).mean()
        
        # Information consolidation (reduction in variance indicates abstraction)
        prev_info = torch.norm(torch.var(prev_states, dim=1), p=2)
        curr_info = torch.norm(torch.var(curr_states, dim=1), p=2)
        consolidation_ratio = max(0, (prev_info - curr_info) / (prev_info + 1e-8))
        
        # Reasoning coherence (smoothness of state transitions)
        state_diff = torch.norm(curr_states - prev_states, p=2, dim=-1).mean()
        coherence_score = torch.sigmoid(1.0 / (state_diff + 0.1))
        
        # Combine metrics with reasoning-specific weights
        abstraction_gain = semantic_advance * 0.5 + consolidation_ratio * 0.3 + coherence_score * 0.2
        return abstraction_gain.item()
        
        # Special tokens for CoT
        self.register_buffer('start_thinking_token', torch.tensor([50256]))
        self.register_buffer('end_thinking_token', torch.tensor([50257]))
        self.register_buffer('reflection_token', torch.tensor([50258]))
    
    def forward(self, input_ids=None, attention_mask=None, memory_context=None, **kwargs):
        """Enhanced forward pass with memory integration and multi-step reasoning"""
        
        # Handle input
        if torch.is_tensor(input_ids):
            hidden_states = input_ids
        else:
            hidden_states = torch.randn(1, 1, self.cfg.hidden_size)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Memory context integration for multi-turn dialogue
        if memory_context is not None and len(memory_context) > 0:
            # Convert memory to tensor
            memory_tensor = torch.stack(memory_context).unsqueeze(0) if isinstance(memory_context[0], torch.Tensor) else torch.tensor(memory_context).unsqueeze(0)
            
            # Project memory to key and value spaces
            memory_keys = self.memory_key_proj(memory_tensor)
            memory_values = self.memory_value_proj(memory_tensor)
            
            # Cross-attention: current context attends to memory
            memory_out, attention_weights = self.memory_attention(
                hidden_states, memory_keys, memory_values
            )
            
            # Residual connection
            hidden_states = hidden_states + memory_out
        else:
            attention_weights = None
        
        # Multi-step CoT reasoning with adaptive depth
        reasoning_states = self.layer_norm(hidden_states)
        reasoning_steps = []
        
        # Calculate problem complexity for dynamic depth adjustment
        complexity_score = self._calculate_problem_complexity(reasoning_states)
        
        # Dynamic depth selection based on complexity
        if complexity_score < 0.3:
            # Simple problems: use minimal depth
            num_layers = 1
        elif complexity_score < 0.6:
            # Medium complexity: use moderate depth
            num_layers = 2
        else:
            # Complex problems: use full depth
            num_layers = len(self.reasoning_layers)
        
        # Adaptive reasoning with early termination
        for i, layer in enumerate(self.reasoning_layers[:num_layers]):
            prev_states = reasoning_states.clone()
            reasoning_states = layer(reasoning_states)
            reasoning_steps.append(reasoning_states.clone())
            
            # Check abstraction gain for early termination
            if i > 0:
                abstraction_gain = self._calculate_abstraction_gain(prev_states, reasoning_states)
                if abstraction_gain < 0.1:  # Minimal improvement
                    break
        
        # Get final reasoning state for outputs
        final_state = reasoning_states[:, -1, :]
        
        # Enhanced outputs
        thinking_logits = self.thinking_head(final_state)
        difficulty_logits = self.difficulty_head(final_state)
        reflection_logits = self.reflection_head(final_state)
        confidence_score = torch.sigmoid(self.confidence_head(final_state))
        
        # Hierarchical reflection analysis
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