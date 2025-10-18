#!/usr/bin/env/python3

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

import torch
from torch import nn
import torch.nn.functional as F

class ArcticCoTMemoryReasoner(nn.Module):
    """
    A reasoning module that integrates multi-step Chain-of-Thought (CoT) reasoning with a memory mechanism.
    This module can adaptively control the reasoning depth based on problem complexity and 
    utilize memory context to enhance reasoning performance.
    """
    
    def __init__(self, cfg):
        """
        Initialize the ArcticCoTMemoryReasoner module.

        Args:
            cfg (object): Configuration object containing necessary hyperparameters.
                          Expected attributes include hidden_size, vocab_size, and n_head.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        
        # Initialize a list of Transformer encoder layers for multi-step reasoning
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.n_head,
                dim_feedforward=cfg.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Initialize a depth controller to adaptively adjust the reasoning depth
        self.depth_controller = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize output heads for different prediction tasks
        self.thinking_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.difficulty_head = nn.Linear(cfg.hidden_size, 5)
        self.reflection_head = nn.Linear(cfg.hidden_size, 4)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # Initialize modules related to memory mechanism
        self.memory_key_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_value_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_head,
            dropout=0.1,
            batch_first=True
        )
        
        # Initialize an error analysis module to predict error type, severity, and confidence
        self.error_analyzer = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 3)  # error_type, severity, confidence
        )
        
        # Initialize a correction head to generate correction logits
        self.correction_head = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        
        # Initialize layer normalization to stabilize the training process
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)
    
    def _calculate_problem_complexity(self, hidden_states):
        """
        Calculate the complexity of the input problem based on sequence length, semantic variance, and reasoning gradient.

        Args:
            hidden_states (torch.Tensor): Input hidden states with shape (batch_size, seq_len, hidden_size).

        Returns:
            float: A scalar value representing the problem complexity in the range [0, 1].
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Calculate complexity based on the sequence length of the input
        length_complexity = min(seq_len / 256, 1.0)
        # Calculate the mean state for semantic variance calculation
        mean_state = hidden_states.mean(dim=1, keepdim=True)
        # Calculate the semantic variance of the input states
        semantic_variance = torch.var(hidden_states - mean_state, dim=[1,2]).mean()
        # Calculate diversity complexity using the sigmoid function
        diversity_complexity = torch.sigmoid(semantic_variance * 15)
        if hidden_states.requires_grad:
            # Calculate reasoning complexity based on the magnitude of the gradient
            grad_magnitude = torch.norm(hidden_states.grad) if hidden_states.grad is not None else torch.tensor(0.0)
            reasoning_complexity = torch.sigmoid(grad_magnitude * 5)
        else:
            reasoning_complexity = torch.tensor(0.5)
        # Combine three complexity components to get the final complexity score
        complexity = (length_complexity * 0.4 + 
                      diversity_complexity * 0.4 + 
                      reasoning_complexity * 0.2)
        return complexity.item()
    
    def _calculate_abstraction_gain(self, prev_states, curr_states):
        """
        Calculate the abstraction gain between previous and current states.

        Args:
            prev_states (torch.Tensor): Previous hidden states with shape (batch_size, seq_len, hidden_size).
            curr_states (torch.Tensor): Current hidden states with shape (batch_size, seq_len, hidden_size).

        Returns:
            float: A scalar value representing the abstraction gain.
        """
        # Normalize the mean of the previous states
        prev_pooled = F.normalize(prev_states.mean(dim=1), p=2, dim=-1)
        # Normalize the mean of the current states
        curr_pooled = F.normalize(curr_states.mean(dim=1), p=2, dim=-1)
        # Calculate the semantic advance between previous and current states
        semantic_advance = 1 - F.cosine_similarity(prev_pooled, curr_pooled, dim=-1).mean()
        # Calculate the information norm of the previous states
        prev_info = torch.norm(torch.var(prev_states, dim=1), p=2)
        # Calculate the information norm of the current states
        curr_info = torch.norm(torch.var(curr_states, dim=1), p=2)
        # Calculate the consolidation ratio of information
        consolidation_ratio = max(0, (prev_info - curr_info) / (prev_info + 1e-8))
        # Calculate the mean difference between previous and current states
        state_diff = torch.norm(curr_states - prev_states, p=2, dim=-1).mean()
        # Calculate the coherence score based on the state difference
        coherence_score = torch.sigmoid(1.0 / (state_diff + 0.1))
        # Combine three components to get the abstraction gain
        abstraction_gain = semantic_advance * 0.5 + consolidation_ratio * 0.3 + coherence_score * 0.2
        return abstraction_gain.item()
    
    def forward(self, input_ids=None, attention_mask=None, memory_context=None, **kwargs):
        """
        Perform a forward pass of the ArcticCoTMemoryReasoner module.

        Args:
            input_ids (torch.Tensor, optional): Input tensor. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            memory_context (list or torch.Tensor, optional): Memory context. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing various output tensors, including thinking logits, 
                  difficulty logits, reflection logits, confidence score, etc.
        """
        # Handle the input and initialize hidden states
        if torch.is_tensor(input_ids):
            hidden_states = input_ids
        else:
            hidden_states = torch.randn(1, 1, self.cfg.hidden_size)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Integrate memory context if it is provided
        if memory_context is not None and len(memory_context) > 0:
            memory_tensor = torch.stack(memory_context).unsqueeze(0) if isinstance(memory_context[0], torch.Tensor) else torch.tensor(memory_context).unsqueeze(0)
            memory_keys = self.memory_key_proj(memory_tensor)
            memory_values = self.memory_value_proj(memory_tensor)
            memory_out, attention_weights = self.memory_attention(hidden_states, memory_keys, memory_values)
            hidden_states = hidden_states + memory_out
        else:
            attention_weights = None
        
        # Perform multi-step CoT reasoning with adaptive depth
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