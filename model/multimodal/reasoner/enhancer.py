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

class ArcticMultiModalReasoningEnhancer(nn.Module):
    """
    A PyTorch module for multi-modal reasoning enhancement based on the Arctic architecture.
    This module performs cross-modal reasoning with temporal consistency to enhance reasoning states.
    """
    def __init__(self, cfg):
        """
        Initialize the ArcticMultiModalReasoningEnhancer.

        Args:
            cfg (object): Configuration object containing necessary hyperparameters.
                Expected attributes include `hidden_size`, `n_head`, 
                and optionally `num_reasoning_steps` (defaults to 4 if not specified).
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_reasoning_steps = getattr(cfg, 'num_reasoning_steps', 4)
        
        # ModuleDict storing multi-head attention modules for different cross-modal and temporal reasoning tasks
        self.cross_modal_reasoner = nn.ModuleDict({
            'visual_textual': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head // 4,
                batch_first=True,
                dropout=0.1
            ),
            'audio_textual': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head // 4,
                batch_first=True,
                dropout=0.1
            ),
            'temporal_reasoning': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head // 2,
                batch_first=True,
                dropout=0.1
            ),
            'efficient_linear': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
        })
        
        # ModuleList containing reasoning layers for iterative reasoning process
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            ) for _ in range(self.num_reasoning_steps)
        ])
        
        # Sequential module to aggregate evidence from text, visual, and audio modalities
        self.evidence_aggregator = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Sequential module to estimate the confidence score of the final reasoning state
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, visual_features=None, audio_features=None, temporal_context=None, return_intermediates=False):
        """
        Perform a forward pass through the ArcticMultiModalReasoningEnhancer.

        Args:
            text_features (torch.Tensor): Input text features with shape (batch_size, seq_len, hidden_size).
            visual_features (torch.Tensor, optional): Input visual features. Defaults to None.
            audio_features (torch.Tensor, optional): Input audio features. Defaults to None.
            temporal_context (torch.Tensor, optional): Temporal context features. Defaults to None.
            return_intermediates (bool, optional): Whether to return intermediate results. Defaults to False.

        Returns:
            tuple: 
                - reasoning_state (torch.Tensor): Final reasoning state with shape (batch_size, 1, hidden_size).
                - confidence (torch.Tensor): Confidence score of the final state with shape (batch_size, 1).
                - intermediates (dict, optional): Dictionary containing intermediate results if `return_intermediates` is True.
        """
        # Extract batch size, device, and sequence length from text features
        batch_size = text_features.shape[0]
        device = text_features.device
        seq_len = text_features.shape[1]
        
        # Initialize the reasoning state with a copy of text features
        reasoning_state = text_features.clone()
        
        # List to collect evidence features from different modalities
        evidence_features = [text_features]
        
        # Determine whether to use efficient linear attention based on sequence length
        use_linear_attention = seq_len > 512
        
        # Initialize a dictionary to store intermediate results
        intermediates = {
            'initial_state': reasoning_state.clone(),
            'modal_contributions': {},
            'step_outputs': [],
            'attention_maps': {}
        }
        
        # Process visual features if they are provided
        if visual_features is not None:
            if use_linear_attention:
                # Apply efficient linear layer for visual reasoning when sequence length is large
                visual_reasoning = self.cross_modal_reasoner['efficient_linear'](text_features)
                reasoning_state = reasoning_state + 0.3 * visual_reasoning
            else:
                # Use multi-head attention for visual-textual reasoning
                visual_reasoning, visual_attn = self.cross_modal_reasoner['visual_textual'](
                    text_features, visual_features, visual_features
                )
                reasoning_state = reasoning_state + 0.3 * visual_reasoning
                # Store the attention map for visual-textual reasoning
                intermediates['attention_maps']['visual_textual'] = visual_attn.detach().cpu()
            evidence_features.append(visual_features)
            intermediates['modal_contributions']['visual'] = 0.3
        else:
            evidence_features.append(torch.zeros_like(text_features))
            intermediates['modal_contributions']['visual'] = 0.0
        
        # Process audio features if they are provided
        if audio_features is not None:
            if use_linear_attention:
                # Apply efficient linear layer for audio reasoning when sequence length is large
                audio_reasoning = self.cross_modal_reasoner['efficient_linear'](text_features)
                reasoning_state = reasoning_state + 0.2 * audio_reasoning
            else:
                # Use multi-head attention for audio-textual reasoning
                audio_reasoning, audio_attn = self.cross_modal_reasoner['audio_textual'](
                    text_features, audio_features, audio_features
                )
                reasoning_state = reasoning_state + 0.2 * audio_reasoning
                # Store the attention map for audio-textual reasoning
                intermediates['attention_maps']['audio_textual'] = audio_attn.detach().cpu()
            evidence_features.append(audio_features)
            intermediates['modal_contributions']['audio'] = 0.2
        else:
            evidence_features.append(torch.zeros_like(text_features))
            intermediates['modal_contributions']['audio'] = 0.0
        
        # Combine evidence features by taking the mean along the sequence dimension and concatenating them
        combined_evidence = torch.cat([feat.mean(dim=1) for feat in evidence_features], dim=-1)
        
        # Aggregate the combined evidence using the evidence aggregator module
        aggregated_evidence = self.evidence_aggregator(combined_evidence)
        
        # Perform iterative reasoning steps
        for i, reasoning_layer in enumerate(self.reasoning_layers):
            # Store the reasoning state before the current step
            pre_step_state = reasoning_state.clone()
            
            # Apply the reasoning layer to the mean of the current reasoning state
            step_output = reasoning_layer(reasoning_state.mean(dim=1))
            
            # Integrate the step output with the aggregated evidence
            integrated = step_output + 0.1 * aggregated_evidence
            
            # Process temporal context if it is provided
            if temporal_context is not None:
                if use_linear_attention:
                    # Apply efficient linear layer for temporal reasoning when sequence length is large
                    temporal_enhanced = self.cross_modal_reasoner['efficient_linear'](integrated.unsqueeze(1))
                    integrated = integrated + 0.2 * temporal_enhanced.squeeze(1)
                else:
                    # Use multi-head attention for temporal reasoning
                    temporal_enhanced, temporal_attn = self.cross_modal_reasoner['temporal_reasoning'](
                        integrated.unsqueeze(1), temporal_context, temporal_context
                    )
                    integrated = integrated + 0.2 * temporal_enhanced.squeeze(1)
                    # Store the attention map for temporal reasoning at the current step
                    intermediates['attention_maps'][f'temporal_step_{i}'] = temporal_attn.detach().cpu()
            
            # Update the reasoning state
            reasoning_state = integrated.unsqueeze(1)
            
            # Record information about the current reasoning step
            step_info = {
                'step_id': i,
                'input_state': pre_step_state.mean(dim=1),
                'output_state': reasoning_state.mean(dim=1),
                'state_change': torch.norm(reasoning_state.mean(dim=1) - pre_step_state.mean(dim=1)).item()
            }
            intermediates['step_outputs'].append(step_info)
        
        # Estimate the confidence of the final reasoning state
        confidence = self.confidence_estimator(reasoning_state.mean(dim=1))
        intermediates['final_confidence'] = confidence
        intermediates['final_state'] = reasoning_state.clone()
        intermediates['sequence_length'] = seq_len
        intermediates['used_linear_attention'] = use_linear_attention
        
        if return_intermediates:
            return reasoning_state, confidence, intermediates
        else:
            return reasoning_state, confidence