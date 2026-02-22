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

"""Multi-modal reasoning enhancer utilities for Yv agents.

This module provides cross-modal reasoning enhancement capabilities that fuse
text, vision, audio, and temporal context through attention-driven reasoning
loops. It produces calibrated confidence estimates for the final reasoning
state through iterative refinement.

Architecture Overview:
    The enhancer operates through a multi-stage pipeline:
    
    1. **Cross-Modal Attention**:
       - Visual-textual attention for image-text fusion
       - Audio-textual attention for sound-text integration
       - Temporal reasoning attention for time-series consistency
       - Efficient linear fallback for long sequences
    
    2. **Evidence Aggregation**:
       - Collects features from all available modalities
       - Combines through learned aggregation network
       - Produces unified evidence representation
    
    3. **Iterative Reasoning**:
       - Multiple reasoning refinement steps
       - Each step integrates evidence with current state
       - Temporal context enhances step-wise consistency
    
    4. **Confidence Estimation**:
       - Final confidence score from refined reasoning state
       - Calibrated through sigmoid activation

Key Features:
    - **Adaptive Attention**: Switches to efficient linear attention for
      sequences longer than 512 tokens to manage memory
    - **Multi-Modal Fusion**: Seamlessly integrates visual, audio, and
      temporal features with text representations
    - **Iterative Refinement**: Progressive improvement through multiple
      reasoning steps with evidence integration
    - **Diagnostic Output**: Optional intermediate states for analysis

Modality Contributions:
    - Visual: 0.3 weight in reasoning state update
    - Audio: 0.2 weight in reasoning state update
    - Temporal: 0.2 weight per reasoning step

Example:
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Config:
    ...     hidden_size: int = 2048
    ...     n_head: int = 16
    ...     num_reasoning_steps: int = 4
    >>> 
    >>> enhancer = YvMultiModalReasoningEnhancer(Config())
    >>> text = torch.randn(2, 100, 2048)
    >>> visual = torch.randn(2, 100, 2048)
    >>> audio = torch.randn(2, 100, 2048)
    >>> state, conf, intermediates = enhancer(
    ...     text, visual, audio, return_intermediates=True
    ... )
    >>> print(f"Confidence: {conf.mean():.2f}")

Dependencies:
    - torch: Neural network operations
    - torch.nn: Module definitions

Note:
    The module automatically handles missing modalities by using zero tensors
    as placeholders, ensuring consistent tensor shapes throughout processing.
"""

import torch
from torch import nn

class YvMultiModalReasoningEnhancer(nn.Module):
    """Enhance reasoning states via cross-modal attention and iterative refinement.
    
    This class implements a sophisticated multi-modal reasoning enhancement
    pipeline that fuses information from text, visual, audio, and temporal
    modalities through attention mechanisms and iterative refinement steps.
    
    Architecture:
        1. **Cross-Modal Reasoner ModuleDict**:
           - visual_textual: Multi-head attention for visual-text fusion
           - audio_textual: Multi-head attention for audio-text fusion
           - temporal_reasoning: Multi-head attention for temporal consistency
           - efficient_linear: Fallback for long sequences (>512 tokens)
        
        2. **Reasoning Layers**:
           - Stack of iterative refinement blocks
           - Each block: Linear -> SiLU -> Dropout -> Linear -> LayerNorm
           - Number of blocks configurable via num_reasoning_steps
        
        3. **Evidence Aggregator**:
           - Combines modality features into unified evidence
           - 4-layer network with SiLU activation and dropout
        
        4. **Confidence Estimator**:
           - Produces scalar confidence score
           - 3-layer network with sigmoid output
    
    Attributes:
        cfg: Configuration namespace with hidden_size, n_head, and
            optional num_reasoning_steps parameters.
        hidden_size (int): Dimension of hidden representations.
        num_reasoning_steps (int): Number of iterative refinement steps.
        cross_modal_reasoner (nn.ModuleDict): Multi-modal attention modules.
        reasoning_layers (nn.ModuleList): Iterative refinement blocks.
        evidence_aggregator (nn.Sequential): Evidence fusion network.
        confidence_estimator (nn.Sequential): Confidence prediction network.
    
    Example:
        >>> enhancer = YvMultiModalReasoningEnhancer(config)
        >>> text = torch.randn(2, 100, 2048)
        >>> visual = torch.randn(2, 100, 2048)
        >>> state, conf = enhancer(text, visual_features=visual)
        >>> print(state.shape)  # torch.Size([2, 1, 2048])
        >>> print(conf.shape)   # torch.Size([2, 1])
    
    Note:
        For sequences longer than 512 tokens, the module automatically
        switches to efficient linear attention to manage memory usage.
    """
    
    def __init__(self, cfg):
        """Instantiate the enhancer using configuration-provided hyperparameters.
        
        This constructor initializes all cross-modal attention modules,
        reasoning layers, evidence aggregator, and confidence estimator
        based on the provided configuration.
        
        Args:
            cfg: Configuration namespace providing the following attributes:
                - hidden_size (int): Dimension of hidden representations.
                - n_head (int): Number of attention heads for full attention.
                    Cross-modal attention uses n_head // 4 for visual and audio,
                    and n_head // 2 for temporal reasoning.
                - num_reasoning_steps (int, optional): Number of iterative
                    reasoning refinement steps. Defaults to 4 if not provided.
        
        Initializes:
            - cross_modal_reasoner: ModuleDict with 4 attention modules
            - reasoning_layers: ModuleList with num_reasoning_steps blocks
            - evidence_aggregator: 4-layer fusion network
            - confidence_estimator: 3-layer confidence network
        
        Example:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Config:
            ...     hidden_size: int = 2048
            ...     n_head: int = 16
            ...     num_reasoning_steps: int = 6
            >>> enhancer = YvMultiModalReasoningEnhancer(Config())
        
        Note:
            The efficient_linear module provides a fallback path for
            memory-efficient processing of long sequences.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_reasoning_steps = getattr(cfg, 'num_reasoning_steps', 4)
        
        # Attention modules addressing visual/audio/temporal reasoning and a fallback linear path.
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
        
        # Iterative reasoning blocks that refine the fused hidden state.
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            ) for _ in range(self.num_reasoning_steps)
        ])
        
        # Aggregator combining modality averages into a shared evidence vector.
        self.evidence_aggregator = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Confidence estimator producing the final scalar score.
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_features, visual_features=None, audio_features=None, temporal_context=None, return_intermediates=False):
        """Fuse multi-modal cues and iterate reasoning to yield a final state.
        
        This method implements the complete multi-modal reasoning enhancement
        pipeline, processing text, visual, audio, and temporal features through
        cross-modal attention and iterative refinement.
        
        Processing Pipeline:
            1. **State Initialization**: Clone text features as initial state
            2. **Visual Processing**: Apply visual-textual attention (0.3 weight)
            3. **Audio Processing**: Apply audio-textual attention (0.2 weight)
            4. **Evidence Aggregation**: Combine modality features
            5. **Iterative Refinement**: Multiple reasoning steps with temporal
            6. **Confidence Estimation**: Final confidence score
        
        Args:
            text_features (torch.Tensor): Textual embeddings of shape
                (batch_size, seq_len, hidden_size). This is the primary input
                that serves as the initial reasoning state.
            visual_features (Optional[torch.Tensor]): Visual embeddings aligned
                with the text sequence of shape (batch_size, seq_len, hidden_size).
                If None, zero tensor is used as placeholder. Default: None.
            audio_features (Optional[torch.Tensor]): Audio embeddings aligned
                with the text sequence of shape (batch_size, seq_len, hidden_size).
                If None, zero tensor is used as placeholder. Default: None.
            temporal_context (Optional[torch.Tensor]): Temporal features for
                consistency across reasoning steps of shape (batch_size, seq_len,
                hidden_size). Applied during each reasoning step. Default: None.
            return_intermediates (bool): If True, return diagnostic metadata
                including attention maps, step outputs, and modal contributions.
                Default: False.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
                - reasoning_state (torch.Tensor): Final reasoning state of shape
                  (batch_size, 1, hidden_size).
                - confidence (torch.Tensor): Confidence score of shape
                  (batch_size, 1) with values in [0, 1].
                - intermediates (Optional[Dict[str, Any]]): If return_intermediates
                  is True, contains:
                    - 'initial_state': Initial reasoning state
                    - 'modal_contributions': Dict of modality weights
                    - 'step_outputs': List of per-step information
                    - 'attention_maps': Dict of attention tensors
                    - 'final_confidence': Final confidence tensor
                    - 'final_state': Final reasoning state
                    - 'sequence_length': Input sequence length
                    - 'used_linear_attention': Whether linear attention was used
        
        Adaptive Attention:
            For sequences longer than 512 tokens, the method automatically
            switches to efficient linear attention to manage GPU memory.
            This trades some expressiveness for memory efficiency.
        
        Example:
            >>> enhancer = YvMultiModalReasoningEnhancer(config)
            >>> text = torch.randn(2, 100, 2048)
            >>> visual = torch.randn(2, 100, 2048)
            >>> state, conf, inter = enhancer(
            ...     text, visual, return_intermediates=True
            ... )
            >>> print(state.shape)  # torch.Size([2, 1, 2048])
            >>> print(conf.shape)   # torch.Size([2, 1])
            >>> print(inter['modal_contributions'])  # {'visual': 0.3, ...}
        
        Note:
            Missing modalities are handled gracefully with zero tensor placeholders.
            The confidence score is calibrated through sigmoid activation for
            reliable probability interpretation.
        """
        # Extract batch size, device, and sequence length from text features.
        batch_size = text_features.shape[0]
        device = text_features.device
        seq_len = text_features.shape[1]
        
        # Initialize the reasoning state with a copy of text features.
        reasoning_state = text_features.clone()
        
        # Collect evidence features from available modalities.
        evidence_features = [text_features]
        
        # Determine whether to use efficient linear attention based on sequence length.
        use_linear_attention = seq_len > 512
        
        # Initialize a dictionary to store intermediate results.
        intermediates = {
            'initial_state': reasoning_state.clone(),
            'modal_contributions': {},
            'step_outputs': [],
            'attention_maps': {}
        }
        
        # Process visual features if they are provided.
        if visual_features is not None:
            if use_linear_attention:
                # Apply efficient linear layer for visual reasoning when sequence length is large.
                visual_reasoning = self.cross_modal_reasoner['efficient_linear'](text_features)
                reasoning_state = reasoning_state + 0.3 * visual_reasoning
            else:
                # Use multi-head attention for visual-textual reasoning.
                visual_reasoning, visual_attn = self.cross_modal_reasoner['visual_textual'](
                    text_features, visual_features, visual_features
                )
                reasoning_state = reasoning_state + 0.3 * visual_reasoning
                # Store the attention map for visual-textual reasoning.
                intermediates['attention_maps']['visual_textual'] = visual_attn.detach().cpu()
            evidence_features.append(visual_features)
            intermediates['modal_contributions']['visual'] = 0.3
        else:
            evidence_features.append(torch.zeros_like(text_features))
            intermediates['modal_contributions']['visual'] = 0.0
        
        # Process audio features if they are provided.
        if audio_features is not None:
            if use_linear_attention:
                # Apply efficient linear layer for audio reasoning when sequence length is large.
                audio_reasoning = self.cross_modal_reasoner['efficient_linear'](text_features)
                reasoning_state = reasoning_state + 0.2 * audio_reasoning
            else:
                # Use multi-head attention for audio-textual reasoning.
                audio_reasoning, audio_attn = self.cross_modal_reasoner['audio_textual'](
                    text_features, audio_features, audio_features
                )
                reasoning_state = reasoning_state + 0.2 * audio_reasoning
                # Store the attention map for audio-textual reasoning.
                intermediates['attention_maps']['audio_textual'] = audio_attn.detach().cpu()
            evidence_features.append(audio_features)
            intermediates['modal_contributions']['audio'] = 0.2
        else:
            evidence_features.append(torch.zeros_like(text_features))
            intermediates['modal_contributions']['audio'] = 0.0
        
        # Combine evidence features by taking the mean along the sequence dimension and concatenating them.
        combined_evidence = torch.cat([feat.mean(dim=1) for feat in evidence_features], dim=-1)
        
        # Aggregate the combined evidence using the evidence aggregator module.
        aggregated_evidence = self.evidence_aggregator(combined_evidence)
        
        # Perform iterative reasoning steps.
        for i, reasoning_layer in enumerate(self.reasoning_layers):
            # Store the reasoning state before the current step.
            pre_step_state = reasoning_state.clone()
            
            # Apply the reasoning layer to the mean of the current reasoning state.
            step_output = reasoning_layer(reasoning_state.mean(dim=1))
            
            # Integrate the step output with the aggregated evidence.
            integrated = step_output + 0.1 * aggregated_evidence
            
            # Process temporal context if it is provided.
            if temporal_context is not None:
                if use_linear_attention:
                    # Apply efficient linear layer for temporal reasoning when sequence length is large.
                    temporal_enhanced = self.cross_modal_reasoner['efficient_linear'](integrated.unsqueeze(1))
                    integrated = integrated + 0.2 * temporal_enhanced.squeeze(1)
                else:
                    # Use multi-head attention for temporal reasoning.
                    temporal_enhanced, temporal_attn = self.cross_modal_reasoner['temporal_reasoning'](
                        integrated.unsqueeze(1), temporal_context, temporal_context
                    )
                    integrated = integrated + 0.2 * temporal_enhanced.squeeze(1)
                    # Store the attention map for temporal reasoning at the current step.
                    intermediates['attention_maps'][f'temporal_step_{i}'] = temporal_attn.detach().cpu()
            
            # Update the reasoning state.
            reasoning_state = integrated.unsqueeze(1)
            
            # Record information about the current reasoning step.
            step_info = {
                'step_id': i,
                'input_state': pre_step_state.mean(dim=1),
                'output_state': reasoning_state.mean(dim=1),
                'state_change': torch.norm(reasoning_state.mean(dim=1) - pre_step_state.mean(dim=1)).item()
            }
            intermediates['step_outputs'].append(step_info)
        
        # Estimate the confidence of the final reasoning state.
        confidence = self.confidence_estimator(reasoning_state.mean(dim=1))
        intermediates['final_confidence'] = confidence
        intermediates['final_state'] = reasoning_state.clone()
        intermediates['sequence_length'] = seq_len
        intermediates['used_linear_attention'] = use_linear_attention
        
        if return_intermediates:
            return reasoning_state, confidence, intermediates
        else:
            return reasoning_state, confidence
