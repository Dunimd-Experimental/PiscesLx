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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""Enhanced reasoning system with routing, uncertainty quantification, and multi-strategy fusion.

This module provides an advanced reasoning system that combines multiple reasoning
strategies (recursive depth reasoning and thought tree search) with intelligent
routing and comprehensive uncertainty quantification. It represents the top-level
reasoning orchestration layer for the Yv architecture.

Architecture Overview:
    The enhanced reasoning system integrates several key components:
    
    1. **Reasoning Router** (YvEnhancedReasoningRouter):
       - Analyzes problem complexity and task type
       - Routes to appropriate reasoning strategies
       - Recommends optimal reasoning depth
    
    2. **Uncertainty Quantifier** (YvUncertaintyQuantifier):
       - Estimates knowledge uncertainty
       - Measures reasoning uncertainty
       - Evaluates completeness uncertainty
       - Fuses uncertainty estimates for overall assessment
    
    3. **Recursive Depth Reasoner** (YvRecursiveDepthReasoner):
       - Conditionally activated for complex problems (complexity > 0.4)
       - Provides subproblem decomposition and aggregation
       - Includes optional bidirectional verification
    
    4. **Thought Tree Reasoner** (YvThoughtTreeReasoner):
       - Conditionally activated for highly complex problems (complexity > 0.6)
       - MCTS-style exploration of reasoning paths
       - UCB-based node selection

Key Features:
    - **Adaptive Strategy Selection**: Routes to appropriate reasoning strategies
      based on problem complexity and task type
    - **Multi-Strategy Fusion**: Combines outputs from multiple reasoners through
      learned projection layers
    - **Uncertainty-Aware**: Provides detailed uncertainty estimates for
      confidence calibration
    - **Task Specialization**: Different configurations for planning, mathematical,
      creative, analytical, and general reasoning tasks

Configuration Options:
    - use_recursive_reasoning: Enable recursive depth reasoning (default: True)
    - use_thought_tree: Enable thought tree search (default: True)
    - use_bidirectional_verification: Enable solution verification (default: True)
    - enable_router: Enable intelligent routing (default: True)
    - enable_uncertainty_quantification: Enable uncertainty estimation (default: True)
    - max_recursive_depth: Maximum recursion depth (default: 10)
    - tree_depth: Thought tree depth (default: 5)
    - beam_width: Tree branching factor (default: 3)
    - num_simulations: MCTS simulations (default: 100)

Example:
    >>> config = YvEnhancedReasoningConfig(
    ...     hidden_size=2048,
    ...     use_recursive_reasoning=True,
    ...     use_thought_tree=True
    ... )
    >>> system = YvEnhancedReasoningSystem(config)
    >>> hidden = torch.randn(2, 10, 2048)
    >>> result = system(hidden, task_type="mathematical")
    >>> print(f"Confidence: {result['confidence'].mean():.2f}")
    >>> print(f"Complexity: {result['complexity'].mean():.2f}")

Dependencies:
    - recursive_depth: YvRecursiveDepthReasoner and YvThoughtTreeReasoner
    - torch: Neural network operations
    - dataclasses: Configuration management

Note:
    The system conditionally activates reasoning strategies based on complexity
    thresholds. Recursive reasoning activates at complexity > 0.4, and thought
    tree search activates at complexity > 0.6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class YvEnhancedReasoningConfig:
    """Configuration dataclass for the enhanced reasoning system.
    
    This dataclass encapsulates all hyperparameters controlling the behavior
    of the enhanced reasoning system, including which reasoning strategies to
    enable and their respective parameters.
    
    Attributes:
        hidden_size (int): Dimension of hidden state representations.
            Must match the model's hidden dimension. Default: 2048.
        use_recursive_reasoning (bool): Whether to enable recursive depth
            reasoning for complex problem decomposition. Default: True.
        use_thought_tree (bool): Whether to enable thought tree search for
            exploration-based reasoning. Default: True.
        use_bidirectional_verification (bool): Whether to enable bidirectional
            verification for solution consistency checking. Default: True.
        max_recursive_depth (int): Maximum recursion depth for the recursive
            reasoner. Higher values allow more complex decomposition. Default: 10.
        tree_depth (int): Maximum depth of the thought tree. Controls how many
            reasoning steps can be explored. Default: 5.
        beam_width (int): Number of children per node in thought tree.
            Controls branching factor. Default: 3.
        num_simulations (int): Number of MCTS simulations for thought tree.
            More simulations improve solution quality. Default: 100.
        enable_router (bool): Whether to enable intelligent routing between
            reasoning strategies based on complexity. Default: True.
        enable_uncertainty_quantification (bool): Whether to enable uncertainty
            estimation for confidence calibration. Default: True.
    
    Example:
        >>> config = YvEnhancedReasoningConfig(
        ...     hidden_size=4096,
        ...     max_recursive_depth=12,
        ...     use_thought_tree=False
        ... )
        >>> system = YvEnhancedReasoningSystem(config)
    
    Note:
        Disabling unused reasoning strategies reduces memory footprint and
        computation time. For simple tasks, consider disabling thought tree
        search and using only recursive reasoning.
    """
    hidden_size: int = 2048
    use_recursive_reasoning: bool = True
    use_thought_tree: bool = True
    use_bidirectional_verification: bool = True
    max_recursive_depth: int = 10
    tree_depth: int = 5
    beam_width: int = 3
    num_simulations: int = 100
    enable_router: bool = True
    enable_uncertainty_quantification: bool = True


class YvEnhancedReasoningRouter(nn.Module):
    """Neural router for adaptive reasoning strategy selection.
    
    This module analyzes problem characteristics and routes to appropriate
    reasoning strategies. It predicts problem complexity and generates routing
    weights for different reasoning approaches based on task type.
    
    Architecture:
        1. **Task Encoder**:
           - Encodes pooled hidden states into task-specific embeddings
           - Two-layer MLP with GELU activation
        
        2. **Complexity Predictor**:
           - Estimates problem complexity from hidden states
           - Three-layer MLP with sigmoid output for [0, 1] range
        
        3. **Router Network**:
           - Combines task embedding with pooled features
           - Produces softmax weights over task types
    
    Task Types:
        - planning: Forward-looking, goal-oriented reasoning
        - mathematical: Symbolic and numerical reasoning
        - creative: Generative and imaginative reasoning
        - analytical: Logical and systematic reasoning
        - reasoning: General-purpose reasoning (default)
    
    Attributes:
        hidden_size (int): Dimension of hidden representations.
        task_encoder (nn.Sequential): Task embedding network.
        complexity_predictor (nn.Sequential): Complexity estimation network.
        router (nn.Sequential): Routing weight generation network.
    
    Example:
        >>> router = YvEnhancedReasoningRouter(hidden_size=2048)
        >>> hidden = torch.randn(2, 10, 2048)
        >>> info = router(hidden, task_type="mathematical")
        >>> print(f"Complexity: {info['complexity'].mean():.2f}")
        >>> print(f"Recommended depth: {info['recommended_depth']}")
    
    Note:
        The recommended_depth is calculated as complexity * 10, clamped to [1, 10].
    """
    
    def __init__(self, hidden_size: int, num_task_types: int = 5):
        """Initialize the reasoning router.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
            num_task_types (int): Number of distinct task types for routing.
                Default: 5 (planning, mathematical, creative, analytical, reasoning).
        
        Initializes:
            - task_encoder: 2-layer MLP for task embedding
            - complexity_predictor: 4-layer MLP with sigmoid output
            - router: 4-layer MLP with softmax output
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.task_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.router = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_task_types),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_type: str = "reasoning"
    ) -> Dict[str, Any]:
        """Analyze problem and generate routing information.
        
        This method processes hidden states to predict complexity, generate
        task embeddings, and produce routing weights for strategy selection.
        
        Args:
            hidden_states (torch.Tensor): Hidden state tensor of shape
                (batch_size, seq_len, hidden_size).
            task_type (str): Task type identifier. Options are "planning",
                "mathematical", "creative", "analytical", or "reasoning".
                Default: "reasoning".
        
        Returns:
            Dict[str, Any]: Routing information containing:
                - 'complexity' (torch.Tensor): Problem complexity score of shape
                  (batch_size, 1) with values in [0, 1].
                - 'task_embedding' (torch.Tensor): Task-specific embedding of
                  shape (batch_size, hidden_size // 4).
                - 'routing_weights' (torch.Tensor): Softmax weights over task
                  types of shape (batch_size, num_task_types).
                - 'recommended_depth' (torch.Tensor): Recommended reasoning depth
                  of shape (batch_size,) with values in [1, 10].
                - 'task_type' (str): Input task type identifier.
        
        Example:
            >>> router = YvEnhancedReasoningRouter(2048)
            >>> hidden = torch.randn(2, 10, 2048)
            >>> info = router(hidden, "mathematical")
            >>> print(info['complexity'].shape)  # torch.Size([2, 1])
        """
        batch_size = hidden_states.size(0)
        
        pooled = hidden_states.mean(dim=1)
        
        complexity = self.complexity_predictor(pooled)
        
        task_embedding = self.task_encoder(pooled)
        
        if task_type == "planning":
            task_onehot = torch.tensor([1, 0, 0, 0, 0], device=hidden_states.device)
        elif task_type == "mathematical":
            task_onehot = torch.tensor([0, 1, 0, 0, 0], device=hidden_states.device)
        elif task_type == "creative":
            task_onehot = torch.tensor([0, 0, 1, 0, 0], device=hidden_states.device)
        elif task_type == "analytical":
            task_onehot = torch.tensor([0, 0, 0, 1, 0], device=hidden_states.device)
        else:
            task_onehot = torch.tensor([0, 0, 0, 0, 1], device=hidden_states.device)
        
        task_onehot = task_onehot.unsqueeze(0).expand(batch_size, -1)
        
        routing_input = torch.cat([task_embedding, pooled], dim=-1)
        routing_weights = self.router(routing_input)
        
        return {
            'complexity': complexity,
            'task_embedding': task_embedding,
            'routing_weights': routing_weights,
            'recommended_depth': (complexity * 10).clamp(1, 10).long(),
            'task_type': task_type
        }


class YvUncertaintyQuantifier(nn.Module):
    """Multi-component uncertainty quantification for reasoning confidence.
    
    This module estimates uncertainty from multiple perspectives to provide
    comprehensive confidence calibration. It decomposes uncertainty into
    knowledge, reasoning, and completeness components.
    
    Uncertainty Components:
        1. **Knowledge Uncertainty**: Measures uncertainty in the model's
           knowledge about the input domain. High values indicate the problem
           may be outside the model's training distribution.
        
        2. **Reasoning Uncertainty**: Measures uncertainty in the reasoning
           process itself. High values indicate the reasoning chain may be
           unreliable or inconsistent.
        
        3. **Completeness Uncertainty**: Measures whether the output fully
           addresses the input. High values indicate missing information
           or incomplete coverage.
    
    Architecture:
        Each uncertainty head is a 4-layer MLP with GELU activation and
        sigmoid output for [0, 1] range. The fusion layer combines all
        uncertainty estimates into a unified representation.
    
    Attributes:
        hidden_size (int): Dimension of hidden representations.
        knowledge_uncertainty (nn.Sequential): Knowledge uncertainty estimator.
        reasoning_uncertainty (nn.Sequential): Reasoning uncertainty estimator.
        completeness_uncertainty (nn.Sequential): Completeness uncertainty estimator.
        fusion (nn.Sequential): Uncertainty fusion network.
    
    Example:
        >>> quantifier = YvUncertaintyQuantifier(2048)
        >>> output = torch.randn(2, 10, 2048)
        >>> input_states = torch.randn(2, 10, 2048)
        >>> reasoning = torch.randn(2, 10, 2048)
        >>> uncertainty = quantifier(output, input_states, reasoning)
        >>> print(f"Overall: {uncertainty['overall_uncertainty'].mean():.2f}")
    
    Note:
        Overall uncertainty is computed as the mean of the three components.
        Lower values indicate higher confidence in the reasoning result.
    """
    
    def __init__(self, hidden_size: int):
        """Initialize the uncertainty quantifier.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
        
        Initializes:
            - knowledge_uncertainty: 4-layer MLP with sigmoid output
            - reasoning_uncertainty: 4-layer MLP with sigmoid output
            - completeness_uncertainty: 4-layer MLP with sigmoid output
            - fusion: 4-layer network for uncertainty combination
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.knowledge_uncertainty = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.reasoning_uncertainty = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.completeness_uncertainty = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
    
    def forward(
        self,
        output: torch.Tensor,
        input_states: torch.Tensor,
        reasoning_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Quantify uncertainty across multiple dimensions.
        
        This method computes knowledge, reasoning, and completeness uncertainty
        estimates, then fuses them into a combined representation.
        
        Args:
            output (torch.Tensor): Reasoning output tensor of shape
                (batch_size, seq_len, hidden_size).
            input_states (torch.Tensor): Original input hidden states of shape
                (batch_size, seq_len, hidden_size).
            reasoning_features (torch.Tensor): Intermediate reasoning features
                of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            Dict[str, torch.Tensor]: Uncertainty estimates containing:
                - 'knowledge_uncertainty' (torch.Tensor): Shape (batch_size, 1).
                - 'reasoning_uncertainty' (torch.Tensor): Shape (batch_size, 1).
                - 'completeness_uncertainty' (torch.Tensor): Shape (batch_size, 1).
                - 'combined_uncertainty' (torch.Tensor): Fused uncertainty of
                  shape (batch_size, hidden_size // 2).
                - 'overall_uncertainty' (torch.Tensor): Mean of three components
                  of shape (batch_size, 1).
        
        Example:
            >>> quantifier = YvUncertaintyQuantifier(2048)
            >>> out = torch.randn(2, 10, 2048)
            >>> inp = torch.randn(2, 10, 2048)
            >>> reason = torch.randn(2, 10, 2048)
            >>> result = quantifier(out, inp, reason)
            >>> print(result['overall_uncertainty'].shape)  # torch.Size([2, 1])
        """
        output_pooled = output.mean(dim=1)
        input_pooled = input_states.mean(dim=1)
        
        knowledge_uncertainty = self.knowledge_uncertainty(
            torch.cat([output_pooled, input_pooled], dim=-1)
        )
        
        reasoning_uncertainty = self.reasoning_uncertainty(
            torch.cat([output_pooled, reasoning_features.mean(dim=1)], dim=-1)
        )
        
        completeness_uncertainty = self.completeness_uncertainty(
            torch.cat([output_pooled, input_pooled - output_pooled], dim=-1)
        )
        
        combined_uncertainty = torch.cat([
            knowledge_uncertainty,
            reasoning_uncertainty,
            completeness_uncertainty
        ], dim=-1)
        
        fused_uncertainty = self.fusion(combined_uncertainty)
        
        return {
            'knowledge_uncertainty': knowledge_uncertainty,
            'reasoning_uncertainty': reasoning_uncertainty,
            'completeness_uncertainty': completeness_uncertainty,
            'combined_uncertainty': fused_uncertainty,
            'overall_uncertainty': (knowledge_uncertainty + reasoning_uncertainty + completeness_uncertainty) / 3
        }


class YvEnhancedReasoningSystem(nn.Module):
    """Top-level enhanced reasoning system with multi-strategy fusion.
    
    This class orchestrates multiple reasoning strategies (recursive depth and
    thought tree) with intelligent routing and uncertainty quantification.
    It represents the complete reasoning pipeline for complex problem-solving.
    
    Architecture:
        1. **Routing Stage**:
           - Analyzes problem complexity
           - Determines which reasoning strategies to activate
           - Recommends optimal reasoning depth
        
        2. **Reasoning Stage**:
           - Recursive reasoning for complex problems (complexity > 0.4)
           - Thought tree search for highly complex problems (complexity > 0.6)
           - Both can be activated simultaneously for maximum coverage
        
        3. **Fusion Stage**:
           - Combines outputs from activated reasoners
           - Projects concatenated features to unified representation
           - Handles variable sequence lengths with padding
        
        4. **Uncertainty Stage**:
           - Quantifies multi-dimensional uncertainty
           - Calibrates final confidence estimate
    
    Activation Thresholds:
        - Recursive reasoning: complexity > 0.4
        - Thought tree search: complexity > 0.6
    
    Attributes:
        config (YvEnhancedReasoningConfig): Configuration dataclass.
        hidden_size (int): Dimension of hidden representations.
        router (Optional[YvEnhancedReasoningRouter]): Strategy router.
        recursive_reasoner (Optional[YvRecursiveDepthReasoner]): Recursive reasoner.
        thought_tree_reasoner (Optional[YvThoughtTreeReasoner]): Tree search reasoner.
        uncertainty_quantifier (Optional[YvUncertaintyQuantifier]): Uncertainty estimator.
        output_proj (nn.Sequential): Output projection network.
    
    Example:
        >>> config = YvEnhancedReasoningConfig(
        ...     hidden_size=2048,
        ...     use_recursive_reasoning=True,
        ...     use_thought_tree=True
        ... )
        >>> system = YvEnhancedReasoningSystem(config)
        >>> hidden = torch.randn(2, 10, 2048)
        >>> result = system(hidden, task_type="mathematical")
        >>> print(f"Confidence: {result['confidence'].mean():.2f}")
    
    Note:
        The system conditionally activates reasoning strategies based on
        complexity thresholds. Simple problems may bypass both reasoners.
    """
    
    def __init__(self, config: Optional[YvEnhancedReasoningConfig] = None):
        """Initialize the enhanced reasoning system.
        
        This constructor sets up all submodules based on the provided
        configuration, conditionally creating each reasoning component.
        
        Args:
            config (Optional[YvEnhancedReasoningConfig]): Configuration
                dataclass containing all hyperparameters. If None, uses default
                values from YvEnhancedReasoningConfig.
        
        Initializes:
            - router: If enable_router is True
            - recursive_reasoner: If use_recursive_reasoning is True
            - thought_tree_reasoner: If use_thought_tree is True
            - uncertainty_quantifier: If enable_uncertainty_quantification is True
            - output_proj: Always initialized for feature fusion
        """
        super().__init__()
        if config is None:
            config = YvEnhancedReasoningConfig()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        if config.enable_router:
            self.router = YvEnhancedReasoningRouter(
                hidden_size=self.hidden_size,
                num_task_types=5
            )
        else:
            self.router = None
        
        if config.use_recursive_reasoning:
            from .recursive_depth import YvRecursiveDepthReasoner, YvRecursiveReasoningConfig
            recursive_config = YvRecursiveReasoningConfig(
                hidden_size=self.hidden_size,
                max_depth=config.max_recursive_depth,
                use_verification=config.use_bidirectional_verification
            )
            self.recursive_reasoner = YvRecursiveDepthReasoner(recursive_config)
        else:
            self.recursive_reasoner = None
        
        if config.use_thought_tree:
            from .recursive_depth import YvThoughtTreeReasoner
            self.thought_tree_reasoner = YvThoughtTreeReasoner(
                hidden_size=self.hidden_size,
                tree_depth=config.tree_depth,
                beam_width=config.beam_width
            )
        else:
            self.thought_tree_reasoner = None
        
        if config.enable_uncertainty_quantification:
            self.uncertainty_quantifier = YvUncertaintyQuantifier(self.hidden_size)
        else:
            self.uncertainty_quantifier = None
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all linear and normalization layers.
        
        Uses Xavier uniform initialization for linear layer weights and
        zero initialization for biases. LayerNorm weights are initialized
        to 1.0 and biases to 0.0 for stable initial training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "reasoning",
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """Execute enhanced reasoning with adaptive strategy selection.
        
        This method implements the complete enhanced reasoning pipeline,
        conditionally activating reasoning strategies based on complexity
        and fusing their outputs into a unified representation.
        
        Processing Pipeline:
            1. **Routing Analysis**: Determine problem complexity and
               recommended reasoning depth
            2. **Recursive Reasoning**: Activate if complexity > 0.4
            3. **Thought Tree Search**: Activate if complexity > 0.6
            4. **Feature Fusion**: Combine outputs with padding for alignment
            5. **Uncertainty Quantification**: Estimate confidence components
        
        Args:
            hidden_states (torch.Tensor): Input hidden states of shape
                (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Optional attention mask
                of shape (batch_size, seq_len). Default: None.
            task_type (str): Task type for routing. Options are "planning",
                "mathematical", "creative", "analytical", or "reasoning".
                Default: "reasoning".
            return_intermediate (bool): Whether to return intermediate outputs
                from individual reasoners. Default: False.
        
        Returns:
            Dict[str, Any]: Results dictionary containing:
                - 'reasoning_output' (torch.Tensor): Final fused output of
                  shape (batch_size, seq_len, hidden_size).
                - 'confidence' (torch.Tensor): Confidence score of shape
                  (batch_size, 1).
                - 'uncertainty' (Dict[str, torch.Tensor]): Uncertainty estimates
                  if quantification enabled.
                - 'routing_info' (Dict[str, Any]): Routing decisions and
                  complexity scores.
                - 'task_type' (str): Input task type.
                - 'complexity' (torch.Tensor): Problem complexity score.
                - 'recursive_output' (Optional[torch.Tensor]): Recursive reasoner
                  output if return_intermediate=True.
                - 'thought_tree_output' (Optional[torch.Tensor]): Tree reasoner
                  output if return_intermediate=True.
        
        Example:
            >>> system = YvEnhancedReasoningSystem(config)
            >>> hidden = torch.randn(2, 10, 2048)
            >>> result = system(hidden, task_type="mathematical")
            >>> print(f"Confidence: {result['confidence'].mean():.2f}")
            >>> print(f"Complexity: {result['complexity'].mean():.2f}")
        
        Note:
            Simple problems (complexity <= 0.4) may bypass both reasoners,
            using only the original hidden states with uncertainty estimation.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        routing_info = {}
        if self.router is not None:
            routing_info = self.router(hidden_states, task_type)
        else:
            routing_info = {
                'complexity': torch.tensor(0.5, device=hidden_states.device).expand(batch_size, 1),
                'recommended_depth': torch.tensor(5, device=hidden_states.device).expand(batch_size)
            }
        
        recursive_output = None
        thought_tree_output = None
        reasoning_features = hidden_states
        
        if self.recursive_reasoner is not None and routing_info['complexity'].mean() > 0.4:
            recursive_result = self.recursive_reasoner(
                hidden_states,
                attention_mask,
                task_type,
                return_intermediate=return_intermediate
            )
            recursive_output = recursive_result['reasoning_output']
            routing_info['recursive_depth'] = recursive_result.get('optimal_depth', 5)
            routing_info['recursive_confidence'] = recursive_result.get('confidence', None)
        
        if self.thought_tree_reasoner is not None and routing_info['complexity'].mean() > 0.6:
            problem_embedding = hidden_states.mean(dim=1, keepdim=True)
            context = hidden_states.mean(dim=1, keepdim=True).expand(-1, hidden_states.size(1), -1)
            
            tree_result = self.thought_tree_reasoner(
                problem_embedding,
                context,
                num_simulations=self.config.num_simulations
            )
            thought_tree_output = tree_result['thought_tree_output']
            routing_info['tree_confidence'] = tree_result['confidence']
        
        features_to_fuse = [hidden_states.mean(dim=1, keepdim=True)]
        
        if recursive_output is not None:
            if recursive_output.dim() == 2:
                recursive_output = recursive_output.unsqueeze(1)
            features_to_fuse.append(recursive_output)
        
        if thought_tree_output is not None:
            if thought_tree_output.dim() == 2:
                thought_tree_output = thought_tree_output.unsqueeze(1)
            features_to_fuse.append(thought_tree_output)
        
        max_seq_len = max(f.size(1) for f in features_to_fuse)
        
        padded_features = []
        for f in features_to_fuse:
            if f.size(1) < max_seq_len:
                padding = torch.zeros(
                    batch_size,
                    max_seq_len - f.size(1),
                    self.hidden_size,
                    device=f.device,
                    dtype=f.dtype
                )
                padded = torch.cat([f, padding], dim=1)
            else:
                padded = f
            padded_features.append(padded)
        
        concat_features = torch.cat(padded_features, dim=-1)
        
        output = self.output_proj(concat_features)
        
        uncertainty_info = {}
        if self.uncertainty_quantifier is not None:
            uncertainty_info = self.uncertainty_quantifier(
                output,
                hidden_states,
                reasoning_features
            )
        
        final_confidence = routing_info.get('recursive_confidence', routing_info.get('tree_confidence'))
        if final_confidence is None:
            final_confidence = 1 - uncertainty_info.get('overall_uncertainty', 
                torch.ones(batch_size, 1, device=output.device) * 0.5)
        
        result = {
            'reasoning_output': output,
            'confidence': final_confidence,
            'uncertainty': uncertainty_info,
            'routing_info': routing_info,
            'task_type': task_type,
            'complexity': routing_info['complexity']
        }
        
        if return_intermediate:
            result['recursive_output'] = recursive_output
            result['thought_tree_output'] = thought_tree_output
        
        return result
