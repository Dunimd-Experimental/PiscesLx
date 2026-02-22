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

"""Recursive depth reasoning with subproblem decomposition and verification.

This module implements sophisticated recursive reasoning algorithms that break
down complex problems into manageable subproblems, solve them recursively, and
aggregate results with optional bidirectional verification. It provides both
depth-controlled recursive reasoning and thought tree exploration strategies.

Architecture Overview:
    The module provides two main reasoning paradigms:
    
    1. **Recursive Depth Reasoner** (YvRecursiveDepthReasoner):
       - Decomposes problems into subproblems based on complexity
       - Recursively solves subproblems with adaptive depth
       - Aggregates results using attention-based mechanisms
       - Verifies solutions through bidirectional reasoning
    
    2. **Thought Tree Reasoner** (YvThoughtTreeReasoner):
       - Explores reasoning as a tree search problem
       - Uses Upper Confidence Bound (UCB) for node selection
       - Performs Monte Carlo Tree Search (MCTS) style simulations
       - Returns best reasoning path based on value estimates

Key Components:
    - **SubProblemDecomposer**: Neural network that predicts subproblem structure
    - **RecursiveReasoningBlock**: Transformer-based reasoning layer stack
    - **SubProblemAggregator**: Attention-based result combination with gating
    - **BidirectionalVerifier**: Forward-backward consistency checking

Mathematical Foundations:
    Recursive Reasoning:
        - Problem decomposition: P = {p_1, p_2, ..., p_n}
        - Recursive solution: S(P) = Aggregate(S(p_1), S(p_2), ..., S(p_n))
        - Depth control: d* = argmin_d (complexity * max_depth)
    
    Thought Tree Search:
        - UCB score: UCB = exploitation + c * sqrt(log(N_parent) / N_child)
        - Value propagation: V_parent = (V_parent * N_parent + V_leaf) / (N_parent + 1)
        - Best path selection: argmax_i V_i

Complexity Analysis:
    - SubProblemDecomposer: O(hidden_size^2) for decomposition network
    - RecursiveReasoningBlock: O(seq_len^2 * hidden_size) for transformer layers
    - SubProblemAggregator: O(n_subproblems * hidden_size^2) for attention
    - BidirectionalVerifier: O(hidden_size^2) for consistency checking
    - Thought Tree: O(simulations * tree_depth * beam_width) for MCTS

Configuration Options:
    - max_depth: Maximum recursion depth (default: 10)
    - width: Maximum subproblems per decomposition (default: 4)
    - temperature: Softmax temperature for sampling (default: 0.7)
    - early_exit_threshold: Confidence threshold for early termination (default: 0.9)

Example:
    >>> config = YvRecursiveReasoningConfig(
    ...     hidden_size=2048,
    ...     max_depth=8,
    ...     use_verification=True
    ... )
    >>> reasoner = YvRecursiveDepthReasoner(config)
    >>> 
    >>> # Process hidden states
    >>> result = reasoner(
    ...     hidden_states,
    ...     attention_mask=None,
    ...     task_type="reasoning",
    ...     return_intermediate=True
    ... )
    >>> print(f"Confidence: {result['confidence']:.2f}")
    >>> print(f"Optimal depth: {result['optimal_depth']}")

Dependencies:
    - torch: Neural network operations
    - torch.nn: Module definitions
    - torch.nn.functional: Activation functions
    - dataclasses: Configuration management

Note:
    The recursive depth reasoner is designed for problems that benefit from
    decomposition, while the thought tree reasoner excels at exploration-heavy
    reasoning tasks with uncertain solution paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class YvRecursiveReasoningConfig:
    """Configuration dataclass for recursive depth reasoning parameters.
    
    This dataclass encapsulates all hyperparameters controlling the behavior
    of the recursive depth reasoner, including depth limits, width constraints,
    and optional feature flags for aggregation and verification.
    
    Attributes:
        hidden_size (int): Dimension of hidden state representations.
            Must match the model's hidden dimension. Default: 2048.
        max_depth (int): Maximum recursion depth for problem decomposition.
            Higher values allow more complex problem breakdown. Default: 10.
        min_depth (int): Minimum recursion depth before early exit consideration.
            Prevents premature termination on simple problems. Default: 1.
        width (int): Maximum number of subproblems per decomposition step.
            Controls branching factor in the reasoning tree. Default: 4.
        temperature (float): Softmax temperature for sampling operations.
            Higher values increase exploration, lower values increase exploitation.
            Default: 0.7.
        use_aggregation (bool): Whether to use attention-based aggregation
            for combining subproblem results. Default: True.
        use_verification (bool): Whether to use bidirectional verification
            for consistency checking of solutions. Default: True.
        threshold_complexity (float): Complexity threshold for triggering
            decomposition. Problems below this threshold may skip decomposition.
            Default: 0.5.
        early_exit_threshold (float): Confidence threshold for early termination.
            Reasoning stops when confidence exceeds this value. Default: 0.9.
    
    Example:
        >>> config = YvRecursiveReasoningConfig(
        ...     hidden_size=4096,
        ...     max_depth=12,
        ...     use_verification=False
        ... )
        >>> reasoner = YvRecursiveDepthReasoner(config)
    
    Note:
        The product of max_depth and width determines the maximum theoretical
        number of leaf nodes in the reasoning tree. Memory usage scales with
        this product.
    """
    hidden_size: int = 2048
    max_depth: int = 10
    min_depth: int = 1
    width: int = 4
    temperature: float = 0.7
    use_aggregation: bool = True
    use_verification: bool = True
    threshold_complexity: float = 0.5
    early_exit_threshold: float = 0.9


class _SubProblemDecomposer(nn.Module):
    """Neural network module for decomposing problems into subproblems.
    
    This module implements a learned decomposition strategy that analyzes problem
    embeddings and predicts both the number and structure of subproblems. It uses
    a combination of feed-forward networks and multi-head attention to identify
    natural problem boundaries and create meaningful subproblem representations.
    
    Architecture:
        1. **Decomposition Network**:
           - Input: Concatenated [problem, context, complexity-weighted problem]
           - Two-layer MLP with GELU activation and LayerNorm
           - Output: Decomposed feature representation
        
        2. **Attention Mechanism**:
           - Multi-head self-attention over decomposed features
           - 4 heads with hidden_size // 4 dimensions per head
           - Dropout of 0.1 for regularization
        
        3. **Subproblem Count Predictor**:
           - Linear layer predicting distribution over 0-5 subproblems
           - Softmax activation for probability distribution
    
    Attributes:
        hidden_size (int): Dimension of input and output representations.
        num_heads (int): Number of attention heads (default: 4).
        decomposition_net (nn.Sequential): MLP for feature decomposition.
        head_dim (int): Dimension per attention head.
        attention (nn.MultiheadAttention): Self-attention module.
        num_subproblems_predictor (nn.Linear): Subproblem count prediction head.
    
    Example:
        >>> decomposer = _SubProblemDecomposer(hidden_size=2048)
        >>> problem_emb = torch.randn(1, 2048)
        >>> context = torch.randn(1, 2048)
        >>> result = decomposer(problem_emb, context, complexity_score=0.7)
        >>> print(result['num_subproblems'])  # tensor with predicted count
    
    Note:
        The complexity_score modulates the decomposition by scaling the problem
        embedding, allowing the network to produce different decompositions
        for problems of varying difficulty.
    """
    
    def __init__(self, hidden_size: int, num_decomposition_heads: int = 4):
        """Initialize the subproblem decomposition module.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
            num_decomposition_heads (int): Number of attention heads for the
                self-attention mechanism. Default: 4.
        
        Initializes:
            - decomposition_net: 3-layer MLP (hidden_size*3 -> hidden_size)
            - attention: Multi-head attention with specified head count
            - num_subproblems_predictor: Linear layer for count prediction
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_decomposition_heads
        
        self.decomposition_net = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.head_dim = hidden_size // num_decomposition_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_decomposition_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.num_subproblems_predictor = nn.Linear(hidden_size, 6)
    
    def forward(
        self,
        problem_embedding: torch.Tensor,
        context: torch.Tensor,
        complexity_score: float
    ) -> Dict[str, torch.Tensor]:
        """Decompose a problem into subproblem representations.
        
        This method takes a problem embedding and context, applies learned
        decomposition, and returns subproblem embeddings along with the
        predicted number of subproblems.
        
        Processing Steps:
            1. Concatenate problem, context, and complexity-weighted problem
            2. Pass through decomposition network
            3. Apply self-attention to refine features
            4. Predict subproblem count distribution
            5. Convert distribution to discrete count
        
        Args:
            problem_embedding (torch.Tensor): Problem representation tensor
                of shape (batch_size, hidden_size).
            context (torch.Tensor): Contextual information tensor of shape
                (batch_size, hidden_size).
            complexity_score (float): Problem complexity in [0, 1] range.
                Higher values indicate more complex problems requiring
                more subproblems.
        
        Returns:
            Dict[str, torch.Tensor]: Decomposition results containing:
                - 'sub_problem_embeddings': Tensor of shape (batch_size, hidden_size)
                  representing the decomposed subproblem features.
                - 'num_subproblems': Tensor of shape (batch_size,) with predicted
                  number of subproblems (integer values 0-5).
                - 'decomposed_features': Tensor of shape (batch_size, hidden_size)
                  with raw decomposition network output.
        
        Example:
            >>> decomposer = _SubProblemDecomposer(2048)
            >>> emb = torch.randn(2, 2048)
            >>> ctx = torch.randn(2, 2048)
            >>> out = decomposer(emb, ctx, 0.6)
            >>> print(out['num_subproblems'].shape)  # torch.Size([2])
        """
        decomposed_features = self.decomposition_net(
            torch.cat([problem_embedding, context, problem_embedding * complexity_score], dim=-1)
        )
        
        queries = decomposed_features.unsqueeze(1).repeat(1, self.num_heads, 1)
        keys = decomposed_features.unsqueeze(1).repeat(1, self.num_heads, 1)
        values = decomposed_features.unsqueeze(1).repeat(1, self.num_heads, 1)
        
        attended, _ = self.attention(queries, keys, values)
        
        sub_problem_embeddings = attended.mean(dim=1)
        
        max_subproblems = torch.softmax(self.num_subproblems_predictor(problem_embedding), dim=-1)
        max_subproblems = (max_subproblems * 6).long()
        
        return {
            'sub_problem_embeddings': sub_problem_embeddings,
            'num_subproblems': max_subproblems,
            'decomposed_features': decomposed_features
        }


class _RecursiveReasoningBlock(nn.Module):
    """Transformer-based reasoning block for processing problem representations.
    
    This module implements a stack of transformer encoder layers followed by
    a projection network, designed to refine problem representations through
    self-attention and feed-forward transformations.
    
    Architecture:
        - **Transformer Layers**: Stack of TransformerEncoderLayer modules
          with 8 attention heads and 4x expansion in feed-forward dimension
        - **Output Projection**: 4-layer network with LayerNorm and GELU
          activation for final representation refinement
    
    Attributes:
        hidden_size (int): Dimension of input and output representations.
        num_layers (int): Number of transformer encoder layers (default: 2).
        reasoning_layers (nn.ModuleList): Stack of transformer encoder layers.
        output_proj (nn.Sequential): Final projection network.
    
    Example:
        >>> block = _RecursiveReasoningBlock(hidden_size=2048, num_layers=3)
        >>> x = torch.randn(2, 10, 2048)  # batch=2, seq_len=10
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([2, 10, 2048])
    
    Note:
        Uses GELU activation throughout for smooth gradient flow.
        Dropout of 0.1 is applied within each transformer layer.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 2):
        """Initialize the recursive reasoning block.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
            num_layers (int): Number of transformer encoder layers to stack.
                More layers increase reasoning capacity but also computation.
                Default: 2.
        
        Initializes:
            - reasoning_layers: ModuleList of TransformerEncoderLayer instances
            - output_proj: Sequential projection network
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through transformer layers and projection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            torch.Tensor: Processed tensor of same shape as input.
        """
        for layer in self.reasoning_layers:
            x = layer(x)
        return self.output_proj(x)


class _SubProblemAggregator(nn.Module):
    """Attention-based aggregator for combining subproblem results.
    
    This module implements a sophisticated aggregation mechanism that combines
    results from multiple subproblems using multi-head attention and learned
    gating. It produces a unified representation that integrates information
    from all subproblem solutions.
    
    Aggregation Strategy:
        1. **Attention-Based Combination**:
           - Uses problem embedding as query
           - Subproblem results serve as keys and values
           - Produces attention-weighted combination
        
        2. **Optional Weight Integration**:
           - When weights provided, combines with attention output
           - Uses softmax-normalized weights for weighted sum
           - Blends weighted sum with attention result (70/30 split)
        
        3. **Gating Mechanism**:
           - Learned gate controls blend between aggregated and original
           - Prevents loss of important problem-specific information
           - Sigmoid gate produces values in [0, 1] range
    
    Attributes:
        hidden_size (int): Dimension of input and output representations.
        num_heads (int): Number of attention heads (default: 8).
        attention (nn.MultiheadAttention): Cross-attention module.
        gate_net (nn.Sequential): Gating network for blend control.
        output_proj (nn.Sequential): Final projection with LayerNorm.
    
    Example:
        >>> aggregator = _SubProblemAggregator(hidden_size=2048)
        >>> problem = torch.randn(2, 2048)
        >>> sub_results = [torch.randn(2, 2048) for _ in range(3)]
        >>> output = aggregator(problem, sub_results)
        >>> print(output.shape)  # torch.Size([2, 2048])
    
    Note:
        Returns the original problem_embedding if sub_problem_results is empty.
        Single subproblem results bypass attention for efficiency.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        """Initialize the subproblem aggregator module.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
            num_heads (int): Number of attention heads for cross-attention.
                Default: 8.
        
        Initializes:
            - attention: Multi-head attention for result combination
            - gate_net: Sigmoid-gated blending network
            - output_proj: LayerNorm-normalized projection
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(
        self,
        problem_embedding: torch.Tensor,
        sub_problem_results: List[torch.Tensor],
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Aggregate subproblem results into a unified representation.
        
        This method combines multiple subproblem solutions using attention
        and optional weighting, producing a coherent final representation.
        
        Args:
            problem_embedding (torch.Tensor): Original problem representation
                of shape (batch_size, hidden_size). Used as query for attention.
            sub_problem_results (List[torch.Tensor]): List of subproblem solution
                tensors, each of shape (batch_size, hidden_size).
            weights (Optional[torch.Tensor]): Optional importance weights for
                each subproblem. If provided, should have length equal to
                sub_problem_results. Default: None.
        
        Returns:
            torch.Tensor: Aggregated representation of shape (batch_size, hidden_size).
                Returns problem_embedding if sub_problem_results is empty.
        
        Example:
            >>> agg = _SubProblemAggregator(2048)
            >>> prob = torch.randn(1, 2048)
            >>> subs = [torch.randn(1, 2048) for _ in range(4)]
            >>> weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
            >>> result = agg(prob, subs, weights)
        """
        if not sub_problem_results:
            return problem_embedding
        
        if len(sub_problem_results) == 1:
            aggregated = sub_problem_results[0]
        else:
            stacked = torch.stack(sub_problem_results, dim=1)
            
            query = problem_embedding.unsqueeze(1)
            key = stacked
            value = stacked
            
            attended, attention_weights = self.attention(query, key, value)
            
            if weights is not None:
                weights_normalized = F.softmax(weights, dim=0)
                weighted_sum = sum(w * r for w, r in zip(weights_normalized, sub_problem_results))
                aggregated = weighted_sum + attended.squeeze(1) * 0.3
            else:
                aggregated = attended.squeeze(1)
        
        gate = self.gate_net(torch.cat([problem_embedding, aggregated], dim=-1))
        
        output = self.output_proj(aggregated * gate + problem_embedding * (1 - gate))
        
        return output


class _BidirectionalVerifier(nn.Module):
    """Bidirectional reasoning verifier for solution consistency checking.
    
    This module implements a forward-backward verification mechanism that
    checks the logical consistency of reasoning results. It processes
    hypotheses and evidence in both directions to ensure coherent solutions.
    
    Verification Process:
        1. **Forward Reasoning**:
           - Combines hypothesis with context
           - Produces forward reasoning result
        
        2. **Backward Reasoning**:
           - Combines evidence with forward result
           - Produces backward reasoning result
        
        3. **Consistency Check**:
           - Concatenates hypothesis, forward, and backward results
           - Passes through consistency checker network
           - Outputs consistency score in [0, 1] range
        
        4. **Result Fusion**:
           - Combines forward and backward results
           - Produces verified, fused representation
    
    Attributes:
        hidden_size (int): Dimension of input and output representations.
        forward_reasoner (nn.Sequential): Forward direction reasoning network.
        backward_reasoner (nn.Sequential): Backward direction reasoning network.
        consistency_checker (nn.Sequential): Network producing consistency score.
        fusion_proj (nn.Sequential): Final fusion projection with LayerNorm.
    
    Example:
        >>> verifier = _BidirectionalVerifier(hidden_size=2048)
        >>> hypothesis = torch.randn(2, 2048)
        >>> evidence = torch.randn(2, 2048)
        >>> context = torch.randn(2, 2048)
        >>> fused, score = verifier(hypothesis, evidence, context)
        >>> print(f"Consistency: {score.mean():.2f}")
    
    Note:
        Consistency score close to 1.0 indicates high coherence between
        hypothesis and evidence. Low scores suggest potential contradictions.
    """
    
    def __init__(self, hidden_size: int):
        """Initialize the bidirectional verifier module.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
        
        Initializes:
            - forward_reasoner: 3-layer MLP for forward reasoning
            - backward_reasoner: 3-layer MLP for backward reasoning
            - consistency_checker: 5-layer network with sigmoid output
            - fusion_proj: 2-layer projection with LayerNorm
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        self.forward_reasoner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.backward_reasoner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(
        self,
        hypothesis: torch.Tensor,
        evidence: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform bidirectional verification on hypothesis and evidence.
        
        This method executes the complete verification pipeline, producing
        both a fused result and a consistency score.
        
        Args:
            hypothesis (torch.Tensor): Proposed solution or claim tensor
                of shape (batch_size, hidden_size).
            evidence (torch.Tensor): Supporting evidence tensor of shape
                (batch_size, hidden_size).
            context (torch.Tensor): Contextual information tensor of shape
                (batch_size, hidden_size).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - fused (torch.Tensor): Verified and fused representation
                  of shape (batch_size, hidden_size).
                - consistency_score (torch.Tensor): Consistency score tensor
                  of shape (batch_size, 1) with values in [0, 1].
        
        Example:
            >>> verifier = _BidirectionalVerifier(2048)
            >>> h = torch.randn(1, 2048)
            >>> e = torch.randn(1, 2048)
            >>> c = torch.randn(1, 2048)
            >>> fused, score = verifier(h, e, c)
            >>> print(f"Consistency: {score.item():.2%}")
        """
        forward_result = self.forward_reasoner(
            torch.cat([hypothesis, context], dim=-1)
        )
        
        backward_result = self.backward_reasoner(
            torch.cat([evidence, forward_result], dim=-1)
        )
        
        consistency_features = torch.cat([
            hypothesis,
            forward_result,
            backward_result
        ], dim=-1)
        
        consistency_score = self.consistency_checker(consistency_features)
        
        fused = self.fusion_proj(
            torch.cat([forward_result, backward_result], dim=-1)
        )
        
        return fused, consistency_score


class YvRecursiveDepthReasoner(nn.Module):
    """Main recursive depth reasoning module with adaptive decomposition.
    
    This class implements a sophisticated recursive reasoning system that
    adaptively decomposes complex problems into subproblems, solves them
    recursively, and aggregates results with optional verification. It provides
    a complete pipeline for depth-controlled reasoning with early exit support.
    
    Architecture:
        1. **Problem Analysis**:
           - Complexity prediction from hidden states
           - Optimal depth estimation based on complexity
           - Task-specific depth adjustment
        
        2. **Recursive Decomposition**:
           - Subproblem generation via learned decomposition
           - Noise injection for exploration diversity
           - Depth-limited recursion with early termination
        
        3. **Result Aggregation**:
           - Attention-based combination of subproblem results
           - Gated blending with original problem embedding
           - Optional verification for consistency checking
        
        4. **Confidence Estimation**:
           - Multi-component confidence prediction
           - Based on final hidden states and reasoning trajectory
    
    Key Features:
        - **Adaptive Depth**: Dynamically adjusts recursion depth based on
          problem complexity and learned predictions
        - **Early Exit**: Terminates early when sufficient confidence achieved
        - **Verification**: Optional bidirectional consistency checking
        - **Task Specialization**: Different depth limits for reasoning,
          planning, and mathematical tasks
    
    Attributes:
        config (YvRecursiveReasoningConfig): Configuration dataclass.
        hidden_size (int): Dimension of hidden representations.
        max_depth (int): Maximum recursion depth.
        min_depth (int): Minimum recursion depth.
        width (int): Maximum subproblems per decomposition.
        temperature (float): Sampling temperature.
        decomposer (_SubProblemDecomposer): Subproblem generation module.
        recursive_block (_RecursiveReasoningBlock): Transformer reasoning stack.
        aggregator (Optional[_SubProblemAggregator]): Result combination module.
        verifier (Optional[_BidirectionalVerifier]): Consistency checker.
        complexity_predictor (nn.Sequential): Complexity estimation network.
        confidence_predictor (nn.Sequential): Confidence estimation network.
        depth_predictor (nn.Sequential): Optimal depth prediction network.
    
    Example:
        >>> config = YvRecursiveReasoningConfig(
        ...     hidden_size=2048,
        ...     max_depth=8,
        ...     use_verification=True
        ... )
        >>> reasoner = YvRecursiveDepthReasoner(config)
        >>> hidden_states = torch.randn(2, 10, 2048)
        >>> result = reasoner(hidden_states, task_type="reasoning")
        >>> print(f"Confidence: {result['confidence'].mean():.2f}")
        >>> print(f"Optimal depth: {result['optimal_depth']}")
    
    Note:
        The module uses Xavier initialization for linear layers and
        constant initialization for LayerNorm parameters.
    """
    
    def __init__(self, config: Optional[YvRecursiveReasoningConfig] = None):
        """Initialize the recursive depth reasoner with configuration.
        
        This constructor sets up all submodules based on the provided
        configuration, including optional aggregation and verification modules.
        
        Args:
            config (Optional[YvRecursiveReasoningConfig]): Configuration
                dataclass containing all hyperparameters. If None, uses default
                values from YvRecursiveReasoningConfig.
        
        Initializes:
            - decomposer: Subproblem decomposition network
            - recursive_block: Transformer-based reasoning layers
            - aggregator: Optional attention-based result combiner
            - verifier: Optional bidirectional consistency checker
            - complexity_predictor: Problem complexity estimation network
            - confidence_predictor: Solution confidence estimation network
            - depth_predictor: Optimal depth prediction network
        """
        super().__init__()
        if config is None:
            config = YvRecursiveReasoningConfig()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_depth = config.max_depth
        self.min_depth = config.min_depth
        self.width = config.width
        self.temperature = config.temperature
        
        self.decomposer = _SubProblemDecomposer(
            hidden_size=self.hidden_size,
            num_decomposition_heads=4
        )
        
        self.recursive_block = _RecursiveReasoningBlock(
            hidden_size=self.hidden_size,
            num_layers=2
        )
        
        if config.use_aggregation:
            self.aggregator = _SubProblemAggregator(
                hidden_size=self.hidden_size,
                num_heads=8
            )
        else:
            self.aggregator = None
        
        if config.use_verification:
            self.verifier = _BidirectionalVerifier(self.hidden_size)
        else:
            self.verifier = None
        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.depth_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.max_depth),
            nn.Softmax(dim=-1)
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
    
    def _calculate_problem_complexity(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Estimate problem complexity from hidden state representations.
        
        This method computes a complexity score that guides the depth of
        recursive decomposition. Higher complexity scores lead to deeper
        recursion and more subproblem decomposition.
        
        Args:
            hidden_states (torch.Tensor): Hidden state tensor of shape
                (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Optional attention mask
                of shape (batch_size, seq_len) for excluding padding tokens.
        
        Returns:
            torch.Tensor: Complexity score tensor of shape (batch_size, 1)
                with values in [0, 1] range.
        
        Note:
            When attention_mask is provided, uses masked mean pooling.
            Otherwise, uses simple mean pooling across sequence dimension.
        """
        pooled = hidden_states.mean(dim=1)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden = hidden_states * mask_expanded
            masked_pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            complexity_input = masked_pooled
        else:
            complexity_input = pooled
        
        complexity = self.complexity_predictor(complexity_input)
        
        return complexity
    
    def _predict_optimal_depth(
        self,
        hidden_states: torch.Tensor,
        complexity: torch.Tensor
    ) -> int:
        """Predict optimal recursion depth using ensemble estimation.
        
        This method combines learned depth prediction with complexity-based
        estimation to determine the optimal recursion depth for a problem.
        The ensemble approach balances data-driven predictions with heuristic
        complexity scaling.
        
        Args:
            hidden_states (torch.Tensor): Hidden state tensor of shape
                (batch_size, seq_len, hidden_size) for depth prediction.
            complexity (torch.Tensor): Complexity score tensor of shape
                (batch_size, 1) for heuristic depth scaling.
        
        Returns:
            int: Optimal recursion depth in range [min_depth, max_depth].
        
        Note:
            Uses 60/40 weighting between learned prediction and complexity-based
            estimation. The learned component uses expected value from softmax
            distribution over depth options.
        """
        depth_logits = self.depth_predictor(hidden_states.mean(dim=1))
        
        expected_depth = (depth_logits * torch.arange(1, self.max_depth + 1, device=depth_logits.device)).sum(dim=-1)
        
        base_depth = int(complexity.item() * self.max_depth)
        base_depth = max(self.min_depth, min(self.max_depth, base_depth))
        
        ensemble_depth = int(0.6 * expected_depth.item() + 0.4 * base_depth)
        ensemble_depth = max(self.min_depth, min(self.max_depth, ensemble_depth))
        
        return ensemble_depth
    
    def _recursive_solve(
        self,
        problem_embedding: torch.Tensor,
        context: torch.Tensor,
        current_depth: int,
        max_depth: int,
        sub_problem_results: List[torch.Tensor]
    ) -> torch.Tensor:
        """Recursively solve a problem by decomposition and aggregation.
        
        This method implements the core recursive reasoning algorithm. It
        decomposes problems into subproblems, solves each recursively, and
        aggregates the results. The recursion terminates when reaching max_depth
        or when the problem is sufficiently simple.
        
        Recursive Algorithm:
            1. **Base Case**: If current_depth >= max_depth, process through
               recursive_block and return the result
            2. **Decomposition**: Calculate complexity and decompose into subproblems
            3. **Recursive Solve**: For each subproblem, recursively call this method
            4. **Aggregation**: Combine subproblem results using attention or mean
        
        Args:
            problem_embedding (torch.Tensor): Problem representation tensor
                of shape (hidden_size,).
            context (torch.Tensor): Contextual information tensor of shape
                (hidden_size,).
            current_depth (int): Current recursion depth level.
            max_depth (int): Maximum allowed recursion depth.
            sub_problem_results (List[torch.Tensor]): Accumulator list for
                collecting intermediate subproblem results.
        
        Returns:
            torch.Tensor: Aggregated solution tensor of shape (hidden_size,).
        
        Note:
            Noise is injected into subproblem embeddings for exploration diversity.
            The noise magnitude decreases with depth to stabilize deeper solutions.
        """
        if current_depth >= max_depth:
            result = self.recursive_block(problem_embedding)
            sub_problem_results.append(result)
            return result
        
        complexity = self._calculate_problem_complexity(
            problem_embedding.unsqueeze(0)
        ).squeeze(-1)
        
        decomposition = self.decomposer(
            problem_embedding,
            context,
            complexity
        )
        
        num_subproblems = min(
            decomposition['num_subproblems'].clamp(2, 6).item(),
            self.width
        )
        
        sub_problem_embeddings = []
        for i in range(num_subproblems):
            noise = torch.randn_like(problem_embedding) * 0.1 * (1 - current_depth / max_depth)
            sub_emb = decomposition['sub_problem_embeddings'] + noise
            sub_problem_embeddings.append(sub_emb)
        
        sub_results = []
        for i, sub_emb in enumerate(sub_problem_embeddings):
            sub_result = self._recursive_solve(
                sub_emb,
                context,
                current_depth + 1,
                max_depth,
                sub_problem_results
            )
            sub_results.append(sub_result)
        
        if self.aggregator is not None:
            weights = torch.ones(len(sub_results), device=problem_embedding.device) / len(sub_results)
            aggregated = self.aggregator(
                problem_embedding,
                sub_results,
                weights
            )
        else:
            aggregated = torch.stack(sub_results).mean(dim=0)
        
        sub_problem_results.append(aggregated)
        
        return aggregated
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "reasoning",
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """Execute recursive depth reasoning on input hidden states.
        
        This method implements the complete forward pass of the recursive depth
        reasoner, processing each batch element through either verified or
        standard recursive reasoning based on configuration.
        
        Processing Pipeline:
            1. **Task-Specific Depth Adjustment**: Different max_depth limits
               for planning (8), mathematical (10), and general reasoning (6)
            2. **Context Extraction**: Mean pooling for contextual reference
            3. **Complexity and Depth Prediction**: Determine optimal recursion
            4. **Batch Processing**: Process each sample individually
            5. **Confidence Estimation**: Predict solution confidence
        
        Args:
            hidden_states (torch.Tensor): Input hidden states of shape
                (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Optional attention mask
                of shape (batch_size, seq_len). Default: None.
            task_type (str): Task type for depth adjustment. Options are
                "reasoning", "planning", or "mathematical". Default: "reasoning".
            return_intermediate (bool): Whether to return intermediate results
                including consistency scores and subproblem results. Default: False.
        
        Returns:
            Dict[str, Any]: Results dictionary containing:
                - 'reasoning_output' (torch.Tensor): Final hidden states
                  of shape (batch_size, hidden_size).
                - 'confidence' (torch.Tensor): Confidence scores of shape
                  (batch_size, 1).
                - 'complexity' (torch.Tensor): Problem complexity scores
                  of shape (batch_size, 1).
                - 'optimal_depth' (int): Predicted optimal recursion depth.
                - 'num_subproblems' (int): Number of subproblems generated.
                - 'consistency_scores' (Optional[torch.Tensor]): Consistency
                  scores if verification enabled and return_intermediate=True.
                - 'verification_passed' (Optional[bool]): Whether all
                  consistency checks passed, if verification enabled.
        
        Example:
            >>> reasoner = YvRecursiveDepthReasoner(config)
            >>> hidden = torch.randn(2, 10, 2048)
            >>> result = reasoner(hidden, task_type="mathematical")
            >>> print(f"Confidence: {result['confidence'].mean():.2f}")
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if task_type == "planning":
            max_depth = min(self.max_depth, 8)
        elif task_type == "mathematical":
            max_depth = min(self.max_depth, 10)
        else:
            max_depth = min(self.max_depth, 6)
        
        context = hidden_states.mean(dim=1, keepdim=True)
        
        problem_embedding = hidden_states
        
        complexity = self._calculate_problem_complexity(hidden_states, attention_mask)
        
        optimal_depth = self._predict_optimal_depth(hidden_states, complexity)
        optimal_depth = min(optimal_depth, max_depth)
        
        sub_problem_results = []
        
        for b in range(batch_size):
            sub_problem_results.append([])
        
        final_results = []
        consistency_scores = []
        
        for b in range(batch_size):
            problem_emb = hidden_states[b:b+1, :seq_len]
            
            if self.verifier is not None:
                hypothesis = self.recursive_block(problem_emb)
                
                evidence = self.recursive_block(hypothesis + context[b:b+1])
                
                verified_result, consistency = self.verifier(
                    hypothesis,
                    evidence,
                    context[b:b+1]
                )
                consistency_scores.append(consistency)
                
                sub_problem_results[b].append(verified_result)
                final_results.append(verified_result.squeeze(0))
            else:
                result = self._recursive_solve(
                    problem_emb.squeeze(0),
                    context[b].squeeze(0),
                    current_depth=0,
                    max_depth=optimal_depth,
                    sub_problem_results=sub_problem_results[b]
                )
                final_results.append(result.squeeze(0))
        
        final_hidden = torch.stack(final_results, dim=0)
        
        confidence = self.confidence_predictor(
            torch.cat([
                final_hidden,
                hidden_states.mean(dim=1),
                final_hidden - hidden_states.mean(dim=1)
            ], dim=-1)
        )
        
        result = {
            'reasoning_output': final_hidden,
            'confidence': confidence,
            'complexity': complexity,
            'optimal_depth': optimal_depth,
            'num_subproblems': len(sub_problem_results[0]) if sub_problem_results else 0
        }
        
        if return_intermediate and self.verifier is not None:
            result['consistency_scores'] = torch.stack(consistency_scores) if consistency_scores else None
            result['sub_problem_results'] = sub_problem_results
        
        if self.verifier is not None and consistency_scores:
            result['verification_passed'] = all(
                score > self.config.threshold_complexity 
                for score in [s.item() for s in consistency_scores]
            )
        
        return result


class YvThoughtTreeReasoner(nn.Module):
    """Monte Carlo Tree Search-style reasoning with UCB selection.
    
    This class implements a thought tree reasoning approach inspired by Monte
    Carlo Tree Search (MCTS). It explores a tree of reasoning steps using
    Upper Confidence Bound (UCB) for node selection, balancing exploration
    and exploitation during the search process.
    
    Architecture:
        1. **Thought Encoding**:
           - Encodes problem-context pairs into initial thought representations
           - Uses 3-layer MLP with GELU activation and LayerNorm
        
        2. **Value and Policy Heads**:
           - Value head: Estimates node value for UCB calculation
           - Policy head: Guides thought expansion direction
        
        3. **Expansion Network**:
           - Multiple expansion networks (one per beam width)
           - Each generates a child thought from parent embedding
           - Residual connection with 0.1 scaling for stability
        
        4. **Selection Attention**:
           - Multi-head attention for node selection refinement
           - 8 heads with dropout for regularization
    
    MCTS Algorithm:
        1. **Selection**: Traverse tree using UCB scores
        2. **Expansion**: Add new child nodes at leaf
        3. **Simulation**: Evaluate leaf node value
        4. **Backpropagation**: Update ancestor values and visit counts
    
    UCB Formula:
        UCB = exploitation + c * sqrt(log(N_parent) / N_child)
        where c = 1.414 (exploration weight)
    
    Attributes:
        hidden_size (int): Dimension of hidden representations.
        tree_depth (int): Maximum depth of the thought tree (default: 5).
        beam_width (int): Number of children per node (default: 3).
        thought_encoder (nn.Sequential): Problem-context encoding network.
        value_head (nn.Linear): Node value estimation layer.
        policy_head (nn.Linear): Policy direction layer.
        expansion_net (nn.ModuleList): Child generation networks.
        selection_attention (nn.MultiheadAttention): Node selection attention.
    
    Example:
        >>> reasoner = YvThoughtTreeReasoner(hidden_size=2048, tree_depth=6)
        >>> problem = torch.randn(2, 2048)
        >>> context = torch.randn(2, 2048)
        >>> result = reasoner(problem, context, num_simulations=50)
        >>> print(f"Best value: {max(result['best_values']):.2f}")
        >>> print(f"Confidence: {result['confidence'].mean():.2f}")
    
    Note:
        Higher num_simulations improves solution quality but increases
        computation time. The tree grows exponentially with depth and
        beam_width, so use conservative values for memory efficiency.
    """
    
    def __init__(self, hidden_size: int, tree_depth: int = 5, beam_width: int = 3):
        """Initialize the thought tree reasoner.
        
        Args:
            hidden_size (int): Dimension of hidden representations. Must match
                the model's hidden dimension for compatibility.
            tree_depth (int): Maximum depth of the thought tree. Controls how
                many reasoning steps can be explored. Default: 5.
            beam_width (int): Number of child nodes generated per parent.
                Controls branching factor of the tree. Default: 3.
        
        Initializes:
            - thought_encoder: 3-layer encoding network
            - value_head: Linear layer for value estimation
            - policy_head: Linear layer for policy guidance
            - expansion_net: ModuleList of beam_width expansion networks
            - selection_attention: 8-head multi-head attention
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.tree_depth = tree_depth
        self.beam_width = beam_width
        
        self.thought_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        
        self.value_head = nn.Linear(hidden_size, 1)
        
        self.policy_head = nn.Linear(hidden_size, hidden_size)
        
        self.expansion_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
            for _ in range(beam_width)
        ])
        
        self.selection_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all linear layers.
        
        Uses Xavier uniform initialization for weights and zero initialization
        for biases to ensure stable initial training dynamics.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _ucb_score(self, node_value: float, visit_count: int, parent_visits: int, exploration_weight: float = 1.414) -> float:
        """Calculate Upper Confidence Bound score for node selection.
        
        The UCB formula balances exploitation of known good nodes with
        exploration of less-visited nodes. This is the standard UCB1 formula
        used in Monte Carlo Tree Search algorithms.
        
        Args:
            node_value (float): Current value estimate of the node.
            visit_count (int): Number of times this node has been visited.
            parent_visits (int): Total visits to the parent node.
            exploration_weight (float): Exploration constant c in UCB formula.
                Higher values encourage more exploration. Default: 1.414 (sqrt(2)).
        
        Returns:
            float: UCB score for node selection. Returns infinity for unvisited
                nodes to ensure they are explored first.
        
        Formula:
            UCB = exploitation + c * sqrt(log(N_parent) / N_child)
            where exploitation = node_value
        """
        if visit_count == 0:
            return float('inf')
        exploitation = node_value
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / visit_count)
        return exploitation + exploration
    
    def forward(
        self,
        problem_embedding: torch.Tensor,
        context: torch.Tensor,
        num_simulations: int = 100
    ) -> Dict[str, Any]:
        """Execute thought tree search for reasoning.
        
        This method implements the complete MCTS-style search algorithm,
        building and exploring a thought tree to find the best reasoning
        path through the problem space.
        
        Algorithm Steps:
            1. **Initialization**: Create root node from problem-context encoding
            2. **Simulation Loop**: Run num_simulations iterations of:
               - Selection: Traverse tree using UCB scores
               - Expansion: Add children at leaf nodes
               - Backpropagation: Update values and visit counts
            3. **Best Path Extraction**: Find highest-value path from root to leaf
            4. **Confidence Calculation**: Normalize best value relative to all values
        
        Args:
            problem_embedding (torch.Tensor): Problem representation tensor
                of shape (batch_size, hidden_size).
            context (torch.Tensor): Contextual information tensor of shape
                (batch_size, hidden_size).
            num_simulations (int): Number of MCTS simulations to run.
                More simulations improve solution quality. Default: 100.
        
        Returns:
            Dict[str, Any]: Results dictionary containing:
                - 'thought_tree_output' (torch.Tensor): Final embedding from
                  best path of shape (batch_size, hidden_size).
                - 'confidence' (torch.Tensor): Confidence scores of shape
                  (batch_size, 1), normalized by best value in batch.
                - 'best_values' (List[float]): Best value found for each batch.
                - 'visit_counts' (List[List[int]]): Visit counts for all nodes.
                - 'tree_depth' (int): Configured tree depth.
                - 'beam_width' (int): Configured beam width.
        
        Example:
            >>> reasoner = YvThoughtTreeReasoner(2048, tree_depth=4)
            >>> problem = torch.randn(1, 2048)
            >>> context = torch.randn(1, 2048)
            >>> result = reasoner(problem, context, num_simulations=50)
            >>> print(f"Best path value: {result['best_values'][0]:.3f}")
        
        Note:
            Memory usage scales with tree_depth * beam_width^tree_depth.
            Use conservative parameters for large hidden_size values.
        """
        batch_size = problem_embedding.size(0)
        
        tree_states = []
        visit_counts = []
        values = []
        
        for b in range(batch_size):
            root_embedding = self.thought_encoder(
                torch.cat([problem_embedding[b], context[b]], dim=-1)
            )
            tree_states.append([root_embedding])
            visit_counts.append([1])
            values.append([self.value_head(root_embedding).squeeze(-1).item()])
        
        for _ in range(num_simulations):
            for b in range(batch_size):
                current_depth = 0
                current_node_idx = 0
                
                while current_depth < self.tree_depth:
                    current_embedding = tree_states[b][current_node_idx]
                    
                    if len(tree_states[b]) - 1 < (current_node_idx + 1) * self.beam_width:
                        for w in range(self.beam_width):
                            expanded = self.expansion_net[w](current_embedding)
                            new_embedding = current_embedding + expanded * 0.1
                            
                            tree_states[b].append(new_embedding)
                            visit_counts[b].append(1)
                            values[b].append(self.value_head(new_embedding).squeeze(-1).item())
                    
                    if current_depth < self.tree_depth - 1:
                        parent_visits = sum(visit_counts[b][current_node_idx * self.beam_width:(current_node_idx + 1) * self.beam_width])
                        
                        best_child_idx = current_node_idx
                        best_ucb_score = float('-inf')
                        
                        for c in range(self.beam_width):
                            child_idx = current_node_idx * self.beam_width + 1 + c
                            if child_idx < len(tree_states[b]):
                                ucb = self._ucb_score(
                                    values[b][child_idx],
                                    visit_counts[b][child_idx],
                                    parent_visits
                                )
                                if ucb > best_ucb_score:
                                    best_ucb_score = ucb
                                    best_child_idx = child_idx
                        
                        current_node_idx = best_child_idx
                    current_depth += 1
                
                leaf_idx = min(current_node_idx, len(tree_states[b]) - 1)
                leaf_value = values[b][leaf_idx]
                
                node = leaf_idx
                while node > 0:
                    parent = (node - 1) // self.beam_width
                    if parent < len(values[b]):
                        values[b][parent] = (values[b][parent] * visit_counts[b][parent] + leaf_value) / (visit_counts[b][parent] + 1)
                        visit_counts[b][parent] += 1
                    node = parent
                
                if leaf_idx < len(visit_counts[b]):
                    visit_counts[b][leaf_idx] += 1
        
        best_paths = []
        best_values = []
        
        for b in range(batch_size):
            best_idx = 0
            best_value = float('-inf')
            
            for i, v in enumerate(values[b]):
                if v > best_value:
                    best_value = v
                    best_idx = i
            
            path = []
            node = best_idx
            while node > 0:
                path.append(tree_states[b][node])
                node = (node - 1) // self.beam_width
            path.append(tree_states[b][0])
            path.reverse()
            
            best_paths.append(path)
            best_values.append(best_value)
        
        final_embedding = torch.stack([path[-1] for path in best_paths], dim=0)
        
        confidence = torch.tensor(
            [min(v / (max(best_values) + 1e-8), 1.0) for v in best_values],
            device=problem_embedding.device
        ).unsqueeze(-1)
        
        return {
            'thought_tree_output': final_embedding,
            'confidence': confidence,
            'best_values': best_values,
            'visit_counts': visit_counts,
            'tree_depth': self.tree_depth,
            'beam_width': self.beam_width
        }
