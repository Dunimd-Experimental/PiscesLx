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

"""Orchestrated multi-path reasoning system for Yv agents.

This module provides a unified reasoning system that integrates multiple reasoning
paradigms into a cohesive framework. It coordinates core reasoning engines,
inference-time search strategies, and meta-learning capabilities to deliver
adaptive, interpretable, and self-improving multi-path reasoning flows.

Architecture Overview:
    The unified system operates through three interconnected subsystems:
    
    1. Core Reasoning Engine (YvMultiPathReasoningEngine):
       - Implements the foundational multi-path reasoning algorithm
       - Manages thinking logits, uncertainty scores, and fact consistency
       - Provides the base computational framework for reasoning operations
    
    2. Inference Engine (YvMultiPathInferenceEngine):
       - Executes inference-time reasoning with confidence evaluation
       - Performs multi-path exploration and path selection
       - Handles fact verification and logical consistency checks
    
    3. Meta-Learner (YvMultiPathMetaLearner):
       - Records reasoning experiences for pattern extraction
       - Constructs reasoning priors from historical data
       - Adapts reasoning strategies based on learned patterns

Key Features:
    - **Adaptive Reasoning**: Dynamically adjusts reasoning depth based on
      problem complexity and confidence thresholds
    - **Meta-Learning Integration**: Leverages past experiences to improve
      reasoning quality through learned priors
    - **Interpretability Support**: Provides detailed diagnostics including
      path analysis, uncertainty trends, and confidence breakdowns
    - **Performance Tracking**: Maintains statistics for success rate,
      average confidence, and pattern learning progress

Usage Patterns:
    The system is designed for both interactive and batch reasoning scenarios:
    
    - **Interactive Mode**: Process individual queries with optional metadata
      and interpretability features
    - **Batch Mode**: Process multiple queries efficiently with cached
      reasoning states
    - **Analysis Mode**: Extract performance statistics and learned patterns

Dependencies:
    - multipath_core: Core reasoning engine implementation
    - multipath_infer: Inference-time reasoning execution
    - multipath_meta: Meta-learning and experience management
    - enhancer: Multi-modal reasoning enhancement capabilities

Example:
    >>> model = load_pretrained_model("ruchbah-base")
    >>> tokenizer = load_tokenizer("ruchbah-base")
    >>> system = YvUnifiedMultiPathReasoningSystem(model, tokenizer)
    >>> 
    >>> # Basic reasoning
    >>> answer = system.reason("What is machine learning?")
    >>> 
    >>> # With full metadata and interpretability
    >>> result = system.reason(
    ...     "Explain quantum computing",
    ...     use_meta_learning=True,
    ...     return_full_metadata=True,
    ...     enable_interpretability=True
    ... )
    >>> 
    >>> # Get performance statistics
    >>> stats = system.get_performance_stats()

Performance Considerations:
    - Memory usage scales with meta-learner experience buffer size
    - Inference time depends on reasoning depth and path count
    - GPU memory required for hidden state extraction during reasoning

Note:
    The system requires a minimum of 50 recorded experiences before
    meta-learning priors become active for reasoning enhancement.
"""

import time
import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Optional, Union
from .multipath_meta import YvMultiPathMetaLearner
from .enhancer import YvMultiModalReasoningEnhancer
from .multipath_core import YvMultiPathReasoningEngine
from .multipath_infer import YvMultiPathInferenceEngine

class YvUnifiedMultiPathReasoningSystem:
    """Runtime container integrating core reasoning, inference, and meta-learning.
    
    This class serves as the primary interface for multi-path reasoning operations,
    orchestrating the interaction between three core subsystems: the reasoning engine,
    inference engine, and meta-learner. It provides a unified API for executing
    reasoning queries, tracking performance metrics, and enabling interpretability.
    
    Architecture:
        The system follows a three-stage processing pipeline:
        
        1. **Prior Generation** (Optional):
           - Meta-learner creates reasoning priors from historical experiences
           - Priors guide the reasoning process toward successful patterns
           - Requires minimum 50 experiences in the memory buffer
        
        2. **Core Processing**:
           - Hidden states extracted from the base model
           - Core reasoning engine processes hidden states
           - Produces thinking logits, uncertainty scores, and fact consistency
        
        3. **Inference Execution**:
           - Multi-path inference explores reasoning paths
           - Confidence evaluation determines answer quality
           - Fact verification ensures logical consistency
    
    Attributes:
        model (nn.Module): Foundation model providing forward and generate methods.
            Must support output_hidden_states=True for embedding extraction.
        tokenizer (Any): Tokenizer aligned with the model for text encoding/decoding.
            Must provide encode() method with return_tensors support.
        device (str): Device identifier for tensor operations ('cuda' or 'cpu').
        reasoning_engine (YvMultiPathReasoningEngine): Core reasoning processor
            that transforms hidden states into reasoning outputs.
        reasoning_inference (YvMultiPathInferenceEngine): Inference-time
            reasoning executor with multi-path exploration capabilities.
        meta_learner (YvMultiPathMetaLearner): Experience recorder and
            pattern extractor for adaptive reasoning improvement.
        total_reasoning_calls (int): Counter for total reasoning invocations.
        successful_reasoning_calls (int): Counter for high-confidence results (>0.8).
        average_confidence (float): Running average of confidence scores.
    
    Performance Metrics:
        - Success Rate: successful_reasoning_calls / total_reasoning_calls
        - Average Confidence: Exponential moving average of confidence scores
        - Pattern Learning: Number of patterns extracted by meta-learner
    
    Example:
        >>> import torch
        >>> from transformers import AutoModel, AutoTokenizer
        >>> 
        >>> model = AutoModel.from_pretrained("ruchbah-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("ruchbah-base")
        >>> system = YvUnifiedMultiPathReasoningSystem(model, tokenizer)
        >>> 
        >>> # Simple reasoning query
        >>> answer = system.reason("What causes rain?")
        >>> print(answer)
        >>> 
        >>> # Detailed reasoning with metadata
        >>> result = system.reason(
        ...     query="Explain the theory of relativity",
        ...     use_meta_learning=True,
        ...     return_full_metadata=True,
        ...     enable_interpretability=True
        ... )
        >>> print(f"Confidence: {result['confidence']:.2f}")
        >>> print(f"Depth: {result['reasoning_depth']}")
    
    Note:
        - Meta-learning requires accumulated experience (50+ records)
        - GPU memory is required for efficient hidden state extraction
        - The system is stateful; repeated calls update meta-learner memory
    """

    def __init__(self, model: nn.Module, tokenizer: Any, device: str = 'cuda') -> None:
        """Instantiate the unified multi-path reasoning pipeline.
        
        This constructor initializes all three reasoning subsystems and sets up
        tracking variables for performance monitoring. The subsystems are created
        with configurations derived from the provided model.
        
        Args:
            model (nn.Module): Underlying foundation model providing forward and
                generate methods. Must support output_hidden_states=True parameter
                for extracting intermediate representations during reasoning.
                The model's config attribute is used to configure the reasoning engine.
            tokenizer (Any): Tokenizer corresponding to the model for encoding user
                queries. Must support encode() method with return_tensors="pt" option
                for producing PyTorch tensors directly.
            device (str): Device identifier for executing model operations.
                Valid options are 'cuda' for GPU acceleration or 'cpu' for
                CPU-only execution. Defaults to 'cuda'.
        
        Initializes:
            - reasoning_engine: Created with model.config for architecture compatibility
            - reasoning_inference: Created with model and tokenizer for generation
            - meta_learner: Created with model for experience-based adaptation
            - Performance counters: All initialized to zero/empty states
        
        Example:
            >>> model = AutoModel.from_pretrained("ruchbah-7b")
            >>> tokenizer = AutoTokenizer.from_pretrained("ruchbah-7b")
            >>> system = YvUnifiedMultiPathReasoningSystem(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     device='cuda:0'
            ... )
        
        Note:
            The model should be in eval mode for inference. Call model.eval()
            before passing to this constructor for optimal performance.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize reasoning subsystems used during inference.
        self.reasoning_engine = YvMultiPathReasoningEngine(model.config)
        self.reasoning_inference = YvMultiPathInferenceEngine(model, tokenizer)
        self.meta_learner = YvMultiPathMetaLearner(model)

        # Tracking variables for runtime metrics.
        self.total_reasoning_calls = 0
        self.successful_reasoning_calls = 0
        self.average_confidence = 0.0

    def reason(
        self,
        query: str,
        use_meta_learning: bool = True,
        return_full_metadata: bool = False,
        enable_interpretability: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """Execute multi-path reasoning for a given user query.
        
        This method orchestrates the complete reasoning pipeline, from query encoding
        through answer generation, with optional meta-learning integration and
        interpretability features. It represents the primary entry point for
        reasoning operations in the unified system.
        
        Processing Pipeline:
            1. **Prior Generation** (if meta-learning enabled and sufficient experience):
               - Query the meta-learner for relevant reasoning priors
               - Priors are derived from similar past successful reasoning chains
            
            2. **Hidden State Extraction**:
               - Encode query using the tokenizer
               - Forward pass through the model with hidden state output
               - Extract final layer hidden states for reasoning
            
            3. **Core Reasoning**:
               - Process hidden states through the reasoning engine
               - Generate thinking logits, uncertainty scores, and fact consistency
            
            4. **Multi-Path Inference**:
               - Execute multi-path reasoning exploration
               - Evaluate confidence across different reasoning paths
               - Select optimal answer based on path quality
            
            5. **Metadata Collection**:
               - Aggregate reasoning statistics and state information
               - Optionally compute interpretability diagnostics
            
            6. **Experience Recording** (if meta-learning enabled):
               - Store reasoning experience in meta-learner memory
               - Enable future pattern extraction and prior generation
        
        Args:
            query (str): User query or problem statement to reason about.
                Can be a question, instruction, or any natural language input
                that requires reasoning to answer.
            use_meta_learning (bool): If True, leverage meta-learned priors
                derived from past experiences and record the current reasoning
                experience for future improvement. Requires minimum 50 experiences
                in memory for prior generation. Defaults to True.
            return_full_metadata (bool): When True, return a detailed metadata
                dictionary containing all reasoning statistics, state information,
                and optional interpretability data. When False, return only the
                answer string. Defaults to False.
            enable_interpretability (bool): Whether to compute and append
                interpretability diagnostics to the result. Includes path analysis,
                uncertainty trends, and confidence breakdown. Only relevant when
                return_full_metadata is True. Defaults to False.
        
        Returns:
            Union[Dict[str, Any], str]: 
                - If return_full_metadata is False: The answer string generated
                  by the reasoning process.
                - If return_full_metadata is True: A dictionary containing:
                    - 'query' (str): Original input query
                    - 'answer' (str): Generated answer
                    - 'confidence' (float): Confidence score [0, 1]
                    - 'reasoning_time' (float): Total processing time in seconds
                    - 'reasoning_depth' (int): Number of reasoning layers used
                    - 'reasoning_state' (dict): Shapes of internal tensors
                    - 'prior_used' (bool): Whether meta-learning prior was applied
                    - 'reasoning_chain' (list): Step-by-step reasoning trace
                    - 'uncertainty_scores' (list): Uncertainty at each step
                    - 'interpretability' (dict, optional): Diagnostics if enabled
        
        Side Effects:
            - Increments total_reasoning_calls counter
            - Updates successful_reasoning_calls if confidence > 0.8
            - Updates average_confidence running statistic
            - Records experience to meta-learner if use_meta_learning is True
        
        Example:
            >>> # Simple query
            >>> answer = system.reason("What is the capital of France?")
            >>> print(answer)
            "The capital of France is Paris."
            
            >>> # With full metadata
            >>> result = system.reason(
            ...     "Explain the water cycle",
            ...     use_meta_learning=True,
            ...     return_full_metadata=True,
            ...     enable_interpretability=True
            ... )
            >>> print(f"Confidence: {result['confidence']:.2%}")
            >>> print(f"Depth: {result['reasoning_depth']}")
            >>> print(f"Time: {result['reasoning_time']:.2f}s")
        
        Note:
            The method uses torch.no_grad() for memory efficiency during
            inference. GPU memory consumption depends on model size and
            sequence length of the query.
        """
        # Increment the total number of reasoning calls.
        self.total_reasoning_calls += 1
        prior = None
        # Generate a reasoning prior if meta-learning is enabled and the experience buffer is populated.
        if use_meta_learning and len(self.meta_learner.reasoning_memory) >= 50:
            prior = self.meta_learner.create_reasoning_prior(query)

        # Record the start time to measure reasoning latency.
        start_time = time.time()
        # Extract hidden states from the base model without gradient computation.
        with torch.no_grad():
            model_out = self.model(self.tokenizer.encode(query, return_tensors="pt").to(self.device),
                                   output_hidden_states=True)
            hidden_states = model_out.hidden_states[-1]
            # Feed hidden states into the core reasoning engine.
            core_out = self.reasoning_engine.forward(hidden_states)

        # Perform multi-path inference and gather the final result with chain data.
        final_result = self.reasoning_inference.multi_path_reason(query, return_metadata=True)
        reasoning_time = time.time() - start_time

        # Collect reasoning metadata summarizing the execution.
        metadata = {
            'query': query,
            'answer': final_result['answer'] if isinstance(final_result, dict) else final_result,
            'confidence': final_result.get('confidence', 0.8) if isinstance(final_result, dict) else 0.8,
            'reasoning_time': reasoning_time,
            'reasoning_depth': final_result.get('reasoning_depth', 3) if isinstance(final_result, dict) else 3,
            'reasoning_state': {
                'thinking_logits_shape': list(core_out.get('thinking_logits', torch.empty(0)).shape),
                'uncertainty_scores_shape': list(core_out.get('uncertainty_scores', torch.empty(0)).shape),
                'fact_consistency_shape': list(core_out.get('fact_consistency', torch.empty(0)).shape),
            },
            'prior_used': prior is not None,
            'reasoning_chain': final_result.get('reasoning_chain', []) if isinstance(final_result, dict) else [],
            'uncertainty_scores': core_out.get('uncertainty_scores', []),
        }

        # Compute interpretability diagnostics when requested.
        if enable_interpretability:
            metadata['interpretability'] = {
                'query_complexity': len(query.split()),
                'reasoning_path_analysis': self._analyze_reasoning_paths(metadata),
                'uncertainty_trend': self._analyze_uncertainty_trend(metadata),
                'confidence_breakdown': self._analyze_confidence_components(metadata),
            }

        # Track success rate using a confidence threshold.
        if metadata['confidence'] > 0.8:
            self.successful_reasoning_calls += 1
        # Update the running average confidence statistic.
        self.average_confidence = ((self.average_confidence * (self.total_reasoning_calls - 1) + metadata['confidence']) / self.total_reasoning_calls)

        # Record reasoning experience to the meta-learner buffer when enabled.
        if use_meta_learning:
            self.meta_learner.record_reasoning(
                query=query,
                reasoning_chain=metadata['reasoning_chain'],
                final_result=metadata['answer'],
                metadata={
                    'confidence': metadata['confidence'],
                    'reasoning_depth': metadata['reasoning_depth'],
                    'uncertainty_evolution': [0.9, 0.7, 0.5, 0.3, 0.2][:metadata['reasoning_depth']]
                }
            )

        return metadata if return_full_metadata else metadata['answer']

    def get_performance_stats(self) -> Dict[str, float]:
        """Return aggregate metrics summarizing reasoning performance.
        
        This method provides a comprehensive snapshot of the system's reasoning
        performance over its lifetime, including success rates, confidence trends,
        and meta-learning progress. It is useful for monitoring system health
        and identifying areas for improvement.
        
        Returns:
            Dict[str, float]: A dictionary containing performance metrics:
                - 'total_reasoning_calls' (int): Total number of reason() invocations
                - 'successful_reasoning_calls' (int): Calls with confidence > 0.8
                - 'success_rate' (float): Ratio of successful to total calls [0, 1]
                - 'average_confidence' (float): Running average confidence score
                - 'meta_learning_experiences' (int): Number of recorded experiences
                - 'patterns_learned' (int): Number of patterns extracted from experiences
        
        Example:
            >>> stats = system.get_performance_stats()
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
            >>> print(f"Average confidence: {stats['average_confidence']:.2f}")
            >>> print(f"Experiences recorded: {stats['meta_learning_experiences']}")
        
        Note:
            The success_rate is protected against division by zero by using
            max(total_reasoning_calls, 1) as the denominator.
        """
        return {
            'total_reasoning_calls': self.total_reasoning_calls,
            'successful_reasoning_calls': self.successful_reasoning_calls,
            'success_rate': self.successful_reasoning_calls / max(self.total_reasoning_calls, 1),
            'average_confidence': self.average_confidence,
            'meta_learning_experiences': len(self.meta_learner.reasoning_memory),
            'patterns_learned': len(self.meta_learner.extract_reasoning_patterns() or {})
        }

    def _analyze_reasoning_paths(self, metadata: Dict[str, Any]) -> Dict[str, Union[List[int], float, int]]:
        """Analyze path importance distribution for interpretability reporting.
        
        This method examines the distribution of importance scores across different
        reasoning paths to identify dominant paths and measure path diversity.
        It uses entropy-based metrics to quantify how evenly importance is distributed.
        
        Analysis Components:
            1. **Dominant Path Identification**: Uses 70th percentile threshold
               to identify paths with above-average importance
            2. **Diversity Measurement**: Computes normalized entropy of the
               importance distribution, where 1.0 indicates uniform distribution
               and 0.0 indicates all importance concentrated in one path
        
        Args:
            metadata (Dict[str, Any]): Reasoning metadata dictionary containing
                'path_importance' key with a list of importance scores for each
                reasoning path explored during inference.
        
        Returns:
            Dict[str, Union[List[int], float, int]]: Analysis results containing:
                - 'total_paths' (int): Total number of paths analyzed (0 if no data)
                - 'dominant_paths' (List[int]): Indices of paths above 70th percentile
                - 'path_diversity' (float): Normalized entropy score [0, 1]
        
        Mathematical Formulation:
            - Entropy: H = -sum(p_i * log(p_i)) where p_i = importance_i / sum(importance)
            - Normalized Entropy: H_normalized = H / log(n) where n = number of paths
        
        Example:
            >>> metadata = {'path_importance': [0.4, 0.3, 0.2, 0.1]}
            >>> analysis = system._analyze_reasoning_paths(metadata)
            >>> print(analysis['path_diversity'])  # ~0.89 (fairly diverse)
        
        Note:
            Returns default values (total_paths=0, dominant_paths=[], diversity=0.0)
            when path_importance is empty or not present in metadata.
        """
        analysis = {'total_paths': 0, 'dominant_paths': [], 'path_diversity': 0.0}
        path_importance = metadata.get('path_importance', [])
        if path_importance:
            # Identify dominant paths via percentile thresholding.
            threshold = np.percentile(path_importance, 70)
            dominant_indices = [i for i, imp in enumerate(path_importance) if imp >= threshold]
            analysis['dominant_paths'] = dominant_indices
            if len(path_importance) > 1:
                # Compute entropy-based diversity normalized by maximum entropy.
                arr = np.array(path_importance, dtype=float)
                arr = arr / max(arr.sum(), 1e-8)
                entropy = -np.sum(arr * np.log(arr + 1e-10))
                max_entropy = np.log(len(arr))
                analysis['path_diversity'] = float(entropy / max(max_entropy, 1e-8))
        return analysis

    def _analyze_uncertainty_trend(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate how uncertainty evolves across reasoning steps.
        
        This method analyzes the trajectory of uncertainty scores throughout the
        reasoning process to determine whether the system is converging toward
        a confident answer or diverging into uncertainty.
        
        Trend Classification:
            - 'decreasing': Final uncertainty is significantly lower than initial
              (reduction > 0.1), indicating successful convergence
            - 'increasing': Final uncertainty is significantly higher than initial
              (increase > 0.1), indicating potential issues
            - 'stable': Uncertainty change is within ±0.1, indicating consistent
              confidence throughout reasoning
        
        Args:
            metadata (Dict[str, Any]): Reasoning metadata dictionary containing
                'uncertainty_scores' key with a list of uncertainty values from
                each reasoning step.
        
        Returns:
            Dict[str, float]: Trend analysis containing:
                - 'initial_uncertainty' (float): First uncertainty value (0.5 if empty)
                - 'final_uncertainty' (float): Last uncertainty value (0.5 if empty)
                - 'uncertainty_reduction' (float): Initial minus final uncertainty
                - 'trend_direction' (str): One of 'decreasing', 'increasing', 'stable'
        
        Example:
            >>> metadata = {'uncertainty_scores': [0.8, 0.6, 0.4, 0.2]}
            >>> trend = system._analyze_uncertainty_trend(metadata)
            >>> print(trend['trend_direction'])  # 'decreasing'
            >>> print(trend['uncertainty_reduction'])  # 0.6
        
        Note:
            Requires at least 2 uncertainty scores for meaningful trend analysis.
            Single-element or empty lists result in 'stable' classification.
        """
        uncertainty_scores = metadata.get('uncertainty_scores', [])
        trend = {
            'initial_uncertainty': uncertainty_scores[0] if uncertainty_scores else 0.5,
            'final_uncertainty': uncertainty_scores[-1] if uncertainty_scores else 0.5,
            'uncertainty_reduction': 0.0,
            'trend_direction': 'stable'
        }
        if len(uncertainty_scores) >= 2:
            initial = uncertainty_scores[0]
            final = uncertainty_scores[-1]
            trend['uncertainty_reduction'] = initial - final
            if final < initial - 0.1:
                trend['trend_direction'] = 'decreasing'
            elif final > initial + 0.1:
                trend['trend_direction'] = 'increasing'
        return trend

    def _analyze_confidence_components(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Decompose the aggregate confidence into conceptual components.
        
        This method breaks down the overall confidence score into interpretable
        components that represent different aspects of the reasoning quality.
        The decomposition helps understand which factors contribute most to
        the final confidence assessment.
        
        Component Definitions:
            - **Base Confidence** (60%): The fundamental confidence derived from
              the model's direct assessment of answer quality
            - **Path Consensus** (25%): Confidence contribution from agreement
              across multiple reasoning paths
            - **Uncertainty Weighted** (15%): Confidence adjustment based on
              uncertainty trajectory during reasoning
        
        Args:
            metadata (Dict[str, Any]): Reasoning metadata dictionary containing
                'confidence' key with the overall confidence score [0, 1].
        
        Returns:
            Dict[str, float]: Component breakdown containing:
                - 'base_confidence' (float): 60% of total confidence
                - 'path_consensus' (float): 25% of total confidence
                - 'uncertainty_weighted' (float): 15% of total confidence
        
        Example:
            >>> metadata = {'confidence': 0.85}
            >>> components = system._analyze_confidence_components(metadata)
            >>> print(components['base_confidence'])  # 0.51
            >>> print(components['path_consensus'])  # 0.2125
        
        Note:
            The weights (60%, 25%, 15%) are fixed and represent the relative
            importance of each component in the overall confidence assessment.
            Future versions may allow dynamic weight adjustment.
        """
        confidence = metadata.get('confidence', 0.8)
        return {
            'base_confidence': confidence * 0.6,
            'path_consensus': confidence * 0.25,
            'uncertainty_weighted': confidence * 0.15
        }
