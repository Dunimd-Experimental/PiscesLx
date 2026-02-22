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

"""Meta-learning utilities for Yv multi-path reasoning systems.

This module provides the YvMultiPathMetaLearner class which implements
experience recording, pattern extraction, and parameter adaptation for
improving multi-path reasoning performance over time.

Architecture:
    1. Experience Recording:
       - Query and reasoning chain storage
       - Result and metadata persistence
       - Query embedding computation
    
    2. Pattern Extraction:
       - Depth pattern analysis
       - Confidence pattern analysis
       - Strategy pattern identification
    
    3. Parameter Adaptation:
       - Confidence threshold adjustment
       - Max depth optimization
       - Strategy preference learning
    
    4. Prior Construction:
       - Similar experience retrieval
       - Expected depth estimation
       - Uncertainty pattern extraction

Key Features:
    - Persistent reasoning memory (up to 10,000 experiences)
    - Pattern mining from successful episodes
    - Adaptive parameter tuning
    - Reasoning prior generation
    - Strategy identification and ranking

Performance Characteristics:
    - Recording: O(1) amortized with memory limit
    - Pattern Extraction: O(N) where N = successful experiences
    - Prior Construction: O(M * K) where M = memory, K = top-k

Usage Example:
    >>> from model.reasoning.reasoner import YvMultiPathMetaLearner
    >>> 
    >>> # Initialize meta-learner
    >>> meta_learner = YvMultiPathMetaLearner(
    ...     model=ruchbah_model,
    ...     learning_rate=1e-5
    >>> )
    >>> 
    >>> # Record reasoning experience
    >>> meta_learner.record_reasoning(
    ...     query="What is AI?",
    ...     reasoning_chain=chain,
    ...     final_result=result,
    ...     metadata={"confidence": 0.95, "reasoning_depth": 3}
    >>> )
    >>> 
    >>> # Extract patterns and adapt parameters
    >>> patterns = meta_learner.extract_reasoning_patterns()
    >>> new_params = meta_learner.adapt_reasoning_parameters(patterns)

Dependencies:
    - time: Timestamp generation for experiences
    - torch: Tensor operations and neural network modules
    - numpy: Numerical computations
    - torch.nn: Neural network components
    - torch.nn.functional: Activation functions
    - collections: Counter and defaultdict utilities
    - typing: Type hints for documentation

Note:
    The meta-learner uses SentenceTransformer for query embeddings when
    available, with hash-based fallback for deterministic embeddings.
"""

import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


class YvMultiPathMetaLearner:
    """Record, analyze, and adapt multi-path reasoning behavior.
    
    This class implements a meta-learning system that records reasoning
    experiences, extracts successful patterns, and adapts reasoning
    parameters for improved performance.
    
    Architecture:
        1. Memory Management:
           - Experience buffer with 10,000 limit
           - Automatic pruning to 5,000 when full
           - Timestamp-based tracking
        
        2. Pattern Extraction:
           - Filters high-confidence experiences (> 0.9)
           - Analyzes depth, confidence, and strategy patterns
           - Computes statistics and distributions
        
        3. Strategy Identification:
           - Analogical reasoning detection
           - Decomposition strategy detection
           - Assumption testing detection
           - Direct reasoning fallback
        
        4. Prior Construction:
           - Cosine similarity for experience retrieval
           - Top-10 similar experiences aggregation
           - Uncertainty pattern averaging
    
    Strategy Types:
        - analogical: Uses analogies for reasoning
        - decomposition: Breaks down complex problems
        - assumption_testing: Tests assumptions systematically
        - direct: Direct reasoning without special strategies
    
    Attributes:
        model (nn.Module): Underlying model for meta-learning.
        learning_rate (float): Learning rate for adaptive updates.
        reasoning_memory (list): Buffer of recorded experiences.
        pattern_extractor (nn.Sequential): Neural network for pattern encoding.
    
    Example:
        >>> learner = YvMultiPathMetaLearner(model)
        >>> learner.record_reasoning(query, chain, result, metadata)
        >>> patterns = learner.extract_reasoning_patterns()
        >>> prior = learner.create_reasoning_prior("What is AI?")
    
    Note:
        Minimum 100 experiences required for pattern extraction.
        Minimum 20 high-confidence experiences for pattern mining.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-5) -> None:
        """Initialize the meta-learner with a base reasoning model.
        
        Args:
            model (nn.Module): Underlying model used during meta-learning routines.
                Must be compatible with the reasoning system.
            learning_rate (float): Learning rate governing adaptive updates.
                Default: 1e-5.
        
        Note:
            The reasoning_memory is initialized as an empty list.
            The pattern_extractor is a 3-layer MLP (768 -> 512 -> 256 -> 128).
        """
        self.model = model
        self.learning_rate = learning_rate
        self.reasoning_memory = []
        self.pattern_extractor = self._build_pattern_extractor()

    def _build_pattern_extractor(self) -> nn.Sequential:
        """Construct the neural network used to encode reasoning patterns."""
        return nn.Sequential(
            # First linear layer reduces feature dimensionality.
            nn.Linear(768, 512),
            # Non-linear activation for expressive capacity.
            nn.ReLU(),
            # Dropout regularization to mitigate overfitting.
            nn.Dropout(0.1),
            # Compress hidden representation further.
            nn.Linear(512, 256),
            # Additional non-linearity for deeper feature extraction.
            nn.ReLU(),
            # Final projection into the pattern embedding space.
            nn.Linear(256, 128)
        )

    def record_reasoning(
        self,
        query: str,
        reasoning_chain: List[Any],
        final_result: Any,
        metadata: Dict[str, Any],
    ) -> None:
        """Persist a reasoning episode into the experience buffer.

        Args:
            query (str): Input query processed by the reasoning system.
            reasoning_chain (List[Any]): Sequence describing intermediate reasoning steps.
            final_result (Any): Output produced by the reasoning engine.
            metadata (Dict[str, Any]): Supplemental information such as confidence metrics.
        """
        experience = {
            'query': query,
            'reasoning_chain': reasoning_chain,
            'final_result': final_result,
            'metadata': metadata,
            'timestamp': time.time(),
            'query_embedding': self._embed_query(query)
        }
        self.reasoning_memory.append(experience)
        if len(self.reasoning_memory) > 10000:
            self.reasoning_memory = self.reasoning_memory[-5000:]

    def _embed_query(self, query: str) -> torch.Tensor:
        """Encode an input query to a fixed-length embedding.

        Args:
            query (str): Free-form query string requiring representation.

        Returns:
            torch.Tensor: Query embedding derived from SentenceTransformer or a hash-based fallback.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(query, convert_to_tensor=True)
            return embedding
        except Exception:
            import hashlib
            # Generate a deterministic hash for the query content.
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            # Seed torch RNG to obtain repeatable pseudo-random embeddings.
            torch.manual_seed(query_hash % 2147483647)
            # Create a base embedding following a normal distribution.
            base_embedding = torch.randn(768)
            # Length factor reflects query verbosity.
            length_factor = min(len(query) / 100, 1.0)
            # Complexity captures diversity of unique tokens.
            complexity_factor = len(set(query.split())) / max(len(query.split()), 1)
            # Combine base embedding with structural scalars for stability.
            structured_embedding = base_embedding * (0.5 + 0.5 * length_factor * complexity_factor)
            return structured_embedding

    def extract_reasoning_patterns(self) -> Optional[Dict[str, Any]]:
        """Mine recurring reasoning patterns from high-confidence episodes.

        Returns:
            Optional[Dict[str, Any]]: Extracted pattern summary or ``None`` when support is insufficient.
        """
        if len(self.reasoning_memory) < 100:
            return None
        successful = [exp for exp in self.reasoning_memory if exp['metadata']['confidence'] > 0.9]
        if len(successful) < 20:
            return None

        patterns = {
            'optimal_depth_distribution': self._analyze_depth_patterns(successful),
            'confidence_indicators': self._analyze_confidence_patterns(successful),
            'reasoning_strategies': self._analyze_strategy_patterns(successful)
        }
        return patterns

    def _analyze_depth_patterns(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize typical reasoning depth statistics among successful episodes."""
        depths = [exp['metadata']['reasoning_depth'] for exp in experiences]
        return {
            'mean_depth': float(np.mean(depths)),
            'depth_variance': float(np.var(depths)),
            'optimal_depth_range': (float(np.percentile(depths, 25)), float(np.percentile(depths, 75)))
        }

    def _analyze_confidence_patterns(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute confidence-related indicators from successful episodes."""
        confidences = [exp['metadata']['confidence'] for exp in experiences]
        uncertainty_evol = [exp['metadata']['uncertainty_evolution'] for exp in experiences]
        return {
            'mean_confidence': float(np.mean(confidences)),
            'confidence_growth_rate': float(self._calculate_confidence_growth(uncertainty_evol)),
            'stability_threshold': float(np.percentile(confidences, 10))
        }

    def _analyze_strategy_patterns(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify prominent reasoning strategies and associated success metrics."""
        strategies = []
        for exp in experiences:
            chain = exp['reasoning_chain']
            strategy = self._identify_strategy(chain)
            strategies.append(strategy)

        return {
            'most_common_strategies': Counter(strategies).most_common(5),
            'strategy_success_rates': self._calculate_strategy_success(strategies, experiences)
        }

    def _calculate_confidence_growth(self, uncertainty_evolutions: List[List[float]]) -> float:
        """Average the reduction in uncertainty over time for successful runs."""
        growth_rates = []
        for evolution in uncertainty_evolutions:
            if len(evolution) > 1:
                reduction = evolution[0] - evolution[-1]
                steps = len(evolution)
                growth_rates.append(reduction / steps)
        return np.mean(growth_rates) if growth_rates else 0.0

    def _identify_strategy(self, reasoning_chain: List[Any]) -> str:
        """Infer the dominant reasoning strategy from descriptive steps."""
        strategies = []
        for step in reasoning_chain:
            s = str(step).lower()
            if 'analog' in s:
                strategies.append('analogical')

            elif 'break down' in s or 'decompose' in s:
                strategies.append('decomposition')
            elif 'assume' in s:
                strategies.append('assumption_testing')
            else:
                strategies.append('direct')
        return max(set(strategies), key=strategies.count)

    def _calculate_strategy_success(self, strategies: List[str], experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute average confidence per reasoning strategy."""
        strategy_success = defaultdict(list)
        for strategy, exp in zip(strategies, experiences):
            strategy_success[strategy].append(exp['metadata']['confidence'])
        return {strategy: float(np.mean(confidences)) for strategy, confidences in strategy_success.items()}

    def adapt_reasoning_parameters(self, patterns: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Derive updated reasoning thresholds from extracted patterns."""
        if not patterns:
            return None
        new_threshold = max(0.7, patterns['confidence_indicators']['stability_threshold'])
        optimal_depth = patterns['optimal_depth_distribution']['mean_depth']
        new_max_depth = max(3, min(8, int(optimal_depth * 1.2)))
        return {
            'confidence_threshold': new_threshold,

            'max_depth': new_max_depth,
            'preferred_strategies': [
                strategy for strategy, _ in patterns['reasoning_strategies']['most_common_strategies'][:3]
            ]
        }

    def create_reasoning_prior(self, query: str) -> Optional[Dict[str, Any]]:
        """Construct a reasoning prior leveraging similar historical episodes."""
        if len(self.reasoning_memory) < 50:
            return None
        query_embedding = self._embed_query(query)
        similarities = []
        for exp in self.reasoning_memory[-500:]:

            sim = F.cosine_similarity(query_embedding.unsqueeze(0), exp['query_embedding'].unsqueeze(0))
            similarities.append((sim, exp))
        top_experiences = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]
        if not top_experiences:
            return None
        prior = {
            'expected_depth': float(np.mean([exp[1]['metadata']['reasoning_depth'] for exp in top_experiences])),
            'expected_confidence': float(np.mean([exp[1]['metadata']['confidence'] for exp in top_experiences])),
            'recommended_strategies': [self._identify_strategy(exp[1]['reasoning_chain']) for exp in top_experiences],
            'uncertainty_pattern': self._extract_uncertainty_pattern(top_experiences)
        }
        return prior

    def _extract_uncertainty_pattern(self, experiences: List[Any]) -> Optional[Dict[str, List[float]]]:
        """Average uncertainty trajectories across similar experiences."""
        uncertainties = []
        for _, exp in experiences:
            if 'uncertainty_evolution' in exp['metadata']:
                uncertainties.append(exp['metadata']['uncertainty_evolution'])
        if not uncertainties:

            return None
        max_len = max(len(u) for u in uncertainties)
        padded_uncertainties = [u + [u[-1]] * (max_len - len(u)) if u else [0.5] * max_len for u in uncertainties]
        return {
            'typical_pattern': np.mean(padded_uncertainties, axis=0).tolist(),
            'variance': np.var(padded_uncertainties, axis=0).tolist()
        }
