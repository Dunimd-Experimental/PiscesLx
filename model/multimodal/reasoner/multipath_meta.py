#!/usr/bin/env python3

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

"""Meta-learning utilities for Arctic multi-path reasoning systems.

The module implements :class:`ArcticMultiPathMetaLearner`, which stores
reasoning experiences, mines successful patterns, and adapts inference
parameters or priors to improve downstream multi-path reasoning engines.
"""

import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

class ArcticMultiPathMetaLearner:
    """Record, analyze, and adapt multi-path reasoning behavior."""

    def __init__(self, model: nn.Module, learning_rate: float = 1e-5) -> None:
        """Initialize the meta-learner with a base reasoning model.

        Args:
            model (nn.Module): Underlying model used during meta-learning routines.
            learning_rate (float): Learning rate governing adaptive updates. Defaults to ``1e-5``.
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
