#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

class ArcticMultiPathMetaLearner:
    """A meta-learner class for multi-path reasoning. It records reasoning experiences, 
    extracts patterns from them, and adapts reasoning parameters accordingly.
    """
    def __init__(self, model, learning_rate=1e-5):
        """Initialize an instance of ArcticMultiPathMetaLearner.

        Args:
            model: The base model used for reasoning.
            learning_rate (float, optional): The learning rate for the learner. Defaults to 1e-5.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.reasoning_memory = []
        self.pattern_extractor = self._build_pattern_extractor()

    def _build_pattern_extractor(self):
        """Construct a neural network for pattern extraction.

        Returns:
            nn.Sequential: A sequential neural network model designed for pattern extraction.
        """
        return nn.Sequential(
            # First linear layer: reduce feature dimension from 768 to 512
            nn.Linear(768, 512),
            # Apply ReLU activation function to introduce non-linearity
            nn.ReLU(),
            # Apply dropout with a rate of 0.1 to prevent overfitting
            nn.Dropout(0.1),
            # Second linear layer: reduce feature dimension from 512 to 256
            nn.Linear(512, 256),
            # Apply ReLU activation function to introduce non-linearity
            nn.ReLU(),
            # Third linear layer: reduce feature dimension from 256 to 128
            nn.Linear(256, 128)
        )

    def record_reasoning(self, query, reasoning_chain, final_result, metadata):
        """Record a reasoning experience into the memory.
        If the number of experiences exceeds 10000, keep only the last 5000 experiences.

        Args:
            query (str): The input query for reasoning.
            reasoning_chain (list): A list of reasoning steps.
            final_result: The final result of the reasoning process.
            metadata (dict): Additional metadata about the reasoning process.
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

    def _embed_query(self, query):
        """Embed a query into a vector representation.
        Attempt to use SentenceTransformer first. If it fails, use a hash-based method to generate the embedding.

        Args:
            query (str): The input query to be embedded.

        Returns:
            torch.Tensor: A tensor representing the embedded query.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(query, convert_to_tensor=True)
            return embedding
        except Exception:
            import hashlib
            # Generate a hash value for the query
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            # Set the random seed based on the hash value
            torch.manual_seed(query_hash % 2147483647)
            # Generate a random embedding vector
            base_embedding = torch.randn(768)
            # Calculate the length factor of the query
            length_factor = min(len(query) / 100, 1.0)
            # Calculate the complexity factor of the query
            complexity_factor = len(set(query.split())) / max(len(query.split()), 1)
            # Generate the structured embedding vector
            structured_embedding = base_embedding * (0.5 + 0.5 * length_factor * complexity_factor)
            return structured_embedding

    def extract_reasoning_patterns(self):
        """Extract reasoning patterns from successful experiences.
        Only perform extraction if there are at least 100 experiences and 20 successful experiences.

        Returns:
            dict: A dictionary containing reasoning patterns, or None if the conditions are not met.
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

    def _analyze_depth_patterns(self, experiences):
        """Analyze the depth patterns of successful reasoning experiences.

        Args:
            experiences (list): A list of successful reasoning experiences.

        Returns:
            dict: A dictionary containing the mean depth, depth variance, and optimal depth range.
        """
        depths = [exp['metadata']['reasoning_depth'] for exp in experiences]
        return {
            'mean_depth': float(np.mean(depths)),
            'depth_variance': float(np.var(depths)),
            'optimal_depth_range': (float(np.percentile(depths, 25)), float(np.percentile(depths, 75)))
        }

    def _analyze_confidence_patterns(self, experiences):
        """Analyze the confidence patterns of successful reasoning experiences.

        Args:
            experiences (list): A list of successful reasoning experiences.

        Returns:
            dict: A dictionary containing the mean confidence, confidence growth rate, and stability threshold.
        """
        confidences = [exp['metadata']['confidence'] for exp in experiences]
        uncertainty_evol = [exp['metadata']['uncertainty_evolution'] for exp in experiences]
        return {
            'mean_confidence': float(np.mean(confidences)),
            'confidence_growth_rate': float(self._calculate_confidence_growth(uncertainty_evol)),
            'stability_threshold': float(np.percentile(confidences, 10))
        }

    def _analyze_strategy_patterns(self, experiences):
        """Analyze the strategy patterns of successful reasoning experiences.

        Args:
            experiences (list): A list of successful reasoning experiences.

        Returns:
            dict: A dictionary containing the most common strategies and their success rates.
        """
        strategies = []
        for exp in experiences:
            chain = exp['reasoning_chain']
            strategy = self._identify_strategy(chain)
            strategies.append(strategy)
        return {
            'most_common_strategies': Counter(strategies).most_common(5),
            'strategy_success_rates': self._calculate_strategy_success(strategies, experiences)
        }

    def _calculate_confidence_growth(self, uncertainty_evolutions):
        """Calculate the average confidence growth rate based on uncertainty evolution sequences.

        Args:
            uncertainty_evolutions (list): A list of uncertainty evolution sequences.

        Returns:
            float: The average confidence growth rate, or 0.0 if no valid sequences are provided.
        """
        growth_rates = []
        for evolution in uncertainty_evolutions:
            if len(evolution) > 1:
                reduction = evolution[0] - evolution[-1]
                steps = len(evolution)
                growth_rates.append(reduction / steps)
        return np.mean(growth_rates) if growth_rates else 0.0

    def _identify_strategy(self, reasoning_chain):
        """Identify the dominant reasoning strategy in a reasoning chain.

        Args:
            reasoning_chain (list): A list of reasoning steps.

        Returns:
            str: The dominant reasoning strategy, e.g., 'analogical', 'decomposition', etc.
        """
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

    def _calculate_strategy_success(self, strategies, experiences):
        """Calculate the average success rate (confidence) for each reasoning strategy.

        Args:
            strategies (list): A list of reasoning strategies.
            experiences (list): A list of corresponding reasoning experiences.

        Returns:
            dict: A dictionary mapping each strategy to its average success rate.
        """
        strategy_success = defaultdict(list)
        for strategy, exp in zip(strategies, experiences):
            strategy_success[strategy].append(exp['metadata']['confidence'])
        return {strategy: float(np.mean(confidences)) for strategy, confidences in strategy_success.items()}

    def adapt_reasoning_parameters(self, patterns):
        """Adapt reasoning parameters based on the extracted patterns.

        Args:
            patterns (dict): A dictionary containing reasoning patterns.

        Returns:
            dict: A dictionary containing adapted reasoning parameters, or None if no patterns are provided.
        """
        if not patterns:
            return
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

    def create_reasoning_prior(self, query):
        """Create a reasoning prior based on similar past experiences.

        Args:
            query (str): The input query.

        Returns:
            dict: A dictionary containing reasoning prior information, or None if the conditions are not met.
        """
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

    def _extract_uncertainty_pattern(self, experiences):
        """Extract typical uncertainty patterns from a list of experiences.

        Args:
            experiences (list): A list of tuples containing similarity scores and experiences.

        Returns:
            dict: A dictionary containing typical uncertainty patterns and their variances, or None if no data is available.
        """
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