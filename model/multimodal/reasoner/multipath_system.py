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

import time
import torch
import numpy as np
from torch import nn
from .multipath_meta import ArcticMultiPathMetaLearner
from .enhancer import ArcticMultiModalReasoningEnhancer
from .multipath_core import ArcticMultiPathReasoningEngine
from .multipath_infer import ArcticMultiPathInferenceEngine

class ArcticUnifiedMultiPathReasoningSystem:
    """
    A unified multi-path reasoning system that integrates a reasoning engine, inference engine, and meta-learner.
    Tracks the performance of reasoning processes and provides interpretability analysis.
    """
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize the unified multi-path reasoning system.

        Args:
            model: The base model for performing reasoning tasks.
            tokenizer: The tokenizer used to encode input queries.
            device (str, optional): The device on which the model will run. Defaults to 'cuda'.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize sub-components of the reasoning system
        self.reasoning_engine = ArcticMultiPathReasoningEngine(model.config)
        self.reasoning_inference = ArcticMultiPathInferenceEngine(model, tokenizer)
        self.meta_learner = ArcticMultiPathMetaLearner(model)

        # Initialize performance tracking variables
        self.total_reasoning_calls = 0
        self.successful_reasoning_calls = 0
        self.average_confidence = 0.0

    def reason(self, query, use_meta_learning=True, return_full_metadata=False, enable_interpretability=False):
        """
        Perform multi-path reasoning on the given query.

        Args:
            query (str): The input query for reasoning.
            use_meta_learning (bool, optional): Whether to use meta-learning during reasoning. Defaults to True.
            return_full_metadata (bool, optional): Whether to return full metadata. Defaults to False.
            enable_interpretability (bool, optional): Whether to enable interpretability analysis. Defaults to False.

        Returns:
            Union[dict, str]: Returns full metadata if return_full_metadata is True, otherwise returns the answer string.
        """
        # Increment the total number of reasoning calls
        self.total_reasoning_calls += 1
        prior = None
        # Generate a reasoning prior if meta-learning is enabled and there are enough experiences
        if use_meta_learning and len(self.meta_learner.reasoning_memory) >= 50:
            prior = self.meta_learner.create_reasoning_prior(query)

        # Record the start time to measure reasoning duration
        start_time = time.time()
        # Extract hidden states from the base model without gradient computation
        with torch.no_grad():
            model_out = self.model(self.tokenizer.encode(query, return_tensors="pt").to(self.device),
                                   output_hidden_states=True)
            hidden_states = model_out.hidden_states[-1]
            # Pass hidden states to the reasoning engine
            core_out = self.reasoning_engine.forward(hidden_states)

        # Perform multi-path reasoning and get the final result
        final_result = self.reasoning_inference.multi_path_reason(query, return_metadata=True)
        reasoning_time = time.time() - start_time

        # Collect reasoning metadata
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

        # Add interpretability analysis to metadata if enabled
        if enable_interpretability:
            metadata['interpretability'] = {
                'query_complexity': len(query.split()),
                'reasoning_path_analysis': self._analyze_reasoning_paths(metadata),
                'uncertainty_trend': self._analyze_uncertainty_trend(metadata),
                'confidence_breakdown': self._analyze_confidence_components(metadata),
            }

        # Increment successful reasoning calls if confidence is high enough
        if metadata['confidence'] > 0.8:
            self.successful_reasoning_calls += 1
        # Update the average confidence
        self.average_confidence = ((self.average_confidence * (self.total_reasoning_calls - 1) + metadata['confidence']) / self.total_reasoning_calls)

        # Record reasoning experience if meta-learning is enabled
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

    def get_performance_stats(self):
        """
        Get the performance statistics of the reasoning system.

        Returns:
            dict: A dictionary containing various performance metrics.
        """
        return {
            'total_reasoning_calls': self.total_reasoning_calls,
            'successful_reasoning_calls': self.successful_reasoning_calls,
            'success_rate': self.successful_reasoning_calls / max(self.total_reasoning_calls, 1),
            'average_confidence': self.average_confidence,
            'meta_learning_experiences': len(self.meta_learner.reasoning_memory),
            'patterns_learned': len(self.meta_learner.extract_reasoning_patterns() or {})
        }

    def _analyze_reasoning_paths(self, metadata):
        """
        Analyze the reasoning paths based on the given metadata.

        Args:
            metadata (dict): Metadata containing path importance information.

        Returns:
            dict: Analysis results including the total number of paths, dominant paths, and path diversity.
        """
        analysis = {'total_paths': 0, 'dominant_paths': [], 'path_diversity': 0.0}
        path_importance = metadata.get('path_importance', [])
        if path_importance:
            # Calculate the threshold to determine dominant paths
            threshold = np.percentile(path_importance, 70)
            dominant_indices = [i for i, imp in enumerate(path_importance) if imp >= threshold]
            analysis['dominant_paths'] = dominant_indices
            if len(path_importance) > 1:
                # Calculate path diversity using entropy
                arr = np.array(path_importance, dtype=float)
                arr = arr / max(arr.sum(), 1e-8)
                entropy = -np.sum(arr * np.log(arr + 1e-10))
                max_entropy = np.log(len(arr))
                analysis['path_diversity'] = float(entropy / max(max_entropy, 1e-8))
        return analysis

    def _analyze_uncertainty_trend(self, metadata):
        """
        Analyze the uncertainty trend based on the given metadata.

        Args:
            metadata (dict): Metadata containing uncertainty scores.

        Returns:
            dict: Analysis results including initial uncertainty, final uncertainty, 
                  uncertainty reduction, and trend direction.
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

    def _analyze_confidence_components(self, metadata):
        """
        Analyze the components of the confidence score based on the given metadata.

        Args:
            metadata (dict): Metadata containing the confidence score.

        Returns:
            dict: Breakdown of the confidence score into base confidence, path consensus, and uncertainty weighted components.
        """
        confidence = metadata.get('confidence', 0.8)
        return {
            'base_confidence': confidence * 0.6,
            'path_consensus': confidence * 0.25,
            'uncertainty_weighted': confidence * 0.15
        }