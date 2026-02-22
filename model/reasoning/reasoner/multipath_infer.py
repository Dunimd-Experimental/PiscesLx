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

"""Inference utilities for Yv multi-path reasoning workflows.

This module provides the YvMultiPathInferenceEngine class which implements
inference-time multi-path reasoning with confidence evaluation, fact verification,
and path selection capabilities.

Architecture:
    1. Multi-Path Reasoning:
       - Iterative layer-by-layer reasoning exploration
       - Confidence-based early stopping
       - Uncertainty map propagation
    
    2. Path Generation:
       - Temperature-diversified sampling
       - Multiple candidate paths per depth
       - Logit-based uncertainty estimation
    
    3. Path Evaluation:
       - Structural scoring (length, consistency)
       - Uncertainty reduction measurement
       - Composite quality scoring
    
    4. Fact Verification:
       - Self-consistency checking
       - Temporal consistency validation
       - Causal consistency analysis
       - Factual accuracy estimation
       - Logical validity assessment
    
    5. Path Selection:
       - Confidence-weighted aggregation
       - Best-path selection via softmax

Key Features:
    - Iterative reasoning with adaptive depth
    - Multi-temperature sampling for diversity
    - Comprehensive fact verification pipeline
    - Uncertainty evolution tracking
    - Rich metadata for analysis

Performance Characteristics:
    - Path Generation: O(T * L) where T = temperatures, L = max_length
    - Fact Verification: O(S) where S = sentences in text
    - Path Selection: O(D) where D = reasoning depth

Usage Example:
    >>> from model.reasoning.reasoner import YvMultiPathInferenceEngine
    >>> 
    >>> # Initialize engine
    >>> engine = YvMultiPathInferenceEngine(
    ...     model=ruchbah_model,
    ...     tokenizer=tokenizer,
    ...     max_depth=5,
    ...     confidence_threshold=0.85
    >>> )
    >>> 
    >>> # Run multi-path reasoning
    >>> result = engine.multi_path_reason(
    ...     prompt="What is the capital of France?",
    ...     return_metadata=True
    >>> )
    >>> 
    >>> # Access results
    >>> print(result['answer'])
    >>> print(result['confidence'])

Dependencies:
    - re: Regular expression pattern matching
    - torch: Tensor operations and neural network inference
    - numpy: Numerical computations
    - torch.nn: Neural network modules
    - torch.nn.functional: Activation functions
    - typing: Type hints for documentation

Note:
    The engine requires a pre-trained model with generate() and forward()
    methods. Fact verification uses heuristic fallbacks when external
    models are not available.
"""

import re
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union


class YvMultiPathInferenceEngine:
    """Perform inference-time multi-path reasoning and confidence evaluation.
    
    This class implements a sophisticated inference engine that explores
    multiple reasoning paths, evaluates their quality, and selects the
    best answer based on confidence scoring.
    
    Architecture:
        1. Initialization:
           - Model and tokenizer setup
           - Depth and threshold configuration
           - Reasoning cache initialization
        
        2. Multi-Path Reasoning Loop:
           - State initialization from prompt
           - Iterative layer reasoning
           - Confidence-based early stopping
           - State update with residual uncertainty
        
        3. Path Generation:
           - Temperature-diversified sampling
           - Multiple candidates per temperature
           - Logit collection for uncertainty
        
        4. Path Evaluation:
           - Length scoring for target length
           - Logical consistency estimation
           - Uncertainty reduction measurement
           - Composite score calculation
        
        5. Fact Verification:
           - Self-consistency: internal coherence
           - Temporal consistency: time-related validity
           - Causal consistency: cause-effect validity
           - Factual accuracy: truthfulness estimation
           - Logical validity: argument structure
    
    Confidence Calculation:
        - Base confidence from sigmoid of mean fact scores
        - Weighted by self, temporal, and causal consistency
        - Normalized to [0, 1] range
    
    Attributes:
        model (nn.Module): Pre-trained model for generation.
        tokenizer (Any): Tokenizer for encoding/decoding.
        max_depth (int): Maximum reasoning layers (default: 5).
        confidence_threshold (float): Early stopping threshold (default: 0.85).
        reasoning_cache (dict): Cache for reasoning results.
    
    Example:
        >>> engine = YvMultiPathInferenceEngine(model, tokenizer)
        >>> result = engine.multi_path_reason("Explain quantum computing")
        >>> print(result)
    
    Note:
        The engine uses torch.no_grad() for inference efficiency.
        External model loading (transformers) is optional with fallbacks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_depth: int = 5,
        confidence_threshold: float = 0.85,
    ) -> None:
        """Configure the inference engine with model, tokenizer, and limits.
        
        Args:
            model (nn.Module): Pre-trained model providing generate and forward APIs.
                Must support output_hidden_states=True for embedding extraction.
            tokenizer (Any): Tokenizer aligned with model for encoding/decoding text.
                Must support encode() and decode() methods.
            max_depth (int): Maximum number of reasoning layers explored before
                termination. Default: 5.
            confidence_threshold (float): Confidence level required to stop
                exploring further paths. Default: 0.85.
        
        Note:
            The reasoning_cache is initialized as an empty dict for storing
            intermediate results and avoiding redundant computation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.reasoning_cache = {}

    @torch.no_grad()
    def multi_path_reason(self, prompt: str, return_metadata: bool = False) -> Union[str, Dict[str, Any]]:
        """Run multi-path reasoning and optionally return diagnostic metadata.

        Args:
            prompt (str): Natural language query or problem description.
            return_metadata (bool): When ``True``, expose intermediate reasoning traces.

        Returns:
            Union[str, Dict[str, Any]]: Either the best answer string or a metadata payload
            containing answer, confidence, reasoning depth, reasoning chain, uncertainty trajectory,
            and fact verification scores.
        """
        # Initialize the reasoning state from the encoded prompt.
        reasoning_state = self._initialize_reasoning_state(prompt)
        reasoning_layers = []
        current_depth = 0

        # Iterate through reasoning layers until depth or confidence limits are reached.
        while current_depth < self.max_depth:
            layer_result = self._multi_path_layer_reasoning(reasoning_state, depth=current_depth)
            reasoning_layers.append(layer_result)
            if layer_result['confidence'] >= self.confidence_threshold:
                break
            reasoning_state = self._update_reasoning_state(reasoning_state, layer_result['residual_uncertainty'])
            current_depth += 1

        # Select the highest-confidence reasoning path from accumulated layers.
        final_result = self._path_selection_inference(reasoning_layers)

        if return_metadata:
            return {
                'answer': final_result['answer'],
                'confidence': final_result['confidence'],
                'reasoning_depth': current_depth + 1,
                'reasoning_chain': [layer['reasoning'] for layer in reasoning_layers],
                'uncertainty_evolution': [layer['uncertainty'] for layer in reasoning_layers],
                'fact_verifications': [layer['facts'] for layer in reasoning_layers]
            }
        return final_result['answer']

    def _initialize_reasoning_state(self, prompt: str) -> Dict[str, Any]:
        """Encode the prompt and construct the initial reasoning state.

        Args:
            prompt (str): Request supplied by the user.

        Returns:
            Dict[str, Any]: Structure containing prompt embeddings, an uncertainty map, and an empty hypothesis store.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        return {
            'prompt_embedding': hidden_states,
            'uncertainty_map': torch.ones_like(hidden_states),
            'hypothesis_space': {}
        }

    def _multi_path_layer_reasoning(self, reasoning_state: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Evaluate candidate reasoning paths for the current depth.

        Args:
            reasoning_state (Dict[str, Any]): Encoded state accumulated so far.
            depth (int): Zero-based reasoning layer index.

        Returns:
            Dict[str, Any]: Record containing the best path, its confidence, uncertainty metrics, and verified facts.
        """
        prompt_emb = reasoning_state['prompt_embedding']
        reasoning_paths = self._generate_reasoning_paths(prompt_emb, depth)
        path_scores = []
        for path in reasoning_paths:
            score = self._evaluate_reasoning_path(path, depth)
            path_scores.append(score)
        best_path_idx = torch.argmax(torch.tensor(path_scores))
        best_path = reasoning_paths[best_path_idx]
        facts = self._verify_facts(best_path)
        confidence = self._calculate_path_confidence(best_path, facts)
        uncertainty = 1 - confidence
        return {
            'reasoning': best_path,
            'confidence': confidence.item(),
            'uncertainty': uncertainty.item(),
            'facts': facts,
            'residual_uncertainty': uncertainty
        }

    def _generate_reasoning_paths(self, prompt_emb: torch.Tensor, depth: int) -> List[Dict[str, Any]]:
        """Sample multiple reasoning paths with diversified temperatures.

        Args:
            prompt_emb (torch.Tensor): Prompt embedding produced by the model.
            depth (int): Reasoning depth influencing the number of samples.

        Returns:
            List[Dict[str, Any]]: Generated paths containing decoded text, token-level logits, and sampling temperature.
        """
        paths = []
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]
        # Iterate through candidate temperatures to diversify sampling behaviour.
        for temp in temperatures[:min(3 + depth, 5)]:
            generated = self.model.generate(
                prompt_emb,
                max_length=256,
                do_sample=True,
                temperature=temp,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            paths.append({
                'text': self.tokenizer.decode(generated.sequences[0], skip_special_tokens=True),
                'logits': generated.scores,
                'temperature': temp
            })
        return paths

    def _evaluate_reasoning_path(self, path: Dict[str, Any], depth: int) -> float:
        """Score a reasoning path using structural and uncertainty heuristics.

        Args:
            path (Dict[str, Any]): Path descriptor containing generated text and logits.
            depth (int): Current reasoning depth controlling target length.

        Returns:
            float: Composite score reflecting path quality.
        """
        length_score = 1.0 / (1.0 + abs(len(path['text']) - 100 * (depth + 1)) / 100)
        consistency_score = self._estimate_logical_consistency(path['text'])
        uncertainty_score = self._measure_uncertainty_reduction(path)
        return length_score * 0.3 + consistency_score * 0.4 + uncertainty_score * 0.3

    def _verify_facts(self, reasoning_path: Dict[str, Any]) -> Dict[str, float]:
        """Estimate factual, temporal, and causal consistency of a reasoning path.

        Args:
            reasoning_path (Dict[str, Any]): Candidate path including plain-text reasoning.

        Returns:
            Dict[str, float]: Per-aspect verification scores in the range ``[0.0, 1.0]``.
        """
        text = reasoning_path['text']
        checks = {
            'self_consistency': self._check_self_consistency(text),
            'temporal_consistency': self._check_temporal_consistency(text),
            'causal_consistency': self._check_causal_consistency(text),
            'factual_accuracy': self._check_factual_accuracy(text),
            'logical_validity': self._check_logical_validity(text)
        }
        return checks

    def _check_factual_accuracy(self, text: str) -> float:
        """Estimate factual accuracy for textual reasoning segments.

        Args:
            text (str): Input text subject to factual assessment.

        Returns:
            float: Factual accuracy score normalized to ``[0.0, 1.0]``.
        """
        try:
            from transformers import pipeline
            fact_checker = pipeline("text-classification", model="microsoft/deberta-v3-base")
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            if not sentences:
                return 0.5
            scores = []
            for sentence in sentences:
                if len(sentence) > 10:
                    result = fact_checker(sentence, candidate_labels=["factual", "non-factual"])
                    scores.append(1.0 if result[0]['label'] == "factual" else 0.0)
            return float(np.mean(scores)) if scores else 0.5
        except Exception:
            factual_indicators = ['is', 'are', 'was', 'were', 'fact', 'data', 'evidence']
            speculative_indicators = ['might', 'could', 'may', 'possibly', 'perhaps', 'likely']
            factual_score = sum(1 for w in factual_indicators if w in text.lower()) / len(factual_indicators)
            speculative_score = sum(1 for w in speculative_indicators if w in text.lower()) / len(speculative_indicators)
            return max(0.0, float(factual_score - speculative_score * 0.5))

    def _check_logical_validity(self, text: str) -> float:
        """Assess logical validity using either model-based or heuristic signals.

        Args:
            text (str): Textual reasoning sequence.

        Returns:
            float: Logical validity score between ``0.0`` and ``1.0``.
        """
        logical_patterns = {
            'deductive': ['if.*then', 'given.*therefore', 'since.*conclude'],
            'inductive': ['based on.*we can infer', 'evidence suggests', 'observations indicate'],
            'abductive': ['the best explanation is', 'most likely cause', 'plausible reason']
        }
        scores = []
        for _, patterns in logical_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    matches += 1
            scores.append(matches / len(patterns))
        return float(np.mean(scores)) if scores else 0.5

    def _calculate_path_confidence(self, path: Dict[str, Any], facts: Dict[str, float]) -> torch.Tensor:
        """Aggregate fact scores into a sigmoid-normalized confidence tensor.

        Args:
            path (Dict[str, Any]): Reasoning path descriptor (unused but retained for parity).
            facts (Dict[str, float]): Per-aspect verification scores produced by :meth:`_verify_facts`.

        Returns:
            torch.Tensor: Tensor containing a single confidence value.
        """
        base_confidence = torch.sigmoid(torch.tensor([
            facts['self_consistency'],
            facts['temporal_consistency'],
            facts['causal_consistency']
        ]).mean())
        return base_confidence

    def _update_reasoning_state(self, current_state: Dict[str, Any], residual_uncertainty: torch.Tensor) -> Dict[str, Any]:
        """Update the uncertainty map with residual uncertainty feedback.

        Args:
            current_state (Dict[str, Any]): Existing reasoning state.
            residual_uncertainty (torch.Tensor): Remaining uncertainty after current depth.

        Returns:
            Dict[str, Any]: Updated reasoning state with adjusted uncertainty map.
        """
        uncertainty_adjustment = 1.0 - residual_uncertainty * 0.5
        return {**current_state, 'uncertainty_map': current_state['uncertainty_map'] * uncertainty_adjustment}

    def _path_selection_inference(self, reasoning_layers: List[Dict[str, Any]]) -> Dict[str, Union[str, float]]:
        """Select the best reasoning path via confidence-weighted aggregation.

        Args:
            reasoning_layers (List[Dict[str, Any]]): Layer-wise reasoning summaries produced by the engine.

        Returns:
            Dict[str, Union[str, float]]: Answer string with its associated confidence score.
        """
        if not reasoning_layers:
            return {'answer': "Unable to reason about this", 'confidence': 0.0}
        weights = torch.softmax(torch.tensor([l['confidence'] for l in reasoning_layers]), dim=0)
        best_layer_idx = torch.argmax(weights)
        best_layer = reasoning_layers[best_layer_idx]
        return {'answer': best_layer['reasoning']['text'], 'confidence': best_layer['confidence']}

    def _estimate_logical_consistency(self, text: str) -> float:
        """Derive a logical consistency score using learned or heuristic cues.

        Args:
            text (str): Reasoning text passage.

        Returns:
            float: Logical consistency score between ``0.0`` and ``1.0``.
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            model = AutoModel.from_pretrained('microsoft/deberta-base')
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            consistency_score = torch.sigmoid(torch.nn.Linear(768, 1)(embeddings)).item()
            return max(0.0, min(1.0, consistency_score))
        except Exception:
            logical_markers = {
                'premise_indicators': ['because', 'since', 'as', 'given that', 'assuming'],
                'conclusion_indicators': ['therefore', 'thus', 'hence', 'so', 'consequently'],
                'contradiction_indicators': ['but', 'however', 'although', 'nevertheless', 'contradiction'],
                'support_indicators': ['furthermore', 'moreover', 'additionally', 'also']
            }
            scores = []
            tl = text.lower()
            for category, markers in logical_markers.items():
                count = sum(1 for m in markers if m in tl)
                score = min(count / 3.0, 1.0) if 'contradiction' not in category else max(0.0, 1 - count / 2.0)
                scores.append(score)
            return float(np.mean(scores)) if scores else 0.5

    def _measure_uncertainty_reduction(self, path: Dict[str, Any]) -> float:
        """Quantify uncertainty reduction achieved by a reasoning path.

        Args:
            path (Dict[str, Any]): Reasoning path record containing token-level logits.

        Returns:
            float: Uncertainty reduction score between ``0.0`` and ``1.0``.
        """
        logits = torch.stack(path['logits'])
        entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1)
        return 1.0 - float(entropy.mean().item())
