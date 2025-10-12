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

import re
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class ArcticMultiPathInferenceEngine:
    """
    A class for performing multi-path reasoning inference. This engine conducts step-by-step reasoning
    with multiple paths at each layer to find the most confident answer.
    """
    def __init__(self, model, tokenizer, max_depth=5, confidence_threshold=0.85):
        """
        Initialize the ArcticMultiPathInferenceEngine.

        Args:
            model: Pre-trained model used for reasoning.
            tokenizer: Tokenizer corresponding to the model.
            max_depth (int): Maximum reasoning depth. Defaults to 5.
            confidence_threshold (float): Confidence threshold to stop reasoning. Defaults to 0.85.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.reasoning_cache = {}

    @torch.no_grad()
    def multi_path_reason(self, prompt, return_metadata=False):
        """
        Perform multi-path reasoning on the given prompt.

        Args:
            prompt (str): Input prompt for reasoning.
            return_metadata (bool): Whether to return additional reasoning metadata. Defaults to False.

        Returns:
            Union[str, dict]: If return_metadata is False, returns the answer string.
                             If True, returns a dictionary containing the answer, confidence,
                             reasoning depth, reasoning chain, uncertainty evolution, and fact verifications.
        """
        # Initialize the reasoning state based on the input prompt
        reasoning_state = self._initialize_reasoning_state(prompt)
        reasoning_layers = []
        current_depth = 0

        # Perform multi-path reasoning layer by layer
        while current_depth < self.max_depth:
            layer_result = self._multi_path_layer_reasoning(reasoning_state, depth=current_depth)
            reasoning_layers.append(layer_result)
            if layer_result['confidence'] >= self.confidence_threshold:
                break
            reasoning_state = self._update_reasoning_state(reasoning_state, layer_result['residual_uncertainty'])
            current_depth += 1

        # Select the best reasoning path from the reasoning layers
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

    def _initialize_reasoning_state(self, prompt):
        """
        Initialize the reasoning state based on the input prompt.

        Args:
            prompt (str): Input prompt.

        Returns:
            dict: A dictionary containing the prompt embedding, uncertainty map, and hypothesis space.
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

    def _multi_path_layer_reasoning(self, reasoning_state, depth):
        """
        Perform multi-path reasoning at a specific layer.

        Args:
            reasoning_state (dict): Current reasoning state.
            depth (int): Current reasoning depth.

        Returns:
            dict: A dictionary containing the reasoning path, confidence, uncertainty,
                  verified facts, and residual uncertainty.
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

    def _generate_reasoning_paths(self, prompt_emb, depth):
        """
        Generate multiple reasoning paths based on the prompt embedding and depth.

        Args:
            prompt_emb (torch.Tensor): Prompt embedding.
            depth (int): Current reasoning depth.

        Returns:
            list: A list of dictionaries, each containing the generated text, logits, and temperature.
        """
        paths = []
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]
        # Generate reasoning paths with different temperatures
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

    def _evaluate_reasoning_path(self, path, depth):
        """
        Evaluate a reasoning path based on length, consistency, and uncertainty reduction.

        Args:
            path (dict): Reasoning path to evaluate.
            depth (int): Current reasoning depth.

        Returns:
            float: Evaluation score of the reasoning path.
        """
        length_score = 1.0 / (1.0 + abs(len(path['text']) - 100 * (depth + 1)) / 100)
        consistency_score = self._estimate_logical_consistency(path['text'])
        uncertainty_score = self._measure_uncertainty_reduction(path)
        return length_score * 0.3 + consistency_score * 0.4 + uncertainty_score * 0.3

    def _verify_facts(self, reasoning_path):
        """
        Verify the facts in a reasoning path.

        Args:
            reasoning_path (dict): Reasoning path containing text to verify.

        Returns:
            dict: A dictionary containing verification scores for different aspects.
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

    def _check_factual_accuracy(self, text):
        """
        Check the factual accuracy of the given text.

        Args:
            text (str): Text to check.

        Returns:
            float: Factual accuracy score in the range [0.0, 1.0].
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

    def _check_logical_validity(self, text):
        """
        Check the logical validity of the given text.

        Args:
            text (str): Text to check.

        Returns:
            float: Logical validity score in the range [0.0, 1.0].
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

    def _calculate_path_confidence(self, path, facts):
        """
        Calculate the confidence of a reasoning path based on verified facts.

        Args:
            path (dict): Reasoning path.
            facts (dict): Verified facts.

        Returns:
            torch.Tensor: Confidence score.
        """
        base_confidence = torch.sigmoid(torch.tensor([
            facts['self_consistency'],
            facts['temporal_consistency'],
            facts['causal_consistency']
        ]).mean())
        return base_confidence

    def _update_reasoning_state(self, current_state, residual_uncertainty):
        """
        Update the reasoning state based on the residual uncertainty.

        Args:
            current_state (dict): Current reasoning state.
            residual_uncertainty (torch.Tensor): Residual uncertainty.

        Returns:
            dict: Updated reasoning state.
        """
        uncertainty_adjustment = 1.0 - residual_uncertainty * 0.5
        return {**current_state, 'uncertainty_map': current_state['uncertainty_map'] * uncertainty_adjustment}

    def _path_selection_inference(self, reasoning_layers):
        """
        Select the best reasoning path from the reasoning layers.

        Args:
            reasoning_layers (list): List of reasoning layer results.

        Returns:
            dict: A dictionary containing the best answer and its confidence.
        """
        if not reasoning_layers:
            return {'answer': "Unable to reason about this", 'confidence': 0.0}
        weights = torch.softmax(torch.tensor([l['confidence'] for l in reasoning_layers]), dim=0)
        best_layer_idx = torch.argmax(weights)
        best_layer = reasoning_layers[best_layer_idx]
        return {'answer': best_layer['reasoning']['text'], 'confidence': best_layer['confidence']}

    def _estimate_logical_consistency(self, text):
        """
        Estimate the logical consistency of the given text.

        Args:
            text (str): Text to estimate.

        Returns:
            float: Logical consistency score in the range [0.0, 1.0].
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

    def _measure_uncertainty_reduction(self, path):
        """
        Measure the uncertainty reduction of a reasoning path.

        Args:
            path (dict): Reasoning path containing logits.

        Returns:
            float: Uncertainty reduction score in the range [0.0, 1.0].
        """
        logits = torch.stack(path['logits'])
        entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1)
        return 1.0 - float(entropy.mean().item())