#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class RuchbahRecursiveReasoningConfig:
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
    def __init__(self, hidden_size: int, num_decomposition_heads: int = 4):
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
    def __init__(self, hidden_size: int, num_layers: int = 2):
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
        for layer in self.reasoning_layers:
            x = layer(x)
        return self.output_proj(x)


class _SubProblemAggregator(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
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
    def __init__(self, hidden_size: int):
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


class RuchbahRecursiveDepthReasoner(nn.Module):
    def __init__(self, config: Optional[RuchbahRecursiveReasoningConfig] = None):
        super().__init__()
        if config is None:
            config = RuchbahRecursiveReasoningConfig()
        
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


class RuchbahThoughtTreeReasoner(nn.Module):
    def __init__(self, hidden_size: int, tree_depth: int = 5, beam_width: int = 3):
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _ucb_score(self, node_value: float, visit_count: int, parent_visits: int, exploration_weight: float = 1.414) -> float:
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
