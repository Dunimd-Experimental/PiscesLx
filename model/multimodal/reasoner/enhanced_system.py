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
class RuchbahEnhancedReasoningConfig:
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


class RuchbahEnhancedReasoningRouter(nn.Module):
    def __init__(self, hidden_size: int, num_task_types: int = 5):
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


class RuchbahUncertaintyQuantifier(nn.Module):
    def __init__(self, hidden_size: int):
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


class RuchbahEnhancedReasoningSystem(nn.Module):
    def __init__(self, config: Optional[RuchbahEnhancedReasoningConfig] = None):
        super().__init__()
        if config is None:
            config = RuchbahEnhancedReasoningConfig()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        if config.enable_router:
            self.router = RuchbahEnhancedReasoningRouter(
                hidden_size=self.hidden_size,
                num_task_types=5
            )
        else:
            self.router = None
        
        if config.use_recursive_reasoning:
            from .recursive_depth import RuchbahRecursiveDepthReasoner, RuchbahRecursiveReasoningConfig
            recursive_config = RuchbahRecursiveReasoningConfig(
                hidden_size=self.hidden_size,
                max_depth=config.max_recursive_depth,
                use_verification=config.use_bidirectional_verification
            )
            self.recursive_reasoner = RuchbahRecursiveDepthReasoner(recursive_config)
        else:
            self.recursive_reasoner = None
        
        if config.use_thought_tree:
            from .recursive_depth import RuchbahThoughtTreeReasoner
            self.thought_tree_reasoner = RuchbahThoughtTreeReasoner(
                hidden_size=self.hidden_size,
                tree_depth=config.tree_depth,
                beam_width=config.beam_width
            )
        else:
            self.thought_tree_reasoner = None
        
        if config.enable_uncertainty_quantification:
            self.uncertainty_quantifier = RuchbahUncertaintyQuantifier(self.hidden_size)
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
