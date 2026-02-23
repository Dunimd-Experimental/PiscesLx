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

"""
Speculative Decoding Inference Operator

Implementation of speculative decoding for accelerated text generation
using a draft model and target model for verification.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

from configs.version import VERSION

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus


@dataclass
class POPSSSpeculativeConfig:
    """Speculative decoding configuration."""
    
    # Model settings
    draft_model_path: str = ".pisceslx/ckpt"
    target_model_path: str = ".pisceslx/ckpt"
    
    # Speculative parameters
    gamma: int = 5  # Number of tokens to speculate
    draft_temperature: float = 0.8
    target_temperature: float = 0.7
    
    # Generation settings
    max_new_tokens: int = 512
    stop_token_id: Optional[int] = None
    
    # Performance settings
    use_cache: bool = True
    early_stopping: bool = True
    
    # Validation settings
    acceptance_threshold: float = 0.1
    fallback_to_regular: bool = True


class POPSSSpeculativeDecodingOperator(PiscesLxOperatorInterface):
    """Complete speculative decoding operator implementation."""
    
    def __init__(self):
        super().__init__()
        self.name = "speculative.decoding"
        self.version = VERSION
        self.type = "inference"
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Infer",file_path=get_log_file("PiscesLx.Opss.Infer"), enable_file=True)
        
    @property
    def description(self) -> str:
        return "Speculative decoding inference operator for accelerated generation"
        
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "draft_model": {"type": "torch.nn.Module", "required": True, "description": "Fast draft model"},
            "target_model": {"type": "torch.nn.Module", "required": True, "description": "Accurate target model"},
            "tokenizer": {"type": "object", "required": True, "description": "Model tokenizer"},
            "prompt": {"type": "str", "required": True, "description": "Input prompt"},
            "config": {"type": "POPSSSpeculativeConfig", "required": False, "description": "Speculative configuration"}
        }
        
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "generated_text": {"type": "str", "description": "Generated text"},
            "generated_tokens": {"type": "list", "description": "Generated token IDs"},
            "acceptance_rate": {"type": "float", "description": "Token acceptance rate"},
            "speedup_ratio": {"type": "float", "description": "Speedup compared to regular decoding"},
            "detailed_stats": {"type": "dict", "description": "Detailed generation statistics"}
        }
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_keys = ['draft_model', 'target_model', 'tokenizer', 'prompt']
        for key in required_keys:
            if key not in inputs or inputs[key] is None:
                self._LOG.error(f"Missing required parameter: {key}")
                return False
        
        # Validate models
        if not isinstance(inputs['draft_model'], nn.Module):
            self._LOG.error("Draft model must be a torch.nn.Module")
            return False
        if not isinstance(inputs['target_model'], nn.Module):
            self._LOG.error("Target model must be a torch.nn.Module")
            return False
            
        return True
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """Execute speculative decoding inference."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Invalid input parameters",
                    execution_time=time.time() - start_time
                )
            
            # Parse inputs
            draft_model = inputs['draft_model']
            target_model = inputs['target_model']
            tokenizer = inputs['tokenizer']
            prompt = inputs['prompt']
            custom_config = inputs.get('config')
            
            # Setup configuration
            if custom_config:
                config = custom_config
            else:
                config = POPSSSpeculativeConfig()
            
            self._LOG.info(f"Starting speculative decoding with gamma={config.gamma}")
            
            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            draft_model = draft_model.to(device)
            target_model = target_model.to(device)
            
            # Encode prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            prompt_length = input_ids.shape[1]
            
            # Execute speculative decoding
            result = self._speculative_decode(
                draft_model, target_model, tokenizer, input_ids, config, device
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(
                result['generated_ids'][prompt_length:], 
                skip_special_tokens=True
            ).strip()
            
            execution_time = time.time() - start_time
            
            result_data = {
                'generated_text': generated_text,
                'generated_tokens': result['generated_ids'][prompt_length:].tolist(),
                'acceptance_rate': result['acceptance_rate'],
                'speedup_ratio': result['speedup_ratio'],
                'detailed_stats': result['stats']
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result_data,
                execution_time=execution_time,
                metadata={
                    'config': config.__dict__,
                    'prompt_length': prompt_length,
                    'generated_length': len(result['generated_ids']) - prompt_length,
                    'gamma': config.gamma
                }
            )
            
        except Exception as e:
            self._LOG.error(f"Speculative decoding failed: {str(e)}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _speculative_decode(self, draft_model: nn.Module, target_model: nn.Module, 
                           tokenizer: Any, input_ids: torch.Tensor, 
                           config: POPSSSpeculativeConfig, device: torch.device) -> Dict[str, Any]:
        """Core speculative decoding algorithm."""
        draft_model.eval()
        target_model.eval()
        
        generated_ids = input_ids.clone()
        accepted_tokens = 0
        total_draft_tokens = 0
        decode_steps = 0
        
        stats = {
            'draft_tokens_proposed': [],
            'tokens_accepted': [],
            'verification_times': [],
            'rejection_points': []
        }
        
        while len(generated_ids[0]) < config.max_new_tokens + input_ids.shape[1]:
            decode_steps += 1
            
            # Draft model generates γ tokens
            draft_tokens = self._draft_generation(draft_model, generated_ids, config, device)
            total_draft_tokens += len(draft_tokens)
            stats['draft_tokens_proposed'].append(len(draft_tokens))
            
            if len(draft_tokens) == 0:
                break
            
            # Target model verifies the draft
            accepted_count, verification_time = self._verify_draft(
                target_model, generated_ids, draft_tokens, config, device
            )
            
            accepted_tokens += accepted_count
            stats['tokens_accepted'].append(accepted_count)
            stats['verification_times'].append(verification_time)
            
            # Append accepted tokens
            generated_ids = torch.cat([
                generated_ids, 
                draft_tokens[:accepted_count].unsqueeze(0)
            ], dim=1)
            
            # Record rejection point
            if accepted_count < len(draft_tokens):
                stats['rejection_points'].append({
                    'step': decode_steps,
                    'position': len(generated_ids[0]),
                    'rejected_tokens': len(draft_tokens) - accepted_count
                })
            
            # Check for stopping condition
            if config.stop_token_id is not None:
                if config.stop_token_id in draft_tokens[:accepted_count]:
                    break
            
            # Early stopping if no tokens accepted
            if accepted_count == 0:
                break
        
        # Calculate metrics
        acceptance_rate = accepted_tokens / max(1, total_draft_tokens)
        speedup_ratio = self._estimate_speedup(acceptance_rate, config.gamma)
        
        return {
            'generated_ids': generated_ids[0],
            'acceptance_rate': acceptance_rate,
            'speedup_ratio': speedup_ratio,
            'stats': stats
        }
    
    def _draft_generation(self, draft_model: nn.Module, input_ids: torch.Tensor, 
                         config: POPSSSpeculativeConfig, device: torch.device) -> torch.Tensor:
        """Generate draft tokens using the fast draft model."""
        generated_tokens = []
        
        current_input = input_ids.clone()
        
        for i in range(config.gamma):
            with torch.no_grad():
                # Enable caching for efficiency
                outputs = draft_model(
                    input_ids=current_input,
                    use_cache=config.use_cache
                )
                
                # Get logits for the last token
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature scaling
                if config.draft_temperature != 1.0:
                    logits = logits / config.draft_temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                generated_tokens.append(next_token)
                current_input = torch.cat([current_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Stop if EOS token is generated
                if config.stop_token_id is not None and next_token.item() == config.stop_token_id:
                    break
        
        return torch.stack(generated_tokens) if generated_tokens else torch.tensor([])
    
    def _verify_draft(self, target_model: nn.Module, input_ids: torch.Tensor,
                     draft_tokens: torch.Tensor, config: POPSSSpeculativeConfig, 
                     device: torch.device) -> Tuple[int, float]:
        """Verify draft tokens using the target model."""
        start_time = time.time()
        
        if len(draft_tokens) == 0:
            return 0, 0.0
        
        # Prepare input for verification
        verification_input = torch.cat([input_ids, draft_tokens.unsqueeze(0)], dim=1)
        
        with torch.no_grad():
            # Get target model predictions for all positions
            target_outputs = target_model(
                input_ids=verification_input,
                use_cache=config.use_cache
            )
            
            target_logits = target_outputs.logits[:, -len(draft_tokens)-1:-1, :]  # Exclude last token
            
            # Compare with draft tokens
            accepted_count = 0
            for i, draft_token in enumerate(draft_tokens):
                target_probs = F.softmax(target_logits[0, i, :], dim=-1)
                draft_prob = target_probs[draft_token].item()
                
                # Accept if probability is above threshold
                if draft_prob > config.acceptance_threshold:
                    accepted_count += 1
                else:
                    break  # Reject remaining tokens
        
        verification_time = time.time() - start_time
        return accepted_count, verification_time
    
    def _estimate_speedup(self, acceptance_rate: float, gamma: int) -> float:
        """Estimate theoretical speedup of speculative decoding."""
        if acceptance_rate <= 0 or gamma <= 1:
            return 1.0
        
        # Theoretical speedup formula
        expected_tokens_per_step = 1 + acceptance_rate * (gamma - 1)
        speedup = expected_tokens_per_step / (1 + (gamma - 1) / gamma)
        
        return max(1.0, speedup)


# Additional helper operators

class POPSSAssistedDecodingOperator(POPSSSpeculativeDecodingOperator):
    """Assisted decoding variant of speculative decoding."""
    
    def __init__(self):
        super().__init__()
        self.name = "assisted.decoding"
        self.version = VERSION
        self.type = "inference"
        self._LOG = PiscesLxLogger("popss.ops.infer.assisted")
    
    @property
    def description(self) -> str:
        return "Assisted decoding inference operator with teacher-student framework"
    
    def _verify_draft(self, target_model: nn.Module, input_ids: torch.Tensor,
                     draft_tokens: torch.Tensor, config: POPSSSpeculativeConfig, 
                     device: torch.device) -> Tuple[int, float]:
        """Enhanced verification with assisted decoding logic."""
        start_time = time.time()
        
        if len(draft_tokens) == 0:
            return 0, 0.0
        
        # More sophisticated verification with confidence scoring
        verification_input = torch.cat([input_ids, draft_tokens.unsqueeze(0)], dim=1)
        
        with torch.no_grad():
            target_outputs = target_model(
                input_ids=verification_input,
                use_cache=config.use_cache
            )
            
            target_logits = target_outputs.logits[:, -len(draft_tokens)-1:-1, :]
            
            accepted_count = 0
            confidence_scores = []
            
            for i, draft_token in enumerate(draft_tokens):
                target_probs = F.softmax(target_logits[0, i, :], dim=-1)
                draft_prob = target_probs[draft_token].item()
                confidence_scores.append(draft_prob)
                
                # Adaptive acceptance based on confidence
                threshold = config.acceptance_threshold * (0.8 ** i)  # Decreasing threshold
                if draft_prob > threshold:
                    accepted_count += 1
                else:
                    break
        
        verification_time = time.time() - start_time
        return accepted_count, verification_time


# Export operators
__all__ = ['POPSSSpeculativeDecodingOperator', 'POPSSAssistedDecodingOperator', 'POPSSSpeculativeConfig']