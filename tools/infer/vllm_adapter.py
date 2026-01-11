#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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
vLLM Model Adapter for PiscesL1 Ruchbah Model.

This module provides the necessary interfaces to make PiscesL1
compatible with vLLM's inference engine.

Key Components:
- PiscesLxVLLMForCausalLM: vLLM-compatible causal LM head
- Input/output adapters for vLLM integration
- Attention mask and position ID handlers
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils import PiscesLxCoreLog
logger = PiscesLxCoreLog("pisceslx.tools.infer.vllm_adapter")


def _getattr_with_path(model: nn.Module, name: str):
    """Get attribute with full path for error messages."""
    for module_name, module in model.named_modules():
        if module_name == name or name.endswith(f".{module_name}"):
            if hasattr(module, name.split(".")[-1]):
                return getattr(module, name.split(".")[-1])
    raise AttributeError(f"Model has no attribute '{name}'")


class PiscesLxVLLMForCausalLM(nn.Module):
    """vLLM-compatible causal LM wrapper for PiscesL1 Ruchbah Model.
    
    This wrapper provides the interface required by vLLM:
    - Supports PagedAttention via get_attention_blocks()
    - Provides proper LM head output
    - Handles position IDs and attention masks
    """
    
    def __init__(self, model, config):
        """Initialize vLLM-compatible wrapper.
        
        Args:
            model: RuchbahModel instance
            config: Model configuration
        """
        super().__init__()
        
        self.model = model
        self.config = config
        self.lm_head = _getattr_with_path(model, "lm_head")
        self.logits_processor = None
        
        self._verify_model_compatibility()
    
    def _verify_model_compatibility(self):
        """Verify model has required components for vLLM."""
        required_attrs = ["lm_head", "embed_tokens"]
        for attr in required_attrs:
            try:
                _getattr_with_path(self.model, attr)
            except AttributeError:
                logger.warning(f"Model missing {attr}, may cause issues with vLLM")
    
    def get_attention_blocks(self):
        """Get attention blocks for PagedAttention support.
        
        Returns:
            List of attention blocks with forward() method
        """
        attention_blocks = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'forward') and 'attention' in name.lower():
                attention_blocks.append((name, module))
        
        logger.info(f"Found {len(attention_blocks)} attention blocks for vLLM")
        return attention_blocks
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with vLLM.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            attention_mask: Attention mask
            kv_cache: Key-value cache (key_cache, value_cache)
            inputs_embeds: Input embeddings
            
        Returns:
            Dict with 'logits' and other outputs
        """
        batch_size, seq_len = input_ids.shape
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=input_ids.device
            )
        
        model_outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        
        hidden_states = model_outputs
        
        if isinstance(model_outputs, dict):
            hidden_states = model_outputs.get('last_hidden_state', model_outputs)
        
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
        }
    
    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        """Compute logits from hidden states.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            sampling_metadata: Sampling metadata from vLLM
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        logits = self.lm_head(hidden_states)
        
        if sampling_metadata is not None and hasattr(sampling_metadata, 'remove_padding'):
            logits = sampling_metadata.remove_padding(logits)
        
        return logits
    
    def load_kv_cache(self, kv_cache: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Load KV cache for inference.
        
        Args:
            kv_cache: Tuple of (key_cache, value_cache)
        """
        pass
    
    def get_kv_cache(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get current KV cache.
        
        Returns:
            Tuple of (key_cache, value_cache) or None
        """
        return None


class PiscesLxVLLMInputAdapter:
    """Adapter for converting vLLM inputs to PiscesL1 format."""
    
    def __init__(self, config):
        """Initialize input adapter.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.tokenizer = None
    
    def get_tokenizer(self):
        """Get or create tokenizer."""
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                if hasattr(self.config, 'tokenizer_name'):
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        return self.tokenizer
    
    def convert_vllm_input_to_model_input(
        self,
        prompt_token_ids: List[int],
        vllm_sampling_metadata: Any = None,
    ) -> Dict[str, Any]:
        """Convert vLLM input to model input format.
        
        Args:
            prompt_token_ids: Token IDs from vLLM
            vllm_sampling_metadata: Sampling metadata
            
        Returns:
            Dictionary with input_ids, attention_mask, position_ids
        """
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long)
        seq_len = len(prompt_token_ids)
        
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        
        attention_mask = torch.ones(1, seq_len, dtype=torch.bool)
        
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }
    
    def convert_model_output_to_vllm_output(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Convert model output to vLLM output format.
        
        Args:
            logits: Logits from model [1, seq_len, vocab_size]
            sampling_metadata: Sampling metadata
            
        Returns:
            Logits for vLLM sampling
        """
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        
        return logits


class PiscesLxVLLMConfigAdapter:
    """Adapter for converting vLLM config to PiscesL1 config."""
    
    @staticmethod
    def from_vllm_config(vllm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert vLLM config to PiscesL1 config format.
        
        Args:
            vllm_config: vLLM configuration dictionary
            
        Returns:
            PiscesL1 configuration dictionary
        """
        pisces_config = {
            'hidden_size': vllm_config.get('hidden_size', 4096),
            'intermediate_size': vllm_config.get('intermediate_size', 14336),
            'num_hidden_layers': vllm_config.get('num_hidden_layers', 32),
            'num_attention_heads': vllm_config.get('num_attention_heads', 32),
            'num_key_value_heads': vllm_config.get('num_key_value_heads', None),
            'vocab_size': vllm_config.get('vocab_size', 131072),
            'max_position_embeddings': vllm_config.get('max_model_len', 32768),
            'rms_norm_eps': vllm_config.get('rms_norm_eps', 1e-5),
            'torch_dtype': vllm_config.get('dtype', 'bfloat16'),
        }
        
        if 'rope_theta' in vllm_config:
            pisces_config['rope_theta'] = vllm_config['rope_theta']
        
        return pisces_config
    
    @staticmethod
    def to_vllm_config(pisces_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PiscesL1 config to vLLM config format.
        
        Args:
            pisces_config: PiscesL1 configuration dictionary
            
        Returns:
            vLLM configuration dictionary
        """
        vllm_config = {
            'hidden_size': pisces_config.get('hidden_size', 4096),
            'intermediate_size': pisces_config.get('intermediate_size', 14336),
            'num_hidden_layers': pisces_config.get('num_hidden_layers', 32),
            'num_attention_heads': pisces_config.get('num_attention_heads', 32),
            'num_key_value_heads': pisces_config.get('num_key_value_heads', None),
            'vocab_size': pisces_config.get('vocab_size', 131072),
            'max_model_len': pisces_config.get('max_position_embeddings', 32768),
            'rms_norm_eps': pisces_config.get('rms_norm_eps', 1e-5),
            'dtype': str(pisces_config.get('torch_dtype', 'bfloat16')),
        }
        
        return vllm_config


def convert_ruchbah_for_vllm(
    model_path: str,
    output_path: Optional[str] = None,
    save_tensor_parallel: bool = False,
) -> str:
    """Convert Ruchbah model checkpoint to vLLM-compatible format.
    
    Args:
        model_path: Path to Ruchbah checkpoint
        output_path: Output path (auto-generated if None)
        save_tensor_parallel: Save with tensor parallelism shards
        
    Returns:
        Path to converted checkpoint
    """
    from model.config import RuchbahConfig
    from model.modeling import RuchbahModel
    
    logger.info(f"Loading Ruchbah model from: {model_path}")
    
    config = RuchbahConfig.from_pretrained(model_path)
    model = RuchbahModel.from_pretrained(model_path)
    model.eval()
    
    if output_path is None:
        output_path = f"{model_path}_vllm"
    
    os.makedirs(output_path, exist_ok=True)
    
    config.save_pretrained(output_path)
    model.save_pretrained(
        output_path,
        save_tensor_parallel=save_tensor_parallel,
    )
    
    import json
    config_dict = config.to_dict()
    config_dict['_name_or_path'] = config.name_or_path or "piscesl1-ruchbah"
    
    with open(os.path.join(output_path, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.success(f"Model converted and saved to: {output_path}")
    logger.info("Model is now ready for vLLM inference")
    
    return output_path


def create_vllm_engine_from_native(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    quantization: Optional[str] = None,
) -> "PiscesLxVLLMEngine":
    """Create vLLM engine from native PiscesL1 model.
    
    Args:
        model_path: Path to PiscesL1 checkpoint
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory usage ratio
        max_model_len: Maximum sequence length
        quantization: Quantization method
        
    Returns:
        Initialized vLLM engine
    """
    converted_path = convert_ruchbah_for_vllm(model_path)
    
    config = PiscesLxVLLMConfig(
        model_path=converted_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        quantization=quantization,
    )
    
    engine = create_vllm_engine(config)
    engine.initialize()
    
    return engine
