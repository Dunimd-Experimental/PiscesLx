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

import torch
from torch import nn
from typing import Optional
from .norms import ArcticRMSNorm
from .attention import ArcticAttention
from utils.log.core import PiscesLxCoreLog
from ..moe import ArcticMoELayer as MoELayer
from ..moe_dynamic import ArcticDynamicMoELayer

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Blocks", file_path="logs/ArcticCore.log")

class ArcticTransformerBlock(nn.Module):
    """
    Transformer block implementation for Arctic model with support for 
    Mixture-of-Experts (MoE), gradient checkpointing, and quantization.

    This block contains attention and MLP layers with pre-normalization,
    residual connections, and optional caching for inference.
    """

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None):
        """
        Initialize ArcticTransformerBlock.

        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place the module on.
            dtype: Data type for the module parameters.
            quantization_config: Configuration for model quantization.
        """
        super().__init__()
        self.attn = ArcticAttention(cfg, device=device, dtype=dtype)
        self.cache_manager = None
        self.layer_idx = -1

        # Select MoE layer based on configuration
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype, 
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            self.mlp = ArcticDynamicMoELayer(cfg, device=device, dtype=dtype)

        # Normalization layers
        self.norm1 = ArcticRMSNorm(cfg.hidden_size)      # Post-attention norm
        self.norm2 = ArcticRMSNorm(cfg.hidden_size)      # Post-MLP norm
        self.pre_norm1 = ArcticRMSNorm(cfg.hidden_size)  # Pre-attention norm
        self.pre_norm2 = ArcticRMSNorm(cfg.hidden_size)  # Pre-MLP norm

        # Residual connection scaling and dropout
        self.residual_scale = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))

        # Gradient checkpointing settings
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        self.adaptive_checkpointing = getattr(cfg, 'adaptive_gradient_checkpointing', True)
        self.memory_threshold_high = getattr(cfg, 'memory_threshold_high', 0.85)
        self.memory_threshold_low = getattr(cfg, 'memory_threshold_low', 0.60)
        self.checkpoint_frequency = getattr(cfg, 'checkpoint_frequency', 1)
        self.current_checkpoint_freq = self.checkpoint_frequency

        # Quantization configuration
        self.quantization_config = quantization_config

        # Apply quantization if configured
        if self.quantization_config is not None:
            try:
                import bitsandbytes as bnb
                layer_importance = self._get_layer_importance()
                
                def convert_linear_to_mixed_precision(module, layer_type='standard'):
                    """
                    Recursively converts linear layers to mixed precision based on importance.
                    
                    Args:
                        module: Module to process.
                        layer_type: Importance level ('critical', 'important', or 'standard').
                    """
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
                            # Choose quantization based on layer importance
                            if layer_importance == 'critical':
                                new_mod = bnb.nn.Linear8bit(
                                    child.in_features, child.out_features,
                                    bias=child.bias is not None,
                                    threshold=getattr(self.quantization_config, 'bnb_8bit_threshold', 6.0),
                                )
                            elif layer_importance == 'important':
                                new_mod = bnb.nn.Linear4bit(
                                    child.in_features, child.out_features,
                                    bias=child.bias is not None,
                                    quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                    compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.float16),
                                    compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                                )
                            else:
                                new_mod = bnb.nn.Linear4bit(
                                    child.in_features, child.out_features,
                                    bias=child.bias is not None,
                                    quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                    compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                                    compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                                )
                            setattr(module, name, new_mod)
                        else:
                            # Recurse into children with appropriate layer type
                            child_layer_type = self._get_child_layer_type(name, layer_type)
                            convert_linear_to_mixed_precision(child, child_layer_type)
                            
                convert_linear_to_mixed_precision(self)
            except Exception as e:
                logger.error(f"Mixed precision quantization failed: {e}")
                self._fallback_to_4bit_quantization()

    def _get_layer_importance(self):
        """
        Get the importance level for layer quantization.
        
        Returns:
            str: Layer importance level ('critical', 'important', or 'standard').
        """
        return getattr(self.quantization_config, 'layer_importance', 'standard')

    def _get_child_layer_type(self, child_name, parent_type):
        """
        Determine the importance type of a child layer based on its name.
        
        Args:
            child_name: Name of the child module.
            parent_type: Importance type of the parent module.
            
        Returns:
            str: Determined importance level.
        """
        if 'attn' in child_name.lower() or 'attention' in child_name.lower():
            return 'critical'
        elif 'mlp' in child_name.lower() or 'feedforward' in child_name.lower():
            return 'important'
        else:
            return parent_type

    def _fallback_to_4bit_quantization(self):
        """
        Fallback method to apply uniform 4-bit quantization when mixed precision fails.
        """
        try:
            import bitsandbytes as bnb
            
            def convert_linear_to_4bit(module):
                """
                Recursively converts all linear layers to 4-bit quantized versions.
                
                Args:
                    module: Module to process.
                """
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        new_mod = bnb.nn.Linear4bit(
                            child.in_features, child.out_features,
                            bias=child.bias is not None,
                            quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                            compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                            compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                        )
                        setattr(module, name, new_mod)
                    else:
                        convert_linear_to_4bit(child)
                        
            convert_linear_to_4bit(self)
            logger.info("Fallback to 4-bit quantization successful")
        except Exception as e:
            logger.error(f"Fallback 4-bit quantization also failed: {e}")

    def _should_use_checkpoint(self):
        """
        Determine whether gradient checkpointing should be used based on memory usage.
        
        Returns:
            bool: True if checkpointing should be used, False otherwise.
        """
        # Skip adaptive logic if checkpointing is disabled or not adaptive
        if not self.use_checkpoint or not self.adaptive_checkpointing:
            return self.use_checkpoint
            
        try:
            if torch.cuda.is_available():
                # Calculate current GPU memory usage ratio
                allocated = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_usage = allocated / total_memory
                
                # Adjust checkpoint frequency based on memory pressure
                if memory_usage > self.memory_threshold_high:
                    self.current_checkpoint_freq = max(1, self.checkpoint_frequency // 2)
                    return True
                elif memory_usage < self.memory_threshold_low:
                    self.current_checkpoint_freq = self.checkpoint_frequency * 2
                    return False
                else:
                    self.current_checkpoint_freq = self.checkpoint_frequency
                    # Randomly decide based on frequency setting
                    import torch as _t
                    return (self.checkpoint_frequency <= 1) or (_t.randint(0, self.checkpoint_frequency, (1,)).item() == 0)
            else:
                # Default behavior when CUDA is not available
                return self.use_checkpoint
        except Exception as e:
            logger.error(f"Adaptive checkpointing memory check failed: {e}")
            return self.use_checkpoint

    def _apply_with_checkpoint(self, x, mask, past_key_values=None, use_cache=False):
        """
        Apply the transformer block with optional gradient checkpointing.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs from previous forward passes.
            use_cache: Whether to use and update key/value cache.
            
        Returns:
            Output tensor(s) from the transformer block.
        """
        import torch.utils.checkpoint as cp
        
        # Prepare past key values for attention
        attn_past_key_values = past_key_values if past_key_values is not None else None
        should_checkpoint = self._should_use_checkpoint()
        
        def _inner(xc, kv=None):
            """Inner function for checkpointing."""
            return self._forward_core(xc, mask, kv, use_cache)
            
        # Apply checkpointing during training if needed
        if should_checkpoint and self.training:
            out = cp.checkpoint(_inner, x, attn_past_key_values, use_reentrant=False)
        else:
            out = _inner(x, attn_past_key_values)
            
        return out

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs from previous forward passes.
            use_cache: Whether to use and update key/value cache.
            
        Returns:
            Tuple of (output_tensor, auxiliary_loss) or 
            (output_tensor, auxiliary_loss, updated_cache) if use_cache=True.
        """
        return self._apply_with_checkpoint(x, mask, past_key_values, use_cache)

    def _forward_core(self, x, mask, attn_past_key_values=None, use_cache=False):
        """
        Core forward computation without checkpointing wrapper.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            mask: Attention mask tensor.
            attn_past_key_values: Key/value cache for attention mechanism.
            use_cache: Whether to use and update key/value cache.
            
        Returns:
            Tuple of (output_tensor, auxiliary_loss) or 
            (output_tensor, auxiliary_loss, updated_cache) if use_cache=True.
        """
        # Attention block with residual connection
        residual = x
        x_norm = self.pre_norm1(x)
        attn_cache = None
        past_for_attn = attn_past_key_values
        
        # Retrieve cached keys/values if using cache and manager exists
        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, attn_past_key_values)
            if got is not None:
                past_for_attn = got
                
        # Compute attention output
        if use_cache:
            attn_out, present_kv = self.attn(x_norm, mask, past_key_values=past_for_attn, use_cache=True, cache_manager=self.cache_manager)
            # Update cache if manager exists and valid key/value pairs returned
            if self.cache_manager is not None and self.layer_idx >= 0 and present_kv is not None:
                self.cache_manager.update_kv_cache(self.layer_idx, present_kv[0], present_kv[1], current_pos=x_norm.shape[1], use_h2o=getattr(self.attn, 'use_h2o', False))
            attn_cache = present_kv
        else:
            attn_out = self.attn(x_norm, mask, past_key_values=past_for_attn, use_cache=False, cache_manager=self.cache_manager)
            
        # Apply scaled residual connection and normalization
        x_out = residual + self.residual_dropout(self.residual_scale * attn_out)
        x_out = self.norm1(x_out)
        
        # MLP block with residual connection
        residual = x_out
        x_norm = self.pre_norm2(x_out)
        mlp_out, aux_loss = self.mlp(x_norm)
        x_out = residual + self.residual_dropout(self.residual_scale * mlp_out)
        x_out = self.norm2(x_out)
        
        # Return cache along with outputs if using cache
        if use_cache:
            return x_out, aux_loss, attn_cache
        return x_out, aux_loss
