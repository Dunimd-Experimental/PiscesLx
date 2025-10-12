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

import torch
from torch import nn
from ..moe import MoELayer
from typing import Optional
from .norms import ArcticRMSNorm
from .attention import ArcticAttention
from utils.log.core import PiscesLxCoreLog
from ..moe_dynamic import ArcticDynamicMoELayer

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Blocks")

class ArcticTransformerBlock(nn.Module):
    def __init__(self, cfg, device=None, dtype=None, quantization_config=None):
        super().__init__()
        self.attn = ArcticAttention(cfg, device=device, dtype=dtype)
        self.cache_manager = None
        self.layer_idx = -1

        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype, 
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            self.mlp = ArcticDynamicMoELayer(cfg, device=device, dtype=dtype)
        self.norm1 = ArcticRMSNorm(cfg.hidden_size)
        self.norm2 = ArcticRMSNorm(cfg.hidden_size)
        self.pre_norm1 = ArcticRMSNorm(cfg.hidden_size)
        self.pre_norm2 = ArcticRMSNorm(cfg.hidden_size)
        self.residual_scale = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        self.adaptive_checkpointing = getattr(cfg, 'adaptive_gradient_checkpointing', True)
        self.memory_threshold_high = getattr(cfg, 'memory_threshold_high', 0.85)
        self.memory_threshold_low = getattr(cfg, 'memory_threshold_low', 0.60)
        self.checkpoint_frequency = getattr(cfg, 'checkpoint_frequency', 1)
        self.current_checkpoint_freq = self.checkpoint_frequency
        self.quantization_config = quantization_config

        if self.quantization_config is not None:
            try:
                import bitsandbytes as bnb
                layer_importance = self._get_layer_importance()
                def convert_linear_to_mixed_precision(module, layer_type='standard'):
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
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
                            child_layer_type = self._get_child_layer_type(name, layer_type)
                            convert_linear_to_mixed_precision(child, child_layer_type)
                convert_linear_to_mixed_precision(self)
            except Exception as e:
                logger.error(f"Mixed precision quantization failed: {e}")
                self._fallback_to_4bit_quantization()

    def _get_layer_importance(self):
        return getattr(self.quantization_config, 'layer_importance', 'standard')

    def _get_child_layer_type(self, child_name, parent_type):
        if 'attn' in child_name.lower() or 'attention' in child_name.lower():
            return 'critical'
        elif 'mlp' in child_name.lower() or 'feedforward' in child_name.lower():
            return 'important'
        else:
            return parent_type

    def _fallback_to_4bit_quantization(self):
        try:
            import bitsandbytes as bnb
            def convert_linear_to_4bit(module):
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
        if not self.use_checkpoint or not self.adaptive_checkpointing:
            return self.use_checkpoint
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_usage = allocated / total_memory
                if memory_usage > self.memory_threshold_high:
                    self.current_checkpoint_freq = max(1, self.checkpoint_frequency // 2)
                    return True
                elif memory_usage < self.memory_threshold_low:
                    self.current_checkpoint_freq = self.checkpoint_frequency * 2
                    return False
                else:
                    self.current_checkpoint_freq = self.checkpoint_frequency
                    import torch as _t
                    return (self.checkpoint_frequency <= 1) or (_t.randint(0, self.checkpoint_frequency, (1,)).item() == 0)
            else:
                return self.use_checkpoint
        except Exception as e:
            logger.error(f"Adaptive checkpointing memory check failed: {e}")
            return self.use_checkpoint

    def _apply_with_checkpoint(self, x, mask, past_key_values=None, use_cache=False):
        import torch.utils.checkpoint as cp
        attn_past_key_values = past_key_values if past_key_values is not None else None
        should_checkpoint = self._should_use_checkpoint()
        def _inner(xc, kv=None):
            return self._forward_core(xc, mask, kv, use_cache)
        if should_checkpoint and self.training:
            out = cp.checkpoint(_inner, x, attn_past_key_values, use_reentrant=False)
        else:
            out = _inner(x, attn_past_key_values)
        return out

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        return self._apply_with_checkpoint(x, mask, past_key_values, use_cache)

    def _forward_core(self, x, mask, attn_past_key_values=None, use_cache=False):
        residual = x
        x_norm = self.pre_norm1(x)
        attn_cache = None
        past_for_attn = attn_past_key_values
        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, attn_past_key_values)
            if got is not None:
                past_for_attn = got
        if use_cache:
            attn_out, present_kv = self.attn(x_norm, mask, past_key_values=past_for_attn, use_cache=True, cache_manager=self.cache_manager)
            if self.cache_manager is not None and self.layer_idx >= 0 and present_kv is not None:
                self.cache_manager.update_kv_cache(self.layer_idx, present_kv[0], present_kv[1], current_pos=x_norm.shape[1], use_h2o=getattr(self.attn, 'use_h2o', False))
            attn_cache = present_kv
        else:
            attn_out = self.attn(x_norm, mask, past_key_values=past_for_attn, use_cache=False, cache_manager=self.cache_manager)
        x_out = residual + self.residual_dropout(self.residual_scale * attn_out)
        x_out = self.norm1(x_out)
        residual = x_out
        x_norm = self.pre_norm2(x_out)
        mlp_out, aux_loss = self.mlp(x_norm)
        x_out = residual + self.residual_dropout(self.residual_scale * mlp_out)
        x_out = self.norm2(x_out)
        if use_cache:
            return x_out, aux_loss, attn_cache
        return x_out, aux_loss