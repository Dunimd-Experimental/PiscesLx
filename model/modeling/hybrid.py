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
Hybrid transformer block module combining attention and Mamba-3.

This module implements a hybrid architecture that combines transformer attention
mechanisms with Mamba-3 state space models, using gating mechanisms to adaptively
fuse their outputs based on sequence characteristics.
"""

import torch
import torch.nn as nn
from .norms import RuchbahRMSNorm
from ..config import RuchbahConfig
from .attention import RuchbahAttention
from typing import Optional, Dict, Any
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from .mamba3 import RuchbahMamba3Integration, RuchbahMamba3Config

logger = PiscesLxCoreLog("Ruchbah.Core.Modeling.Hybrid", file_path="logs/RuchbahCore.log")

class RuchbahProgressiveHybridGate(nn.Module):
    """
    Progressive hybrid gating mechanism for stable Transformer-Mamba3 training.
    
    Implements gradual transition from Transformer-only to full hybrid mode
    based on training progress and sequence characteristics.
    """
    
    def __init__(self, d_model: int, total_steps: int = 100000):
        super().__init__()
        self.d_model = d_model
        self.total_steps = total_steps
        
        self.seq_stats_proj = nn.Linear(4, d_model // 4)
        self.content_proj = nn.Linear(d_model, d_model // 4)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 8),
            nn.SiLU(),
            nn.Linear(d_model // 8, 2),
            nn.Softmax(dim=-1)
        )
        
        self.register_buffer('current_step', torch.tensor(0))
        self.hybrid_ratio = nn.Parameter(torch.tensor(0.0))
        
        self.fixed_short = torch.tensor([[0.7, 0.3]])
        self.fixed_long = torch.tensor([[0.3, 0.7]])
    
    def _compute_sequence_stats(self, x: torch.Tensor) -> torch.Tensor:
        seq_mean = x.mean(dim=1)
        seq_std = x.std(dim=1)
        seq_max = x.max(dim=1)[0]
        seq_min = x.min(dim=1)[0]
        stats = torch.stack([
            seq_mean.mean(dim=1),
            seq_std.mean(dim=1),
            seq_max.mean(dim=1),
            seq_min.mean(dim=1)
        ], dim=-1)
        return stats
    
    def forward(
        self,
        attention_out: torch.Tensor,
        mamba_out: torch.Tensor,
        hidden_states: torch.Tensor,
        sequence_length: int
    ) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        
        if self.training:
            self.current_step += 1
        
        progress = min(1.0, self.current_step.item() / self.total_steps)
        current_hybrid_ratio = torch.sigmoid(self.hybrid_ratio) * progress
        
        if sequence_length > 4096:
            fixed_weights = self.fixed_long.to(hidden_states.device)
        else:
            fixed_weights = self.fixed_short.to(hidden_states.device)
        
        seq_stats = self._compute_sequence_stats(hidden_states)
        seq_features = self.seq_stats_proj(seq_stats)
        content_features = hidden_states.mean(dim=1)
        content_features = self.content_proj(content_features)
        combined_features = torch.cat([seq_features, content_features], dim=-1)
        adaptive_weights = self.gate_proj(combined_features)
        
        final_attn = fixed_weights[0, 0] * (1 - current_hybrid_ratio) + adaptive_weights[:, 0] * current_hybrid_ratio
        final_mamba = fixed_weights[0, 1] * (1 - current_hybrid_ratio) + adaptive_weights[:, 1] * current_hybrid_ratio
        
        attn_weight = final_attn.unsqueeze(0).unsqueeze(0) if final_attn.dim() == 0 else final_attn.unsqueeze(0)
        mamba_weight = final_mamba.unsqueeze(0).unsqueeze(0) if final_mamba.dim() == 0 else final_mamba.unsqueeze(0)
        
        fused_output = attn_weight * attention_out + mamba_weight * mamba_out
        
        return {
            "fused_output": fused_output,
            "attention_weight": attn_weight.squeeze() if attn_weight.dim() > 1 else attn_weight,
            "mamba_weight": mamba_weight.squeeze() if mamba_weight.dim() > 1 else mamba_weight,
            "gate_type": "progressive_hybrid",
            "hybrid_ratio": current_hybrid_ratio
        }


class RuchbahHybridBlock(nn.Module):
    """
    Hybrid transformer block combining attention and Mamba-3.

    Implements a hybrid architecture that uses attention for all sequences and
    conditionally applies Mamba-3 for long sequences. Outputs are fused using
    an intelligent gating mechanism.
    """

    def __init__(self, cfg: RuchbahConfig, device=None, dtype=None, quantization_config=None):
        """
        Initialize the hybrid block.

        Args:
            cfg (RuchbahConfig): Configuration object containing model hyperparameters.
            device (torch.device, optional): Device to place the module on.
            dtype (torch.dtype, optional): Data type for the module parameters.
            quantization_config (object, optional): Configuration for model quantization.
                Currently not used but kept for API compatibility.
        """
        super().__init__()

        # Core components: attention and Mamba-3
        self.attention = RuchbahAttention(cfg, device=device, dtype=dtype)
        self.mamba3_config = RuchbahMamba3Config(
            d_model=cfg.hidden_size,
            d_state=getattr(cfg, 'mamba3_d_state', 128),
            d_conv=getattr(cfg, 'mamba3_d_conv', 4),
            expand=getattr(cfg, 'mamba3_expand', 2),
            dt_rank=getattr(cfg, 'mamba3_dt_rank', 'auto'),
            use_trapezoidal=getattr(cfg, 'mamba3_use_trapezoidal', True),
            use_complex=getattr(cfg, 'mamba3_use_complex', True),
            use_mimo=getattr(cfg, 'mamba3_use_mimo', True)
        )
        self.mamba3 = RuchbahMamba3Integration(cfg.hidden_size, self.mamba3_config)

        # Gating mechanism for fusion
        self.progressive_gate = RuchbahProgressiveHybridGate(cfg.hidden_size, total_steps=getattr(cfg, 'hybrid_total_steps', 100000))

        # Normalization layers
        self.norm_attention = RuchbahRMSNorm(cfg.hidden_size)
        self.norm_mamba = RuchbahRMSNorm(cfg.hidden_size)
        self.norm_fusion = RuchbahRMSNorm(cfg.hidden_size)

        # MoE layer for feedforward network
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            from ..moe import RuchbahMoELayer as MoELayer
            self.mlp = MoELayer(
                cfg,
                device=device,
                dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            from ..moe_dynamic import RuchbahDynamicMoELayer
            self.mlp = RuchbahDynamicMoELayer(cfg, device=device, dtype=dtype)

        # Additional normalization for MLP
        self.norm_mlp = RuchbahRMSNorm(cfg.hidden_size)

        # Residual connection scaling factors: (2 * n_layers)^(-0.5)
        self.residual_scale_attn = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        self.residual_scale_mamba = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        self.residual_scale_mlp = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)

        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))

        # Configuration parameters
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        self.sequence_threshold = getattr(cfg, 'sequence_threshold', 4096)
        self.hybrid_layers = getattr(
            cfg, 'mamba3_layers', list(range(cfg.n_layer // 2, cfg.n_layer))
        )

        # Cache management
        self.cache_manager = None
        self.layer_idx = -1

        logger.info(
            f"Initialized RuchbahHybridBlock with gate_type={gate_type}, "
            f"sequence_threshold={self.sequence_threshold}"
        )

    def set_cache_manager(self, cache_manager, layer_idx: int):
        """
        Set cache manager for efficient inference.

        Args:
            cache_manager: Cache manager instance for KV cache management.
            layer_idx (int): Index of this layer in the model.
        """
        self.cache_manager = cache_manager
        self.layer_idx = layer_idx
        self.attention.cache_manager = cache_manager

    def _should_use_mamba3(self, seq_len: int) -> bool:
        """
        Determine whether to use Mamba-3 based on sequence length and layer index.

        Args:
            seq_len (int): Current sequence length.

        Returns:
            bool: True if Mamba-3 should be used, False otherwise.
        """
        return seq_len > self.sequence_threshold and self.layer_idx in self.hybrid_layers

    def _forward_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_key_values=None,
        use_cache=False
    ):
        """
        Forward pass through attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            mask (Optional[torch.Tensor]): Attention mask tensor.
            past_key_values (tuple, optional): Cached key/value pairs from previous passes.
            use_cache (bool): Whether to use and update key/value cache.

        Returns:
            tuple: If use_cache=True, returns (attention_output, present_key_values).
                Otherwise returns (attention_output, None).
        """
        x_norm = self.norm_attention(x)

        # Retrieve cached keys/values if using cache and manager exists
        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, past_key_values)
            if got is not None:
                past_for_attn = got
            else:
                past_for_attn = past_key_values
        else:
            past_for_attn = past_key_values

        if use_cache:
            attn_out, present_kv = self.attention(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=True,
                cache_manager=self.cache_manager
            )
            # Update cache if manager exists and valid key/value pairs returned
            if self.cache_manager is not None and self.layer_idx >= 0 and present_kv is not None:
                self.cache_manager.update_kv_cache(
                    self.layer_idx,
                    present_kv[0],
                    present_kv[1],
                    current_pos=x_norm.shape[1],
                    use_h2o=getattr(self.attention, 'use_h2o', False)
                )
            return attn_out, present_kv
        else:
            attn_out = self.attention(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=False,
                cache_manager=self.cache_manager
            )
            return attn_out, None

    def _forward_mamba3(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass through Mamba-3 state space model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            mask (Optional[torch.Tensor]): Optional attention mask.

        Returns:
            torch.Tensor: Mamba-3 output tensor of shape [batch, seq_len, d_model].
        """
        x_norm = self.norm_mamba(x)
        mamba_out = self.mamba3(x_norm, mask)
        return mamba_out

    def _forward_mlp(self, x: torch.Tensor):
        """
        Forward pass through MoE MLP layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].

        Returns:
            tuple: (mlp_output, auxiliary_loss) where mlp_output has shape
                [batch, seq_len, d_model] and auxiliary_loss is a scalar tensor.
        """
        x_norm = self.norm_mlp(x)
        mlp_out, aux_loss = self.mlp(x_norm)
        return mlp_out, aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache=False
    ) -> Dict[str, Any]:
        """
        Forward pass of the hybrid block.

        Processes input through attention (always) and conditionally through Mamba-3
        for long sequences. Outputs are fused using intelligent gating, then passed
        through MoE feedforward network.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            attention_mask (Optional[torch.Tensor]): Optional attention mask tensor.
            past_key_values (tuple, optional): Cached key/value pairs from previous passes.
            use_cache (bool): Whether to use and update key/value cache.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - output: Final output tensor [batch, seq_len, d_model]
                - aux_loss: Auxiliary loss from MoE layer
                - use_mamba3: Boolean indicating if Mamba-3 was used
                - fusion_stats: Dictionary with fusion statistics (gate weights, etc.)
                - sequence_length: Current sequence length
                - past_key_values: Cached key/value pairs (if use_cache=True)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Determine whether to use Mamba-3 based on sequence length
        use_mamba3 = self._should_use_mamba3(seq_len)

        # Forward through attention (always used)
        attn_out, attn_cache = self._forward_attention(
            hidden_states, attention_mask, past_key_values, use_cache
        )

        if use_mamba3:
            # Forward through Mamba-3
            mamba_out = self._forward_mamba3(hidden_states, attention_mask)

            # Intelligent fusion of attention and Mamba outputs
            fusion_result = self.progressive_gate(
                attn_out, mamba_out, hidden_states, seq_len
            )

            # Apply residual connection with learned scaling
            hybrid_out = hidden_states + self.residual_dropout(
                self.residual_scale_attn * fusion_result["fused_output"]
            )

            # Log fusion statistics for debugging
            logger.debug(
                f"Layer {self.layer_idx}: Sequence length {seq_len}, "
                f"Attention weight: {fusion_result['attention_weight'].mean():.3f}, "
                f"Mamba weight: {fusion_result['mamba_weight'].mean():.3f}, "
                f"Hybrid ratio: {fusion_result.get('hybrid_ratio', 0):.3f}"
            )

        else:
            # Pure attention path (no Mamba-3)
            hybrid_out = hidden_states + self.residual_dropout(
                self.residual_scale_attn * attn_out
            )

            fusion_result = {
                "attention_weight": torch.ones(batch_size),
                "mamba_weight": torch.zeros(batch_size),
                "gate_type": "attention_only"
            }

        # Normalize after hybrid fusion
        hybrid_out = self.norm_fusion(hybrid_out)

        # Feedforward network (MoE)
        mlp_out, aux_loss = self._forward_mlp(hybrid_out)

        # Final residual connection
        output = hybrid_out + self.residual_dropout(
            self.residual_scale_mlp * mlp_out
        )

        # Final normalization
        output = self.norm_mlp(output)

        result = {
            "output": output,
            "aux_loss": aux_loss,
            "use_mamba3": use_mamba3,
            "fusion_stats": fusion_result,
            "sequence_length": seq_len
        }

        if use_cache:
            result["past_key_values"] = attn_cache

        return result
