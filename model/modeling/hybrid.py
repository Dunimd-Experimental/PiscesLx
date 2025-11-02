"""
Arctic Hybrid Block - Mamba-3 and Attention Integration

This module implements the hybrid architecture combining Mamba-3 State Space Model
with traditional attention mechanisms using intelligent gating fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import ArcticRMSNorm
from ..config import ArcticConfig
from .attention import ArcticAttention
from typing import Optional, Dict, Any
from utils.log.core import PiscesLxCoreLog
from .mamba3 import Mamba3Integration, Mamba3Config

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Hybrid", file_path="logs/ArcticCore.log")


class IntelligentGate(nn.Module):
    """
    Intelligent gating mechanism for fusing attention and Mamba-3 outputs.
    Adapts to sequence length, content complexity, and model state.
    """
    
    def __init__(self, d_model: int, gate_type: str = "learned"):
        super().__init__()
        self.d_model = d_model
        self.gate_type = gate_type
        
        if gate_type == "learned":
            # Learned gating with multiple features
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model * 3, d_model),  # attention_out, mamba_out, hidden_states
                nn.SiLU(),
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 2),  # Two gates: attention_weight, mamba_weight
                nn.Softmax(dim=-1)
            )
        elif gate_type == "adaptive":
            # Adaptive gating based on sequence statistics
            self.seq_stats_proj = nn.Linear(4, d_model // 4)  # mean, std, max, min
            self.content_proj = nn.Linear(d_model, d_model // 4)
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 8),
                nn.SiLU(),
                nn.Linear(d_model // 8, 2),
                nn.Softmax(dim=-1)
            )
            
    def _compute_sequence_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sequence-level statistics"""
        # x: [batch, seq_len, d_model]
        seq_mean = x.mean(dim=1)  # [batch, d_model]
        seq_std = x.std(dim=1)    # [batch, d_model]
        seq_max = x.max(dim=1)[0]  # [batch, d_model]
        seq_min = x.min(dim=1)[0]  # [batch, d_model]
        
        # Global statistics across dimensions
        stats = torch.stack([
            seq_mean.mean(dim=1),  # Global mean
            seq_std.mean(dim=1),   # Global std
            seq_max.mean(dim=1),   # Global max
            seq_min.mean(dim=1)    # Global min
        ], dim=-1)  # [batch, 4]
        
        return stats
        
    def forward(self, attention_out: torch.Tensor, mamba_out: torch.Tensor, 
                hidden_states: torch.Tensor, sequence_length: int) -> Dict[str, torch.Tensor]:
        """
        Compute intelligent gating weights
        
        Args:
            attention_out: Attention output [batch, seq_len, d_model]
            mamba_out: Mamba-3 output [batch, seq_len, d_model]
            hidden_states: Original hidden states [batch, seq_len, d_model]
            sequence_length: Current sequence length
            
        Returns:
            Dictionary with fused output and gate weights
        """
        batch_size = hidden_states.shape[0]
        
        if self.gate_type == "learned":
            # Global pooling for efficient gating
            attn_pooled = attention_out.mean(dim=1)  # [batch, d_model]
            mamba_pooled = mamba_out.mean(dim=1)     # [batch, d_model]
            hidden_pooled = hidden_states.mean(dim=1)  # [batch, d_model]
            
            # Concatenate features
            gate_input = torch.cat([attn_pooled, mamba_pooled, hidden_pooled], dim=-1)
            gate_weights = self.gate_proj(gate_input)  # [batch, 2]
            
            attn_weight = gate_weights[:, 0:1].unsqueeze(1)  # [batch, 1, 1]
            mamba_weight = gate_weights[:, 1:2].unsqueeze(1)  # [batch, 1, 1]
            
        elif self.gate_type == "adaptive":
            # Compute sequence statistics
            seq_stats = self._compute_sequence_stats(hidden_states)  # [batch, 4]
            seq_features = self.seq_stats_proj(seq_stats)  # [batch, d_model//4]
            
            # Content features
            content_features = hidden_states.mean(dim=1)  # [batch, d_model]
            content_features = self.content_proj(content_features)  # [batch, d_model//4]
            
            # Combine features
            combined_features = torch.cat([seq_features, content_features], dim=-1)
            gate_weights = self.gate_proj(combined_features)  # [batch, 2]
            
            attn_weight = gate_weights[:, 0:1].unsqueeze(1).unsqueeze(1)  # [batch, 1, 1]
            mamba_weight = gate_weights[:, 1:2].unsqueeze(1).unsqueeze(1)  # [batch, 1, 1]
            
        else:  # Fixed gating
            # Simple fixed weights based on sequence length
            if sequence_length > 4096:  # Long sequences favor Mamba-3
                attn_weight = torch.full((batch_size, 1, 1), 0.3, device=hidden_states.device)
                mamba_weight = torch.full((batch_size, 1, 1), 0.7, device=hidden_states.device)
            else:  # Short sequences favor attention
                attn_weight = torch.full((batch_size, 1, 1), 0.7, device=hidden_states.device)
                mamba_weight = torch.full((batch_size, 1, 1), 0.3, device=hidden_states.device)
        
        # Apply gating
        fused_output = attn_weight * attention_out + mamba_weight * mamba_out
        
        return {
            "fused_output": fused_output,
            "attention_weight": attn_weight.squeeze(),
            "mamba_weight": mamba_weight.squeeze(),
            "gate_type": self.gate_type
        }


class ArcticHybridBlock(nn.Module):
    """
    Arctic Hybrid Block combining Attention and Mamba-3 with intelligent fusion
    """
    
    def __init__(self, cfg: ArcticConfig, device=None, dtype=None, quantization_config=None):
        super().__init__()
        
        # Core components
        self.attention = ArcticAttention(cfg, device=device, dtype=dtype)
        self.mamba3_config = Mamba3Config(
            d_model=cfg.hidden_size,
            d_state=getattr(cfg, 'mamba3_d_state', 128),
            d_conv=getattr(cfg, 'mamba3_d_conv', 4),
            expand=getattr(cfg, 'mamba3_expand', 2),
            dt_rank=getattr(cfg, 'mamba3_dt_rank', 'auto'),
            use_trapezoidal=getattr(cfg, 'mamba3_use_trapezoidal', True),
            use_complex=getattr(cfg, 'mamba3_use_complex', True),
            use_mimo=getattr(cfg, 'mamba3_use_mimo', True)
        )
        self.mamba3 = Mamba3Integration(cfg.hidden_size, self.mamba3_config)
        
        # Intelligent gating
        gate_type = getattr(cfg, 'hybrid_gate_type', 'adaptive')
        self.intelligent_gate = IntelligentGate(cfg.hidden_size, gate_type)
        
        # Normalization layers
        self.norm_attention = ArcticRMSNorm(cfg.hidden_size)
        self.norm_mamba = ArcticRMSNorm(cfg.hidden_size)
        self.norm_fusion = ArcticRMSNorm(cfg.hidden_size)
        
        # MoE layer (retained for feedforward)
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            from ..moe import ArcticMoELayer as MoELayer
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            from ..moe_dynamic import ArcticDynamicMoELayer
            self.mlp = ArcticDynamicMoELayer(cfg, device=device, dtype=dtype)
        
        # Additional normalization
        self.norm_mlp = ArcticRMSNorm(cfg.hidden_size)
        
        # Residual connections
        self.residual_scale_attn = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        self.residual_scale_mamba = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        self.residual_scale_mlp = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        
        # Configuration
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        self.sequence_threshold = getattr(cfg, 'sequence_threshold', 4096)
        self.hybrid_layers = getattr(cfg, 'mamba3_layers', list(range(cfg.n_layer // 2, cfg.n_layer)))
        
        # Cache management
        self.cache_manager = None
        self.layer_idx = -1
        
        logger.info(f"Initialized ArcticHybridBlock with gate_type={gate_type}, "
                   f"sequence_threshold={self.sequence_threshold}")
    
    def set_cache_manager(self, cache_manager, layer_idx: int):
        """Set cache manager for efficient inference"""
        self.cache_manager = cache_manager
        self.layer_idx = layer_idx
        self.attention.cache_manager = cache_manager
        
    def _should_use_mamba3(self, seq_len: int) -> bool:
        """Determine whether to use Mamba-3 based on sequence length"""
        return seq_len > self.sequence_threshold and self.layer_idx in self.hybrid_layers
    
    def _forward_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor], 
                          past_key_values=None, use_cache=False):
        """Forward pass through attention mechanism"""
        x_norm = self.norm_attention(x)
        
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
                x_norm, mask, past_key_values=past_for_attn, 
                use_cache=True, cache_manager=self.cache_manager
            )
            if self.cache_manager is not None and self.layer_idx >= 0 and present_kv is not None:
                self.cache_manager.update_kv_cache(
                    self.layer_idx, present_kv[0], present_kv[1], 
                    current_pos=x_norm.shape[1], 
                    use_h2o=getattr(self.attention, 'use_h2o', False)
                )
            return attn_out, present_kv
        else:
            attn_out = self.attention(
                x_norm, mask, past_key_values=past_for_attn, 
                use_cache=False, cache_manager=self.cache_manager
            )
            return attn_out, None
    
    def _forward_mamba3(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass through Mamba-3"""
        x_norm = self.norm_mamba(x)
        mamba_out = self.mamba3(x_norm, mask)
        return mamba_out
    
    def _forward_mlp(self, x: torch.Tensor):
        """Forward pass through MoE MLP"""
        x_norm = self.norm_mlp(x)
        mlp_out, aux_loss = self.mlp(x_norm)
        return mlp_out, aux_loss
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values=None, use_cache=False) -> Dict[str, Any]:
        """
        Forward pass of Arctic Hybrid Block
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            past_key_values: Optional past key values for caching
            use_cache: Whether to use KV caching
            
        Returns:
            Dictionary with output and auxiliary information
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Determine architecture based on sequence length
        use_mamba3 = self._should_use_mamba3(seq_len)
        
        # Forward through attention (always used)
        attn_out, attn_cache = self._forward_attention(
            hidden_states, attention_mask, past_key_values, use_cache
        )
        
        if use_mamba3:
            # Forward through Mamba-3
            mamba_out = self._forward_mamba3(hidden_states, attention_mask)
            
            # Intelligent fusion
            fusion_result = self.intelligent_gate(
                attn_out, mamba_out, hidden_states, seq_len
            )
            
            # Apply residual connections with learned scales
            hybrid_out = hidden_states + self.residual_dropout(
                self.residual_scale_attn * fusion_result["fused_output"]
            )
            
            # Log fusion statistics
            logger.debug(f"Layer {self.layer_idx}: Sequence length {seq_len}, "
                        f"Attention weight: {fusion_result['attention_weight'].mean():.3f}, "
                        f"Mamba weight: {fusion_result['mamba_weight'].mean():.3f}")
            
        else:
            # Pure attention path
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