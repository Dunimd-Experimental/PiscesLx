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
FP8 Training Support for PiscesL1 using Transformer Engine.

FP8 (8-bit Floating Point) training provides:
- 2x memory savings compared to BF16
- 1.5-2x throughput improvement
- Minimal accuracy loss (<1%)

Key Components:
- FP8 linear layers
- FP8 attention
- Dynamic loss scaling
- Graceful fallback to BF16

Requirements:
- NVIDIA GPU with Hopper architecture (H100) or newer
- CUDA 12.0+
- Transformer Engine: pip install transformer-engine[pytorch]
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PiscesLxCoreLog
logger = PiscesLxCoreLog("pisceslx.tools.train.fp8_train")


@dataclass
class PiscesLxFP8Config:
    """FP8 training configuration."""
    
    fp8_format: str = "E4M3"
    fp8_recipe: str = "delayed_scaling"
    amax_compute_algo: str = "max"
    amax_history_len: int = 1024
    scaling_factor_compute_algo: str = "max"
    use_fp8: bool = True
    fallback_to_bf16: bool = True
    
    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxFP8Config":
        config_dict = {}
        
        if getattr(args, "fp8", None):
            config_dict["use_fp8"] = args.fp8
        
        if getattr(args, "fp8_format", None):
            config_dict["fp8_format"] = args.fp8_format
        
        if getattr(args, "fp8_recipe", None):
            config_dict["fp8_recipe"] = args.fp8_recipe
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fp8_format": self.fp8_format,
            "fp8_recipe": self.fp8_recipe,
            "amax_compute_algo": self.amax_compute_algo,
            "amax_history_len": self.amax_history_len,
            "use_fp8": self.use_fp8,
        }


class PiscesLxFP8Linear(nn.Module):
    """FP8-compatible linear layer.
    
    Uses Transformer Engine's FP8 linear when available,
    falls back to standard linear for unsupported hardware.
    """
    
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, fp8: bool = True):
        """Initialize FP8 linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Use bias term
            fp8: Use FP8 precision
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8 = fp8
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.fp8_linear = None
        self._use_fp8 = False
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_correct_fan(self.weight, 'fan_in')
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _try_init_fp8(self):
        """Initialize FP8 linear if hardware supports it."""
        if self.fp8_linear is not None:
            return
        
        try:
            import transformer_engine.pytorch as te
            self.fp8_linear = te.Linear(
                self.in_features,
                self.out_features,
                bias=self.bias is not None,
            )
            self._use_fp8 = True
            logger.debug("FP8 linear initialized successfully")
        except (ImportError, RuntimeError) as e:
            self._use_fp8 = False
            if self.fp8:
                logger.warning(f"FP8 not available, using BF16: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, *, in_features]
            
        Returns:
            Output tensor [B, *, out_features]
        """
        if self.fp8:
            self._try_init_fp8()
        
        if self._use_fp8 and self.fp8_linear is not None:
            return self.fp8_linear(x)
        else:
            return F.linear(x, self.weight, self.bias)


class PiscesLxFP8Attention(nn.Module):
    """FP8-compatible multi-head attention.
    
    Uses Transformer Engine's FP8 attention when available.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 attention_dropout: float = 0.0, bias: bool = True,
                 fp8: bool = True):
        """Initialize FP8 attention.
        
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout rate
            bias: Use bias in projections
            fp8: Use FP8 precision
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        
        self.head_dim = hidden_size // num_attention_heads
        
        self.query = PiscesLxFP8Linear(hidden_size, hidden_size, bias=bias, fp8=fp8)
        self.key = PiscesLxFP8Linear(hidden_size, hidden_size, bias=bias, fp8=fp8)
        self.value = PiscesLxFP8Linear(hidden_size, hidden_size, bias=bias, fp8=fp8)
        
        self.output = PiscesLxFP8Linear(hidden_size, hidden_size, bias=bias, fp8=fp8)
        
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.fp8_attention = None
        self._use_fp8 = False
        
        self._init_fp8_attention()
    
    def _init_fp8_attention(self):
        """Initialize FP8 attention if available."""
        try:
            import transformer_engine.pytorch as te
            self.fp8_attention = te.Attention(
                self.hidden_size,
                self.num_attention_heads,
                self.head_dim,
                attention_dropout=self.attention_dropout,
            )
            self._use_fp8 = True
        except (ImportError, RuntimeError):
            self._use_fp8 = False
    
    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, attention_mask: torch.Tensor = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            query: Query tensor [B, T, H]
            key: Key tensor [B, T, H]
            value: Value tensor [B, T, H]
            attention_mask: Attention mask [B, T] or None
            need_weights: Return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size = query.shape[0]
        
        if self._use_fp8 and self.fp8_attention is not None:
            q = self.query(query)
            k = self.key(key)
            v = self.value(value)
            
            q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            k = k.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            v = v.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            context, _ = self.fp8_attention(
                q, k, v,
                attention_mask=attention_mask,
            )
            
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, -1, self.hidden_size)
            
        else:
            q = self.query(query)
            k = self.key(key)
            v = self.value(value)
            
            q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, v)
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, -1, self.hidden_size)
        
        output = self.output(context)
        
        if need_weights:
            return output, attn_weights
        return output, None


class PiscesLxFP8MLP(nn.Module):
    """FP8-compatible feed-forward network."""
    
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0,
                 bias: bool = True, fp8: bool = True):
        """Initialize FP8 MLP.
        
        Args:
            hidden_size: Hidden dimension
            mlp_ratio: Expansion ratio
            bias: Use bias
            fp8: Use FP8 precision
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * mlp_ratio)
        
        self.gate_proj = PiscesLxFP8Linear(hidden_size, self.intermediate_size, bias=False, fp8=fp8)
        self.up_proj = PiscesLxFP8Linear(hidden_size, self.intermediate_size, bias=bias, fp8=fp8)
        self.down_proj = PiscesLxFP8Linear(self.intermediate_size, hidden_size, bias=bias, fp8=fp8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, H]
            
        Returns:
            Output tensor [B, T, H]
        """
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class PiscesLxFP8TransformerBlock(nn.Module):
    """FP8-compatible transformer block."""
    
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 mlp_ratio: float = 4.0, attention_dropout: float = 0.0,
                 dropout: float = 0.0, bias: bool = True,
                 fp8: bool = True):
        """Initialize FP8 transformer block.
        
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            attention_dropout: Attention dropout rate
            dropout: Dropout rate
            bias: Use bias in projections
            fp8: Use FP8 precision
        """
        super().__init__()
        
        self.attention = PiscesLxFP8Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            fp8=fp8,
        )
        
        self.mlp = PiscesLxFP8MLP(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            bias=bias,
            fp8=fp8,
        )
        
        self.input_norm = nn.LayerNorm(hidden_size)
        self.post_attention_norm = nn.LayerNorm(hidden_size)
        self.post_mlp_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, H]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Output tensor [B, T, H]
        """
        residual = x
        
        x = self.input_norm(x)
        attn_output, _ = self.attention(x, x, x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = residual + attn_output
        
        residual = x
        x = self.post_attention_norm(x)
        mlp_output = self.mlp(x)
        mlp_output = self.dropout(mlp_output)
        x = residual + mlp_output
        
        x = self.post_mlp_norm(x)
        
        return x


class PiscesLxFP8Model(nn.Module):
    """Complete FP8-compatible transformer model."""
    
    def __init__(self, hidden_size: int, num_layers: int,
                 num_attention_heads: int, vocab_size: int,
                 mlp_ratio: float = 4.0, attention_dropout: float = 0.0,
                 dropout: float = 0.0, max_position_embeddings: int = 32768,
                 fp8: bool = True):
        """Initialize FP8 model.
        
        Args:
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            vocab_size: Vocabulary size
            mlp_ratio: MLP expansion ratio
            attention_dropout: Attention dropout rate
            dropout: Dropout rate
            max_position_embeddings: Maximum sequence length
            fp8: Use FP8 precision
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            PiscesLxFP8TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                fp8=fp8,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.zeros_(self.lm_head.weight)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            labels: Target labels [B, T]
            
        Returns:
            Output dict with loss and logits
        """
        x = self.embed_tokens(input_ids)
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        return {"loss": loss, "logits": logits}


class PiscesLxFP8Trainer:
    """FP8 training manager."""
    
    def __init__(self, config: Optional[PiscesLxFP8Config] = None):
        """Initialize FP8 trainer.
        
        Args:
            config: FP8 configuration
        """
        self.config = config or PiscesLxFP8Config()
        self._check_hardware_support()
    
    def _check_hardware_support(self):
        """Check if hardware supports FP8."""
        self.supported = False
        self.reason = ""
        
        if not torch.cuda.is_available():
            self.reason = "CUDA not available"
            return
        
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            self.reason = f"FP8 requires Hopper (9.0+), got {capability}"
            return
        
        try:
            import transformer_engine.pytorch as te
            self.supported = True
            logger.info(f"FP8 supported (CUDA {capability})")
        except ImportError:
            self.reason = "Transformer Engine not installed"
            logger.warning(self.reason)
    
    def get_gradScaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Get gradient scaler for FP8 training."""
        if not self.supported:
            return None
        
        return torch.cuda.amp.GradScaler(
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
    
    def get_autocast_context(self):
        """Get FP8 autocast context manager.
        
        Returns:
            Context manager for FP8 training
        """
        if self.supported and self.config.use_fp8:
            try:
                import transformer_engine.pytorch as te
                return te.fp8_autocast(
                    enabled=True,
                    fp8_format=self.config.fp8_format,
                    amax_compute_algo=self.config.amax_compute_algo,
                    amax_history_len=self.config.amax_history_len,
                )
            except ImportError:
                pass
        
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        
        return torch.cuda.amp.autocast(dtype=torch.float16)
    
    def train_step(self, model: nn.Module, batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
        """Perform one training step with FP8.
        
        Args:
            model: FP8 model
            batch: Input batch
            optimizer: Optimizer
            scaler: Gradient scaler
            
        Returns:
            Loss dict
        """
        model.train()
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        
        input_ids = batch.get("input_ids")
        labels = batch.get("labels", input_ids)
        
        with self.get_autocast_context():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.get("loss")
            
            if loss is None:
                return {"loss": 0.0}
            
            total_loss = loss
        
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def convert_model_to_fp8(self, model: nn.Module) -> nn.Module:
        """Convert existing model to FP8.
        
        Args:
            model: Original model
            
        Returns:
            FP8-compatible model
        """
        if not self.supported:
            logger.warning(f"Cannot convert: {self.reason}")
            return model
        
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                fp8_linear = PiscesLxFP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    fp8=True,
                )
                fp8_linear.weight = module.weight
                if module.bias is not None:
                    fp8_linear.bias = module.bias
                setattr(model, name, fp8_linear)
            
            elif isinstance(module, nn.MultiheadAttention):
                fp8_attention = PiscesLxFP8Attention(
                    hidden_size=module.embed_dim,
                    num_attention_heads=module.num_heads,
                    attention_dropout=module.dropout,
                    bias=module.bias is not None,
                    fp8=True,
                )
                setattr(model, name, fp8_attention)
            
            else:
                self.convert_model_to_fp8(module)
        
        logger.success("Model converted to FP8")
        return model


def create_fp8_model(hidden_size: int = 4096,
                     num_layers: int = 32,
                     num_attention_heads: int = 32,
                     vocab_size: int = 131072,
                     fp8: bool = True) -> PiscesLxFP8Model:
    """Factory function to create FP8 model.
    
    Args:
        hidden_size: Hidden dimension
        num_layers: Number of layers
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        fp8: Use FP8 precision
        
    Returns:
        FP8 model instance
    """
    return PiscesLxFP8Model(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
        fp8=fp8,
    )


def is_fp8_supported() -> Tuple[bool, str]:
    """Check if FP8 training is supported.
    
    Returns:
        Tuple of (supported, reason)
    """
    trainer = PiscesLxFP8Trainer()
    return trainer.supported, trainer.reason
