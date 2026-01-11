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
mHC (Manifold-constrained Hyper-Connection) for PiscesL1.

Manifold-constrained Hyper-Connection is a novel residual connection mechanism
that addresses the training instability of standard hyper-connections while
maintaining their expressive power.

Key Features:
- Manifold constraints for stable training
- Learnable hyper-connection weights
- Orthogonal regularization
- Gradient flow optimization
- Compatible with existing transformer blocks

Reference: DeepSeek mHC Architecture (2025)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_

from utils import PiscesLxCoreLog
logger = PiscesLxCoreLog("pisceslx.model.blocks.mhc")


class RuchbahManifoldConstraint(nn.Module):
    """Manifold constraint for hyper-connection stability.
    
    Implements orthogonal projection onto a constrained manifold
    to ensure stable and well-conditioned hyper-connections.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 4, 
                 constraint_type: str = "soft_orthogonal"):
        """Initialize manifold constraint.
        
        Args:
            hidden_size: Hidden dimension
            num_layers: Number of layers to connect
            constraint_type: Type of constraint
                - "soft_orthogonal": Soft orthogonal constraint
                - "hard_orthogonal": Hard orthogonalization via SVD
                - "norm_bound": Norm bounding constraint
                - "spectral": Spectral norm constraint
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.constraint_type = constraint_type
        
        if constraint_type == "soft_orthogonal":
            self.register_buffer("_eye", torch.eye(num_layers))
        
        elif constraint_type == "spectral":
            self.spectral_norm = nn.Parameter(torch.ones(1))
        
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply manifold constraint to hyper-connection weights.
        
        Args:
            weights: Hyper-connection weights [B, num_layers, H, H]
            
        Returns:
            Constrained weights
        """
        if self.constraint_type == "soft_orthogonal":
            return self._soft_orthogonal_constraint(weights)
        elif self.constraint_type == "hard_orthogonal":
            return self._hard_orthogonal_constraint(weights)
        elif self.constraint_type == "norm_bound":
            return self._norm_bound_constraint(weights)
        elif self.constraint_type == "spectral":
            return self._spectral_constraint(weights)
        return weights
    
    def _soft_orthogonal_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Soft orthogonal constraint via regularization."""
        batch_size = weights.shape[0]
        
        W = weights.mean(dim=0)
        WtW = W @ W.transpose(-1, -2)
        
        ortho_loss = F.mse_loss(WtW, self._eye.expand_as(WtW))
        
        if self.training:
            if not hasattr(self, "_ortho_loss_acc"):
                self._ortho_loss_acc = 0.0
            self._ortho_loss_acc = ortho_loss.item()
        
        return weights
    
    def _hard_orthogonal_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Hard orthogonalization via SVD."""
        batch_size = weights.shape[0]
        constrained = []
        
        for b in range(batch_size):
            W = weights[b]
            
            U, S, Vh = torch.linalg.svd(W)
            
            V = Vh.transpose(-1, -2).conj()
            W_ortho = U @ V
            
            constrained.append(W_ortho)
        
        return torch.stack(constrained)
    
    def _norm_bound_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Norm bounding constraint."""
        norms = torch.linalg.vector_norm(weights, dim=(-1, -2), keepdim=True)
        max_norm = 1.0 / math.sqrt(weights.shape[-1])
        
        scales = torch.clamp(norms / max_norm, min=1.0)
        
        return weights / scales
    
    def _spectral_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Spectral norm constraint."""
        spectral_norm = torch.linalg.svdvals(weights).max(dim=-1, keepdim=True)[0]
        spectral_norm = spectral_norm.unsqueeze(-1).unsqueeze(-1)
        
        scale = self.spectral_norm / (spectral_norm + 1e-6)
        scale = torch.clamp(scale, max=1.0)
        
        return weights * scale
    
    def get_constraint_loss(self) -> torch.Tensor:
        """Get the constraint loss for training."""
        if hasattr(self, "_ortho_loss_acc"):
            loss = self._ortho_loss_acc
            self._ortho_loss_acc = 0.0
            return torch.tensor(loss, device=self._eye.device)
        return torch.tensor(0.0, device=self._eye.device if hasattr(self, "_eye") else "cpu")


class RuchbahHyperConnection(nn.Module):
    """Hyper-Connection layer with manifold constraints.
    
    Standard hyper-connection: y = Σᵢ αᵢ xᵢ where αᵢ are learnable weights
    mHC adds manifold constraints to ensure stable training.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 4,
                 use_manifold_constraint: bool = True,
                 constraint_type: str = "soft_orthogonal",
                 drop_path_rate: float = 0.0):
        """Initialize hyper-connection.
        
        Args:
            hidden_size: Hidden dimension
            num_layers: Number of layers to connect
            use_manifold_constraint: Whether to use manifold constraint
            constraint_type: Type of constraint
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_path_rate = drop_path_rate
        
        self.use_manifold_constraint = use_manifold_constraint
        if use_manifold_constraint:
            self.manifold_constraint = RuchbahManifoldConstraint(
                hidden_size, num_layers, constraint_type
            )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_layers),
        )
        
        self.gate = nn.Sigmoid()
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.weight_generator:
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, layer_outputs: List[torch.Tensor], 
                current_input: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            layer_outputs: List of layer outputs [x₀, x₁, ..., xₙ]
            current_input: Current layer input (optional)
            
        Returns:
            Hyper-connected output and gate weights
        """
        batch_size = layer_outputs[0].shape[0]
        device = layer_outputs[0].device
        
        if current_input is None:
            current_input = layer_outputs[-1]
        
        input_features = current_input.mean(dim=1)
        
        raw_weights = self.weight_generator(input_features)
        raw_weights = raw_weights.view(batch_size, self.num_layers)
        
        if self.use_manifold_constraint:
            hyper_weights = self.manifold_constraint(raw_weights)
        else:
            hyper_weights = raw_weights
        
        gate_weights = self.gate(hyper_weights)
        gate_weights = F.softmax(gate_weights, dim=-1)
        
        hyper_output = torch.zeros_like(layer_outputs[0])
        for i, layer_out in enumerate(layer_outputs):
            weight = gate_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)
            hyper_output = hyper_output + weight * layer_out
        
        output = self.layer_norm(hyper_output)
        
        return output, gate_weights
    
    def get_constraint_loss(self) -> torch.Tensor:
        """Get constraint loss if applicable."""
        if self.use_manifold_constraint:
            return self.manifold_constraint.get_constraint_loss()
        return torch.tensor(0.0, device="cpu")


class RuchbahMHCBlock(nn.Module):
    """Transformer block with mHC (Manifold-constrained Hyper-Connection).
    
    Replaces standard residual connection with mHC for improved
    training stability and model expressiveness.
    """
    
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_layers: int = 4, mlp_ratio: float = 4.0,
                 attention_dropout: float = 0.0, dropout: float = 0.0,
                 use_manifold_constraint: bool = True,
                 constraint_type: str = "soft_orthogonal",
                 drop_path_rate: float = 0.0):
        """Initialize mHC block.
        
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            num_layers: Number of layers for hyper-connection
            mlp_ratio: MLP expansion ratio
            attention_dropout: Attention dropout rate
            dropout: Dropout rate
            use_manifold_constraint: Use manifold constraint
            constraint_type: Type of constraint
            drop_path_rate: Drop path rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout),
        )
        
        self.input_norm = nn.LayerNorm(hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        
        self.hyper_connection = RuchbahHyperConnection(
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_manifold_constraint=use_manifold_constraint,
            constraint_type=constraint_type,
            drop_path_rate=drop_path_rate,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._cache = None
    
    def forward(self, x: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, H]
            attention_mask: Attention mask [B, T] or None
            
        Returns:
            Output tensor [B, T, H]
        """
        if self._cache is None:
            self._cache = []
        
        input_norm = self.input_norm(x)
        
        attention_output, _ = self.attention(
            query=input_norm,
            key=input_norm,
            value=input_norm,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attention_output = self.dropout(attention_output)
        
        attention_norm = self.attention_norm(attention_output)
        
        mlp_output = self.mlp(attention_norm)
        
        residual_attention = x + attention_output
        residual_mlp = residual_attention + mlp_output
        
        self._cache.append(residual_mlp)
        
        if len(self._cache) > 4:
            self._cache = self._cache[-4:]
        
        hyper_output, gate_weights = self.hyper_connection(
            self._cache, input_norm
        )
        
        output = x + hyper_output
        
        return output
    
    def reset_cache(self):
        """Reset hyper-connection cache."""
        self._cache = None
    
    def get_constraint_loss(self) -> torch.Tensor:
        """Get mHC constraint loss."""
        return self.hyper_connection.get_constraint_loss()


class RuchbahMHCTransformer(nn.Module):
    """Complete Transformer with mHC for PiscesL1.
    
    Replaces standard residual connections with mHC throughout
    the transformer architecture.
    """
    
    def __init__(self, hidden_size: int, num_layers: int,
                 num_attention_heads: int, mlp_ratio: float = 4.0,
                 attention_dropout: float = 0.0, dropout: float = 0.0,
                 use_manifold_constraint: bool = True,
                 constraint_type: str = "soft_orthogonal",
                 drop_path_rate: float = 0.0):
        """Initialize mHC Transformer.
        
        Args:
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            attention_dropout: Attention dropout rate
            dropout: Dropout rate
            use_manifold_constraint: Use manifold constraint
            constraint_type: Type of constraint
            drop_path_rate: Drop path rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.blocks = nn.ModuleList([
            RuchbahMHCBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_layers=4,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                use_manifold_constraint=use_manifold_constraint,
                constraint_type=constraint_type,
                drop_path_rate=drop_path_rate * i / num_layers,
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, H]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Output tensor [B, T, H]
        """
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.final_norm(x)
        
        return x
    
    def reset_all_caches(self):
        """Reset all block caches."""
        for block in self.blocks:
            block.reset_cache()
    
    def get_total_constraint_loss(self) -> torch.Tensor:
        """Get total mHC constraint loss from all blocks."""
        total_loss = torch.tensor(0.0, device="cpu")
        for block in self.blocks:
            total_loss = total_loss + block.get_constraint_loss()
        return total_loss


class RuchbahMHCLayerReplacement:
    """Utility to replace standard residual connections with mHC.
    
    This can be used to upgrade existing transformer blocks.
    """
    
    @staticmethod
    def replace_attention_block(block: nn.Module, 
                               num_layers: int = 4,
                               use_manifold_constraint: bool = True) -> nn.Module:
        """Replace standard attention block with mHC block.
        
        Args:
            block: Standard transformer block
            num_layers: Number of layers for hyper-connection
            use_manifold_constraint: Use manifold constraint
            
        Returns:
            mHC block
        """
        if not hasattr(block, 'attention') or not hasattr(block, 'mlp'):
            logger.warning("Block doesn't have attention/mlp, skipping")
            return block
        
        config = {
            'hidden_size': block.attention.embed_dim,
            'num_attention_heads': block.attention.num_heads,
            'num_layers': num_layers,
            'mlp_ratio': block.mlp[0].out_features // block.attention.embed_dim if len(block.mlp) > 1 else 4,
            'attention_dropout': block.attention.dropout,
            'dropout': getattr(block, 'dropout', nn.Dropout(0.0)).p if hasattr(getattr(block, 'dropout', None), 'p') else 0.0,
            'use_manifold_constraint': use_manifold_constraint,
        }
        
        mhc_block = RuchbahMHCBlock(**config)
        
        return mhc_block
    
    @staticmethod
    def upgrade_transformer(model: nn.Module, 
                           num_layers: int = 4,
                           use_manifold_constraint: bool = True) -> nn.Module:
        """Upgrade entire transformer model with mHC.
        
        Args:
            model: Transformer model
            num_layers: Number of layers for hyper-connection
            use_manifold_constraint: Use manifold constraint
            
        Returns:
            Upgraded model
        """
        if hasattr(model, 'blocks') or hasattr(model, 'layers'):
            blocks_attr = 'blocks' if hasattr(model, 'blocks') else 'layers'
            blocks = getattr(model, blocks_attr)
            
            for i, block in enumerate(blocks):
                upgraded = RuchbahMHCLayerReplacement.replace_attention_block(
                    block, num_layers, use_manifold_constraint
                )
                blocks[i] = upgraded
            
            logger.success(f"Upgraded {len(blocks)} transformer blocks with mHC")
        
        return model


def create_mhc_transformer(hidden_size: int = 4096, 
                           num_layers: int = 32,
                           num_attention_heads: int = 32,
                           mlp_ratio: float = 4.0,
                           use_manifold_constraint: bool = True,
                           constraint_type: str = "soft_orthogonal") -> RuchbahMHCTransformer:
    """Factory function to create mHC Transformer.
    
    Args:
        hidden_size: Hidden dimension
        num_layers: Number of layers
        num_attention_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        use_manifold_constraint: Use manifold constraint
        constraint_type: Type of constraint
        
    Returns:
        mHC Transformer instance
    """
    return RuchbahMHCTransformer(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        mlp_ratio=mlp_ratio,
        use_manifold_constraint=use_manifold_constraint,
        constraint_type=constraint_type,
    )


class RuchbahMHCLoss(nn.Module):
    """Loss function with mHC constraint penalty.
    
    Combines standard language modeling loss with mHC constraint loss.
    """
    
    def __init__(self, lambda_constraint: float = 0.01):
        """Initialize mHC loss.
        
        Args:
            lambda_constraint: Weight for constraint loss
        """
        super().__init__()
        self.lambda_constraint = lambda_constraint
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                constraint_loss: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            logits: Model logits [B, T, V]
            labels: Target labels [B, T]
            constraint_loss: mHC constraint loss (optional)
            
        Returns:
            Loss dict
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        lm_loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        total_loss = lm_loss
        
        if constraint_loss is not None and self.lambda_constraint > 0:
            constraint = constraint_loss * self.lambda_constraint
            total_loss = total_loss + constraint
        else:
            constraint = torch.tensor(0.0, device=lm_loss.device)
        
        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "constraint_loss": constraint,
        }
