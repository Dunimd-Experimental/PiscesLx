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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from __future__ import annotations

"""
Predictive Delta Coding for Sequential KV Cache Compression.

Based on:
    Magarshak, "Sequential KV Cache Compression via Probabilistic Language Tries:
    Beyond the Per-Vector Shannon Limit", arXiv:2604.15356, 2026, Section 5.

Theory:
    Each KV vector KV_{t+1} is not stored directly. Instead, we store only the
    residual of KV_{t+1} from its model-predicted value:

        residual_{t+1} = KV_actual_{t+1} - KV_predicted_{t+1}

    where KV_predicted_{t+1} is computed by a lightweight predictor network
    from the previous step's hidden state:

        KV_predicted_{t+1} = predictor(hidden_state_t)

    The entropy of the residual is bounded by the model's per-token surprisal:

        H(KV_{t+1} | KV_{<=t}) <= H(token_{t+1} | token_{<=t})

    At typical language model perplexity of 10-20 (fluent English text),
    this gives 3.3-4.3 bits per entire token position, compared to 3 bits
    per vector component for per-vector quantization methods.

    Working in the MLA (Multi-head Latent Attention) latent space further
    reduces the dimension from O(2*L*H_head*d) to O(kv_lora_rank), making
    the predictor computationally negligible.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any


# Paper: Original contribution by Dunimd Team (Yv Architecture — predictive delta coding)
class YvPredictiveDeltaCoder(nn.Module):
    """Predictive delta coder for sequential KV cache compression.

    Implements Layer 2 of the sequential KV compression architecture:
    predictive delta coding. A lightweight neural predictor estimates
    the next KV vector from the current hidden state, and only the
    residual (difference) is stored in the cache.

    Architecture:
        The predictor is a small 1-2 layer MLP that maps from the
        transformer's hidden state to the MLA compressed KV space:

            hidden_state (R^hidden_size) -> Linear -> R^kv_lora_rank

        Optionally, a bottleneck layer can be added for further
        parameter efficiency:

            hidden_state -> Linear -> R^bottleneck -> ReLU -> Linear -> R^kv_lora_rank

    Key Features:
        - Neural predictor with minimal parameter overhead
        - Per-layer independent predictors
        - Adaptive quantization of residuals (2-4 bits)
        - Zero-parameter warmup mode using previous-frame prediction
        - Works in MLA latent space for maximum efficiency

    Compression Flow:
        Write:
            predicted_kv = predictor(prev_hidden_state)
            residual = actual_kv - predicted_kv
            (residual_quant, scale) = adaptive_quantize(residual, bits)
            store(residual_quant, scale)

        Read:
            (residual_quant, scale) = load()
            residual = adaptive_dequantize(residual_quant, scale)
            kv = predicted_kv + residual

    Reference:
        Magarshak, "Sequential KV Cache Compression via Probabilistic Language Tries",
        arXiv:2604.15356, 2026, Section 5.
    """

    def __init__(
        self,
        hidden_size: int,
        kv_lora_rank: int = 512,
        num_layers: int = 32,
        predictor_bottleneck: Optional[int] = None,
        delta_bits: int = 2,
        use_layer_specific_predictors: bool = True,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize the predictive delta coder.

        Args:
            hidden_size: Transformer hidden size dimension.
            kv_lora_rank: MLA KV compression rank (default: 512).
            num_layers: Number of transformer layers.
            predictor_bottleneck: Optional bottleneck dimension for predictor.
                If None, uses a single linear layer. If set, uses a 2-layer MLP.
            delta_bits: Bits for residual quantization (default: 2).
            use_layer_specific_predictors: If True, each layer has its own
                predictor. If False, a single predictor is shared across layers.
            dtype: Data type for predictor weights.
            device: Device for predictor weights.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.kv_lora_rank = kv_lora_rank
        self.num_layers = num_layers
        self.delta_bits = delta_bits
        self.use_layer_specific_predictors = use_layer_specific_predictors
        self.dtype = dtype
        self.device = device

        if use_layer_specific_predictors:
            self.predictors = nn.ModuleList([
                self._build_predictor(predictor_bottleneck, device, dtype)
                for _ in range(num_layers)
            ])
        else:
            self.shared_predictor = self._build_predictor(predictor_bottleneck, device, dtype)

        self._pending_predictions: Dict[int, torch.Tensor] = {}
        self._enable_delta: bool = True
        self._delta_encode_count: int = 0
        self._delta_skip_count: int = 0

    def _build_predictor(
        self,
        bottleneck: Optional[int],
        device: str,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Build a predictor module.

        If bottleneck is specified, builds a 2-layer MLP:
            hidden_size -> bottleneck -> ReLU -> kv_lora_rank
        Otherwise, builds a single linear layer:
            hidden_size -> kv_lora_rank

        Args:
            bottleneck: Optional bottleneck dimension.
            device: Device for weights.
            dtype: Data type for weights.

        Returns:
            Predictor module.
        """
        if bottleneck is not None:
            predictor = nn.Sequential(
                nn.Linear(self.hidden_size, bottleneck, bias=False, device=device, dtype=dtype),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, self.kv_lora_rank, bias=False, device=device, dtype=dtype),
            )
        else:
            predictor = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False, device=device, dtype=dtype)

        self._init_predictor_weights(predictor)
        return predictor

    def _init_predictor_weights(self, predictor: nn.Module):
        """Initialize predictor weights with small values for stable delta coding.

        Uses very small initialization to ensure the predictor starts close
        to zero, meaning residuals initially equal the full KV vectors. This
        provides a safe warmup period before delta compression kicks in.

        Args:
            predictor: Predictor module to initialize.
        """
        for m in predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def _get_predictor(self, layer_idx: int) -> nn.Module:
        """Get the predictor for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Predictor module for the specified layer.
        """
        if self.use_layer_specific_predictors:
            return self.predictors[layer_idx % self.num_layers]
        return self.shared_predictor

    def predict_next_kv(self, hidden_state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Predict the KV latent vector for the next token position.

        Uses the current layer's output hidden state to predict what
        the KV latent will be for the next token.

        Args:
            hidden_state: Output hidden state from the transformer layer
                at position t, shape [batch, hidden_size].
            layer_idx: Layer index for per-layer predictor selection.

        Returns:
            Predicted KV latent vector, shape [batch, kv_lora_rank].
        """
        predictor = self._get_predictor(layer_idx)
        hidden_state = hidden_state.to(dtype=self.dtype, device=self.device)
        predicted = predictor(hidden_state)
        return predicted

    def compute_pending_prediction(
        self,
        layer_idx: int,
        hidden_state: torch.Tensor,
    ):
        """Compute and store the predicted KV for the next step.

        This is called after a full forward pass through the layer.
        The prediction will be consumed when the next token's KV is stored.

        Args:
            layer_idx: Layer index.
            hidden_state: Layer output hidden state at current position.
                Shape [batch, hidden_size] or [batch, seq_len, hidden_size].
                If seq_len > 1, uses the last position.
        """
        if not self._enable_delta:
            return

        with torch.no_grad():
            if hidden_state.dim() == 3:
                hidden_state = hidden_state[:, -1, :]

            hidden_state = hidden_state.contiguous()
            predicted = self.predict_next_kv(hidden_state, layer_idx)
            self._pending_predictions[layer_idx] = predicted.detach()

    def encode_residual(
        self,
        kv_actual: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode KV via delta coding: compute residual and quantize.

        If no pending prediction exists for this layer, the full KV is
        stored as-is (identity encoding with zero prediction).

        Args:
            kv_actual: Actual KV latent tensor.
                Shape varies by context: [1, n_heads, n_tokens, head_dim]
                or [batch, n_tokens, kv_lora_rank].
            layer_idx: Layer index.

        Returns:
            Tuple of (residual_quant, scale, kv_predicted) where:
                residual_quant: Quantized residual tensor.
                scale: Per-element scale factor for dequantization.
                kv_predicted: The prediction that was subtracted (needed for decode).
        """
        kv_predicted = self._pending_predictions.get(layer_idx, None)

        if kv_predicted is None or not self._enable_delta:
            self._delta_skip_count += 1
            kv_predicted = torch.zeros_like(kv_actual)
            residual = kv_actual
        else:
            self._pending_predictions.pop(layer_idx)
            self._delta_encode_count += 1
            residual = kv_actual - kv_predicted

        residual_quant, scale = self._adaptive_quantize(residual, self.delta_bits)
        return residual_quant, scale, kv_predicted.detach()

    def decode_residual(
        self,
        residual_quant: torch.Tensor,
        scale: torch.Tensor,
        kv_predicted: torch.Tensor,
    ) -> torch.Tensor:
        """Decode delta-coded KV: dequantize residual and add prediction.

        Args:
            residual_quant: Quantized residual tensor.
            scale: Scale factor from encoding.
            kv_predicted: Prediction tensor that was subtracted during encoding.

        Returns:
            Reconstructed KV latent tensor.
        """
        residual = self._adaptive_dequantize(residual_quant, scale)
        return kv_predicted + residual

    def _adaptive_quantize(
        self,
        tensor: torch.Tensor,
        bits: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor with adaptive per-channel scaling.

        Uses uniform quantization with per-channel max-based scaling.
        For very small residuals (magnitude near zero), uses even fewer
        effective bits through scale clamping.

        Args:
            tensor: Input tensor to quantize.
            bits: Number of quantization bits.

        Returns:
            Tuple of (quantized_tensor, scale_tensor).
        """
        if bits >= 16:
            return tensor, torch.ones_like(tensor[:, :, :, :1])

        orig_shape = tensor.shape

        if tensor.dim() == 4:
            tensor_flat = tensor.reshape(orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3])
            scale = tensor_flat.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            tensor_flat = tensor_flat.reshape(orig_shape[0] * orig_shape[1], -1)
            scale = scale.reshape(-1, 1)
        elif tensor.dim() == 3:
            tensor_flat = tensor.reshape(orig_shape[0], -1)
            scale = tensor_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            tensor_flat = tensor.reshape(-1)
            scale = tensor_flat.abs().amax().clamp(min=1e-8).view(1, 1)

        num_levels = 2 ** bits
        max_val = num_levels - 1
        half_range = max_val / 2

        q = torch.clamp(torch.round((tensor_flat / scale) * half_range), -half_range, half_range)
        q = q.reshape(orig_shape)

        scale_out = scale.view(scale.shape[0], *([1] * (len(orig_shape) - 1)))

        return q, scale_out

    def _adaptive_dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize a tensor from its quantized representation.

        Args:
            quantized: Quantized integer tensor.
            scale: Scale tensor from encoding.

        Returns:
            Dequantized floating-point tensor.
        """
        bits = self.delta_bits
        if bits >= 16:
            return quantized

        num_levels = 2 ** bits
        half_range = (num_levels - 1) / 2

        orig_shape = quantized.shape
        flat = quantized.reshape(*([-1] if quantized.dim() > 1 else [1]), quantized.numel() // (
            quantized.shape[0] if quantized.dim() > 1 else 1
        ))

        s = scale.reshape(-1, 1)
        qf = flat.reshape(s.shape[0], -1)

        dequant = (qf / half_range) * s
        return dequant.reshape(orig_shape)

    def forward(
        self,
        kv_actual: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode KV via delta coding.

        Convenience wrapper around encode_residual. Used when the
        delta coder is called directly from the cache manager.

        Args:
            kv_actual: Actual KV tensor.
            layer_idx: Layer index.

        Returns:
            (residual_quant, scale, kv_predicted) for storage.
        """
        return self.encode_residual(kv_actual, layer_idx)

    def set_enabled(self, enabled: bool):
        """Enable or disable delta coding.

        Args:
            enabled: True to enable, False to disable.
        """
        self._enable_delta = enabled

    def get_stats(self) -> Dict[str, Any]:
        """Return encoding statistics for monitoring."""
        total = max(1, self._delta_encode_count + self._delta_skip_count)
        return {
            'delta_encode_count': self._delta_encode_count,
            'delta_skip_count': self._delta_skip_count,
            'delta_utilization': self._delta_encode_count / total,
            'delta_bits': self.delta_bits,
            'kv_lora_rank': self.kv_lora_rank,
            'num_predictors': self.num_layers if self.use_layer_specific_predictors else 1,
        }

    def reset_stats(self):
        """Reset encoding statistics."""
        self._delta_encode_count = 0
        self._delta_skip_count = 0


class YvDeltaCacheEntry:
    """Container for a single delta-encoded cache block entry.

    Stores the quantized residual and the metadata needed for reconstruction.
    This is the on-disk / in-memory format for delta-encoded KV blocks.

    Attributes:
        residual_k: Quantized key residual tensor.
        residual_v: Quantized value residual tensor.
        scale_k: Scale for key residual dequantization.
        scale_v: Scale for value residual dequantization.
        pred_k: The key prediction that was subtracted during encoding.
        pred_v: The value prediction that was subtracted during encoding.
        num_tokens: Number of tokens in this entry.
    """

    def __init__(
        self,
        residual_k: torch.Tensor,
        residual_v: torch.Tensor,
        scale_k: torch.Tensor,
        scale_v: torch.Tensor,
        pred_k: torch.Tensor,
        pred_v: torch.Tensor,
    ):
        self.residual_k = residual_k
        self.residual_v = residual_v
        self.scale_k = scale_k
        self.scale_v = scale_v
        self.pred_k = pred_k
        self.pred_v = pred_v

    @property
    def num_tokens(self) -> int:
        return self.residual_k.shape[2] if self.residual_k.dim() >= 3 else 1

    def to(self, device: torch.device) -> 'YvDeltaCacheEntry':
        """Move all tensors to the specified device."""
        self.residual_k = self.residual_k.to(device)
        self.residual_v = self.residual_v.to(device)
        self.scale_k = self.scale_k.to(device)
        self.scale_v = self.scale_v.to(device)
        self.pred_k = self.pred_k.to(device)
        self.pred_v = self.pred_v.to(device)
        return self

    def detach(self) -> 'YvDeltaCacheEntry':
        """Detach all tensors from the computation graph."""
        self.residual_k = self.residual_k.detach()
        self.residual_v = self.residual_v.detach()
        self.scale_k = self.scale_k.detach()
        self.scale_v = self.scale_v.detach()
        self.pred_k = self.pred_k.detach()
        self.pred_v = self.pred_v.detach()
        return self

    def clone(self) -> 'YvDeltaCacheEntry':
        """Create a deep copy of the entry."""
        return YvDeltaCacheEntry(
            residual_k=self.residual_k.clone(),
            residual_v=self.residual_v.clone(),
            scale_k=self.scale_k.clone(),
            scale_v=self.scale_v.clone(),
            pred_k=self.pred_k.clone(),
            pred_v=self.pred_v.clone(),
        )
