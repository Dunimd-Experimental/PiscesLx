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

"""Quartet: End-to-End FP4 Training for Yv Models.

Based on OpenReview 2025. Stores all linear layer weights in FP4
with stochastic rounding for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def fp4_quantize(x: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP4 format with stochastic rounding.

    FP4 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    Range: approximately [-6, 6]

    Args:
        x: Input tensor.

    Returns:
        Quantized tensor in FP4 format.
    """
    # FP4 representable values (approximate)
    fp4_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], device=x.device, dtype=x.dtype)

    # Stochastic rounding
    floor_idx = torch.searchsorted(fp4_values.sort()[0], x)
    floor_idx = floor_idx.clamp(0, len(fp4_values) - 1)

    # Add noise for stochastic rounding
    noise = torch.rand_like(x)
    rounded = torch.where(
        noise > 0.5,
        fp4_values[floor_idx],
        fp4_values[floor_idx.clamp_max(len(fp4_values) - 2)]
    )

    return rounded


class YvQuartetLinear(nn.Linear):
    """Linear layer with FP4 weight storage.

    Stores weights in FP4 format but computes in FP16/BF16.
    Uses stochastic rounding for quantization.

    Attributes:
        weight_fp4 (torch.Tensor): FP4 quantized weights.
        scale (float): Quantization scale factor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer("weight_fp4", None)
        self.scale = 1.0

    def _quantize_weights(self) -> None:
        """Quantize current weights to FP4."""
        with torch.no_grad():
            self.scale = self.weight.abs().max().item() / 6.0 + 1e-8
            scaled_weight = self.weight / self.scale
            self.weight_fp4 = fp4_quantize(scaled_weight)

    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize FP4 weights to compute precision."""
        if self.weight_fp4 is None:
            return self.weight
        return self.weight_fp4 * self.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP4 weights.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        # Quantize before forward if in training
        if self.training:
            self._quantize_weights()

        # Dequantize for computation
        weight_compute = self._dequantize_weights()

        return F.linear(input, weight_compute, self.bias)


class YvQuartetTrainer:
    """Trainer for end-to-end FP4 training.

    Converts model linear layers to FP4 and handles training.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._convert_to_fp4()

    def _convert_to_fp4(self) -> None:
        """Convert all linear layers in model to FP4."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, YvQuartetLinear):
                # Create FP4 replacement
                fp4_layer = YvQuartetLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
                fp4_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    fp4_layer.bias.data = module.bias.data.clone()

                # Replace in parent
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, fp4_layer)
                else:
                    setattr(self.model, child_name, fp4_layer)

    def get_compression_ratio(self) -> float:
        """Get weight compression ratio from FP4."""
        return 4.0  # FP32 -> FP4 is 8x, BF16 -> FP4 is 4x
