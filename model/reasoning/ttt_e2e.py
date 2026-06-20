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

"""Test-Time Training (TTT-E2E) for Yv Reasoning.

Based on arXiv 2512.23675. End-to-end test-time training during inference
with self-supervised loss (LM + consistency + uncertainty regularization).

Performs 3-5 gradient steps on last N layers with very small learning rate
to adapt to new domains without catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class YvTestTimeTrainer:
    """Test-time training for online model adaptation.

    Adapts model weights during inference on unfamiliar inputs using
    self-supervised signals. Restricts updates to last N layers with
    very small learning rate to prevent forgetting.

    Attributes:
        model (nn.Module): Model to adapt.
        update_layers (int): Number of last layers to update.
        lr (float): Learning rate for test-time updates.
        max_steps (int): Maximum gradient steps per adaptation.
        ewc_module: Optional EWC module for forgetting prevention.

    Example:
        >>> ttt = YvTestTimeTrainer(model, update_layers=2, lr=1e-5)
        >>> if ttt.should_adapt(confidence=0.4, complexity=0.8):
        ...     adapted_model = ttt.adapt(batch_input)
    """

    def __init__(
        self,
        model: nn.Module,
        update_layers: int = 2,
        lr: float = 1e-5,
        max_steps: int = 5,
        ewc_module=None
    ):
        self.model = model
        self.update_layers = update_layers
        self.lr = lr
        self.max_steps = max_steps
        self.ewc_module = ewc_module

        # Identify parameters to update (last N layers)
        total_layers = 0
        for name in model.state_dict():
            if "layers" in name:
                parts = name.split(".")
                for part in parts:
                    if part.isdigit():
                        total_layers = max(total_layers, int(part) + 1)

        import re
        self.params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                for i in range(update_layers):
                    target_layer = total_layers - 1 - i
                    if re.search(rf'layers\.{target_layer}\.', name) or re.search(rf'layers\.{target_layer}$', name):
                        self.params_to_update.append(param)
                        break

    def compute_self_supervised_loss(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute self-supervised loss for test-time training.

        Combines:
        1. LM loss: standard next-token prediction
        2. Consistency loss: similar inputs should have similar hidden states
        3. Uncertainty regularization: encourage high-confidence predictions

        Args:
            hidden_states: Hidden states [batch, seq, hidden].
            logits: Output logits [batch, seq, vocab_size].
            input_ids: Input token IDs for LM loss.

        Returns:
            Combined self-supervised loss.
        """
        loss = torch.tensor(0.0, device=hidden_states.device)

        # 1. LM loss (if input_ids provided)
        if input_ids is not None and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
                ignore_index=-100
            )
            loss = loss + lm_loss

        # 2. Consistency loss: hidden states should be smooth
        if hidden_states.shape[1] > 1:
            h_shifted = hidden_states[:, 1:, :]
            h_prev = hidden_states[:, :-1, :]
            consistency_loss = F.mse_loss(h_shifted, h_prev, reduction="mean")
            loss = loss + 0.1 * consistency_loss

        # 3. Uncertainty regularization: encourage peaky distributions
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            uncertainty_loss = -entropy  # Negative entropy = encourage certainty
            loss = loss + 0.01 * uncertainty_loss

        return loss

    def should_adapt(
        self,
        confidence: float,
        complexity: float,
        confidence_threshold: float = 0.6,
        complexity_threshold: float = 0.7
    ) -> bool:
        """Determine if test-time adaptation is needed.

        Args:
            confidence: Model confidence (0-1).
            complexity: Estimated task complexity (0-1).
            confidence_threshold: Confidence below which to adapt.
            complexity_threshold: Complexity above which to adapt.

        Returns:
            True if adaptation should be triggered.
        """
        return complexity > complexity_threshold and confidence < confidence_threshold

    def adapt(
        self,
        batch_input: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        steps: Optional[int] = None
    ) -> nn.Module:
        """Perform test-time adaptation on the model.

        Args:
            batch_input: Input tensor for adaptation.
            input_ids: Optional token IDs for LM loss.
            steps: Number of gradient steps (default: self.max_steps).

        Returns:
            Adapted model.
        """
        if not self.params_to_update:
            return self.model

        steps = steps or self.max_steps
        optimizer = torch.optim.SGD(self.params_to_update, lr=self.lr)

        self.model.train()

        for _ in range(steps):
            optimizer.zero_grad()

            outputs = self.model(batch_input)

            # Extract hidden states and logits
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]
            else:
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = None

            loss = self.compute_self_supervised_loss(hidden_states, logits, input_ids)

            # Add EWC loss if available
            if self.ewc_module is not None:
                ewc_loss = self.ewc_module.compute_ewc_loss(self.model)
                loss = loss + ewc_loss

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.params_to_update, max_norm=0.1)

            optimizer.step()

        self.model.eval()
        return self.model

    def adapt_step(
        self,
        batch_input: torch.Tensor,
        depth: int = 5,
        confidence_target: float = 0.7
    ) -> torch.Tensor:
        """Single-step test-time adaptation with complexity-aware depth.

        Performs one gradient step of self-supervised learning, adapting
        to the current input distribution. The depth parameter controls
        reasoning intensity and confidence_target guides regularization.

        Args:
            batch_input: Input tensor for adaptation [batch, seq, hidden].
            depth: Reasoning depth guiding adaptation intensity.
            confidence_target: Target confidence for entropy regularization.

        Returns:
            torch.Tensor: The adaptation loss value.
        """
        if not self.params_to_update:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        optimizer = torch.optim.SGD(self.params_to_update, lr=self.lr)
        optimizer.zero_grad()

        self.model.train()
        # Forward pass on the adaptation batch
        with torch.set_grad_enabled(True):
            outputs = self.model(batch_input)

            if hasattr(outputs, "logits") and outputs.logits is not None:
                logits = outputs.logits
            else:
                vocab_size = getattr(self.model.cfg, 'vocab_size', None) if hasattr(self, 'model') and hasattr(self.model, 'cfg') else None
                if vocab_size and batch_input.size(-1) == vocab_size:
                    logits = batch_input
                else:
                    logits = None

            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            elif isinstance(outputs, torch.Tensor):
                hidden_states = outputs
            else:
                hidden_states = batch_input

            loss = self.compute_self_supervised_loss(hidden_states, logits)

            # Scale loss by depth for intensity control
            depth_factor = min(depth / 10.0, 1.0)
            loss = loss * (0.5 + 0.5 * depth_factor)

            if self.ewc_module is not None:
                ewc_loss = self.ewc_module.compute_ewc_loss(self.model)
                loss = loss + ewc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params_to_update, max_norm=0.1)
            optimizer.step()

        self.model.eval()
        return loss.detach()
