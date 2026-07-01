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

"""Elastic Weight Consolidation (EWC) for Yv Models.

Prevents catastrophic forgetting during online learning by penalizing
changes to important parameters. Based on Kirkpatrick et al., PNAS 2017.

Mathematical Formulation:
    EWC_loss = lambda * sum(F_i * (theta_i - theta_opt_i)^2)
    where F_i is the Fisher Information (diagonal) for parameter i.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Iterator


# Paper: Kirkpatrick et al., "Overcoming Catastrophic Forgetting", PNAS 2017
class YvEWC(nn.Module):
    """Elastic Weight Consolidation module.

    Tracks Fisher Information Matrix diagonal and optimal parameters
    to prevent catastrophic forgetting during online weight updates.

    Attributes:
        lambda_ewc (float): EWC regularization strength.
        fisher_dict (dict): Fisher Information per parameter.
        optimal_params (dict): Optimal parameter values.

    Example:
        >>> ewc = YvEWC(model.parameters(), lambda_ewc=1000.0)
        >>> ewc.compute_fisher_information(data_loader, model, loss_fn)
        >>> loss = task_loss + ewc.compute_ewc_loss(current_params)
    """

    def __init__(
        self,
        params,
        lambda_ewc: float = 1000.0
    ):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Determine if params are named (Iterator[Tuple[str, nn.Parameter]]) or plain (Iterator[nn.Parameter])
        param_list = list(params)
        if param_list and isinstance(param_list[0], (list, tuple)) and len(param_list[0]) == 2:
            for name, param in param_list:
                if param.requires_grad:
                    self.optimal_params[name] = param.data.clone()
                    self.fisher_dict[name] = torch.zeros_like(param.data)
        else:
            for i, param in enumerate(param_list):
                if param.requires_grad:
                    name = f"param_{i}"
                    self.optimal_params[name] = param.data.clone()
                    self.fisher_dict[name] = torch.zeros_like(param.data)

    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader,
        loss_fn,
        num_samples: int = 200
    ) -> None:
        """Compute Fisher Information Matrix diagonal.

        Approximates Fisher as E[(dL/dtheta)^2] via squared gradients.

        Args:
            model: Model to compute Fisher for.
            data_loader: Data loader for sampling.
            loss_fn: Loss function.
            num_samples: Number of samples to use.
        """
        # Reset Fisher
        for name in self.fisher_dict:
            self.fisher_dict[name].zero_()

        model.train()
        samples_processed = 0

        for batch in data_loader:
            if samples_processed >= num_samples:
                break

            model.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and name in self.fisher_dict and param.grad is not None:
                    self.fisher_dict[name] += param.grad.data ** 2

            samples_processed += 1

        # Average over samples
        for name in self.fisher_dict:
            self.fisher_dict[name] /= max(samples_processed, 1)

    def compute_ewc_loss(
        self,
        model: nn.Module
    ) -> torch.Tensor:
        """Compute EWC regularization loss.

        Args:
            model: Current model with updated parameters.

        Returns:
            EWC loss scalar.
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.optimal_params:
                optimal = self.optimal_params[name].to(param.device)
                fisher = self.fisher_dict[name].to(param.device)

                loss += (fisher * (param - optimal) ** 2).sum()

        return self.lambda_ewc * loss

    def update_optimal_params(self, model: nn.Module) -> None:
        """Store current parameters as optimal (call after training phase).

        Args:
            model: Model with current parameters.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.optimal_params:
                self.optimal_params[name] = param.data.clone()

    def get_importance(self, param_name: str) -> torch.Tensor:
        """Get Fisher importance for a parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            Fisher importance tensor.
        """
        return self.fisher_dict.get(param_name, torch.tensor(0.0))
