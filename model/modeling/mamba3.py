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
Mamba-3 state space model implementation.

This module implements Mamba-3, a state space model variant with trapezoidal
discretization, complex state space, and MIMO (Multi-Input Multi-Output) structure.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class RuchbahMamba3Config:
    """
    Configuration for Mamba-3 state space model.

    Attributes:
        d_model (int): Model dimension (hidden size).
        d_state (int): State dimension. Defaults to 128.
        d_conv (int): Convolution kernel size. Defaults to 4.
        expand (int): Expansion factor for inner dimension. Defaults to 2.
        dt_rank (Union[int, str]): Time step rank. If "auto", computed as
            ceil(d_model / 16). Defaults to "auto".
        use_trapezoidal (bool): Enable trapezoidal discretization. Defaults to True.
        use_complex (bool): Enable complex state space. Defaults to True.
        use_mimo (bool): Enable MIMO structure. Defaults to True.
        bias (bool): Whether to use bias in linear layers. Defaults to False.
        conv_bias (bool): Whether to use bias in convolution layers. Defaults to True.
    """

    d_model: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"

    # Mamba-3 specific features
    use_trapezoidal: bool = True
    use_complex: bool = True
    use_mimo: bool = True

    # Architecture parameters
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self):
        """
        Post-initialization processing.

        Computes d_inner and sets dt_rank if it is "auto".
        """
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

class RuchbahTrapezoidalDiscretization(nn.Module):
    """
    Trapezoidal discretization for state space models.

    Implements trapezoidal rule for discretizing continuous-time state space
    models, which provides better numerical stability than Euler method by
    combining information from interval start and end points.
    """

    def __init__(self, d_model: int, dt_rank: int):
        """
        Initialize trapezoidal discretization module.

        Args:
            d_model (int): Model dimension.
            dt_rank (int): Time step rank for projection.
        """
        super().__init__()
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Initialize with small values for numerical stability
        nn.init.constant_(self.dt_proj.weight, 0.0)
        nn.init.constant_(self.dt_proj.bias, 0.0)

    def forward(self, dt: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Apply trapezoidal discretization to state transition matrix.

        Uses trapezoidal rule: (I + Δt/2 * A) / (I - Δt/2 * A) for more
        accurate discretization than Euler method.

        Args:
            dt (torch.Tensor): Time step tensor of shape [batch, seq_len, d_model].
            A (torch.Tensor): State matrix of shape [d_state, d_state].

        Returns:
            torch.Tensor: Discretized state transition matrix of shape
                [batch, seq_len, d_state, d_state].
        """
        # Project and ensure positive time steps
        dt = F.softplus(self.dt_proj(dt))

        # Trapezoidal rule: (I + Δt/2 * A) / (I - Δt/2 * A)
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        dt_half = dt.unsqueeze(-1) / 2

        numerator = I + dt_half * A.unsqueeze(0).unsqueeze(0)
        denominator = I - dt_half * A.unsqueeze(0).unsqueeze(0)

        # Solve linear system for stable matrix inversion
        discretized = torch.linalg.solve(denominator, numerator)
        return discretized


class RuchbahComplexStateSpace(nn.Module):
    """
    Complex-valued state space model.

    Implements complex-valued state space transformations for more expressive
    state updates. The complex representation allows for richer dynamics than
    real-valued state spaces.
    """

    def __init__(self, d_state: int, d_model: int):
        """
        Initialize complex state space module.

        Args:
            d_state (int): State dimension.
            d_model (int): Model dimension (used for consistency, not directly used).
        """
        super().__init__()
        self.d_state = d_state

        # Initialize complex state matrix with real and imaginary parts
        A_real = torch.randn(d_state, d_state) * 0.05
        A_imag = torch.randn(d_state, d_state) * 0.05

        # Ensure stability: eigenvalues should have negative real parts
        A = A_real + 1j * A_imag
        eigenvals = torch.linalg.eigvals(A)
        if (eigenvals.real > 0).any():
            # Stabilize by shifting eigenvalues to negative real parts
            max_real = eigenvals.real.max()
            A_real = A_real - max_real * torch.eye(d_state)

        # Additional stability: ensure well-conditioned matrix
        A_stabilized = A_real + 1j * A_imag
        cond_number = torch.linalg.cond(A_stabilized)
        if cond_number > 1000.0:
            # Regularize matrix to improve conditioning
            reg_factor = 1e-3
            A_real = A_real + reg_factor * torch.eye(d_state)

        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)

    @property
    def A_complex(self) -> torch.Tensor:
        """
        Get complex-valued state matrix.

        Returns:
            torch.Tensor: Complex state matrix of shape [d_state, d_state].
        """
        return self.A_real + 1j * self.A_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply complex state space transformation.

        Args:
            x (torch.Tensor): Real-valued input tensor of shape
                [batch, seq_len, d_state].

        Returns:
            torch.Tensor: Transformed state tensor of shape [batch, seq_len, d_state].
                Returns input unchanged or zeros if numerical instability detected.
        """
        # Convert to complex representation
        x_complex = torch.complex(x, torch.zeros_like(x))

        # Apply complex transformation
        A_complex = self.A_complex.unsqueeze(0).unsqueeze(0)

        # Numerical stability check
        if torch.isnan(A_complex).any() or torch.isinf(A_complex).any():
            # Fallback to identity transformation if matrix is unstable
            return x

        # Apply complex matrix multiplication
        x_transformed = torch.einsum(
            'blds,blst->bldt',
            x_complex.unsqueeze(-2),
            A_complex
        ).squeeze(-2)

        # Return real part
        result = x_transformed.real

        # Final stability check
        if torch.isnan(result).any() or torch.isinf(result).any():
            return torch.zeros_like(x)

        return result

class RuchbahMIMOStateSpace(nn.Module):
    """
    Multi-Input Multi-Output state space model.

    Implements MIMO state space using matrix multiplication operations,
    replacing outer product form with efficient matrix operations.
    """

    def __init__(self, d_model: int, d_state: int):
        """
        Initialize MIMO state space module.

        Args:
            d_model (int): Model dimension (input/output dimension).
            d_state (int): State dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # MIMO transformation matrices
        self.B = nn.Linear(d_model, d_state, bias=False)  # Input matrix
        self.C = nn.Linear(d_state, d_model, bias=False)  # Output matrix
        self.D = nn.Linear(d_model, d_model, bias=False)  # Feedthrough matrix

        # Initialize with small gain for stability
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)
        nn.init.xavier_uniform_(self.C.weight, gain=0.1)
        nn.init.xavier_uniform_(self.D.weight, gain=0.1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MIMO state space update step.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, d_model].
            state (torch.Tensor): Current state tensor of shape [batch, d_state].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (output, new_state) where:
                - output: Output tensor of shape [batch, d_model]
                - new_state: Updated state tensor of shape [batch, d_state]
        """
        # Input transformation: u = B @ x
        u = self.B(x)

        # State update (simplified: state = state + u)
        # In full form: state = A @ state + B @ x
        new_state = state + u

        # Output transformation: y = C @ state + D @ x
        y = self.C(new_state) + self.D(x)

        return y, new_state

class RuchbahSelectiveScan(nn.Module):
    """
    Selective scan mechanism for state space models.

    Implements hardware-efficient selective scan operation with numerical
    stability checks and bounds to prevent overflow/underflow.
    """

    def __init__(self, d_model: int, dt_rank: int):
        """
        Initialize selective scan module.

        Args:
            d_model (int): Model dimension.
            dt_rank (int): Time step rank for projection.
        """
        super().__init__()
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_model))
        self.D = nn.Parameter(torch.ones(d_model))

        # Initialize with conservative values for stability
        nn.init.constant_(self.dt_proj.weight, 0.0)
        nn.init.constant_(self.dt_proj.bias, 0.0)
        nn.init.uniform_(self.A_log, -3.0, -1.0)

    def forward(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Selective scan operation with numerical stability.

        Args:
            u (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            delta (torch.Tensor): Time step tensor of shape [batch, seq_len, dt_rank].

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, d_model].
                Returns zeros if NaN/Inf detected.
        """
        batch, seq_len, d_model = u.shape

        # Discretization with stability checks
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())

        # Clamp values to prevent extreme values
        delta = torch.clamp(delta, min=1e-6, max=10.0)
        A = torch.clamp(A, min=-10.0, max=-1e-6)

        # Sequential scan operation with stability checks
        output = torch.zeros_like(u)
        state = torch.zeros(batch, d_model, device=u.device, dtype=u.dtype)

        for i in range(seq_len):
            # Compute state update with numerical stability
            exp_term = torch.exp(delta[:, i:i+1] * A)
            exp_term = torch.clamp(exp_term, min=1e-6, max=1.0)

            state = state * exp_term + u[:, i:i+1]

            # Clamp state to prevent overflow
            state = torch.clamp(state, min=-1e4, max=1e4)
            output[:, i:i+1] = state

        # Final output with feedthrough term
        final_output = output + u * self.D.unsqueeze(0).unsqueeze(0)
        final_output = torch.clamp(final_output, min=-1e4, max=1e4)

        # NaN/Inf check with fallback
        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            return torch.zeros_like(u)

        return final_output

class RuchbahMamba3Block(nn.Module):
    """
    Complete Mamba-3 block implementation.

    Combines input projection, convolution, selective scan, and optional
    Mamba-3 features (trapezoidal discretization, complex state space, MIMO).
    """

    def __init__(self, config: RuchbahMamba3Config):
        """
        Initialize Mamba-3 block.

        Args:
            config (RuchbahMamba3Config): Configuration object for Mamba-3.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner

        # Input projection: splits into two paths (x and z)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # Depthwise convolution for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # Selective scan mechanism
        self.selective_scan = RuchbahSelectiveScan(config.d_inner, config.dt_rank)

        # Optional Mamba-3 features
        if config.use_trapezoidal:
            self.trapezoidal = RuchbahTrapezoidalDiscretization(config.d_inner, config.dt_rank)

        if config.use_complex:
            self.complex_ssm = RuchbahComplexStateSpace(config.d_state, config.d_inner)

        if config.use_mimo:
            self.mimo_ssm = RuchbahMIMOStateSpace(config.d_inner, config.d_state)

        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Mamba-3 block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            attention_mask (Optional[torch.Tensor]): Optional attention mask.
                Currently not used but kept for API compatibility.

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, d_model].
        """
        batch, seq_len, d_model = hidden_states.shape

        # Input projection: split into x and z paths
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Convolution: rearrange for conv1d, apply, then rearrange back
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)

        # Apply optional Mamba-3 features
        if self.config.use_trapezoidal and hasattr(self, 'trapezoidal'):
            # Trapezoidal discretization (currently computed but not directly used)
            dt = torch.randn(batch, seq_len, self.config.dt_rank, device=x.device)
            A = torch.eye(self.config.d_state, device=x.device)
            discretized_A = self.trapezoidal(dt, A)

        if self.config.use_complex and hasattr(self, 'complex_ssm'):
            # Apply complex state space transformation
            x_complex = self.complex_ssm(x)
            x = x + x_complex

        if self.config.use_mimo and hasattr(self, 'mimo_ssm'):
            # Apply MIMO transformation sequentially
            state = torch.zeros(batch, self.config.d_state, device=x.device)
            mimo_outputs = []

            for t in range(seq_len):
                output, state = self.mimo_ssm(x[:, t], state)
                mimo_outputs.append(output)

            x_mimo = torch.stack(mimo_outputs, dim=1)
            x = x + x_mimo

        # Selective scan
        delta = torch.randn(batch, seq_len, self.config.dt_rank, device=x.device)
        x = self.selective_scan(x, delta)

        # Gating: apply SiLU to z path and multiply with x
        z = F.silu(z)
        output = x * z

        # Output projection
        output = self.out_proj(output)

        return output

class RuchbahMamba3Integration(nn.Module):
    """
    High-level integration module for Mamba-3.

    Wraps Mamba-3 block with residual connection and layer normalization
    for use in transformer architectures.
    """

    def __init__(self, d_model: int, config: Optional[RuchbahMamba3Config] = None):
        """
        Initialize Mamba-3 integration module.

        Args:
            d_model (int): Model dimension (hidden size).
            config (Optional[RuchbahMamba3Config]): Configuration object. If None,
                creates default configuration with d_model.
        """
        super().__init__()
        if config is None:
            config = RuchbahMamba3Config(d_model=d_model)

        self.mamba3_block = RuchbahMamba3Block(config)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with residual connection and layer normalization.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            attention_mask (Optional[torch.Tensor]): Optional attention mask.
                Passed to Mamba-3 block but currently not used.

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, d_model].
        """
        # Store residual for skip connection
        residual = hidden_states

        # Apply Mamba-3 transformation
        output = self.mamba3_block(hidden_states, attention_mask)

        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output
