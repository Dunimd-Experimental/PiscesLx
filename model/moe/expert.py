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

"""
Expert Network Implementations for Mixture-of-Experts.

This module provides various expert network architectures for MoE layers,
supporting different activation functions, gating mechanisms, and depths.

Expert Types:
    1. Standard Expert:
       - Basic feed-forward network with SiLU activation
       - Simple up-projection -> activation -> down-projection
       - Fast computation, moderate expressiveness
    
    2. SwiGLU Expert:
       - SwiGLU gating mechanism for enhanced expressiveness
       - gate_proj * up_proj with SiLU gating
       - Recommended for most use cases
    
    3. GeGLU Expert:
       - GeGLU gating mechanism with GELU activation
       - Similar to SwiGLU but with GELU instead of SiLU
       - Good for tasks requiring smoother gradients
    
    4. Multi-Layer Expert:
       - Deep expert with multiple hidden layers
       - Higher capacity for complex transformations
       - Use sparingly due to increased computation
    
    5. Shared Expert:
       - Expert that is always active (not routed)
       - Used in DeepSeek-V3 style MoE
       - Provides base transformation for all tokens

Key Features:
    - Configurable hidden and intermediate sizes
    - Optional dropout for regularization
    - Optional bias in linear layers
    - Factory pattern for easy expert creation
    - Registry for custom expert types

Performance Characteristics:
    - Standard: O(2 * hidden * intermediate) parameters
    - SwiGLU/GeGLU: O(3 * hidden * intermediate) parameters
    - Multi-Layer: O((num_layers + 1) * hidden * intermediate) parameters
    - Computation: O(batch_size * seq_len * intermediate_size)

Usage Example:
    >>> from model.moe.expert import (
    ...     YvExpertFactory,
    ...     YvExpertConfig,
    ...     YvExpertType,
    ...     create_expert_module
    ... )
    >>> 
    >>> # Create expert using factory
    >>> config = YvExpertConfig(
    ...     hidden_size=4096,
    ...     intermediate_size=11008,
    ...     expert_type=YvExpertType.SWIGLU,
    ...     dropout=0.0
    ... )
    >>> expert = YvExpertFactory.create(config)
    >>> output = expert(hidden_states)
    >>> 
    >>> # Create expert using convenience function
    >>> expert = create_expert_module(
    ...     hidden_size=4096,
    ...     intermediate_size=11008,
    ...     expert_type="swiglu"
    ... )

Note:
    All classes follow the YvXxx naming convention.
    SwiGLU experts are recommended for best performance/quality trade-off.
    Expert intermediate_size is typically 2-4x hidden_size.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Moe", file_path=get_log_file("Yv.Moe"), enable_file=True)


class YvExpertType(Enum):
    """Enumeration of available expert network types.
    
    Defines the different architectures for expert networks in MoE layers.
    Each type offers different trade-offs between expressiveness and efficiency.
    
    Attributes:
        STANDARD: Basic feed-forward network with SiLU activation.
            Fastest computation, moderate expressiveness.
            Parameters: 2 * hidden * intermediate
        SWIGLU: SwiGLU-based expert with gated linear unit.
            Enhanced expressiveness through gating mechanism.
            Recommended for most use cases.
            Parameters: 3 * hidden * intermediate
        GEGLU: GeGLU-based expert with GELU-gated linear unit.
            Similar to SwiGLU but with GELU activation.
            Good for tasks requiring smoother gradients.
            Parameters: 3 * hidden * intermediate
        MULTI_LAYER: Deep expert with multiple hidden layers.
            Highest capacity for complex transformations.
            Use sparingly due to increased computation.
        SHARED: Shared expert that is always active.
            Used in DeepSeek-V3 style MoE architecture.
            Provides base transformation for all tokens.
        ATTENTION: Attention-based expert (reserved for future use).
    
    Example:
        >>> expert_type = YvExpertType.SWIGLU
        >>> if expert_type == YvExpertType.STANDARD:
        ...     print("Using standard FFN expert")
    
    Note:
        SWIGLU is recommended as the default for best quality/efficiency balance.
        MULTI_LAYER experts should be used sparingly due to computation cost.
    """
    STANDARD = "standard"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    MULTI_LAYER = "multi_layer"
    SHARED = "shared"
    ATTENTION = "attention"


@dataclass
class YvExpertConfig:
    """Configuration dataclass for expert network parameters.
    
    Encapsulates all parameters needed to configure an expert network,
    supporting various expert types and architectural choices.
    
    Attributes:
        hidden_size (int): Input and output hidden dimension.
            Must match the model's hidden size. Default: 2048.
        intermediate_size (int): Intermediate dimension for expert.
            Typically 2-4x hidden_size. Default: 5632.
        expert_type (YvExpertType): Type of expert architecture.
            Default: SWIGLU.
        dropout (float): Dropout probability for regularization.
            0.0 for no dropout. Default: 0.0.
        use_bias (bool): Whether to use bias in linear layers.
            False for most modern architectures. Default: False.
        activation (str): Activation function name.
            Options: "silu", "gelu", "relu". Default: "silu".
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.SWIGLU,
        ...     dropout=0.1
        ... )
    
    Note:
        intermediate_size should be tuned based on model capacity needs.
        dropout > 0 is rarely needed in MoE experts due to implicit regularization.
    """
    hidden_size: int = 2048
    intermediate_size: int = 5632
    expert_type: YvExpertType = YvExpertType.SWIGLU
    dropout: float = 0.0
    use_bias: bool = False
    activation: str = "silu"

    def __post_init__(self):
        """Post-initialization to convert string expert_type to enum."""
        if isinstance(self.expert_type, str):
            self.expert_type = YvExpertType(self.expert_type)


class YvExpertBase(nn.Module):
    """Abstract base class for all expert network implementations.
    
    Defines the interface that all expert networks must implement.
    Subclasses should implement the forward method with their specific
    architecture.
    
    Attributes:
        config (YvExpertConfig): Configuration for the expert.
    
    Note:
        This is an abstract class and cannot be instantiated directly.
        Use YvExpertFactory or specific expert classes instead.
    """

    def __init__(self, config: YvExpertConfig):
        """Initialize base expert with configuration.
        
        Args:
            config: Expert configuration containing architecture parameters.
        """
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert network.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size] or
               [num_tokens, hidden_size].
        
        Returns:
            Output tensor with same shape as input.
        
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement forward")


class YvStandardExpert(YvExpertBase):
    """Standard feed-forward network expert with SiLU activation.
    
    Implements a simple two-layer feed-forward network:
        output = down_proj(silu(up_proj(x)))
    
    This is the most computationally efficient expert type but has
    lower expressiveness compared to gated variants.
    
    Architecture:
        Input [hidden_size] -> Linear -> SiLU -> Dropout -> Linear -> Output [hidden_size]
    
    Attributes:
        up_proj (nn.Linear): Up-projection layer.
        down_proj (nn.Linear): Down-projection layer.
        dropout (nn.Module): Dropout layer or identity.
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.STANDARD
        ... )
        >>> expert = YvStandardExpert(config)
        >>> output = expert(hidden_states)
    
    Note:
        Use SwiGLU experts for better quality at slight computational cost.
    """

    def __init__(self, config: YvExpertConfig, device=None, dtype=None):
        """Initialize standard expert.
        
        Args:
            config: Expert configuration.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__(config)

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through standard expert.
        
        Args:
            x: Input tensor [*, hidden_size].
        
        Returns:
            Output tensor [*, hidden_size].
        """
        x = self.up_proj(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class YvSwiGLUExpert(YvExpertBase):
    """SwiGLU-based expert network with gated linear unit.
    
    Implements the SwiGLU architecture from "GLU Variants Improve Transformer":
        output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    SwiGLU provides enhanced expressiveness through the gating mechanism
    while maintaining computational efficiency. This is the recommended
    expert type for most use cases.
    
    Architecture:
        Input -> gate_proj -> SiLU ─┐
                                   ├─> * -> down_proj -> Output
        Input -> up_proj ──────────┘
    
    Attributes:
        gate_proj (nn.Linear): Gate projection for gating signal.
        up_proj (nn.Linear): Value projection.
        down_proj (nn.Linear): Output projection.
        dropout (nn.Module): Dropout layer or identity.
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.SWIGLU
        ... )
        >>> expert = YvSwiGLUExpert(config)
        >>> output = expert(hidden_states)
    
    Note:
        Recommended expert type for best quality/efficiency trade-off.
        Uses 50% more parameters than standard expert.
    """

    def __init__(self, config: YvExpertConfig, device=None, dtype=None):
        """Initialize SwiGLU expert.
        
        Args:
            config: Expert configuration.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__(config)

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU expert.
        
        Args:
            x: Input tensor [*, hidden_size].
        
        Returns:
            Output tensor [*, hidden_size].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class YvGeGLUExpert(YvExpertBase):
    """GeGLU-based expert network with GELU-gated linear unit.
    
    Implements the GeGLU architecture similar to SwiGLU but with GELU:
        output = down_proj(gelu(gate_proj(x)) * up_proj(x))
    
    GeGLU provides similar benefits to SwiGLU but with GELU activation
    which may be preferred for certain tasks requiring smoother gradients.
    
    Architecture:
        Input -> gate_proj -> GELU ─┐
                                    ├─> * -> down_proj -> Output
        Input -> up_proj ───────────┘
    
    Attributes:
        gate_proj (nn.Linear): Gate projection for gating signal.
        up_proj (nn.Linear): Value projection.
        down_proj (nn.Linear): Output projection.
        dropout (nn.Module): Dropout layer or identity.
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.GEGLU
        ... )
        >>> expert = YvGeGLUExpert(config)
        >>> output = expert(hidden_states)
    
    Note:
        Similar to SwiGLU but with GELU instead of SiLU.
        GELU may provide smoother gradients in some cases.
    """

    def __init__(self, config: YvExpertConfig, device=None, dtype=None):
        """Initialize GeGLU expert.
        
        Args:
            config: Expert configuration.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__(config)

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            device=device,
            dtype=dtype
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GeGLU expert.
        
        Args:
            x: Input tensor [*, hidden_size].
        
        Returns:
            Output tensor [*, hidden_size].
        """
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class YvMultiLayerExpert(YvExpertBase):
    """Multi-layer expert network for complex transformations.
    
    Implements a deep expert with multiple hidden layers, providing
    higher capacity for complex transformations at the cost of
    increased computation.
    
    Architecture:
        Input -> [Linear -> SiLU -> Dropout] * (num_layers-1) -> Linear -> Output
    
    Use this expert type sparingly as it significantly increases
    computational cost compared to single-layer experts.
    
    Attributes:
        num_layers (int): Number of layers in the expert.
        layers (nn.Sequential): Sequential container of all layers.
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.MULTI_LAYER
        ... )
        >>> expert = YvMultiLayerExpert(config, num_layers=3)
        >>> output = expert(hidden_states)
    
    Note:
        Use sparingly due to increased computation.
        num_layers=2 is often sufficient for additional capacity.
    """

    def __init__(
        self,
        config: YvExpertConfig,
        num_layers: int = 2,
        device=None,
        dtype=None
    ):
        """Initialize multi-layer expert.
        
        Args:
            config: Expert configuration.
            num_layers: Number of layers (minimum 2). Default: 2.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__(config)

        self.num_layers = num_layers

        layers = []
        in_size = config.hidden_size
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_size, config.intermediate_size, bias=config.use_bias, device=device, dtype=dtype))
            layers.append(nn.SiLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_size = config.intermediate_size

        layers.append(nn.Linear(in_size, config.hidden_size, bias=config.use_bias, device=device, dtype=dtype))

        self.layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization."""
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-layer expert.
        
        Args:
            x: Input tensor [*, hidden_size].
        
        Returns:
            Output tensor [*, hidden_size].
        """
        return self.layers(x)


class YvSharedExpert(YvExpertBase):
    """Shared expert that is always active in MoE layers.
    
    Implements a shared expert used in DeepSeek-V3 style MoE architecture.
    Unlike routed experts, the shared expert processes all tokens without
    routing decisions, providing a base transformation.
    
    Architecture:
        Uses YvSwiGLUExpert internally for the transformation.
    
    Purpose:
        - Provides base transformation for all tokens
        - Ensures all tokens receive some processing
        - Complements routed experts with shared knowledge
    
    Attributes:
        expert (YvSwiGLUExpert): Internal SwiGLU expert.
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.SHARED
        ... )
        >>> shared_expert = YvSharedExpert(config)
        >>> # All tokens pass through this expert
        >>> output = shared_expert(all_hidden_states)
    
    Note:
        Used in DeepSeek-V3 style MoE with shared expert isolation.
        Output is typically combined with routed expert outputs.
    """

    def __init__(self, config: YvExpertConfig, device=None, dtype=None):
        """Initialize shared expert.
        
        Args:
            config: Expert configuration.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__(config)

        self.expert = YvSwiGLUExpert(config, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared expert.
        
        Args:
            x: Input tensor [*, hidden_size].
        
        Returns:
            Output tensor [*, hidden_size].
        """
        return self.expert(x)


class YvExpertFactory:
    """Factory class for creating expert network instances.
    
    Provides a centralized way to create different types of expert networks
    based on configuration. Supports registration of custom expert types.
    
    Class Attributes:
        _registry (Dict[YvExpertType, type]): Registry mapping
            expert types to their implementing classes.
    
    Example:
        >>> config = YvExpertConfig(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type=YvExpertType.SWIGLU
        ... )
        >>> expert = YvExpertFactory.create(config)
        >>> 
        >>> # Register custom expert type
        >>> YvExpertFactory.register(
        ...     YvExpertType.ATTENTION,
        ...     MyCustomExpert
        ... )
    
    Note:
        Use create_expert_module() for simpler expert creation.
    """

    _registry: Dict[YvExpertType, type] = {
        YvExpertType.STANDARD: YvStandardExpert,
        YvExpertType.SWIGLU: YvSwiGLUExpert,
        YvExpertType.GEGLU: YvGeGLUExpert,
        YvExpertType.MULTI_LAYER: YvMultiLayerExpert,
        YvExpertType.SHARED: YvSharedExpert,
    }

    @classmethod
    def create(
        cls,
        config: YvExpertConfig,
        expert_type: Optional[YvExpertType] = None,
        device=None,
        dtype=None,
        **kwargs
    ) -> YvExpertBase:
        """Create an expert instance based on configuration.
        
        Args:
            config: Expert configuration.
            expert_type: Optional expert type override. Uses config's type if None.
            device: Device to place parameters on.
            dtype: Data type for parameters.
            **kwargs: Additional arguments for specific expert types
                (e.g., num_layers for MULTI_LAYER).
        
        Returns:
            Configured expert instance.
        
        Raises:
            ValueError: If expert_type is not registered.
        
        Example:
            >>> expert = YvExpertFactory.create(
            ...     config,
            ...     expert_type=YvExpertType.SWIGLU
            ... )
        """
        expert_type = expert_type or config.expert_type

        if expert_type not in cls._registry:
            raise ValueError(f"Unknown expert type: {expert_type}")

        expert_class = cls._registry[expert_type]

        if expert_type == YvExpertType.MULTI_LAYER:
            return expert_class(config, num_layers=kwargs.get('num_layers', 2), device=device, dtype=dtype)

        return expert_class(config, device=device, dtype=dtype)

    @classmethod
    def register(cls, expert_type: YvExpertType, expert_class: type):
        """Register a custom expert type.
        
        Args:
            expert_type: Expert type enum value.
            expert_class: Expert class to register.
        
        Example:
            >>> YvExpertFactory.register(
            ...     YvExpertType.ATTENTION,
            ...     MyAttentionExpert
            ... )
        """
        cls._registry[expert_type] = expert_class


def create_expert_module(
    hidden_size: int,
    intermediate_size: int,
    expert_type: str = "swiglu",
    dropout: float = 0.0,
    use_bias: bool = False,
    device=None,
    dtype=None,
    **kwargs
) -> YvExpertBase:
    """Convenience function to create an expert module.
    
    Creates an expert network with the specified parameters without
    requiring explicit configuration object creation.
    
    Args:
        hidden_size: Input and output hidden dimension.
        intermediate_size: Intermediate dimension for expert.
        expert_type: Expert type name string. Default: "swiglu".
        dropout: Dropout probability. Default: 0.0.
        use_bias: Whether to use bias in linear layers. Default: False.
        device: Device to place parameters on.
        dtype: Data type for parameters.
        **kwargs: Additional arguments for specific expert types.
    
    Returns:
        Configured expert instance.
    
    Example:
        >>> expert = create_expert_module(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     expert_type="swiglu"
        ... )
    """
    config = YvExpertConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_type=expert_type,
        dropout=dropout,
        use_bias=use_bias,
    )
    return YvExpertFactory.create(config, device=device, dtype=dtype, **kwargs)
