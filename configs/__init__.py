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

"""
Configuration Package for PiscesL1 Large Language Model Framework.

This module serves as the central configuration management hub for the entire
PiscesL1 ecosystem. It provides a unified interface for accessing and managing
all configuration parameters across model architectures, training pipelines,
dataset specifications, watermarking systems, and system-level settings.

ARCHITECTURAL OVERVIEW:
    The PiscesL1 configuration system is designed with a hierarchical and
    modular structure that separates concerns while maintaining tight
    integration across components. This design enables:

    1. Separation of Configuration Concerns:
       - Model configurations define architecture-specific parameters
       - Training configurations control optimization and workflow
       - Dataset configurations manage data sources and preprocessing
       - Watermark configurations handle AI-generated content tracking
       - System settings control runtime behavior and development tools

    2. Hierarchical Configuration Loading:
       - Base configurations provide sensible defaults
       - Model-specific configurations override base settings
       - Training-specific configurations fine-tune for specific stages
       - Runtime configurations allow for dynamic adjustments

    3. Cross-Module Integration:
       - Version tracking ensures compatibility across components
       - Configuration validation prevents invalid parameter combinations
       - Schema enforcement maintains configuration integrity

CONFIGURATION FILE STRUCTURE:
    configs/
    ├── __init__.py              # Package initialization, exports, and documentation
    ├── version.py               # Version identifiers and release metadata
    ├── settings.yaml            # System-level and developer mode settings
    ├── dataset.yaml             # Dataset registry with sources and preprocessing
    ├── watermark.yaml           # Comprehensive watermarking configuration
    │
    ├── model/                   # Model architecture configurations
    │   ├── 0.5B.yaml           # 0.5 billion parameters (compact model)
    │   ├── 1.5B.yaml           # 1.5 billion parameters (small model)
    │   ├── 7B.yaml             # 7 billion parameters (standard model)
    │   ├── 32B.yaml            # 32 billion parameters (large model)
    │   ├── 64B.yaml            # 64 billion parameters (extra-large model)
    │   ├── 70B.yaml            # 70 billion parameters (enterprise model)
    │   ├── 128B.yaml           # 128 billion parameters (super model)
    │   ├── 314B.yaml           # 314 billion parameters (ultra model)
    │   ├── 671B.yaml           # 671 billion parameters ( flagship model)
    │   └── 1T.yaml             # 1 trillion parameters (massive model)
    │
    └── train/                   # Training stage configurations
        ├── default.yaml          # Base training configuration template
        ├── pretrain.yaml         # Pre-training stage configuration
        ├── continued_pretrain.yaml  # Continued pre-training configuration
        ├── sft.yaml             # Supervised Fine-Tuning configuration
        ├── alignment_ppo.yaml    # PPO-based alignment configuration
        ├── alignment_dpo.yaml    # DPO-based alignment configuration
        ├── alignment_orpo.yaml   # ORPO-based alignment configuration
        └── specialized.yaml      # Domain-specific training configuration

CONFIGURATION HIERARCHY AND PRECEDENCE:
    Level 1: Framework Defaults (lowest priority)
        - Built-in default values in each configuration schema
        - Provides baseline functionality for all configurations

    Level 2: Model-Specific Configurations
        - Inherits from framework defaults
        - Overrides with model-specific hyperparameters
        - Defines architecture: hidden_size, n_layer, n_head, etc.
        - Specifies MoE parameters: num_experts, top_k, capacity_factor

    Level 3: Training Stage Configurations
        - Inherits from model configuration
        - Overrides with stage-specific settings
        - Defines optimizer, scheduler, batch_size, sequence_length
        - Controls training dynamics: gradient_accumulation, mixed_precision

    Level 4: Runtime Overrides (highest priority)
        - Command-line arguments
        - Environment variables
        - Dynamic configuration updates during execution

CONFIGURATION PARAMETER CATEGORIES:

    A. Model Architecture Parameters:
       - model_type: Unique identifier for the model architecture variant
       - vocab_size: Size of the vocabulary embedding table
       - hidden_size: Dimensionality of the hidden state vectors
       - n_layer: Number of transformer layers in the model
       - n_head: Number of attention heads for multi-head attention
       - n_kv_head: Number of key-value heads (GQA optimization)
       - intermediate_size: FFN layer hidden dimensionality
       - max_position_embeddings: Maximum sequence length supported
       - rope_theta: Base rotation frequency for RoPE embeddings

    B. Mixture-of-Experts (MoE) Parameters:
       - moe_num_experts: Total number of expert networks available
       - moe_top_k: Number of experts to activate per token
       - moe_capacity_factor: Expert capacity multiplier for load balancing
       - moe_load_balance_alpha: Weight for load balancing auxiliary loss
       - moe_noise_std: Standard deviation for routing noise injection
       - moe_prediction_horizon:look-ahead steps for expert prediction
       - moe_routing_temperature: Temperature for softmax routing distribution
       - moe_temperature_min/max: Temperature bounds for dynamic adjustment

    C. Knowledge Density Optimization Parameters:
       - expert_init_method: Initialization strategy for expert weights
       - diversity_weight: Weight for diversity-promoting auxiliary loss
       - mi_weight: Mutual information regularization weight
       - online_clustering: Enable online expert clustering adaptation
       - orthogonality_weight: Expert orthogonality regularization weight
       - routing_entropy_weight: Entropy regularization for routing stability
       - activation_variance_weight: Variance regularization for expert outputs
       - expert_warmup_steps: Steps before expert adaptation begins

    D. Multimodal Configuration Parameters:
       - image_res: Input image resolution (-1 for dynamic)
       - image_patch: Patch size for vision tokenization
       - max_image_res: Maximum supported image resolution
       - mm_tokens: Number of multimodal tokens per image
       - audio_tokens: Number of audio tokens per audio segment
       - enable_patch_pack: Enable packing multiple patches for efficiency

    E. Training Optimization Parameters:
       - optimizer.name: Optimizer algorithm (adamw, sgd, etc.)
       - learning_rate: Base learning rate for training
       - weight_decay: L2 regularization coefficient
       - batch_size: Number of samples per training batch
       - sequence_length: Maximum sequence length per sample
       - gradient_accumulation_steps: Virtual batch size multiplier
       - mixed_precision: Precision mode (bf16, fp16, fp32)
       - flash_attention: Enable memory-efficient attention kernel

    F. Advanced Training Operators:
       - moe_gradient: Enable MoE-specific gradient handling
       - kfac: Enable Kronecker Factorization for preconditioning
       - multitask: Enable multi-task learning support
       - watermark: Enable watermarking during generation
       - modality_scheduler: Enable dynamic modality scheduling

    G. Distributed Training Parameters:
       - distributed.enabled: Enable distributed training mode
       - world_size: Total number of processes in training cluster
       - parallel_3d: Enable 3D parallelism (DP, TP, PP)
       - dp_size: Data parallel group size
       - tp_size: Tensor parallel group size
       - pp_size: Pipeline parallel group size
       - sequence_parallel: Enable sequence parallelism across TP group
       - zero_stage: ZeRO optimization stage (0, 1, 2, 3)

    H. Quantization Parameters:
       - quantization.enabled: Enable weight quantization
       - quant_method: Quantization algorithm (nf4, fp8, int8, etc.)
       - bits: Number of bits for quantized weights
       - group_size: Quantization group size for accuracy preservation
       - symmetric: Use symmetric vs asymmetric quantization

    I. LoRA Adaptation Parameters:
       - lora.enabled: Enable Low-Rank Adaptation
       - lora.r: Rank dimension for LoRA matrices
       - lora_alpha: Scaling factor for LoRA updates
       - lora_dropout: Dropout probability for LoRA layers
       - target_modules: List of module names to apply LoRA

USAGE PATTERNS:

    Pattern 1: Direct Configuration Access
        import yaml
        with open('configs/model/7B.yaml') as f:
            config = yaml.safe_load(f)
        model = PiscesL1Model(config)

    Pattern 2: Configuration Inheritance
        base_config = load_config('configs/train/default.yaml')
        sft_config = load_config('configs/train/sft.yaml')
        merged_config = deep_merge(base_config, sft_config)

    Pattern 3: Version-Aware Loading
        from configs.version import VERSION, CVERSION
        config = load_config('configs/model/7B.yaml')
        assert config['version'] == VERSION

    Pattern 4: Runtime Configuration Override
        config = load_config('configs/train/sft.yaml')
        config['training']['learning_rate'] = 1e-4  # Override

CONFIGURATION VALIDATION:
    All configurations undergo validation against predefined schemas before use:

    1. Type Checking:
       - Ensures parameter values match expected data types
       - Catches type mismatches before runtime errors

    2. Range Validation:
       - Validates numeric parameters within acceptable bounds
       - Prevents unstable configurations (e.g., negative learning rates)

    3. Compatibility Checking:
       - Verifies parameter combinations are compatible
       - Checks version compatibility across components

    4. Required Parameter Verification:
       - Ensures all mandatory parameters are present
       - Provides clear error messages for missing parameters

THREAD SAFETY:
    Configuration objects are immutable after initialization. The framework
    uses copy-on-write semantics to ensure thread-safe access. Multiple
    threads can safely read configurations without additional synchronization.

MEMORY EFFICIENCY:
    Configurations are loaded lazily and cached appropriately. Large
    configuration files (e.g., dataset registries) are processed in
    streaming fashion to minimize memory footprint.

INTEGRATION POINTS:

    Model Module (model/):
        - Loads model/*.yaml for architecture definitions
        - Validates configuration against model schema
        - Extracts hyperparameters for model initialization

    Training Module (opss/train/):
        - Loads train/*.yaml for training stage configurations
        - Merges with model configuration for complete setup
        - Uses training config for optimizer/scheduler setup

    Data Module:
        - Loads dataset.yaml for dataset registry
        - Extracts preprocessing configurations
        - Manages dataset caching and downloading

    Watermark Module:
        - Loads watermark.yaml for watermarking configuration
        - Extracts jurisdiction-specific parameters
        - Configures detection and audit systems

    Inference Module:
        - Loads inference_config from model configs
        - Configures sampling parameters (temperature, top_p, etc.)
        - Sets up vLLM and speculative decoding parameters

PERFORMANCE CONSIDERATIONS:
    - Configuration loading is optimized for minimal overhead
    - YAML parsing uses C-based loader for speed
    - Large lists (datasets) use lazy loading where possible
    - Configuration caching reduces repeated file I/O

EXTENSIBILITY:
    To add new configuration categories:
    1. Create new YAML file in appropriate subdirectory
    2. Define schema according to configuration hierarchy
    3. Add loading/validation logic in relevant module
    4. Update this documentation with new category details

SECURITY CONSIDERATIONS:
    - Configuration files are treated as trusted input
    - No dynamic code execution from configuration values
    - Sensitive parameters (API keys) should use environment variables
    - Configuration validation prevents injection attacks

TROUBLESHOOTING:
    Common issues and solutions:

    Issue: "Missing required parameter: learning_rate"
    Solution: Ensure training config includes all required parameters

    Issue: "Incompatible model and training configurations"
    Solution: Verify model size matches available hardware resources

    Issue: "Configuration version mismatch"
    Solution: Update configuration to match current framework version

REFERENCES:
    - YAML Specification: https://yaml.org/spec/1.2.2/
    - Transformers Library Configuration Patterns
    - DeepSpeed Configuration Guide
    - vLLM Configuration Reference
"""