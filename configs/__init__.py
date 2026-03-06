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

This package provides centralized configuration management for all components
of the PiscesL1 ecosystem, including model architectures, training parameters,
dataset specifications, and system-level settings.

Package Structure:
    configs/
    ├── __init__.py          # Package initialization and exports
    ├── version.py           # Version and release information
    ├── dataset.yaml         # Dataset configuration registry
    ├── watermark.yaml       # Watermarking configuration
    └── model/               # Model architecture configurations
        ├── 0.5B.yaml        # 0.5 billion parameter model config
        ├── 1.5B.yaml        # 1.5 billion parameter model config
        ├── 7B.yaml          # 7 billion parameter model config
        ├── 70B.yaml         # 70 billion parameter model config
        ├── 128B.yaml        # 128 billion parameter model config
        ├── 314B.yaml        # 314 billion parameter model config
        ├── 671B.yaml        # 671 billion parameter model config
        └── 1T.yaml          # 1 trillion parameter model config

Configuration Hierarchy:
    The configuration system follows a hierarchical approach:
    
    1. System-Level Configuration:
       - Hardware resources (GPU count, memory limits)
       - Distributed training settings
       - Logging and monitoring parameters
    
    2. Model-Level Configuration:
       - Architecture specifications (layers, hidden size, attention heads)
       - Mixture-of-Experts parameters (num experts, top-k routing)
       - Multimodal component settings (vision, audio, video encoders)
       - Tokenization parameters
    
    3. Training-Level Configuration:
       - Optimizer settings (learning rate, weight decay, scheduler)
       - Batch size and gradient accumulation
       - Precision settings (FP32, BF16, FP8, quantization)
       - Checkpoint and logging intervals
    
    4. Dataset-Level Configuration:
       - Dataset paths and formats
       - Preprocessing pipelines
       - Data augmentation settings

Model Configuration Schema:
    Each model configuration JSON file follows this schema:
    
    {
        "model_type": "ruchbah",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "rope_theta": 500000.0,
        "moe": {
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "shared_experts": 2
        },
        "multimodal": {
            "vision_encoder": {...},
            "audio_encoder": {...},
            "video_encoder": {...}
        }
    }

Usage Examples:
    Loading a model configuration:
        import json
        with open('configs/model/7B.json') as f:
            config = json.load(f)
    
    Accessing version information:
        from configs.version import VERSION, CVERSION, AUTHOR
    
    Loading dataset configuration:
        import yaml
        with open('configs/dataset.yaml') as f:
            dataset_config = yaml.safe_load(f)

Configuration Validation:
    All configurations are validated at load time against predefined schemas.
    Invalid configurations will raise descriptive error messages indicating
    the specific validation failure.

Thread Safety:
    Configuration objects are immutable after initialization and can be
    safely shared across threads without additional synchronization.

Integration Points:
    - Model initialization: configs/model/*.yaml
    - Training pipeline: configs/dataset.yaml
    - Watermarking system: configs/watermark.yaml
    - Version tracking: configs/version.py
"""
