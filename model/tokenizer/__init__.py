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

"""Yv Tokenizer Module - Multi-backend tokenization for the Yv architecture.

This module provides comprehensive tokenization utilities for the Yv model,
supporting multiple tokenization backends including BPE-based text tokenization
and H-Network visual tokenization. It integrates seamlessly with the model's
multimodal, reasoning, and agent capabilities.

Architecture Overview:
    The tokenizer module is organized into four main components:
    
    1. **YvTokenizer** (tokenizer.py):
       - Primary tokenizer interface with multiple backend support
       - Supports Qwen3 tokenizer for 100+ language coverage
       - Supports H-Network visual tokenizer for image-based processing
       - Unified API for encode/decode operations
    
    2. **YvTokenizerBuilder** (builder.py):
       - Factory class for creating and configuring tokenizers
       - Supports training from corpus using BPE algorithm
       - Supports loading pre-trained tokenizers
       - Supports merging multiple tokenizers
    
    3. **YvSpecialTokens** (special_tokens.py):
       - Comprehensive special token definitions
       - Categories: standard, multimodal, reasoning, agent, control
       - Token-to-ID and ID-to-token mappings
       - Serialization support for saving/loading
    
    4. **Backend Implementations**:
       - _YvQwenTokenizer: Qwen3-based tokenizer with multimodal support
       - _YvHNetworkTokenizer: Visual tokenizer for H-Network processing

Token Categories:
    - **Standard Tokens**: BOS, EOS, PAD, UNK, MASK, SEP, CLS
    - **Multimodal Tokens**: <image>, <audio>, <video>, <document>
    - **Reasoning Tokens**: <think/>, <answer/>, <verify/>, <reasoning/>
    - **Agent Tokens**: <tool/>, <action/>, <observation/>
    - **Control Tokens**: Separators, masks, and structural markers

Supported Backends:
    - **qwen3**: Qwen3 tokenizer with 100+ language support (recommended)
    - **h_network**: Visual tokenizer that renders text as images and
      compresses them into visual token representations

Example:
    >>> from model.tokenizer import YvTokenizer, YvTokenizerBuilder
    >>> 
    >>> # Load pre-trained tokenizer
    >>> tokenizer = YvTokenizerBuilder.from_pretrained("./tokenizers/base")
    >>> 
    >>> # Or create with specific backend
    >>> tokenizer = YvTokenizer(tokenizer_type="qwen3")
    >>> 
    >>> # Encode and decode
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> text = tokenizer.decode(tokens)
    >>> print(f"Tokens: {tokens}, Decoded: {text}")

Dependencies:
    - transformers: For Qwen3 tokenizer backend
    - PIL/Pillow: For H-Network visual tokenization
    - torch: For tensor operations
    - numpy: For array operations

Note:
    The Qwen3 backend requires the transformers library and downloads
    tokenizer files on first use. The H-Network backend provides a
    fallback mechanism when visual processing fails.
"""

from .tokenizer import YvTokenizer, _YvQwenTokenizer, _YvHNetworkTokenizer
from .builder import YvTokenizerBuilder, YvTokenizerConfig
from .special_tokens import YvSpecialTokens, YvSpecialTokenType

__all__ = [
    "YvTokenizer",
    "_YvQwenTokenizer",
    "_YvHNetworkTokenizer",
    "YvTokenizerBuilder",
    "YvTokenizerConfig",
    "YvSpecialTokens",
    "YvSpecialTokenType",
]
