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

"""Yv Tokenizer Module - Unified tokenization for the Yv architecture.

This module provides comprehensive tokenization utilities for the Yv model,
using the tokenizer.json format from tokenizer/ directory.

Architecture:
    The tokenizer module is organized into three main components:

    1. **YvTokenizer** (tokenizer.py):
       - Unified tokenizer interface using GLM5.1 tokenizer.json
       - Singleton pattern for efficient resource usage
       - Full support for GLM special tokens and chat template
       - Multimodal token support (vision, audio, video)

    2. **YvTokenizerBuilder** (builder.py):
       - Factory class for creating tokenizers
       - Loading from pre-trained tokenizer directories
       - BPE training from corpus (legacy support)

    3. **YvSpecialTokens** (special_tokens.py):
       - Special token definitions for Yv architecture
       - Token-to-ID and ID-to-token mappings
       - Serialization support

Special Tokens (Loaded from tokenizer_config.json):
    =======================  =====================================
    See tokenizer_config.json for complete list.
    Categories: Native, Agentic, Vision, Audio, etc.
    =======================  =====================================

Example:
    >>> from model.tokenizer import YvTokenizer, get_tokenizer
    >>>
    >>> tokenizer = YvTokenizer()
    >>>
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> text = tokenizer.decode(tokens)
    >>>
    >>> messages = [{"role": "user", "content": "Hi"}]
    >>> chat_text = tokenizer.apply_chat_template(messages)

Dependencies:
    - transformers: For AutoTokenizer
    - torch: For tensor operations

Note:
    The tokenizer uses tokenizer.json from the local tokenizer/ directory.
    This file embeds vocabulary, merge rules, and configuration.
"""

from .tokenizer import YvTokenizer, get_tokenizer, EXTENDED_VOCAB_SIZE, POPSSExtendedTokenizerConfig, PiscesLx160KTokenizer
from .builder import YvTokenizerBuilder, YvTokenizerConfig
from .special_tokens import YvSpecialTokens, YvSpecialTokenType

__all__ = [
    "YvTokenizer",
    "get_tokenizer",
    "EXTENDED_VOCAB_SIZE",
    "POPSSExtendedTokenizerConfig",
    "PiscesLx160KTokenizer",
    "YvTokenizerBuilder",
    "YvTokenizerConfig",
    "YvSpecialTokens",
    "YvSpecialTokenType",
]
