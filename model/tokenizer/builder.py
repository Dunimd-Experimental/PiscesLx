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

"""Tokenizer builder for Yv model with tokenizer support.

This module provides utilities for building and configuring tokenizers
for the Yv architecture, with support for tokenizer.json format.

Architecture:
    The builder module consists of two main components:

    1. **YvTokenizerConfig**:
       - Dataclass encapsulating tokenizer configuration
       - Serialization support via to_dict/from_dict methods
       - Integrates with YvSpecialTokens for special token management

    2. **YvTokenizerBuilder**:
       - Factory class for tokenizer creation
       - Loading from pre-trained tokenizer directories
       - Legacy BPE training support

Key Features:
    - **GLM5.1 Loading**: Load tokenizer directly from tokenizer.json
    - **Pre-trained Loading**: Load existing tokenizers from directories
    - **Configuration Management**: Persist and restore tokenizer config
    - **Special Token Management**: Automatic special token handling

Example:
    >>> from model.tokenizer import YvTokenizerBuilder, YvTokenizerConfig
    >>>
    >>> config = YvTokenizerConfig()
    >>> tokenizer = YvTokenizerBuilder.from_pretrained("./tokenizer")
    >>>
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> text = tokenizer.decode(tokens)

Dependencies:
    - json: Configuration serialization
    - pathlib: File path handling
    - dataclasses: Configuration management

Note:
    The primary method for loading GLM5.1 tokenizer is through
    YvTokenizer directly, which uses the singleton pattern for
    efficient resource usage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .tokenizer import YvTokenizer
from .special_tokens import YvSpecialTokens
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Tokenizer.Builder", file_path=get_log_file("Yv.Tokenizer.Builder"), enable_file=True)


@dataclass
class YvTokenizerConfig:
    """Configuration dataclass for tokenizer building and initialization.

    This dataclass encapsulates all hyperparameters controlling tokenizer
    behavior, including vocabulary size, model max length, and special
    token configuration. It provides serialization methods for persistence.

    Attributes:
        model_max_length (int): Maximum sequence length. Default: 131072.
        add_prefix_space (bool): Whether to add space at text beginning.
            Default: False.
        trim_offsets (bool): Whether to trim whitespace from token offsets.
            Default: True.
        special_tokens (YvSpecialTokens): Special tokens configuration
            containing all special token definitions.

    Example:
        >>> config = YvTokenizerConfig()
        >>> config_dict = config.to_dict()
        >>> restored = YvTokenizerConfig.from_dict(config_dict)

    Note:
        Default settings are optimized for the tokenizer.
    """

    model_max_length: int = 131072
    add_prefix_space: bool = False
    trim_offsets: bool = True
    special_tokens: YvSpecialTokens = field(default_factory=YvSpecialTokens)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation suitable for JSON.
        """
        return {
            "model_max_length": self.model_max_length,
            "add_prefix_space": self.add_prefix_space,
            "trim_offsets": self.trim_offsets,
            "special_tokens": self.special_tokens.to_dict(),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "YvTokenizerConfig":
        """Deserialize configuration from dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration.

        Returns:
            YvTokenizerConfig: Reconstructed configuration instance.
        """
        if "special_tokens" in config:
            config["special_tokens"] = YvSpecialTokens.from_dict(config["special_tokens"])
        return cls(**config)


class YvTokenizerBuilder:
    """Builder class for creating and configuring Yv tokenizers.

    This class provides a factory interface for tokenizer creation,
    supporting loading from pre-trained tokenizer directories.

    Supported Construction Methods:
        - **from_pretrained**: Load tokenizer from a directory containing
          tokenizer.json (GLM5.1 format) or vocab.json + merges.txt
        - **build**: Create tokenizer with specific configuration

    Attributes:
        config (YvTokenizerConfig): Configuration for tokenizer building.

    Example:
        >>> tokenizer = YvTokenizerBuilder.from_pretrained("./tokenizer")
        >>> tokens = tokenizer.encode("Hello, world!")

    Note:
        For tokenizer, the preferred method is to use YvTokenizer directly
        which uses the singleton pattern for efficient resource usage.
    """

    def __init__(self, config: Optional[YvTokenizerConfig] = None):
        """Initialize the tokenizer builder.

        Args:
            config (Optional[YvTokenizerConfig]): Configuration for
                tokenizer building. If None, uses default configuration.
        """
        self.config = config or YvTokenizerConfig()

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_dir: Union[str, Path],
        **kwargs
    ) -> YvTokenizer:
        """Load a pre-trained tokenizer from directory.

        This class method loads a tokenizer from a directory containing the
        necessary files. Supports tokenizer.json format as well
        as traditional vocab.json + merges.txt format.

        Args:
            tokenizer_dir (Union[str, Path]): Path to tokenizer directory
                containing tokenizer.json or vocab.json + merges.txt.
            **kwargs: Additional arguments passed to YvTokenizer
                constructor.

        Returns:
            YvTokenizer: Initialized tokenizer instance.

        Example:
            >>> tokenizer = YvTokenizerBuilder.from_pretrained(
            ...     "./tokenizer"
            ... )
            >>> tokens = tokenizer.encode("Hello, world!")
        """
        tokenizer_dir = Path(tokenizer_dir)

        config_path = tokenizer_dir / "tokenizer_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = YvTokenizerConfig.from_dict(json.load(f))
                _LOG.info(f"Loaded tokenizer config from {config_path}")
            except Exception as e:
                _LOG.warning(f"Failed to load config: {e}, using defaults")
                config = YvTokenizerConfig()
        else:
            config = YvTokenizerConfig()

        return YvTokenizer(
            tokenizer_dir=tokenizer_dir,
            **kwargs
        )

    def build(
        self,
        tokenizer_dir: Optional[Union[str, Path]] = None,
    ) -> YvTokenizer:
        """Build tokenizer from directory.

        Args:
            tokenizer_dir (Optional[Union[str, Path]]): Path to tokenizer
                directory. If None, uses project tokenizer/ directory.

        Returns:
            YvTokenizer: Initialized tokenizer instance.
        """
        if tokenizer_dir is None:
            tokenizer_dir = Path("tokenizer")

        return YvTokenizer(tokenizer_dir=tokenizer_dir)

    @staticmethod
    def save_config(
        config: YvTokenizerConfig,
        save_path: Union[str, Path],
    ) -> None:
        """Save tokenizer configuration to file.

        Args:
            config (YvTokenizerConfig): Configuration to save.
            save_path (Union[str, Path]): Path to save configuration.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        _LOG.info(f"Tokenizer config saved to {save_path}")
