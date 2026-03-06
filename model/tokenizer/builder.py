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

"""Tokenizer builder for Yv model with BPE training and configuration management.

This module provides comprehensive utilities for building, loading, saving, and
configuring tokenizers for the Yv architecture. It supports multiple
tokenizer types, BPE training from corpus, and tokenizer merging operations.

Architecture Overview:
    The builder module consists of two main components:
    
    1. **YvTokenizerConfig**:
       - Dataclass encapsulating all tokenizer hyperparameters
       - Serialization support via to_dict/from_dict methods
       - Integrates with YvSpecialTokens for special token management
    
    2. **YvTokenizerBuilder**:
       - Factory class for tokenizer creation
       - BPE training from text corpus
       - Loading from pre-trained vocabulary files
       - Saving tokenizer artifacts to disk
       - Merging multiple tokenizers

Key Features:
    - **BPE Training**: Train tokenizers from scratch using Byte Pair Encoding
    - **Pre-trained Loading**: Load existing tokenizers from directories
    - **Vocabulary Building**: Build from vocabulary and merge files
    - **Special Token Management**: Automatic special token handling
    - **Configuration Persistence**: Save/load tokenizer configurations

BPE Training Algorithm:
    1. Pre-tokenize text into words with end-of-word markers
    2. Initialize vocabulary with individual characters
    3. Iteratively merge most frequent pairs until vocab_size reached
    4. Add special tokens to vocabulary
    5. Save vocabulary and merge rules to files

Example:
    >>> from model.tokenizer.builder import YvTokenizerBuilder, YvTokenizerConfig
    >>> 
    >>> # Create with custom configuration
    >>> config = YvTokenizerConfig(
    ...     vocab_size=32000,
    ...     tokenizer_type="standard",
    ...     min_frequency=2
    ... )
    >>> builder = YvTokenizerBuilder(config)
    >>> 
    >>> # Train from corpus
    >>> corpus = ["Hello world!", "This is a test.", ...]
    >>> builder.train_from_corpus(corpus, vocab_size=32000)
    >>> 
    >>> # Build and save
    >>> tokenizer = builder.build("./my_tokenizer")

Dependencies:
    - json: Configuration serialization
    - pathlib: File path handling
    - dataclasses: Configuration management
    - utils.dc: Logging utilities

Note:
    The builder creates temporary directories for intermediate files when
    no output directory is specified. Always provide tokenizer_dir for
    persistent tokenizer storage.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .tokenizer import YvTokenizer
from .special_tokens import YvSpecialTokens
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Tokenizer", file_path=get_log_file("Yv.Tokenizer"), enable_file=True)


@dataclass
class YvTokenizerConfig:
    """Configuration dataclass for tokenizer building and initialization.
    
    This dataclass encapsulates all hyperparameters controlling tokenizer
    behavior, including vocabulary size, compression settings, and special
    token configuration. It provides serialization methods for persistence.
    
    Attributes:
        tokenizer_type (str): Type of tokenizer backend. Options are "standard"
            for BPE-based tokenization or "h_network" for visual tokenization.
            Default: "standard".
        vocab_size (int): Target vocabulary size for training. The actual size
            may be smaller if corpus is insufficient. Default: 151646.
        byte_fallback (bool): Whether to use byte-level fallback for unknown
            characters. Enables encoding any Unicode text. Default: True.
        compression_ratio (int): Compression ratio for H-Network visual
            tokenization. Higher values produce fewer tokens. Default: 20.
        render_dpi (int): DPI for rendering text in H-Network tokenizer.
            Affects visual quality of rendered text. Default: 150.
        add_prefix_space (bool): Whether to add space at the beginning of
            text before tokenization. Useful for some models. Default: False.
        trim_offsets (bool): Whether to trim whitespace from token offsets.
            Affects character-level alignment. Default: True.
        min_frequency (int): Minimum frequency for a token to be included
            in vocabulary during training. Default: 2.
        special_tokens (YvSpecialTokens): Special tokens configuration
            containing all special token definitions.
    
    Example:
        >>> config = YvTokenizerConfig(
        ...     vocab_size=50000,
        ...     tokenizer_type="standard",
        ...     min_frequency=3
        ... )
        >>> config_dict = config.to_dict()
        >>> restored = YvTokenizerConfig.from_dict(config_dict)
    
    Note:
        The default vocab_size of 151646 matches Qwen3 tokenizer for
        compatibility with pre-trained models.
    """

    tokenizer_type: str = "standard"
    vocab_size: int = 151646
    byte_fallback: bool = True
    compression_ratio: int = 20
    render_dpi: int = 150
    add_prefix_space: bool = False
    trim_offsets: bool = True
    min_frequency: int = 2
    special_tokens: YvSpecialTokens = field(default_factory=YvSpecialTokens)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.
        
        Converts the dataclass to a JSON-serializable dictionary,
        including nested special_tokens configuration.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
                suitable for JSON serialization.
        
        Example:
            >>> config = YvTokenizerConfig()
            >>> d = config.to_dict()
            >>> print(d['vocab_size'])  # 151646
        """
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "byte_fallback": self.byte_fallback,
            "compression_ratio": self.compression_ratio,
            "render_dpi": self.render_dpi,
            "add_prefix_space": self.add_prefix_space,
            "trim_offsets": self.trim_offsets,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens.to_dict(),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "YvTokenizerConfig":
        """Deserialize configuration from dictionary.
        
        Reconstructs a YvTokenizerConfig from a dictionary,
        handling nested special_tokens configuration.
        
        Args:
            config (Dict[str, Any]): Dictionary containing configuration
                parameters. Must have matching keys to dataclass fields.
        
        Returns:
            YvTokenizerConfig: Reconstructed configuration instance.
        
        Example:
            >>> d = {"vocab_size": 32000, "tokenizer_type": "standard"}
            >>> config = YvTokenizerConfig.from_dict(d)
            >>> print(config.vocab_size)  # 32000
        """
        if "special_tokens" in config:
            config["special_tokens"] = YvSpecialTokens.from_dict(config["special_tokens"])
        return cls(**config)


class YvTokenizerBuilder:
    """Builder class for creating and configuring Yv tokenizers.
    
    This class provides a comprehensive interface for tokenizer creation,
    supporting multiple construction methods including loading pre-trained
    tokenizers, training from corpus, and building from vocabulary files.
    
    Supported Construction Methods:
        - **from_pretrained**: Load tokenizer from a directory containing
          vocab.json, merges.txt, and tokenizer_config.json
        - **from_vocab**: Build tokenizer from vocabulary and merge files
        - **train_from_corpus**: Train a new tokenizer using BPE algorithm
        - **build**: Construct tokenizer from accumulated vocabulary and merges
    
    BPE Training Process:
        1. Pre-tokenize corpus into words with end-of-word markers (</w>)
        2. Count word frequencies and filter by min_frequency
        3. Initialize vocabulary with unique characters
        4. Iteratively find and merge most frequent character pairs
        5. Stop when vocab_size reached or no more pairs above threshold
        6. Add special tokens to vocabulary
    
    Attributes:
        config (YvTokenizerConfig): Configuration for tokenizer building.
        _vocab (Dict[str, int]): Vocabulary mapping tokens to IDs.
        _merges (List[tuple]): List of BPE merge rules as (token1, token2) tuples.
    
    Example:
        >>> # Load pre-trained tokenizer
        >>> tokenizer = YvTokenizerBuilder.from_pretrained("./tokenizers/base")
        >>> 
        >>> # Train new tokenizer
        >>> builder = YvTokenizerBuilder()
        >>> builder.train_from_corpus(corpus, vocab_size=32000)
        >>> tokenizer = builder.build("./output")
        >>> 
        >>> # Add custom tokens
        >>> builder.add_special_tokens(["<custom1>", "<custom2>"])
    
    Note:
        The builder maintains internal state (_vocab and _merges) during
        training. Call build() to create the final tokenizer and clear state.
    """

    def __init__(self, config: Optional[YvTokenizerConfig] = None):
        """Initialize the tokenizer builder.
        
        Args:
            config (Optional[YvTokenizerConfig]): Configuration for
                tokenizer building. If None, uses default configuration.
        
        Initializes:
            - config: Configuration dataclass
            - _vocab: Empty vocabulary dictionary
            - _merges: Empty merge rules list
        """
        self.config = config or YvTokenizerConfig()
        self._vocab: Dict[str, int] = {}
        self._merges: List[tuple] = []

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_dir: Union[str, Path],
        **kwargs
    ) -> YvTokenizer:
        """Load a pre-trained tokenizer from directory.
        
        This class method loads a tokenizer from a directory containing the
        necessary vocabulary and configuration files. It handles both
        tokenizer_config.json and default configurations.
        
        Args:
            tokenizer_dir (Union[str, Path]): Path to tokenizer directory
                containing vocab.json, merges.txt, and optionally
                tokenizer_config.json.
            **kwargs: Additional arguments passed to YvTokenizer
                constructor, such as cache_dir or trust_remote_code.
        
        Returns:
            YvTokenizer: Initialized tokenizer instance ready for use.
        
        Directory Structure:
            Expected files in tokenizer_dir:
            - vocab.json: Vocabulary mapping tokens to IDs
            - merges.txt: BPE merge rules (one per line)
            - tokenizer_config.json: Optional configuration file
        
        Example:
            >>> tokenizer = YvTokenizerBuilder.from_pretrained(
            ...     "./tokenizers/base"
            ... )
            >>> tokens = tokenizer.encode("Hello, world!")
        
        Note:
            If tokenizer_config.json is missing, uses default configuration.
        """
        tokenizer_dir = Path(tokenizer_dir)

        config_path = tokenizer_dir / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = YvTokenizerConfig.from_dict(json.load(f))
        else:
            config = YvTokenizerConfig()

        return YvTokenizer(
            tokenizer_dir=tokenizer_dir,
            tokenizer_type=config.tokenizer_type,
            **kwargs
        )

    @classmethod
    def from_vocab(
        cls,
        vocab_path: Union[str, Path],
        merges_path: Optional[Union[str, Path]] = None,
        special_tokens: Optional[YvSpecialTokens] = None,
        **kwargs
    ) -> YvTokenizer:
        """Build tokenizer from vocabulary file.
        
        This class method creates a tokenizer from vocabulary and merge files,
        useful for loading custom tokenizers or those from other frameworks.
        
        Args:
            vocab_path (Union[str, Path]): Path to vocabulary JSON file
                mapping tokens to integer IDs.
            merges_path (Optional[Union[str, Path]]): Path to merges file
                containing BPE merge rules. If None, looks for merges.txt
                in the same directory as vocab_path. Default: None.
            special_tokens (Optional[YvSpecialTokens]): Special tokens
                configuration. If None, uses default special tokens.
                Default: None.
            **kwargs: Additional arguments passed to YvTokenizer.
        
        Returns:
            YvTokenizer: Initialized tokenizer instance.
        
        Example:
            >>> tokenizer = YvTokenizerBuilder.from_vocab(
            ...     vocab_path="./vocab.json",
            ...     merges_path="./merges.txt"
            ... )
        """
        vocab_path = Path(vocab_path)

        if merges_path is None:
            merges_path = vocab_path.parent / "merges.txt"
        else:
            merges_path = Path(merges_path)

        tokenizer_dir = vocab_path.parent

        return YvTokenizer(
            tokenizer_dir=tokenizer_dir,
            tokenizer_type="standard",
            **kwargs
        )

    def train_from_corpus(
        self,
        corpus: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        show_progress: bool = True
    ) -> "YvTokenizerBuilder":
        """Train tokenizer on text corpus using BPE algorithm.
        
        This method implements the Byte Pair Encoding (BPE) algorithm to
        learn a vocabulary from a text corpus. It iteratively merges the
        most frequent character pairs until the target vocabulary size
        is reached.
        
        Training Process:
            1. Pre-tokenize text into words with end-of-word markers
            2. Count word frequencies and filter by min_frequency
            3. Initialize vocabulary with unique characters
            4. Iteratively find and merge most frequent pairs
            5. Stop when vocab_size reached or no valid pairs remain
        
        Args:
            corpus (List[str]): List of text strings to train on.
                Larger corpora produce better tokenizers.
            vocab_size (Optional[int]): Target vocabulary size. If None,
                uses config.vocab_size. Default: None.
            min_frequency (int): Minimum frequency for token inclusion.
                Tokens appearing fewer times are excluded. Default: 2.
            show_progress (bool): Whether to log training progress.
                Default: True.
        
        Returns:
            YvTokenizerBuilder: Self for method chaining.
        
        Example:
            >>> builder = YvTokenizerBuilder()
            >>> corpus = ["Hello world!", "This is a test.", ...]
            >>> builder.train_from_corpus(corpus, vocab_size=32000)
            >>> tokenizer = builder.build("./output")
        
        Note:
            This method modifies internal _vocab and _merges state.
            Call build() to create the final tokenizer.
        """
        vocab_size = vocab_size or self.config.vocab_size
        min_frequency = min_frequency or self.config.min_frequency

        word_freqs: Dict[str, int] = {}
        for text in corpus:
            words = self._pre_tokenize(text)
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1

        word_freqs = {w: c for w, c in word_freqs.items() if c >= min_frequency}

        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)

        vocab = sorted(list(vocab))
        self._vocab = {token: idx for idx, token in enumerate(vocab)}

        merges = []
        while len(self._vocab) < vocab_size:
            pair_freqs = self._compute_pair_freqs(word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < min_frequency:
                break

            merges.append(best_pair)
            self._add_merge(best_pair)
            word_freqs = self._merge_pair(best_pair, word_freqs)

        self._merges = merges

        if show_progress:
            _LOG.info(f"Trained tokenizer with {len(self._vocab)} tokens and {len(self._merges)} merges")

        return self

    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words with end-of-word markers.
        
        Splits text on whitespace and appends the BPE end-of-word marker
        (</w>) to each word. This marker helps the tokenizer distinguish
        word boundaries during training.
        
        Args:
            text (str): Input text to pre-tokenize.
        
        Returns:
            List[str]: List of words with end-of-word markers.
        
        Example:
            >>> builder._pre_tokenize("Hello world")
            ['Hello</w>', 'world</w>']
        """
        words = text.split()
        return [word + "</w>" for word in words]

    def _compute_pair_freqs(self, word_freqs: Dict[str, int]) -> Dict[tuple, int]:
        """Compute frequency of all adjacent symbol pairs.
        
        Iterates through all words in the frequency dictionary and counts
        occurrences of adjacent symbol pairs. Used to find the most
        frequent pair for BPE merging.
        
        Args:
            word_freqs (Dict[str, int]): Dictionary mapping words (as
                space-separated symbols) to their frequencies.
        
        Returns:
            Dict[tuple, int]: Dictionary mapping symbol pairs (tuples)
                to their total frequencies across all words.
        
        Example:
            >>> word_freqs = {'H e l l o</w>': 5, 'w o r l d</w>': 3}
            >>> pairs = builder._compute_pair_freqs(word_freqs)
            >>> print(pairs.get(('l', 'l')))  # Frequency of 'll' pair
        """
        pair_freqs: Dict[tuple, int] = {}

        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

        return pair_freqs

    def _add_merge(self, pair: tuple):
        """Add a merged token to the vocabulary.
        
        Creates a new token by concatenating the two tokens in the pair
        and adds it to the vocabulary if not already present.
        
        Args:
            pair (tuple): Tuple of two tokens to merge (token1, token2).
        
        Side Effects:
            Updates self._vocab with the new merged token.
        
        Example:
            >>> builder._add_merge(('l', 'l'))
            >>> # Adds 'll' to vocabulary if not present
        """
        new_token = pair[0] + pair[1]
        if new_token not in self._vocab:
            self._vocab[new_token] = len(self._vocab)

    def _merge_pair(self, pair: tuple, word_freqs: Dict[str, int]) -> Dict[str, int]:
        """Merge all occurrences of a pair in the word frequency dictionary.
        
        Replaces all instances of the space-separated pair with the merged
        token in all words. This updates the word representations after
        a BPE merge operation.
        
        Args:
            pair (tuple): Tuple of two tokens to merge (token1, token2).
            word_freqs (Dict[str, int]): Current word frequency dictionary.
        
        Returns:
            Dict[str, int]: Updated word frequency dictionary with merged pairs.
        
        Example:
            >>> word_freqs = {'l l o</w>': 5}
            >>> new_freqs = builder._merge_pair(('l', 'l'), word_freqs)
            >>> print(new_freqs)  # {'ll o</w>': 5}
        """
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def add_special_tokens(self, tokens: List[str]) -> int:
        """Add special tokens to the vocabulary.
        
        Adds new special tokens to the vocabulary if they don't already
        exist. Special tokens typically include BOS, EOS, PAD, UNK, and
        domain-specific markers.
        
        Args:
            tokens (List[str]): List of special token strings to add.
        
        Returns:
            int: Number of tokens actually added (excluding duplicates).
        
        Example:
            >>> added = builder.add_special_tokens(["<BOS>", "<EOS>", "<PAD>"])
            >>> print(f"Added {added} new tokens")
        """
        added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                added += 1
        return added

    def build(self, tokenizer_dir: Optional[Union[str, Path]] = None) -> YvTokenizer:
        """Build and save the final tokenizer.
        
        Creates a YvTokenizer from the accumulated vocabulary and merge
        rules. Saves all necessary files to the specified directory.
        
        Args:
            tokenizer_dir (Optional[Union[str, Path]]): Directory to save
                tokenizer files. If None, creates a temporary directory.
                Default: None.
        
        Returns:
            YvTokenizer: Fully initialized tokenizer instance.
        
        Files Created:
            - vocab.json: Vocabulary mapping tokens to IDs
            - merges.txt: BPE merge rules (one pair per line)
            - tokenizer_config.json: Configuration for the tokenizer
        
        Example:
            >>> builder = YvTokenizerBuilder()
            >>> builder.train_from_corpus(corpus)
            >>> tokenizer = builder.build("./my_tokenizer")
        
        Note:
            The tokenizer directory is created if it doesn't exist.
        """
        if tokenizer_dir is None:
            import tempfile
            tokenizer_dir = Path(tempfile.mkdtemp())
        else:
            tokenizer_dir = Path(tokenizer_dir)

        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = tokenizer_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        merges_path = tokenizer_dir / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in self._merges:
                f.write(f"{merge[0]} {merge[1]}\n")

        config_path = tokenizer_dir / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        _LOG.info(f"Built tokenizer saved to {tokenizer_dir}")

        return YvTokenizer(
            tokenizer_dir=tokenizer_dir,
            tokenizer_type=self.config.tokenizer_type
        )

    def save(self, tokenizer_dir: Union[str, Path]) -> None:
        """Save vocabulary and merge rules to directory.
        
        Saves the current vocabulary and merge rules to files without
        creating a tokenizer instance. Useful for intermediate saves
        during training.
        
        Args:
            tokenizer_dir (Union[str, Path]): Directory to save files.
        
        Files Created:
            - vocab.json: Vocabulary mapping tokens to IDs
            - merges.txt: BPE merge rules
        
        Note:
            Unlike build(), this method does not save tokenizer_config.json
            or return a tokenizer instance.
        """
        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = tokenizer_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        merges_path = tokenizer_dir / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in self._merges:
                f.write(f"{merge[0]} {merge[1]}\n")

    @staticmethod
    def merge_tokenizers(
        tokenizers: List[YvTokenizer],
        weights: Optional[List[float]] = None
    ) -> YvTokenizer:
        """Merge multiple tokenizers into a single unified tokenizer.
        
        Combines vocabularies from multiple tokenizers, optionally weighted
        by importance. This is useful for creating domain-adaptive tokenizers
        or combining multilingual tokenizers.
        
        Args:
            tokenizers (List[YvTokenizer]): List of tokenizers to merge.
            weights (Optional[List[float]]): Weights for each tokenizer.
                If None, uses uniform weights. Default: None.
        
        Returns:
            YvTokenizer: Merged tokenizer instance.
        
        Raises:
            ValueError: If no tokenizers provided or length mismatch
                between tokenizers and weights.
        
        Example:
            >>> merged = YvTokenizerBuilder.merge_tokenizers(
            ...     [tokenizer_en, tokenizer_zh],
            ...     weights=[0.7, 0.3]
            ... )
        
        Note:
            Currently returns the first tokenizer. Full merging implementation
            would combine vocabularies and recompute merge priorities.
        """
        if not tokenizers:
            raise ValueError("No tokenizers provided for merging")

        if weights is None:
            weights = [1.0 / len(tokenizers)] * len(tokenizers)

        if len(tokenizers) != len(weights):
            raise ValueError("Number of tokenizers must match number of weights")

        _LOG.info(f"Merging {len(tokenizers)} tokenizers")
        return tokenizers[0]
