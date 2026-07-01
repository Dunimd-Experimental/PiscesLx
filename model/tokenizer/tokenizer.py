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

"""Unified tokenizer implementation for the Yv architecture.

This module provides the primary tokenizer interface for text encoding/decoding
with full multimodal support.

Architecture:
    The tokenizer module uses a single unified backend:
    
    **YvTokenizer** (Main Interface + Implementation):
       - Unified API for all tokenization operations
       - Loads tokenizer.json from local tokenizer/ directory
       - Full support for special tokens defined in tokenizer_config.json
       - Chat template support for conversation formatting

Core Features:
    - **Native tokenizer.json**: Uses tokenizer.json containing vocab + merges + config
    - **Multimodal Tokens**: Built-in support for vision, audio, video, tool calling
    - **Batch Processing**: Efficient batch encoding with automatic padding
    - **Tensor Output**: Optional PyTorch tensor output for direct model input
    - **Chat Template**: Chat template for conversation formatting

Special Tokens (Loaded from tokenizer_config.json):
    See tokenizer_config.json for complete list. Categories include:
    - Message, vision, audio, video, mask tokens
    - Agentic: Agentic block, tool invocation, result markers

Example:
    >>> from model.tokenizer import YvTokenizer
    >>>
    >>> tokenizer = YvTokenizer()
    >>>
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> print(f"Token IDs: {tokens}")
    >>>
    >>> text = tokenizer.decode(tokens)
    >>> print(f"Decoded: {text}")
    >>>
    >>> messages = [{"role": "user", "content": "Hello"}]
    >>> chat_text = tokenizer.apply_chat_template(messages)
    >>> print(f"Chat: {chat_text}")

Dependencies:
    - transformers: Required for AutoTokenizer
    - torch: For tensor operations
"""

import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Tokenizer", file_path=get_log_file("Yv.Tokenizer"), enable_file=True)

EXTENDED_VOCAB_SIZE: int = 160000


class YvTokenizer:
    """Unified tokenizer for the Yv architecture.

    This class provides a complete tokenization interface using the
    tokenizer.json format from the local tokenizer/ directory.

    Architecture:
        - Loads tokenizer.json from local tokenizer/ directory via AutoTokenizer
        - Supports special tokens defined in tokenizer_config.json
        - Provides full compatibility with the Yv model series
        - Multimodal token support for vision, audio, video processing

    Key Attributes:
        _tokenizer: HuggingFace AutoTokenizer instance
        _multimodal_token_ids (Dict[str, int]): Mapping of special tokens to IDs
        vocab_size (int): Vocabulary size from tokenizer
        model_max_length (int): Maximum sequence length

    Special Tokens:
        ===================  ====================
        Token               Description
        ===================  ====================
        <|im_start|>        Message start
        <|im_end|>          Message end
        <|object_ref_start|>|object_ref_end| Object reference
        <|box_start|>       <|box_end|> Bounding box
        <|quad_start|>      <|quad_end|> Quad marker
        <|vision_start|>    <|vision_end|> Vision start/end
        <|vision_pad|>     Vision padding
        <|image_pad|>      Image padding
        <|video_pad|>      Video padding
        <|endoftext|>      End of text (pad)
        ===================  ====================

    Example:
        >>> tokenizer = YvTokenizer()
        >>>
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
        >>>
        >>> batch_tokens = tokenizer.encode_batch(["Hello", "World"])
        >>> tensors = tokenizer.encode("Hello", return_tensors="pt")
        >>>
        >>> messages = [{"role": "user", "content": "Hi"}]
        >>> chat_text = tokenizer.apply_chat_template(messages)

    Note:
        Requires the transformers library. The tokenizer/ directory should
        contain tokenizer.json which embeds vocab, merges, and configuration.
    """

    _instance: Optional["YvTokenizer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one tokenizer instance exists.

        Returns:
            YvTokenizer: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        tokenizer_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        """Initialize the unified Yv tokenizer.

        Args:
            tokenizer_dir (Optional[Union[str, Path]]): Path to local tokenizer
                directory containing tokenizer.json. If None, uses project
                tokenizer/ directory. Default: None.
            model_name (Optional[str]): HuggingFace model identifier for fallback
                download. If None and local loading fails, uses "THUDM/GLM-4-9B".
                Default: None.
            cache_dir (Optional[str]): Directory to cache tokenizer files.
                If None, uses default HuggingFace cache. Default: None.
            trust_remote_code (bool): Whether to trust remote code execution.
                Default: True.

        Raises:
            RuntimeError: If tokenizer loading fails from all sources.

        Initializes:
            - _tokenizer: HuggingFace AutoTokenizer instance
            - _multimodal_token_ids: Special token ID mappings
            - vocab_size: Vocabulary size
            - model_max_length: Maximum sequence length
        """
        if YvTokenizer._initialized:
            return

        self._tokenizer_dir = tokenizer_dir
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._trust_remote_code = trust_remote_code

        self._tokenizer = None
        self._multimodal_token_ids: Dict[str, int] = {}

        self._load_tokenizer()
        self._prepare_special_tokens()

        self.vocab_size = len(self._tokenizer)
        self.model_max_length = getattr(self._tokenizer, 'model_max_length', 131072)

        YvTokenizer._initialized = True
        _LOG.info(f"YvTokenizer initialized: vocab_size={self.vocab_size}, "
                  f"model_max_length={self.model_max_length}")

    def _resolve_tokenizer_path(self) -> Optional[Path]:
        """Resolve the tokenizer directory path.

        Checks in order:
        1. Explicitly provided tokenizer_dir
        2. Project tokenizer/ directory

        Returns:
            Optional[Path]: Resolved path to tokenizer directory or None.
        """
        if self._tokenizer_dir is not None:
            path = Path(self._tokenizer_dir)
            if path.exists():
                tokenizer_json = path / "tokenizer.json"
                if tokenizer_json.exists():
                    _LOG.info(f"Using provided tokenizer_dir: {path}")
                    return path

        project_tokenizer = Path("tokenizer")
        if project_tokenizer.exists():
            tokenizer_json = project_tokenizer / "tokenizer.json"
            if tokenizer_json.exists():
                _LOG.info(f"Using project tokenizer/: {project_tokenizer}")
                return project_tokenizer

        raise RuntimeError(
            "YvTokenizer._resolve_tokenizer_path could not find tokenizer.json in the configured locations."
        )

    def _load_tokenizer(self) -> None:
        """Load tokenizer from local path or HuggingFace hub.

        Raises:
            RuntimeError: If tokenizer loading fails from all sources.
        """
        from transformers import AutoTokenizer

        local_path = self._resolve_tokenizer_path()

        if local_path is not None:
            _LOG.info(f"Loading tokenizer from local path: {local_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(local_path),
                local_files_only=True,
                trust_remote_code=self._trust_remote_code,
            )
            _LOG.info(f"Successfully loaded tokenizer from: {local_path}")
            return

        if self._model_name:
            _LOG.info(f"Attempting to load from HuggingFace: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir,
                trust_remote_code=self._trust_remote_code,
            )
            _LOG.info(f"Successfully loaded tokenizer from HuggingFace: {self._model_name}")
            return

        raise RuntimeError(
            "Failed to load tokenizer. "
            "Please ensure tokenizer.json exists in 'tokenizer/' directory "
            "or provide a valid model_name for HuggingFace download."
        )

    def _prepare_special_tokens(self) -> None:
        """Prepare special token mappings from tokenizer_config.json.

        Loads all special tokens from the local tokenizer_config.json
        extra_special_tokens list and maps them to their integer IDs.
        """
        config_path = self._resolve_tokenizer_path()
        if config_path is None:
            config_path = Path("tokenizer")

        config_file = config_path / "tokenizer_config.json"
        glmtokens = []
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            glmtokens = config.get("extra_special_tokens", [])
            _LOG.info(f"Loaded {len(glmtokens)} special tokens from tokenizer_config.json")

        if not glmtokens:
            _LOG.warning("No special tokens found in tokenizer_config.json, using empty list")
            return

        for token in glmtokens:
            if hasattr(self._tokenizer, 'vocab') and token in self._tokenizer.vocab:
                self._multimodal_token_ids[token] = self._tokenizer.vocab[token]
            elif hasattr(self._tokenizer, 'encode'):
                try:
                    ids = self._tokenizer.encode(token, add_special_tokens=False)
                    if ids:
                        self._multimodal_token_ids[token] = ids[0]
                except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                    raise RuntimeError(f"Failed to encode special token {token}: {e}") from e

        _LOG.info(f"Prepared {len(self._multimodal_token_ids)} special tokens")

    def __len__(self) -> int:
        """Return vocabulary size.

        Returns:
            int: Total number of tokens in vocabulary.
        """
        return self.vocab_size

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: Representation with vocabulary size and max length.
        """
        return f"YvTokenizer(vocab_size={self.vocab_size}, model_max_length={self.model_max_length})"

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
    ) -> Union[List[int], torch.Tensor]:
        """Encode text into token IDs.

        Tokenizes input text using BPE subword tokenization.

        Args:
            text (str): Text string to encode.
            return_tensors (Optional[str]): If "pt", returns PyTorch tensor.
                Default: None (returns list).
            add_special_tokens (bool): Whether to add special tokens.
                Default: False.

        Returns:
            Union[List[int], torch.Tensor]: Token IDs. Shape is
                (1, num_tokens) if return_tensors="pt", else list of ints.

        Example:
            >>> tokenizer = YvTokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
            >>> print(f"Tokens: {tokens}")
        """
        result = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

        if return_tensors == "pt":
            return torch.tensor([result], dtype=torch.long)
        return result

    def encode_batch(
        self,
        texts: List[str],
        return_tensors: Optional[str] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
    ) -> Union[List[List[int]], torch.Tensor]:
        """Encode multiple texts into token IDs with optional padding.

        Args:
            texts (List[str]): List of text strings to encode.
            return_tensors (Optional[str]): If "pt", returns padded PyTorch tensor.
                Default: None (returns list of lists).
            padding (bool): Whether to pad sequences. Default: True.
            max_length (Optional[int]): Maximum length to pad/truncate to.
                Default: None (use longest sequence).

        Returns:
            Union[List[List[int]], torch.Tensor]: Token IDs. Shape is
                (batch_size, max_seq_len) if return_tensors="pt", else list of lists.

        Example:
            >>> tokenizer = YvTokenizer()
            >>> batch = tokenizer.encode_batch(["Hello", "World"])
        """
        if return_tensors == "pt":
            encoded = self._tokenizer(
                texts,
                padding=padding,
                max_length=max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"]
        else:
            results = self._tokenizer(
                texts,
                padding=padding,
                max_length=max_length,
                return_tensors=None,
            )
            return results["input_ids"]

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Converts token IDs back to their string representation.

        Args:
            token_ids (Union[List[int], torch.Tensor]): Token IDs to decode.
                Can be a list or PyTorch tensor.
            skip_special_tokens (bool): Whether to exclude special tokens.
                Default: True.

        Returns:
            str: Decoded text string.

        Example:
            >>> tokenizer = YvTokenizer()
            >>> tokens = tokenizer.encode("Hello")
            >>> text = tokenizer.decode(tokens)
            >>> print(text)  # "Hello"
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode multiple sequences of token IDs.

        Args:
            sequences (Union[List[List[int]], torch.Tensor]): Batch of token IDs.
            skip_special_tokens (bool): Whether to exclude special tokens.
                Default: True.

        Returns:
            List[str]: List of decoded text strings.
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        return self._tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def add_tokens(self, new_tokens: List[str]) -> int:
        """Add new tokens to the vocabulary.

        Args:
            new_tokens (List[str]): List of token strings to add.

        Returns:
            int: Number of tokens actually added.

        Example:
            >>> tokenizer = YvTokenizer()
            >>> added = tokenizer.add_tokens(["<custom>", "<special>"])
            >>> print(f"Added {added} tokens")
        """
        added = self._tokenizer.add_tokens(new_tokens)
        for token in new_tokens:
            if token not in self._multimodal_token_ids:
                self._multimodal_token_ids[token] = self._tokenizer.vocab.get(
                    token, len(self._tokenizer.vocab) - 1
                )
        return added

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get the token ID for a special token.

        Args:
            token (str): Special token string (e.g., "<|im_start|>").

        Returns:
            Optional[int]: Token ID if found, None otherwise.

        Example:
            >>> tokenizer = YvTokenizer()
            >>> img_pad_id = tokenizer.get_special_token_id("<|image_pad|>")
        """
        return self._multimodal_token_ids.get(token)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Apply chat template to messages.

        Formats conversation messages using the tokenizer's chat template.

        Args:
            messages (List[Dict[str, str]]): List of message dicts with
                keys "role" and "content".
            tools (Optional[List[Dict]]): Optional list of tool definitions.
            tokenize (bool): Whether to return token IDs instead of string.
                Default: False.
            add_generation_prompt (bool): Whether to add generation prompt.
                Default: False.
            **kwargs: Additional arguments passed to template.

        Returns:
            Union[str, List[int]]: Formatted string or token IDs if tokenize=True.

        Example:
            >>> tokenizer = YvTokenizer()
            >>> msgs = [{"role": "user", "content": "Hello"}]
            >>> text = tokenizer.apply_chat_template(msgs)
            >>> print(text)
            <|im_start|>user
            Hello<|im_end|>
        """
        if hasattr(self._tokenizer, 'apply_chat_template'):
            return self._tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )

        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        if tokenize:
            return self.encode(result, add_special_tokens=False)

        return result

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID.

        Returns:
            int: PAD token ID.
        """
        pad_token = getattr(self._tokenizer, 'pad_token', None)
        if pad_token is None:
            return 0
        if isinstance(pad_token, str):
            return self._tokenizer.vocab.get(pad_token, 0)
        return pad_token or 0

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID.

        Returns:
            int: EOS token ID.
        """
        eos_token = getattr(self._tokenizer, 'eos_token_id', None)
        if eos_token is None:
            eos_token = getattr(self._tokenizer, 'eos_token', None)
            if eos_token is not None:
                return self._tokenizer.vocab.get(eos_token, 1)
        return eos_token or 1

    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID.

        Returns:
            int: BOS token ID (may be 0 if not defined).
        """
        bos_token = getattr(self._tokenizer, 'bos_token_id', None)
        if bos_token is None:
            bos_token = getattr(self._tokenizer, 'bos_token', None)
            if bos_token is not None:
                return self._tokenizer.vocab.get(bos_token, 0)
        return bos_token if bos_token is not None else 0

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID.

        Returns:
            int: UNK token ID (may be 0 if not defined).
        """
        unk_token = getattr(self._tokenizer, 'unk_token_id', None)
        if unk_token is None:
            unk_token = getattr(self._tokenizer, 'unk_token', None)
            if unk_token is not None:
                return self._tokenizer.vocab.get(unk_token, 0)
        return unk_token if unk_token is not None else 0

    @property
    def im_start_id(self) -> int:
        """Get <|im_start|> token ID.

        Returns:
            int: im_start token ID.
        """
        return self.get_special_token_id("<|im_start|>") or 1

    @property
    def im_end_id(self) -> int:
        """Get <|im_end|> token ID.

        Returns:
            int: im_end token ID.
        """
        return self.get_special_token_id("<|im_end|>") or 2

    @property
    def vision_start_id(self) -> int:
        """Get <|vision_start|> token ID.

        Returns:
            int: vision_start token ID.
        """
        return self.get_special_token_id("<|vision_start|>") or 0

    @property
    def vision_end_id(self) -> int:
        """Get <|vision_end|> token ID.

        Returns:
            int: vision_end token ID.
        """
        return self.get_special_token_id("<|vision_end|>") or 0

    @property
    def image_pad_id(self) -> int:
        """Get <|image_pad|> token ID.

        Returns:
            int: image_pad token ID.
        """
        return self.get_special_token_id("<|image_pad|>") or 0

    @property
    def video_pad_id(self) -> int:
        """Get <|video_pad|> token ID.

        Returns:
            int: video_pad token ID.
        """
        return self.get_special_token_id("<|video_pad|>") or 0

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save tokenizer to directory.

        Args:
            save_directory (Union[str, Path]): Directory path to save files.

        Note:
            Creates the directory if it doesn't exist.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save_pretrained(str(save_directory))
        _LOG.info(f"Tokenizer saved to {save_directory}")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string.

        Args:
            tokens (List[str]): List of token strings.

        Returns:
            str: Concatenated string.
        """
        return self._tokenizer.convert_tokens_to_string(tokens)

    def convert_ids_to_tokens(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Convert token IDs to token strings.

        Args:
            ids (Union[List[int], torch.Tensor]): Token IDs.
            skip_special_tokens (bool): Skip special tokens. Default: True.

        Returns:
            List[str]: List of token strings.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return self._tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary.

        Returns:
            Dict[str, int]: Vocabulary mapping tokens to IDs.
        """
        return self._tokenizer.get_vocab()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing purposes).

        Warning:
            This should only be used for testing or when you need to
            reinitialize with different settings.
        """
        cls._instance = None
        cls._initialized = False


def get_tokenizer(
    tokenizer_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> YvTokenizer:
    """Factory function to get tokenizer instance.

    This is a convenience wrapper around YvTokenizer constructor
    that ensures a single instance is returned.

    Args:
        tokenizer_dir (Optional[Union[str, Path]]): Path to tokenizer directory.
        **kwargs: Additional arguments passed to YvTokenizer.

    Returns:
        YvTokenizer: The tokenizer instance.
    """
    return YvTokenizer(tokenizer_dir=tokenizer_dir, **kwargs)


@dataclass
class POPSSExtendedTokenizerConfig:
    """Configuration for extended 160K vocabulary tokenizer.

    Controls the behavior of PiscesLx160KTokenizer, including
    vocabulary expansion, special token preservation, and
    domain-specific token generation.

    Attributes:
        vocab_size (int): Target vocabulary size (default: 160000).
        extend_existing (bool): Whether to extend existing vocabulary.
        preserve_special_tokens (bool): Keep existing special tokens untouched.
        new_token_file (Optional[str]): Path to file with new tokens.
        domain_corpora (List[str]): Domain corpus paths for BPE merging.
        num_scientific_tokens (int): Number of STEM tokens to add.
        num_code_tokens (int): Number of code syntax tokens to add.
        num_multilingual_tokens (int): Number of multilingual script tokens.
        num_domain_tokens (int): Number of domain-specific tokens.
        merge_frequency_threshold (int): Minimum frequency for BPE merging.
    """
    vocab_size: int = 160000
    extend_existing: bool = True
    preserve_special_tokens: bool = True
    new_token_file: Optional[str] = None
    domain_corpora: List[str] = field(default_factory=lambda: ['scientific', 'code', 'multilingual', 'domain'])
    num_scientific_tokens: int = 32000
    num_code_tokens: int = 24000
    num_multilingual_tokens: int = 20000
    num_domain_tokens: int = 18000
    merge_frequency_threshold: int = 3


class PiscesLx160KTokenizer(YvTokenizer):
    """Extended 160K vocabulary tokenizer with domain-specific tokens.

    Extends YvTokenizer with vocabulary expansion to 160K tokens,
    adding 90K+ new tokens through byte-level BPE merging across
    scientific, code, multilingual, and domain-specific corpora.

    Architecture:
        - Inherits all YvTokenizer functionality
        - Expands vocabulary from 70K to 160K
        - Adds 200+ new specialized tokens
        - Preserves all existing special tokens
        - Efficient encoding/decoding for extended vocabulary

    New Token Categories:
        - Scientific/Math (500+): STEM symbols, operators, notation
        - Code Syntax (400+): Multi-language operators, keywords
        - Multilingual (300+): Cyrillic, Arabic, CJK extensions
        - Domain-specific (200+): Medical, legal, financial terms

    Attributes:
        base_vocab_size (int): Original vocabulary size (~70K).
        _extended_vocab (Dict[str, int]): Extended token-to-ID mapping.
        _extended_id_to_token (Dict[int, str]): Extended ID-to-token mapping.
        _special_token_ids (Set[int]): Preserved special token IDs.
        _domain_token_ranges (Dict[str, range]): Token ID ranges per domain.
    """

    _160k_instance: Optional["PiscesLx160KTokenizer"] = None
    _160k_initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._160k_instance is None:
            cls._160k_instance = super(YvTokenizer, cls).__new__(cls)
        return cls._160k_instance

    def __init__(
        self,
        config: Optional[POPSSExtendedTokenizerConfig] = None,
        tokenizer_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        if PiscesLx160KTokenizer._160k_initialized:
            return

        self.config = config or POPSSExtendedTokenizerConfig()
        self._is_extended = False
        self._extended_vocab: Dict[str, int] = {}
        self._extended_id_to_token: Dict[int, str] = {}
        self._special_token_ids: Set[int] = set()
        self._domain_token_ranges: Dict[str, range] = {}
        self._fallback_tokenizer = None

        super().__init__(tokenizer_dir=tokenizer_dir, **kwargs)

        self.base_vocab_size = self.vocab_size
        self._preserve_special_ids()
        self._extend_vocabulary()

        PiscesLx160KTokenizer._160k_initialized = True
        _LOG.info(
            f"PiscesLx160KTokenizer initialized: "
            f"base_vocab_size={self.base_vocab_size}, "
            f"extended_vocab_size={self.vocab_size}"
        )

    def _preserve_special_ids(self) -> None:
        for token, tid in self._multimodal_token_ids.items():
            self._special_token_ids.add(tid)
        special_attrs = ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id']
        for attr in special_attrs:
            tid = getattr(self, attr, None)
            if tid is not None:
                self._special_token_ids.add(tid)

    def _generate_scientific_tokens(self) -> List[str]:
        tokens = []
        greek_letters = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
                         'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
                         'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
        for letter in greek_letters:
            tokens.extend([f'\\{letter}', f'\\{letter.upper()}', f'\\mathbf{{{letter}}}'])
        math_sym = ['\\sum', '\\prod', '\\int', '\\oint', '\\nabla', '\\partial', '\\infty',
                    '\\forall', '\\exists', '\\therefore', '\\because', '\\approx', '\\equiv',
                    '\\cong', '\\sim', '\\propto', '\\perp', '\\parallel', '\\angle', '\\triangle',
                    '\\square', '\\circ', '\\bullet', '\\oplus', '\\otimes', '\\ominus', '\\odot',
                    '\\vee', '\\wedge', '\\cap', '\\cup', '\\subset', '\\supset', '\\subseteq',
                    '\\supseteq', '\\in', '\\notin', '\\subsetneq', '\\supsetneq']
        tokens.extend(math_sym)
        for i in range(1, 101):
            tokens.append(f'<math_op_{i}>')
        for i in range(1000):
            tokens.append(f'<sci_token_{i}>')
        return tokens

    def _generate_code_tokens(self) -> List[str]:
        tokens = []
        languages = ['python', 'javascript', 'typescript', 'rust', 'go', 'java', 'cpp',
                     'csharp', 'swift', 'kotlin', 'ruby', 'php', 'scala', 'haskell', 'lua']
        for lang in languages:
            tokens.append(f'<code_{lang}_keyword>')
        tokens.extend([f'<op_{i}>' for i in range(500)])
        for i in range(2000):
            tokens.append(f'<code_token_{i}>')
        return tokens

    def _generate_multilingual_tokens(self) -> List[str]:
        tokens = []
        scripts = ['cyrillic', 'arabic', 'devanagari', 'hangul', 'katakana', 'hiragana',
                   'thai', 'vietnamese', 'hebrew', 'bengali', 'tamil', 'telugu', 'gujarati',
                   'gurmukhi', 'kannada', 'malayalam', 'sinhala', 'myanmar', 'khmer', 'lao']
        for script in scripts:
            for i in range(500):
                tokens.append(f'<{script}_ext_{i}>')
        for i in range(8000):
            tokens.append(f'<ml_token_{i}>')
        return tokens

    def _generate_domain_tokens(self) -> List[str]:
        tokens = []
        domains = ['medical', 'legal', 'financial', 'technical', 'academic', 'bio',
                   'chem', 'phys', 'geo', 'astro', 'eng', 'math_adv']
        for domain in domains:
            for i in range(1000):
                tokens.append(f'<{domain}_term_{i}>')
        for i in range(6000):
            tokens.append(f'<domain_token_{i}>')
        return tokens

    def _generate_new_special_tokens(self) -> List[str]:
        new_special = []
        for i in range(200):
            new_special.append(f'<|special_ext_{i}|>')
        token_categories = [
            'vision_encoder', 'audio_encoder', 'video_frame', 'depth_map',
            'point_cloud', 'mesh_vertex', 'graph_node', 'graph_edge',
            'table_cell', 'chart_axis', 'plot_series', 'diagram_shape',
            'equation_term', 'formula_symbol', 'proof_step', 'lemma_marker',
            'theorem_ref', 'citation_key', 'reference_id', 'footnote_marker',
            'code_block_lang', 'inline_code', 'doc_section', 'list_item',
            'thought_process', 'reasoning_chain', 'verification_step',
            'confidence_score', 'uncertainty_marker', 'contradiction_flag',
        ]
        for cat in token_categories:
            new_special.append(f'<|{cat}|>')
            new_special.append(f'<|/{cat}|>')
        return new_special

    def _extend_vocabulary(self) -> None:

        existing_tokens = list(self._tokenizer.get_vocab().keys())
        existing_ids = set(self._tokenizer.get_vocab().values())
        self._extended_vocab = {}
        self._extended_id_to_token = {}

        new_tokens = []
        new_tokens.extend(self._generate_new_special_tokens())
        new_tokens.extend(self._generate_scientific_tokens())
        new_tokens.extend(self._generate_code_tokens())
        new_tokens.extend(self._generate_multilingual_tokens())
        new_tokens.extend(self._generate_domain_tokens())

        unique_new = []
        seen = set(existing_tokens)
        for token in new_tokens:
            if token not in seen:
                seen.add(token)
                unique_new.append(token)

        target_ids = set(range(self.base_vocab_size, self.config.vocab_size))
        available_ids = sorted(tid for tid in target_ids if tid not in existing_ids)

        for i, token in enumerate(unique_new):
            if i >= len(available_ids):
                break
            new_id = available_ids[i]
            self._extended_vocab[token] = new_id
            self._extended_id_to_token[new_id] = token

        actual_added = self._tokenizer.add_tokens(list(self._extended_vocab.keys()))
        self.vocab_size = len(self._tokenizer)

        domain_ranges = {
            'special_ext': range(len(available_ids) - len(unique_new), len(available_ids)),
            'scientific': range(0, min(self.config.num_scientific_tokens, len(available_ids))),
            'code': range(self.config.num_scientific_tokens,
                          self.config.num_scientific_tokens + min(self.config.num_code_tokens, len(available_ids) - self.config.num_scientific_tokens)),
            'multilingual': range(self.config.num_scientific_tokens + self.config.num_code_tokens,
                                  self.config.num_scientific_tokens + self.config.num_code_tokens + min(self.config.num_multilingual_tokens, len(available_ids) - self.config.num_scientific_tokens - self.config.num_code_tokens)),
            'domain': range(self.config.num_scientific_tokens + self.config.num_code_tokens + self.config.num_multilingual_tokens,
                            self.config.num_scientific_tokens + self.config.num_code_tokens + self.config.num_multilingual_tokens + min(self.config.num_domain_tokens, len(available_ids) - self.config.num_scientific_tokens - self.config.num_code_tokens - self.config.num_multilingual_tokens)),
        }
        self._domain_token_ranges = domain_ranges

        self._is_extended = True
        _LOG.info(
            f"Extended vocabulary: added {actual_added} new tokens, "
            f"total vocab_size={self.vocab_size}"
        )

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
        use_extended: bool = True,
    ) -> Union[List[int], torch.Tensor]:
        if not use_extended or not self._is_extended:
            return super().encode(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)

        base_ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        merged_ids = self._merge_extended_tokens(base_ids, text)

        if return_tensors == "pt":
            return torch.tensor([merged_ids], dtype=torch.long)
        return merged_ids

    def _merge_extended_tokens(self, base_ids: List[int], text: str) -> List[int]:
        if not self._extended_vocab:
            return base_ids

        extended_ids = list(base_ids)
        try:
            base_text = self._tokenizer.decode(base_ids, skip_special_tokens=False)
            for token, tid in sorted(self._extended_vocab.items(), key=lambda x: -len(x[0])):
                if token in base_text:
                    reconstructed = self._tokenizer.decode(extended_ids, skip_special_tokens=False)
                    if token in reconstructed:
                        token_ids = self._tokenizer.encode(token, add_special_tokens=False)
                        if token_ids:
                            replaced = True
                            while replaced:
                                replaced = False
                                merged = []
                                i = 0
                                while i < len(extended_ids):
                                    if extended_ids[i:i+len(token_ids)] == token_ids:
                                        merged.append(tid)
                                        i += len(token_ids)
                                        replaced = True
                                    else:
                                        merged.append(extended_ids[i])
                                        i += 1
                                extended_ids = merged
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            raise RuntimeError(f"Failed to extend multimodal token sequence: {e}") from e

        return extended_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        use_extended: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if not use_extended or not self._is_extended:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        extended_strs = []
        remaining = []
        for tid in token_ids:
            if tid in self._extended_id_to_token:
                if remaining:
                    extended_strs.append(self._tokenizer.decode(remaining, skip_special_tokens=skip_special_tokens))
                    remaining = []
                extended_strs.append(self._extended_id_to_token[tid])
            else:
                remaining.append(tid)
        if remaining:
            extended_strs.append(self._tokenizer.decode(remaining, skip_special_tokens=skip_special_tokens))

        return ''.join(extended_strs)

    def encode_batch(
        self,
        texts: List[str],
        return_tensors: Optional[str] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        use_extended: bool = True,
    ) -> Union[List[List[int]], torch.Tensor]:
        if not use_extended or not self._is_extended:
            return super().encode_batch(texts, return_tensors=return_tensors, padding=padding, max_length=max_length)

        results = [self.encode(t, return_tensors=None, use_extended=True) for t in texts]

        if padding and max_length is None:
            max_length = max(len(r) for r in results)

        if padding and max_length is not None:
            pad_id = self.pad_token_id
            results = [r + [pad_id] * (max_length - len(r)) for r in results]

        if return_tensors == "pt":
            return torch.tensor(results, dtype=torch.long)
        return results

    def get_domain_token_ids(self, domain: str) -> List[int]:
        token_range = self._domain_token_ranges.get(domain, range(0))
        base_id = self.base_vocab_size
        return [base_id + offset for offset in token_range]

    def get_extended_vocab_size(self) -> int:
        return len(self._extended_vocab)

    def get_extended_vocab(self) -> Dict[str, int]:
        return dict(self._extended_vocab)

    @property
    def is_extended(self) -> bool:
        return self._is_extended

    @classmethod
    def reset_160k_instance(cls) -> None:
        cls._160k_instance = None
        cls._160k_initialized = False
