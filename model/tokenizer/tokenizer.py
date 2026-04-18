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

import os
import json
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Tokenizer", file_path=get_log_file("Yv.Tokenizer"), enable_file=True)


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

        return None

    def _load_tokenizer(self) -> None:
        """Load tokenizer from local path or HuggingFace hub.

        Raises:
            RuntimeError: If tokenizer loading fails from all sources.
        """
        from transformers import AutoTokenizer

        local_path = self._resolve_tokenizer_path()

        if local_path is not None:
            _LOG.info(f"Loading tokenizer from local path: {local_path}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path),
                    local_files_only=True,
                    trust_remote_code=self._trust_remote_code,
                )
                _LOG.info(f"Successfully loaded tokenizer from: {local_path}")
                return
            except Exception as e:
                _LOG.warning(f"Failed to load from local path: {e}")

        if self._model_name:
            _LOG.info(f"Attempting to load from HuggingFace: {self._model_name}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name,
                    cache_dir=self._cache_dir,
                    trust_remote_code=self._trust_remote_code,
                )
                _LOG.info(f"Successfully loaded tokenizer from HuggingFace: {self._model_name}")
                return
            except Exception as e:
                _LOG.warning(f"Failed to load from HuggingFace: {e}")

        raise RuntimeError(
            f"Failed to load tokenizer. "
            f"Please ensure tokenizer.json exists in 'tokenizer/' directory "
            f"or provide a valid model_name for HuggingFace download."
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
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                glmtokens = config.get("extra_special_tokens", [])
                _LOG.info(f"Loaded {len(glmtokens)} special tokens from tokenizer_config.json")
            except Exception as e:
                _LOG.warning(f"Failed to load tokenizer_config.json: {e}")

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
                except:
                    pass

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
