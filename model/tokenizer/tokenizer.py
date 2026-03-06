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

"""Multi-backend tokenizer implementation for the Yv architecture.

This module provides the primary tokenizer interface with support for multiple
tokenization backends, including Qwen3-based text tokenization and H-Network
visual tokenization. It offers a unified API for encoding and decoding text
across different tokenization strategies.

Architecture Overview:
    The tokenizer module consists of three main components:
    
    1. **YvTokenizer** (Main Interface):
       - Unified API for all tokenization backends
       - Backend selection via tokenizer_type parameter
       - Support for batch encoding and decoding
       - Multimodal token management
    
    2. **_YvQwenTokenizer** (Qwen3 Backend):
       - Wraps HuggingFace transformers Qwen3 tokenizer
       - 100+ language support with excellent multilingual coverage
       - Custom multimodal token integration
       - Recommended for most use cases
    
    3. **_YvHNetworkTokenizer** (Visual Backend):
       - Converts text to visual representations
       - Renders text as images and compresses to visual tokens
       - Fallback mechanism for error handling
       - Experimental feature for H-Network processing

Tokenization Backends:
    - **qwen3**: Qwen3 tokenizer with 151K+ vocabulary, supporting 100+ languages.
      Uses BPE subword tokenization with byte-level fallback. Recommended for
      production use due to excellent multilingual support and efficiency.
    
    - **h_network**: Visual tokenizer that renders text as images and compresses
      them into visual token representations. Useful for H-Network architectures
      that process text through visual pathways.

Key Features:
    - **Multi-Backend Support**: Switch between tokenization strategies
    - **Multimodal Tokens**: Built-in support for image, audio, video tokens
    - **Batch Processing**: Efficient batch encoding with automatic padding
    - **Tensor Output**: Optional PyTorch tensor output for direct model input
    - **Special Token Management**: Automatic handling of special tokens

Example:
    >>> from model.tokenizer import YvTokenizer
    >>> 
    >>> # Initialize with Qwen3 backend (recommended)
    >>> tokenizer = YvTokenizer(tokenizer_type="qwen3")
    >>> 
    >>> # Encode text
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> print(f"Token IDs: {tokens}")
    >>> 
    >>> # Decode back to text
    >>> text = tokenizer.decode(tokens)
    >>> print(f"Decoded: {text}")
    >>> 
    >>> # Batch encoding with tensor output
    >>> texts = ["Hello", "World"]
    >>> tensors = tokenizer.encode_batch(texts, return_tensors="pt")
    >>> print(f"Shape: {tensors.shape}")

Dependencies:
    - transformers: Required for Qwen3 tokenizer backend
    - PIL/Pillow: Required for H-Network visual tokenization
    - torch: For tensor operations
    - numpy: For array operations

Note:
    The Qwen3 backend requires the transformers library and will download
    tokenizer files on first use. Ensure network connectivity or pre-cache
    the tokenizer files for offline use.
"""

import os
import re
import json
import unicodedata
import urllib.request
from utils.dc import PiscesLxLogger
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from typing import Any, Dict, Optional, List, Union
from pathlib import Path

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Tokenizer", file_path=get_log_file("Yv.Tokenizer"), enable_file=True)

MULTIMODAL_TOKENS = [
    "<agentic>", "</agentic>", 
    "<ag>", "</ag>", 
    "<ag_group>", "</ag_group>",
    "<tool>", "</tool>", 
    "<tn>", "</tn>", 
    "<tp>", "</tp>",
    "<result>", "</result>",
    "<error>", "</error>",
    "<image>", "<audio>", "<video>"
]
SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<think|>", "</think|>", "<|think|>", "</|think|>"] + MULTIMODAL_TOKENS


class _YvHNetworkTokenizer:
    """H-Network tokenizer for visual text processing without traditional tokenization.
    
    This tokenizer converts text to visual tokens by rendering text as images
    and compressing them into high-efficiency visual representations. It provides
    an alternative tokenization pathway for H-Network architectures.
    
    Architecture:
        1. **Text Rendering**: Renders text to images using PIL
        2. **Image Processing**: Resizes to 224x224 and extracts patches
        3. **Patch Compression**: Compresses patches using hash-based encoding
        4. **Token Generation**: Produces visual token IDs from compressed patches
    
    Attributes:
        compression_ratio (int): Ratio for compressing visual patches. Higher
            values produce fewer tokens but may lose detail. Default: 20.
        render_dpi (int): DPI for text rendering. Affects visual quality.
            Default: 150.
        fallback_enabled (bool): Whether to use fallback encoding on errors.
            Default: True.
        visual_vocab_size (int): Size of visual token vocabulary. Default: 8192.
        font (ImageFont): Font used for text rendering.
    
    Example:
        >>> tokenizer = _YvHNetworkTokenizer(compression_ratio=15)
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(f"Visual tokens: {tokens}")
    
    Note:
        This is an experimental tokenizer for H-Network processing. For
        production use, consider the Qwen3 backend instead.
    """
    
    def __init__(self, compression_ratio: int = 20, render_dpi: int = 150, 
                 font_path: Optional[str] = None, fallback_enabled: bool = True):
        """Initialize the H-Network visual tokenizer.
        
        Args:
            compression_ratio (int): Compression ratio for visual patches.
                Higher values produce fewer tokens. Default: 20.
            render_dpi (int): DPI for text rendering. Default: 150.
            font_path (Optional[str]): Path to custom font file. If None,
                uses default system font. Default: None.
            fallback_enabled (bool): Whether to use fallback encoding when
                visual processing fails. Default: True.
        
        Initializes:
            - compression_ratio: Patch compression ratio
            - render_dpi: Text rendering DPI
            - fallback_enabled: Fallback mode flag
            - visual_vocab_size: Vocabulary size (8192)
            - font: PIL font object for rendering
        """
        self.compression_ratio = compression_ratio
        self.render_dpi = render_dpi
        self.fallback_enabled = fallback_enabled
        self.visual_vocab_size = 8192
        
        try:
            self.font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
            
        _LOG.info(f"H-Network tokenizer initialized: compression_ratio={compression_ratio}, "
                   f"render_dpi={render_dpi}, fallback_enabled={fallback_enabled}")
    
    def _render_text_to_image(self, text: str, max_width: int = 1024) -> Image.Image:
        """Render text string to a PIL Image.
        
        Creates an image with white background and renders the text using
        the configured font. Handles multi-line text by splitting on newlines.
        
        Args:
            text (str): Text string to render.
            max_width (int): Maximum image width in pixels. Default: 1024.
        
        Returns:
            Image.Image: PIL Image containing the rendered text.
        
        Note:
            Image dimensions are calculated based on text length and line count.
        """
        lines = text.split('\n')
        max_chars = max(len(line) for line in lines) if lines else 80
        img_width = min(max_width, max_chars * 8 + 20)
        img_height = len(lines) * 20 + 20
        
        image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(image)
        
        y_offset = 10
        for line in lines:
            if line.strip():
                draw.text((10, y_offset), line.strip(), font=self.font, fill='black')
            y_offset += 20
            
        return image
    
    def _compress_visual_tokens(self, image: Image.Image) -> List[int]:
        """Compress image into visual token IDs.
        
        Resizes the image to 224x224, extracts 16x16 patches, and compresses
        them into visual tokens using hash-based encoding.
        
        Args:
            image (Image.Image): PIL Image to compress.
        
        Returns:
            List[int]: List of visual token IDs (max 100 tokens).
        
        Compression Process:
            1. Resize image to 224x224
            2. Extract 16x16 patches (14x14 = 196 patches)
            3. Select patches based on compression_ratio
            4. Hash patch mean values to generate token IDs
        """
        img_array = np.array(image.resize((224, 224)))
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0
        
        patch_size = 16
        patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(3, -1, patch_size, patch_size)
        
        num_patches = patches.shape[1]
        target_tokens = max(1, num_patches // self.compression_ratio)
        
        visual_tokens = []
        for i in range(target_tokens):
            patch_idx = (i * num_patches) // target_tokens
            patch_hash = hash(patches[:, patch_idx].mean().item()) % self.visual_vocab_size
            visual_tokens.append(abs(patch_hash))
            
        return visual_tokens[:100]
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """Encode text into visual token IDs.
        
        Renders text to an image and compresses it into visual tokens.
        Falls back to character-level encoding if visual processing fails.
        
        Args:
            text (str): Text string to encode.
            return_tensors (Optional[str]): If "pt", returns PyTorch tensor.
                Default: None (returns list).
        
        Returns:
            Union[List[int], torch.Tensor]: Visual token IDs. Shape is
                (1, num_tokens) if return_tensors="pt", else list of ints.
        
        Raises:
            RuntimeError: If encoding fails and fallback_enabled is False.
        
        Example:
            >>> tokenizer = _YvHNetworkTokenizer()
            >>> tokens = tokenizer.encode("Hello")
            >>> print(f"Tokens: {tokens}")
        """
        try:
            image = self._render_text_to_image(text)
            visual_tokens = self._compress_visual_tokens(image)
            
            _LOG.debug(f"H-Network encoded '{text[:50]}...' to {len(visual_tokens)} visual tokens")
            
            if return_tensors == "pt":
                return torch.tensor([visual_tokens], dtype=torch.long)
            return visual_tokens
            
        except Exception as e:
            _LOG.error(f"H-Network encoding failed: {e}")
            if self.fallback_enabled:
                _LOG.warning("Falling back to standard BPE tokenizer")
                return [ord(c) % self.visual_vocab_size for c in text[:100]]
            else:
                raise RuntimeError(f"H-Network encoding failed: {e}")
    
    def encode_batch(self, texts: List[str], return_tensors: Optional[str] = None) -> List[List[int]]:
        """Encode multiple texts into visual token IDs.
        
        Args:
            texts (List[str]): List of text strings to encode.
            return_tensors (Optional[str]): Currently ignored for batch encoding.
                Returns list of lists.
        
        Returns:
            List[List[int]]: List of token ID lists, one per input text.
        
        Note:
            Unlike encode(), this method does not support tensor output.
        """
        return [self.encode(text, return_tensors=None) for text in texts]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode visual token IDs back to text.
        
        Note: Visual tokens cannot be accurately decoded back to original text.
        This method provides a best-effort character-level reconstruction.
        
        Args:
            token_ids (List[int]): List of visual token IDs.
            skip_special_tokens (bool): Ignored for visual tokenizer.
        
        Returns:
            str: Reconstructed string (may not match original).
        
        Warning:
            Decoding visual tokens is lossy. Use Qwen3 backend if accurate
            round-trip encoding/decoding is required.
        """
        decoded_chars = []
        for token_id in token_ids:
            char_code = token_id % 256
            if 32 <= char_code <= 126:
                decoded_chars.append(chr(char_code))
            else:
                decoded_chars.append('?')
                
        return ''.join(decoded_chars)
    
    def __len__(self):
        """Return vocabulary size.
        
        Returns:
            int: Visual vocabulary size (8192).
        """
        return self.visual_vocab_size
    
    @property
    def pad_token_id(self):
        """Get padding token ID.
        
        Returns:
            int: PAD token ID (0).
        """
        return 0
    
    @property
    def eos_token_id(self):
        """Get end-of-sequence token ID.
        
        Returns:
            int: EOS token ID (1).
        """
        return 1
    
    @property
    def bos_token_id(self):
        """Get beginning-of-sequence token ID.
        
        Returns:
            int: BOS token ID (2).
        """
        return 2
    
    @property
    def unk_token_id(self):
        """Get unknown token ID.
        
        Returns:
            int: UNK token ID (3).
        """
        return 3


class _YvQwenTokenizer:
    """Qwen3-based tokenizer with multimodal token support.
    
    This tokenizer wraps the HuggingFace transformers Qwen3 tokenizer,
    providing 100+ language support with BPE subword tokenization. It adds
    custom multimodal tokens for image, audio, video, and tool calling.
    
    Architecture:
        - Wraps AutoTokenizer from HuggingFace transformers
        - Uses BPE subword tokenization with byte-level fallback
        - Supports 100+ languages with excellent multilingual coverage
        - Adds custom multimodal tokens to the vocabulary
    
    Attributes:
        model_name (str): HuggingFace model identifier.
        cache_dir (Optional[str]): Cache directory for tokenizer files.
        trust_remote_code (bool): Whether to trust remote code.
        _tokenizer: Underlying HuggingFace tokenizer instance.
        _multimodal_token_ids (Dict[str, int]): Mapping of multimodal tokens to IDs.
    
    Example:
        >>> tokenizer = _YvQwenTokenizer("Qwen/Qwen3-8B")
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
    
    Note:
        Requires the transformers library. Downloads tokenizer files on
        first use unless cached.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        """Initialize the Qwen3 tokenizer.
        
        Args:
            model_name (str): HuggingFace model identifier for Qwen3.
                Default: "Qwen/Qwen3-8B".
            cache_dir (Optional[str]): Directory to cache tokenizer files.
                If None, uses default HuggingFace cache. Default: None.
            trust_remote_code (bool): Whether to trust remote code execution.
                Required for some Qwen models. Default: True.
        
        Raises:
            ImportError: If transformers library is not installed.
            RuntimeError: If tokenizer loading fails.
        
        Initializes:
            - model_name: Model identifier
            - cache_dir: Cache directory
            - trust_remote_code: Trust flag
            - _tokenizer: HuggingFace tokenizer instance
            - _multimodal_token_ids: Multimodal token mapping
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
            _LOG.info(f"Qwen3 tokenizer loaded: {model_name}, vocab_size={len(self._tokenizer)}")
        except ImportError:
            raise ImportError("transformers library required for Qwen tokenizer. Install with: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen tokenizer: {e}")
        
        self._multimodal_token_ids = {}
        self._add_multimodal_tokens()
    
    def _add_multimodal_tokens(self):
        """Add multimodal tokens to the tokenizer vocabulary.
        
        Iterates through MULTIMODAL_TOKENS and adds each token to the
        underlying tokenizer's vocabulary if not already present. Maintains
        a mapping from token strings to their IDs.
        
        Side Effects:
            - Adds tokens to _tokenizer vocabulary
            - Populates _multimodal_token_ids mapping
        """
        for token in MULTIMODAL_TOKENS:
            if token not in self._tokenizer.vocab:
                self._tokenizer.add_tokens([token])
            self._multimodal_token_ids[token] = self._tokenizer.vocab.get(token, len(self._tokenizer.vocab) - 1)
        
        _LOG.info(f"Added {len(MULTIMODAL_TOKENS)} multimodal tokens")
    
    def __len__(self):
        """Return vocabulary size.
        
        Returns:
            int: Total number of tokens in vocabulary.
        """
        return len(self._tokenizer)
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """Encode text into token IDs.
        
        Tokenizes the input text using BPE subword tokenization without
        adding special tokens (BOS/EOS).
        
        Args:
            text (str): Text string to encode.
            return_tensors (Optional[str]): If "pt", returns PyTorch tensor.
                Default: None (returns list).
        
        Returns:
            Union[List[int], torch.Tensor]: Token IDs. Shape is
                (1, num_tokens) if return_tensors="pt", else list of ints.
        
        Example:
            >>> tokenizer = _YvQwenTokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
            >>> print(f"Tokens: {tokens}")
        """
        result = self._tokenizer.encode(text, add_special_tokens=False)
        
        if return_tensors == "pt":
            return torch.tensor([result], dtype=torch.long)
        return result
    
    def encode_batch(self, texts: List[str], return_tensors: Optional[str] = None) -> Union[List[List[int]], torch.Tensor]:
        """Encode multiple texts into token IDs with optional padding.
        
        Tokenizes each text in the batch and optionally pads to the same length
        when returning tensors.
        
        Args:
            texts (List[str]): List of text strings to encode.
            return_tensors (Optional[str]): If "pt", returns padded PyTorch tensor.
                Default: None (returns list of lists).
        
        Returns:
            Union[List[List[int]], torch.Tensor]: Token IDs. Shape is
                (batch_size, max_seq_len) if return_tensors="pt", else list of lists.
        
        Note:
            When return_tensors="pt", sequences are padded with pad_token_id
            to match the longest sequence in the batch.
        """
        results = [self.encode(t, return_tensors=None) for t in texts]
        
        if return_tensors == "pt":
            max_len = max(len(r) for r in results)
            padded = [r + [self.pad_token_id] * (max_len - len(r)) for r in results]
            return torch.tensor(padded, dtype=torch.long)
        return results
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Converts token IDs back to their string representation, optionally
        skipping special tokens like BOS, EOS, PAD.
        
        Args:
            token_ids (Union[List[int], torch.Tensor]): Token IDs to decode.
                Can be a list or PyTorch tensor.
            skip_special_tokens (bool): Whether to exclude special tokens
                from the output. Default: True.
        
        Returns:
            str: Decoded text string.
        
        Example:
            >>> tokenizer = _YvQwenTokenizer()
            >>> tokens = tokenizer.encode("Hello")
            >>> text = tokenizer.decode(tokens)
            >>> print(text)  # "Hello"
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def add_tokens(self, new_tokens: List[str]) -> int:
        """Add new tokens to the vocabulary.
        
        Adds new tokens to the tokenizer vocabulary and updates the
        multimodal token mapping.
        
        Args:
            new_tokens (List[str]): List of token strings to add.
        
        Returns:
            int: Number of tokens actually added (excluding duplicates).
        
        Example:
            >>> tokenizer = _YvQwenTokenizer()
            >>> added = tokenizer.add_tokens(["<custom>", "<special>"])
            >>> print(f"Added {added} tokens")
        """
        added = self._tokenizer.add_tokens(new_tokens)
        for token in new_tokens:
            if token not in self._multimodal_token_ids:
                self._multimodal_token_ids[token] = self._tokenizer.vocab.get(token, len(self._tokenizer.vocab) - 1)
        return added
    
    def get_multimodal_token_id(self, token: str) -> Optional[int]:
        """Get the token ID for a multimodal token.
        
        Looks up the integer ID for a given multimodal token string.
        
        Args:
            token (str): Multimodal token string (e.g., "<image>", "<audio>").
        
        Returns:
            Optional[int]: Token ID if found, None otherwise.
        
        Example:
            >>> tokenizer = _YvQwenTokenizer()
            >>> image_id = tokenizer.get_multimodal_token_id("<image>")
        """
        return self._multimodal_token_ids.get(token)
    
    @property
    def pad_token_id(self):
        """Get padding token ID.
        
        Returns:
            int: PAD token ID, defaults to 0 if not set.
        """
        return self._tokenizer.pad_token_id or 0
    
    @property
    def eos_token_id(self):
        """Get end-of-sequence token ID.
        
        Returns:
            int: EOS token ID, defaults to 1 if not set.
        """
        return self._tokenizer.eos_token_id or 1
    
    @property
    def bos_token_id(self):
        """Get beginning-of-sequence token ID.
        
        Returns:
            int: BOS token ID, defaults to 2 if not set.
        """
        return self._tokenizer.bos_token_id or 2
    
    @property
    def unk_token_id(self):
        """Get unknown token ID.
        
        Returns:
            int: UNK token ID, defaults to 3 if not set.
        """
        return self._tokenizer.unk_token_id or 3
    
    @property
    def vocab_size(self):
        """Get vocabulary size.
        
        Returns:
            int: Total number of tokens in vocabulary.
        """
        return len(self._tokenizer)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory.
        
        Saves the tokenizer configuration and vocabulary files to the
        specified directory for later loading.
        
        Args:
            save_directory (str): Directory path to save tokenizer files.
        
        Note:
            Creates the directory if it doesn't exist.
        """
        self._tokenizer.save_pretrained(save_directory)
        _LOG.info(f"Tokenizer saved to {save_directory}")


class YvTokenizer:
    """Primary tokenizer interface with multiple backend support.
    
    This class provides a unified API for tokenization across different
    backends, including Qwen3-based text tokenization and H-Network visual
    tokenization. It delegates all operations to the underlying backend
    implementation.
    
    Supported Backends:
        - **qwen3**: Qwen3 tokenizer with 100+ language support.
          Uses BPE subword tokenization. Recommended for production.
        - **h_network**: Visual tokenizer that renders text as images.
          Experimental feature for H-Network architectures.
    
    Architecture:
        The tokenizer follows a delegation pattern where all encoding and
        decoding operations are forwarded to the backend implementation
        selected at initialization time.
    
    Attributes:
        tokenizer_type (str): Backend type ("qwen3" or "h_network").
        _impl: Backend tokenizer implementation instance.
    
    Example:
        >>> # Initialize with Qwen3 backend (recommended)
        >>> tokenizer = YvTokenizer(tokenizer_type="qwen3")
        >>> 
        >>> # Encode text
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(f"Token IDs: {tokens}")
        >>> 
        >>> # Decode back to text
        >>> text = tokenizer.decode(tokens)
        >>> print(f"Decoded: {text}")
        >>> 
        >>> # Batch encoding with tensor output
        >>> texts = ["Hello", "World"]
        >>> tensors = tokenizer.encode_batch(texts, return_tensors="pt")
        >>> print(f"Shape: {tensors.shape}")
    
    Note:
        The Qwen3 backend requires the transformers library and will
        download tokenizer files on first use.
    """
    
    def __init__(
        self,
        tokenizer_type: str = "qwen3",
        model_name: str = "Qwen/Qwen3-8B",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Yv tokenizer with specified backend.
        
        Args:
            tokenizer_type (str): Backend type. Options:
                - "qwen3": Qwen3 tokenizer (recommended, 100+ languages)
                - "h_network": Visual tokenizer for H-Network processing
                Default: "qwen3".
            model_name (str): HuggingFace model identifier for Qwen3 backend.
                Default: "Qwen/Qwen3-8B".
            cache_dir (Optional[str]): Directory to cache tokenizer files.
                If None, uses default cache location. Default: None.
            **kwargs: Additional arguments passed to backend tokenizer:
                - trust_remote_code (bool): For Qwen3 backend
                - compression_ratio (int): For H-Network backend
                - render_dpi (int): For H-Network backend
                - font_path (str): For H-Network backend
                - fallback_enabled (bool): For H-Network backend
        
        Raises:
            ValueError: If tokenizer_type is not "qwen3" or "h_network".
        
        Example:
            >>> # Qwen3 backend
            >>> tokenizer = YvTokenizer("qwen3", "Qwen/Qwen3-8B")
            >>> 
            >>> # H-Network backend with custom settings
            >>> tokenizer = YvTokenizer("h_network", compression_ratio=15)
        """
        self.tokenizer_type = str(tokenizer_type or "qwen3").strip().lower()
        
        if self.tokenizer_type == "qwen3":
            self._impl = _YvQwenTokenizer(
                model_name=model_name,
                cache_dir=cache_dir,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )
        elif self.tokenizer_type == "h_network":
            self._impl = _YvHNetworkTokenizer(**kwargs)
        else:
            raise ValueError(f"Invalid tokenizer_type: {self.tokenizer_type}. Choose 'qwen3' or 'h_network'")
        
        _LOG.info(f"YvTokenizer initialized: type={self.tokenizer_type}, vocab_size={len(self._impl)}")
    
    def __len__(self):
        """Return vocabulary size.
        
        Returns:
            int: Total number of tokens in vocabulary.
        """
        return len(self._impl)
    
    def __repr__(self):
        """Return string representation.
        
        Returns:
            str: Representation with tokenizer type and vocabulary size.
        """
        return f"YvTokenizer(type={self.tokenizer_type}, vocab_size={len(self)})"
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """Encode text into token IDs.
        
        Tokenizes the input text using the selected backend.
        
        Args:
            text (str): Text string to encode.
            return_tensors (Optional[str]): If "pt", returns PyTorch tensor.
                Default: None (returns list).
        
        Returns:
            Union[List[int], torch.Tensor]: Token IDs.
        
        Example:
            >>> tokenizer = YvTokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
        """
        return self._impl.encode(text, return_tensors=return_tensors)
    
    def encode_batch(self, texts: List[str], return_tensors: Optional[str] = None) -> Union[List[List[int]], torch.Tensor]:
        """Encode multiple texts into token IDs.
        
        Tokenizes each text in the batch using the selected backend.
        
        Args:
            texts (List[str]): List of text strings to encode.
            return_tensors (Optional[str]): If "pt", returns padded tensor.
                Default: None (returns list of lists).
        
        Returns:
            Union[List[List[int]], torch.Tensor]: Token IDs for each text.
        
        Example:
            >>> tokenizer = YvTokenizer()
            >>> tokens = tokenizer.encode_batch(["Hello", "World"])
        """
        return self._impl.encode_batch(texts, return_tensors=return_tensors)
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Converts token IDs back to their string representation.
        
        Args:
            token_ids (Union[List[int], torch.Tensor]): Token IDs to decode.
            skip_special_tokens (bool): Whether to exclude special tokens.
                Default: True.
        
        Returns:
            str: Decoded text string.
        
        Example:
            >>> tokenizer = YvTokenizer()
            >>> tokens = tokenizer.encode("Hello")
            >>> text = tokenizer.decode(tokens)
        """
        return self._impl.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def add_tokens(self, new_tokens: List[str]) -> int:
        """Add new tokens to the vocabulary.
        
        Args:
            new_tokens (List[str]): List of token strings to add.
        
        Returns:
            int: Number of tokens actually added.
        
        Example:
            >>> tokenizer = YvTokenizer()
            >>> added = tokenizer.add_tokens(["<custom>", "<special>"])
        """
        return self._impl.add_tokens(new_tokens)
    
    def get_multimodal_token_id(self, token: str) -> Optional[int]:
        """Get the token ID for a multimodal token.
        
        Args:
            token (str): Multimodal token string (e.g., "<image>").
        
        Returns:
            Optional[int]: Token ID if found, None if not supported or not found.
        
        Note:
            Only supported by Qwen3 backend.
        """
        if hasattr(self._impl, 'get_multimodal_token_id'):
            return self._impl.get_multimodal_token_id(token)
        return None
    
    @property
    def pad_token_id(self):
        """Get padding token ID.
        
        Returns:
            int: PAD token ID.
        """
        return self._impl.pad_token_id
    
    @property
    def eos_token_id(self):
        """Get end-of-sequence token ID.
        
        Returns:
            int: EOS token ID.
        """
        return self._impl.eos_token_id
    
    @property
    def bos_token_id(self):
        """Get beginning-of-sequence token ID.
        
        Returns:
            int: BOS token ID.
        """
        return self._impl.bos_token_id
    
    @property
    def unk_token_id(self):
        """Get unknown token ID.
        
        Returns:
            int: UNK token ID.
        """
        return self._impl.unk_token_id
    
    @property
    def vocab_size(self):
        """Get vocabulary size.
        
        Returns:
            int: Total number of tokens in vocabulary.
        """
        return len(self._impl)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory.
        
        Saves the tokenizer configuration and vocabulary files.
        
        Args:
            save_directory (str): Directory path to save tokenizer files.
        
        Note:
            Only supported by Qwen3 backend. Logs warning for H-Network.
        """
        if hasattr(self._impl, 'save_pretrained'):
            self._impl.save_pretrained(save_directory)
        else:
            _LOG.warning(f"save_pretrained not supported for tokenizer type: {self.tokenizer_type}")
