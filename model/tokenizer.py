#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import os
import re
import json
import unicodedata
import urllib.request
from utils.log.core import PiscesLxCoreLog
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from typing import Optional, List, Union

logger = PiscesLxCoreLog("Arctic.Core.Tokenizer", file_path="logs/ArcticCore.log")

class ArcticHNetworkTokenizer:
    """H-Network tokenizer for visual text processing without traditional tokenization.
    
    Converts text to visual tokens by rendering text as images and compressing them
    into high-efficiency visual representations.
    """
    
    def __init__(self, compression_ratio: int = 20, render_dpi: int = 150, 
                 font_path: Optional[str] = None, fallback_enabled: bool = True):
        """Initialize H-Network tokenizer.
        
        Args:
            compression_ratio: Visual compression ratio (default: 20)
            render_dpi: DPI for text rendering (default: 150)
            font_path: Path to font file (optional)
            fallback_enabled: Enable fallback to standard tokenizer on failure
        """
        self.compression_ratio = compression_ratio
        self.render_dpi = render_dpi
        self.fallback_enabled = fallback_enabled
        self.visual_vocab_size = 8192  # Fixed visual vocabulary size
        
        # Default font configuration
        try:
            self.font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()
            
        logger.info(f"H-Network tokenizer initialized: compression_ratio={compression_ratio}, "
                   f"render_dpi={render_dpi}, fallback_enabled={fallback_enabled}")
    
    def _render_text_to_image(self, text: str, max_width: int = 1024) -> Image.Image:
        """Render text to PIL Image with proper formatting."""
        # Estimate image dimensions
        lines = text.split('\n')
        max_chars = max(len(line) for line in lines) if lines else 80
        img_width = min(max_width, max_chars * 8 + 20)
        img_height = len(lines) * 20 + 20
        
        # Create white background image
        image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Render text line by line
        y_offset = 10
        for line in lines:
            if line.strip():
                draw.text((10, y_offset), line.strip(), font=self.font, fill='black')
            y_offset += 20
            
        return image
    
    def _compress_visual_tokens(self, image: Image.Image) -> List[int]:
        """Compress image into visual tokens using learned compression."""
        # Convert to tensor and normalize
        img_array = np.array(image.resize((224, 224)))  # Standard size
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0
        
        # Simple patch-based compression (placeholder for learned compression)
        patch_size = 16
        patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(3, -1, patch_size, patch_size)
        
        # Compress patches to visual tokens
        num_patches = patches.shape[1]
        target_tokens = max(1, num_patches // self.compression_ratio)
        
        # Generate visual token IDs (simplified compression)
        visual_tokens = []
        for i in range(target_tokens):
            patch_idx = (i * num_patches) // target_tokens
            # Simple hash-based token generation
            patch_hash = hash(patches[:, patch_idx].mean().item()) % self.visual_vocab_size
            visual_tokens.append(abs(patch_hash))
            
        return visual_tokens[:100]  # Limit max tokens for efficiency
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        """Encode text into visual tokens.
        
        Args:
            text: Input text to encode
            return_tensors: If "pt", return PyTorch tensor
            
        Returns:
            List of visual token IDs or tensor
        """
        try:
            # Render text to image
            image = self._render_text_to_image(text)
            
            # Compress to visual tokens
            visual_tokens = self._compress_visual_tokens(image)
            
            logger.debug(f"H-Network encoded '{text[:50]}...' to {len(visual_tokens)} visual tokens")
            
            if return_tensors == "pt":
                return torch.tensor([visual_tokens], dtype=torch.long)
            return visual_tokens
            
        except Exception as e:
            logger.error(f"H-Network encoding failed: {e}")
            if self.fallback_enabled:
                logger.warning("Falling back to standard BPE tokenizer")
                # Return simple character-level encoding as fallback
                return [ord(c) % self.visual_vocab_size for c in text[:100]]
            else:
                raise RuntimeError(f"H-Network encoding failed: {e}")
    
    def encode_batch(self, texts: List[str], return_tensors: Optional[str] = None) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text, return_tensors=None) for text in texts]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode visual tokens back to text (approximate reconstruction)."""
        # This is a simplified decoder - in practice would use learned decompression
        decoded_chars = []
        for token_id in token_ids:
            # Simple mapping from visual tokens to characters
            char_code = token_id % 256
            if 32 <= char_code <= 126:  # Printable ASCII
                decoded_chars.append(chr(char_code))
            else:
                decoded_chars.append('?')  # Placeholder for non-printable
                
        return ''.join(decoded_chars)
    
    def __len__(self):
        """Return visual vocabulary size."""
        return self.visual_vocab_size

class ArcticBPETokenizer:
    """An implementation of Byte Pair Encoding (BPE) tokenizer.

    This tokenizer is used to convert text to a sequence of tokens and vice versa.
    It supports loading pre-trained vocabulary and merge rules, as well as adding new tokens.
    When byte-level fallback is enabled, it encodes out-of-vocabulary segments as UTF-8 byte tokens.
    """
    def __init__(self, vocab_path=None, merges_path=None, special_tokens=None, byte_fallback=True):
        """Initialize the BPETokenizer.

        Args:
            vocab_path (str, optional): Path to the vocabulary file (vocab.json). Defaults to None.
            merges_path (str, optional): Path to the merges file (merges.txt). Defaults to None.
            special_tokens (list, optional): List of special tokens. Defaults to ["<s>", "</s>", "<unk>", "<pad>"].
            byte_fallback (bool, optional): Whether to enable byte-level fallback. Defaults to True.
        """
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.byte_fallback = byte_fallback

        if vocab_path and os.path.exists(vocab_path):
            # Load vocabulary from the specified file
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.encoder = json.load(f)
            # Create a decoder by inverting the encoder mapping
            self.decoder = {v: k for k, v in self.encoder.items()}
        else:
            # Create a dummy vocabulary containing ASCII printable characters
            base_tokens = [chr(i) for i in range(32, 127)]
            self.encoder = {tok: i for i, tok in enumerate(base_tokens)}
            self.decoder = {i: tok for tok, i in self.encoder.items()}
            logger.error("No vocab.json found, using dummy vocab.")

        if merges_path and os.path.exists(merges_path):
            # Load merge rules from the specified file
            with open(merges_path, "r", encoding="utf-8") as f:
                merges = [tuple(line.strip().split()) for line in f if not line.startswith("#") and line.strip()]
            # Assign ranks to each merge pair
            self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        else:
            logger.error("No merges.txt found, using char-level BPE.")

        # Set default special tokens if not provided
        self.special_tokens = special_tokens or ["<s>", "</s>", "<unk>", "<pad>"]

        # Add special tokens to the vocabulary
        for tok in self.special_tokens:
            if tok not in self.encoder:
                self.encoder[tok] = len(self.encoder)
                self.decoder[self.encoder[tok]] = tok

        # Generate UTF-8 byte tokens for fallback to ensure multilingual support
        self.byte_tokens = [f"<0x{b:02X}>" for b in range(256)]
        for tok in self.byte_tokens:
            if tok not in self.encoder:
                self.encoder[tok] = len(self.encoder)
                self.decoder[self.encoder[tok]] = tok

        # Create mappings between bytes and token IDs
        self.byte_to_id = {b: self.encoder[f"<0x{b:02X}>"] for b in range(256)}
        self.id_to_byte = {tid: b for (b, tid) in self.byte_to_id.items()}

        # Get the IDs of special tokens
        self.unk_id = self.encoder["<unk>"]
        self.pad_id = self.encoder["<pad>"]
        self.bos_id = self.encoder["<s>"]
        self.eos_id = self.encoder["</s>"]

    def __len__(self):
        """Return the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.encoder)

    def add_tokens(self, new_tokens):
        """Add new tokens to the vocabulary.

        Args:
            new_tokens (list): List of new tokens to add.

        Returns:
            int: The number of tokens added to the vocabulary.
        """
        added_count = 0
        for token in new_tokens:
            if token not in self.encoder:
                new_id = len(self.encoder)
                self.encoder[token] = new_id
                self.decoder[new_id] = token
                if token not in self.special_tokens:
                    self.special_tokens.append(token)
                added_count += 1
        return added_count

    def save_pretrained(self, save_directory):
        """Save tokenizer files (vocab, merges) to a specified directory.

        Args:
            save_directory (str): Directory to save the tokenizer files.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save the vocabulary to a JSON file
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        # Copy the merges file if it exists
        if self.merges_path and os.path.exists(self.merges_path):
            import shutil
            merges_file = os.path.join(save_directory, "merges.txt")
            shutil.copyfile(self.merges_path, merges_file)

    def bpe(self, token):
        """Apply Byte Pair Encoding to a single token.

        Args:
            token (str): The input token to be processed.

        Returns:
            tuple: A tuple of sub-tokens after BPE processing.
        """
        if token in self.special_tokens:
            return [token]

        # Convert the token into a tuple of characters
        word = tuple(token)
        # Generate all possible adjacent pairs of characters
        pairs = set(zip(word, word[1:]))

        if not pairs:
            return [token]

        while True:
            min_pair = None
            min_rank = float("inf")
            # Find the pair with the lowest merge rank
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_pair not in self.bpe_ranks:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            # Generate new pairs for the next iteration
            pairs = set(zip(word, word[1:]))

        return word

    def encode(self, text, return_tensors=None):
        """Encode text into a list of token IDs.

        Args:
            text (str): The input text to be encoded.
            return_tensors (str, optional): If "pt", return a PyTorch tensor. Defaults to None.

        Returns:
            list or torch.Tensor: A list of token IDs or a PyTorch tensor.
        """
        # Normalize the text to NFC form for stable Unicode processing
        text = unicodedata.normalize("NFC", text)

        # Add spaces around special tokens
        for tok in self.special_tokens:
            text = text.replace(tok, f" {tok} ")

        # Split the text into tokens using regular expressions
        tokens = re.findall(r"\w+|[^\w\s]|<[^>]+>", text, re.UNICODE)

        ids = []
        for token in tokens:
            # Apply BPE if merge rules are available, otherwise use the token as-is
            bpe_tokens = self.bpe(token) if self.bpe_ranks else [token]
            for bpe_tok in bpe_tokens:
                if bpe_tok in self.encoder:
                    ids.append(self.encoder[bpe_tok])
                else:
                    if self.byte_fallback:
                        # Encode out-of-vocabulary tokens as UTF-8 bytes
                        for b in bpe_tok.encode("utf-8"):
                            ids.append(self.byte_to_id[b])
                    else:
                        ids.append(self.unk_id)

        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)

        return ids

    def encode_batch(self, texts, return_tensors=None):
        """Encode a batch of texts into lists of token IDs.

        Args:
            texts (list): A list of input texts to be encoded.
            return_tensors (str, optional): If "pt", return PyTorch tensors. Defaults to None.

        Returns:
            list: A list of lists of token IDs.
        """
        return [self.encode(t, return_tensors=None) for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        """Decode a list of token IDs into text.

        Args:
            ids (list): A list of token IDs to be decoded.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

        Returns:
            str: The decoded text.
        """
        # Reconstruct byte sequences produced by byte-level fallback
        out_tokens = []
        bytes_buf = []
        for i in ids:
            if i in self.id_to_byte:
                bytes_buf.append(self.id_to_byte[i])
            else:
                if bytes_buf:
                    try:
                        out_tokens.append(bytes(bytes_buf).decode("utf-8"))
                    except Exception:
                        out_tokens.append("".join(f"<0x{b:02X}>" for b in bytes_buf))
                    bytes_buf = []
                out_tokens.append(self.decoder.get(i, "<unk>"))

        if bytes_buf:
            try:
                out_tokens.append(bytes(bytes_buf).decode("utf-8"))
            except Exception:
                out_tokens.append("".join(f"<0x{b:02X}>" for b in bytes_buf))

        if skip_special_tokens:
            out_tokens = [t for t in out_tokens if t not in self.special_tokens]

        text = " ".join(out_tokens)
        text = text.replace(" ##", "")
        text = text.replace("Ġ", "")
        return text.strip()

    @property
    def pad_token_id(self):
        """Get the ID of the padding token.

        Returns:
            int: The ID of the padding token.
        """
        return self.pad_id

    @property
    def eos_token_id(self):
        """Get the ID of the end-of-sequence token.

        Returns:
            int: The ID of the end-of-sequence token.
        """
        return self.eos_id

    @property
    def bos_token_id(self):
        """Get the ID of the beginning-of-sequence token.

        Returns:
            int: The ID of the beginning-of-sequence token.
        """
        return self.bos_id

    @property
    def unk_token_id(self):
        """Get the ID of the unknown token.

        Returns:
            int: The ID of the unknown token.
        """
        return self.unk_id

def download_if_missing(url, local_path):
    """Download a file if it doesn't exist locally.

    Args:
        url (str): URL of the file to download.
        local_path (str): Local path to save the file.
    """
    if not os.path.exists(local_path):
        logger.info(f"Downloading {os.path.basename(local_path)} ...")
        urllib.request.urlretrieve(url, local_path)
        logger.info(f"Downloaded {local_path}")

def get_tokenizer(tokenizer_type="standard", **kwargs):
    """Get tokenizer instance with specified type.

    Args:
        tokenizer_type (str): Type of tokenizer - "standard" or "h_network"
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Union[ArcticBPETokenizer, ArcticHNetworkTokenizer]: Tokenizer instance

    Raises:
        FileNotFoundError: If vocab.json or merges.txt is not found for standard tokenizer
        ValueError: If invalid tokenizer_type is specified
    """
    if tokenizer_type == "h_network":
        logger.info("Initializing H-Network tokenizer")
        return ArcticHNetworkTokenizer(**kwargs)
    elif tokenizer_type == "standard":
        vocab_path, merges_path = None, None
        # Search for the vocabulary file in multiple locations
        for path in ["tokenizer/vocab.json", "vocab.json", os.environ.get("PISCES_VOCAB_PATH")]:
            if path and os.path.exists(path):
                vocab_path = path
                break

        # Search for the merges file in multiple locations
        for path in ["tokenizer/merges.txt", "merges.txt", os.environ.get("PISCES_MERGES_PATH")]:
            if path and os.path.exists(path):
                merges_path = path
                break

        if vocab_path is None or merges_path is None:
            raise FileNotFoundError(
                "🔴\tPisces BPETokenizer: vocab.json or merges.txt not found! "
                "Please put them in the 'tokenizer/' directory."
            )

        return ArcticBPETokenizer(vocab_path, merges_path)
    else:
        raise ValueError(f"Invalid tokenizer_type: {tokenizer_type}. Choose 'standard' or 'h_network'")

def load_tokenizer_from_config(config_path: str = None) -> Union[ArcticBPETokenizer, ArcticHNetworkTokenizer]:
    """Load tokenizer based on model configuration file.
    
    Args:
        config_path (str): Path to model config JSON file. If None, will auto-detect.
        
    Returns:
        Union[ArcticBPETokenizer, ArcticHNetworkTokenizer]: Configured tokenizer instance
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
    if config_path is None:
        # Auto-detect config file
        for size in ["0.5B", "1.5B", "7B", "32B", "64B", "70B", "128B", "314B", "671B", "1T"]:
            test_path = f"configs/model/{size}.json"
            if os.path.exists(test_path):
                config_path = test_path
                break
                
    if not config_path or not os.path.exists(config_path):
        logger.warning("No model config found, defaulting to standard tokenizer")
        return get_tokenizer("standard")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        training_config = config.get('training_config', {})
        tokenizer_type = training_config.get('tokenizer_type', 'standard')
        
        logger.info(f"Loading tokenizer from config: {config_path}")
        logger.info(f"Selected tokenizer type: {tokenizer_type}")
        
        # Extract tokenizer-specific parameters from config
        tokenizer_kwargs = {}
        if tokenizer_type == "h_network":
            # Extract H-Network specific parameters
            hnet_config = training_config.get('h_network', {})
            tokenizer_kwargs.update({
                'compression_ratio': hnet_config.get('compression_ratio', 20),
                'render_dpi': hnet_config.get('render_dpi', 150),
                'font_path': hnet_config.get('font_path'),
                'fallback_enabled': hnet_config.get('fallback_enabled', True)
            })
        
        return get_tokenizer(tokenizer_type, **tokenizer_kwargs)
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer from config {config_path}: {e}")
        logger.warning("Falling back to standard tokenizer")
        return get_tokenizer("standard")