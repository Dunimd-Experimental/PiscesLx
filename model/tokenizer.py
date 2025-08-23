#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
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
from utils.log import RIGHT, ERROR

class BPETokenizer:
    """A Byte Pair Encoding (BPE) tokenizer implementation.

    This tokenizer is used to convert text into a sequence of tokens and vice versa.
    It supports loading pre-trained vocabulary and merge rules, as well as adding new tokens.

    With the byte-level fallback enabled, it robustly supports multilingual text
    including the six United Nations languages (Arabic, Chinese, English, French,
    Russian, Spanish) by encoding out-of-vocabulary segments as UTF-8 byte tokens.
    """
    def __init__(self, vocab_path=None, merges_path=None, special_tokens=None, byte_fallback=True):
        """Initialize the BPETokenizer.

        Args:
            vocab_path (str, optional): Path to the vocabulary file (vocab.json). Defaults to None.
            merges_path (str, optional): Path to the merges file (merges.txt). Defaults to None.
            special_tokens (list, optional): List of special tokens. Defaults to ["<s>", "</s>", "<unk>", "<pad>"].
        """
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.byte_fallback = byte_fallback

        if vocab_path and os.path.exists(vocab_path):
            # Load vocabulary from file
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}
        else:
            # Create a dummy vocabulary with ASCII printable characters
            base_tokens = [chr(i) for i in range(32, 127)]
            self.encoder = {tok: i for i, tok in enumerate(base_tokens)}
            self.decoder = {i: tok for tok, i in self.encoder.items()}
            ERROR("No vocab.json found, using dummy vocab.")
        if merges_path and os.path.exists(merges_path):
            # Load merge rules from file
            with open(merges_path, "r", encoding="utf-8") as f:
                merges = [tuple(line.strip().split()) for line in f if not line.startswith("#") and line.strip()]
            self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        else:
            ERROR("No merges.txt found, using char-level BPE.")
        self.special_tokens = special_tokens or ["<s>", "</s>", "<unk>", "<pad>"]
        # Add special tokens to the vocabulary
        for tok in self.special_tokens:
            if tok not in self.encoder:
                self.encoder[tok] = len(self.encoder)
                self.decoder[self.encoder[tok]] = tok
        # Install UTF-8 byte tokens for fallback to ensure multilingual coverage
        # e.g., <0x00> ... <0xFF>
        self.byte_tokens = [f"<0x{b:02X}>" for b in range(256)]
        for tok in self.byte_tokens:
            if tok not in self.encoder:
                self.encoder[tok] = len(self.encoder)
                self.decoder[self.encoder[tok]] = tok
        self.byte_to_id = {b: self.encoder[f"<0x{b:02X}>"] for b in range(256)}
        self.id_to_byte = {tid: b for (b, tid) in self.byte_to_id.items()}
        self.unk_id = self.encoder["<unk>"]
        self.pad_id = self.encoder["<pad>"]
        self.bos_id = self.encoder["<s>"]
        self.eos_id = self.encoder["</s>"]

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.encoder)

    def add_tokens(self, new_tokens):
        """Adds new tokens to the vocabulary.

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
        """Saves tokenizer files (vocab, merges) to a directory.

        Args:
            save_directory (str): Directory to save the tokenizer files.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        # Copy merges file if it exists
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
        word = tuple(token)
        pairs = set(zip(word, word[1:]))
        if not pairs:
            return [token]
        while True:
            min_pair = None
            min_rank = float("inf")
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
                except:
                    new_word.extend(word[i:])
                    break
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
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
        # Normalize to NFC for stable Unicode processing
        text = unicodedata.normalize("NFC", text)
        for tok in self.special_tokens:
            text = text.replace(tok, f" {tok} ")
        tokens = re.findall(r"\w+|[^\w\s]|<[^>]+>", text, re.UNICODE)
        ids = []
        for token in tokens:
            bpe_tokens = self.bpe(token) if self.bpe_ranks else [token]
            for bpe_tok in bpe_tokens:
                if bpe_tok in self.encoder:
                    ids.append(self.encoder[bpe_tok])
                else:
                    if self.byte_fallback:
                        for b in bpe_tok.encode("utf-8"):
                            ids.append(self.byte_to_id[b])
                    else:
                        ids.append(self.unk_id)
                    # print(f"[Tokenizer] OOV token: {bpe_tok}")
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
        # Reconstruct bytes sequences produced by byte_fallback
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
        """Get the ID of the padding token."""
        return self.pad_id

    @property
    def eos_token_id(self):
        """Get the ID of the end-of-sequence token."""
        return self.eos_id

    @property
    def bos_token_id(self):
        """Get the ID of the beginning-of-sequence token."""
        return self.bos_id

    @property
    def unk_token_id(self):
        """Get the ID of the unknown token."""
        return self.unk_id

def download_if_missing(url, local_path):
    """Download a file if it doesn't exist locally.

    Args:
        url (str): URL of the file to download.
        local_path (str): Local path to save the file.
    """
    if not os.path.exists(local_path):
        RIGHT(f"Downloading {os.path.basename(local_path)} ...")
        urllib.request.urlretrieve(url, local_path)
        RIGHT(f"Downloaded {local_path}")

def get_tokenizer():
    """Get a BPETokenizer instance with pre-trained vocabulary and merge rules.

    Returns:
        BPETokenizer: A BPETokenizer instance.

    Raises:
        FileNotFoundError: If vocab.json or merges.txt is not found.
    """
    vocab_path, merges_path = None, None
    for path in ["tokenizer/vocab.json", "vocab.json", os.environ.get("PISCES_VOCAB_PATH")]:
        if path and os.path.exists(path):
            vocab_path = path
            break
    for path in ["tokenizer/merges.txt", "merges.txt", os.environ.get("PISCES_MERGES_PATH")]:
        if path and os.path.exists(path):
            merges_path = path
            break
    if vocab_path is None or merges_path is None:
        raise FileNotFoundError(
            "❌\tPisces BPETokenizer: vocab.json or merges.txt not found! "
            "Please put them in the 'tokenizer/' directory."
        )
    return BPETokenizer(vocab_path, merges_path)