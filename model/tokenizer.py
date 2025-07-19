#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import re
import json
import urllib.request


class BPETokenizer:
    def __init__(self, vocab_path=None, merges_path=None, special_tokens=None):
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}
        else:
            base_tokens = [chr(i) for i in range(32, 127)]
            self.encoder = {tok: i for i, tok in enumerate(base_tokens)}
            self.decoder = {i: tok for tok, i in self.encoder.items()}
            print("❌\tNo vocab.json found, using dummy vocab.")
        self.bpe_ranks = {}
        if merges_path and os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                merges = [tuple(line.strip().split()) for line in f if not line.startswith("#") and line.strip()]
            self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        else:
            print("❌\tNo merges.txt found, using char-level BPE.")
        self.special_tokens = special_tokens or ["<s>", "</s>", "<unk>", "<pad>"]
        for tok in self.special_tokens:
            if tok not in self.encoder:
                self.encoder[tok] = len(self.encoder)
                self.decoder[self.encoder[tok]] = tok
        self.unk_id = self.encoder["<unk>"]
        self.pad_id = self.encoder["<pad>"]
        self.bos_id = self.encoder["<s>"]
        self.eos_id = self.encoder["</s>"]
    def bpe(self, token):
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
                    ids.append(self.unk_id)
                    # print(f"[Tokenizer] OOV token: {bpe_tok}")
        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids
    def encode_batch(self, texts, return_tensors=None):
        return [self.encode(t, return_tensors=None) for t in texts]
    def decode(self, ids, skip_special_tokens=True):
        tokens = [self.decoder.get(i, "<unk>") for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        text = " ".join(tokens)
        text = text.replace(" ##", "")
        text = text.replace("Ġ", "")
        return text.strip()
    @property
    def pad_token_id(self): return self.pad_id
    @property
    def eos_token_id(self): return self.eos_id
    @property
    def bos_token_id(self): return self.bos_id
    @property
    def unk_token_id(self): return self.unk_id

def download_if_missing(url, local_path):
    if not os.path.exists(local_path):
        print(f"✅\tDownloading {os.path.basename(local_path)} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"✅\tDownloaded {local_path}")

def get_tokenizer():
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