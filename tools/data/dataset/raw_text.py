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

"""
Local JSONL/JSON corpora as training data (§04 data stage).

The training orchestrator accepts either a registry dataset name or a path —
a directory, a file, or a ``kaggle://`` locator.  Everything that used to live as
closures inside ``PiscesLxTrainOrchestrator.run`` now lives here, for two reasons:

* ``DataLoader`` workers pickle the dataset, and a class defined inside a function is
  not picklable, so any run with ``num_workers > 0`` died on ``PicklingError``.
* §6.2 — capabilities are exposed through classes, not through nested functions.
"""

import gzip
import io
import json
import os
from typing import Any, Dict, List

import torch

__all__ = ["PiscesLxToolsDataRawTextDataset", "PiscesLxToolsDataJsonSampleReader",
           "PiscesLxToolsDataSampleRecord",
           "PiscesLxToolsDataSupervisedDataset",
           "PiscesLxToolsDataPreferenceDataset",
           "PiscesLxToolsDataClassificationDataset"]


class PiscesLxToolsDataRawTextDataset(torch.utils.data.Dataset):
    """Tokenize text records lazily so large corpora never sit in memory twice."""

    def __init__(self, texts: List[str], tokenizer, max_len: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = PiscesLxToolsDataSampleRecord.encode_ids(
            self.tokenizer, self.texts[idx], self.max_len
        )
        return {"input_ids": ids}


class PiscesLxToolsDataJsonSampleReader:
    """Read samples from paths, Kaggle locators and Hub names for the data stage."""

    KAGGLE_PREFIX = "kaggle://"
    JSON_SUFFIXES = (".jsonl", ".json")
    TEXT_FIELDS = ("text", "content")
    INSTRUCTION_FIELDS = ("instruction", "output")
    HTML_SIGNATURES = (b"<!doctype html", b"<html")
    GZIP_MAGIC = b"\x1f\x8b"
    DIAGNOSTIC_BYTES = 200
    DIAGNOSTIC_CHARS = 120
    KAGGLE_MIN_PATH_PARTS = 2
    KAGGLE_WEB_API = "https://www.kaggle.com/api/v1"

    @staticmethod
    def extract_text(item: Dict[str, Any]) -> str:
        """Return the document text of one sample, whatever schema it uses."""
        for field in PiscesLxToolsDataJsonSampleReader.TEXT_FIELDS:
            if item.get(field):
                return item[field]
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if instruction or output:
            return instruction + "\n" + output
        return json.dumps(item, ensure_ascii=False)

    @classmethod
    def load_samples(cls, file_path: str) -> list:
        """Load one JSON/JSONL/gz file, refusing HTML that a bad slug returned."""
        samples: List[Any] = []
        try:
            with open(file_path, "rb") as raw:
                magic = raw.read(8)
                raw.seek(0)
                head = magic.lower()
                if any(head.startswith(signature) for signature in cls.HTML_SIGNATURES):
                    raise ValueError(
                        f"{file_path} is HTML, not JSON — the dataset may be "
                        "private, deleted, or addressed by a bad slug"
                    )
                if magic[:2] == cls.GZIP_MAGIC:
                    stream = gzip.open(raw, mode="rt", encoding="utf-8-sig")
                else:
                    stream = io.TextIOWrapper(raw, encoding="utf-8-sig")
                with stream as handle:
                    if file_path.endswith(".jsonl"):
                        for line in handle:
                            line = line.strip()
                            if line:
                                samples.append(json.loads(line))
                    else:
                        data = json.load(handle)
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict):
                            samples = [data]
        except Exception as exc:
            cls._log_failure(file_path, exc)
        return samples

    @classmethod
    def _log_failure(cls, file_path: str, exc: Exception) -> None:
        from utils.dc import PiscesLxLogger
        from utils.paths import PiscesLxPaths

        log = PiscesLxLogger(
            "PiscesLx.Tools.Data",
            file_path=PiscesLxPaths.get_log_file("PiscesLx.Tools.Data"),
            enable_file=True,
        )
        try:
            with open(file_path, "rb") as raw:
                head = raw.read(cls.DIAGNOSTIC_BYTES)
            log.error(
                f"Failed to load {file_path}: {exc} | raw_bytes={head.hex()} | "
                f"raw_text={head[:cls.DIAGNOSTIC_CHARS]!r}"
            )
        except Exception:
            log.error(f"Failed to load {file_path}: {exc}")

    @classmethod
    def scan_directory(cls, directory: str) -> list:
        """Collect every sample under *directory*, deepest file first by name."""
        samples: List[Any] = []
        for root, _dirs, files in os.walk(directory):
            for name in sorted(files):
                if name.endswith(cls.JSON_SUFFIXES):
                    samples.extend(cls.load_samples(os.path.join(root, name)))
        return samples

    @classmethod
    def resolve(cls, data_spec: str):
        """Turn a registry name, path or ``kaggle://`` locator into a sample list.

        ``None`` means the trainer should fall back to the dataset manager, which is
        what resolves Hub names.
        """
        if data_spec.startswith(cls.KAGGLE_PREFIX):
            return cls._from_kaggle(data_spec)
        resolved = os.path.abspath(data_spec)
        if os.path.isdir(resolved):
            return cls.scan_directory(resolved)
        if os.path.isfile(resolved):
            return cls.load_samples(resolved)
        return None

    @classmethod
    def _from_kaggle(cls, data_spec: str) -> list:
        subpath = data_spec[len(cls.KAGGLE_PREFIX):]
        try:
            cls._patch_kagglesdk_compat()
            import kagglehub

            parts = subpath.replace("\\", "/").split("/")
            if len(parts) < cls.KAGGLE_MIN_PATH_PARTS:
                raise ValueError(f"invalid Kaggle locator: {data_spec}")
            owner_dataset = f"{parts[0]}/{parts[1]}"
            rest = "/".join(parts[2:]) if len(parts) > cls.KAGGLE_MIN_PATH_PARTS else ""
            local = kagglehub.dataset_download(owner_dataset)
            target = os.path.join(str(local), rest) if rest else str(local)
            if os.path.isfile(target):
                return cls.load_samples(target)
            if os.path.isdir(target):
                return cls.scan_directory(target)
            raise FileNotFoundError(f"Kaggle path not found: {target}")
        except Exception as exc:
            cls._log_failure(data_spec, exc)
            return []

    @staticmethod
    def _patch_kagglesdk_compat() -> None:
        """Backfill the endpoint helper newer kagglesdk builds no longer export."""
        try:
            import kagglesdk.kaggle_env as kenv
        except ImportError:
            return
        if hasattr(kenv, "get_web_endpoint"):
            return

        def get_web_endpoint(config=None):
            if config and hasattr(config, "get"):
                endpoint = config.get("web_endpoint")
                if endpoint:
                    return endpoint
            return PiscesLxToolsDataJsonSampleReader.KAGGLE_WEB_API

        kenv.get_web_endpoint = get_web_endpoint


class PiscesLxToolsDataTextCollator:
    """Turn tokenized samples into a batch: pre-shaped, text, or ragged ids.

    Defined at module level because ``DataLoader`` workers pickle ``collate_fn``;
    the closure this replaces made every run with ``num_workers > 0`` fail with
    ``PicklingError``.
    """

    IGNORED_LABEL_ID = -100

    def __init__(self, seq_len: int, pad_id: int, packing: bool = False,
                 tokenizer=None, pad_to_multiple_of: int = 0) -> None:
        self.seq_len = int(seq_len)
        self.pad_id = int(pad_id)
        self.packing = bool(packing)
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def __call__(self, batch_items):
        if not batch_items:
            return self._empty_batch()

        first = batch_items[0]
        if {"input_ids", "attention_mask", "labels"} <= set(first):
            return {
                key: torch.stack([item[key] for item in batch_items], dim=0)
                for key in ("input_ids", "attention_mask", "labels")
            }

        if self.tokenizer is not None and "text" in first:
            return self._tokenize_texts([item.get("text", "") for item in batch_items])

        return self._pack_or_pad([item.get("input_ids") for item in batch_items])

    def _tokenize_texts(self, texts):
        rows = [PiscesLxToolsDataSampleRecord.encode_ids(self.tokenizer, text, self.seq_len)
                for text in texts]
        width = max(int(row.numel()) for row in rows) or 1
        if self.pad_to_multiple_of > 0:
            width = -(-width // self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            length = min(int(row.numel()), width)
            input_ids[index, :length] = row[:length]
            attention_mask[index, :length] = 1
        labels = input_ids.clone()
        labels[labels == self.pad_id] = self.IGNORED_LABEL_ID
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _pack_or_pad(self, id_lists):
        rows = [ids.long().view(-1) if ids is not None
                else torch.tensor([self.pad_id], dtype=torch.long) for ids in id_lists]
        if self.packing:
            flat = torch.cat(rows) if rows else torch.tensor([self.pad_id], dtype=torch.long)
            chunks = []
            offset = 0
            while offset < flat.numel():
                chunk = flat[offset:offset + self.seq_len]
                if chunk.numel() < self.seq_len:
                    chunk = torch.cat([chunk, torch.full((self.seq_len - chunk.numel(),),
                                                         self.pad_id, dtype=torch.long)])
                chunks.append(chunk)
                offset += self.seq_len
            input_ids = torch.stack(chunks)
            attention_mask = (input_ids != self.pad_id).long()
            labels = input_ids.clone()
            labels[labels == self.pad_id] = self.IGNORED_LABEL_ID
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        input_ids = torch.full((len(rows), self.seq_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(rows), self.seq_len), self.IGNORED_LABEL_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), self.seq_len), dtype=torch.long)
        for index, ids in enumerate(rows):
            length = int(ids.numel())
            ids = ids[:self.seq_len]
            length = min(length, self.seq_len)
            input_ids[index, :length] = ids[:length]
            labels[index, :length] = ids[:length]
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _empty_batch(self):
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "attention_mask": empty.clone(), "labels": empty.clone()}


class PiscesLxToolsDataSampleRecord:
    """Shared record handling for the supervised task datasets below."""

    PROMPT_FIELDS = ("instruction", "prompt", "question", "input", "text", "content")
    RESPONSE_FIELDS = ("output", "response", "answer", "completion")
    CHOSEN_FIELDS = ("chosen", "preferred")
    REJECTED_FIELDS = ("rejected",)
    LABEL_FIELDS = ("label", "class", "target")
    IGNORED_LABEL_ID = -100

    @staticmethod
    def encode_ids(tokenizer, text: str, max_len: int) -> torch.Tensor:
        """Encode *text* with the project tokenizer and cut it to *max_len* tokens.

        ``YvTokenizer.encode`` exposes no truncation option — it returns the whole BPE
        sequence — so the limit is applied here instead of feeding the model over-long
        rows.
        """
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()
        return torch.tensor(list(ids[:max_len]), dtype=torch.long)

    @classmethod
    def field(cls, sample: Dict[str, Any], names) -> str:
        for name in names:
            value = sample.get(name)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def encode_pair(tokenizer, prompt: str, response: str, max_len: int):
        """Tokenize a prompt/response pair, masking the prompt so loss is on the answer."""
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if not response_ids:
            raise ValueError("empty response — a supervised record must have answer text")
        total = len(prompt_ids) + len(response_ids)
        if total > max_len:
            keep = max_len - len(response_ids)
            if keep < 0:
                raise ValueError(
                    f"response alone ({len(response_ids)} tokens) exceeds max_len={max_len}"
                )
            prompt_ids = prompt_ids[:keep]
        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = PiscesLxToolsDataSampleRecord.IGNORED_LABEL_ID
        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}


class PiscesLxToolsDataSupervisedDataset(torch.utils.data.Dataset):
    """Real SFT records: prompt plus answer, with the prompt excluded from the loss."""

    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.records = []
        for sample in samples:
            prompt = PiscesLxToolsDataSampleRecord.field(
                sample, PiscesLxToolsDataSampleRecord.PROMPT_FIELDS
            )
            response = PiscesLxToolsDataSampleRecord.field(
                sample, PiscesLxToolsDataSampleRecord.RESPONSE_FIELDS
            )
            if not prompt and not response:
                continue
            self.records.append((prompt, response or prompt))
        if not self.records:
            raise ValueError(
                "no supervised records: every sample is missing the prompt/response fields"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, response = self.records[idx]
        return PiscesLxToolsDataSampleRecord.encode_pair(
            self.tokenizer, prompt, response, self.max_len
        )


class PiscesLxToolsDataPreferenceDataset(torch.utils.data.Dataset):
    """Real preference pairs for DPO-style stages."""

    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.records = []
        for sample in samples:
            prompt = PiscesLxToolsDataSampleRecord.field(
                sample, PiscesLxToolsDataSampleRecord.PROMPT_FIELDS
            )
            chosen = PiscesLxToolsDataSampleRecord.field(
                sample, PiscesLxToolsDataSampleRecord.CHOSEN_FIELDS
            )
            rejected = PiscesLxToolsDataSampleRecord.field(
                sample, PiscesLxToolsDataSampleRecord.REJECTED_FIELDS
            )
            if not (prompt and chosen and rejected):
                continue
            self.records.append((prompt, chosen, rejected))
        if not self.records:
            raise ValueError("no preference pairs: samples lack prompt/chosen/rejected")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, chosen, rejected = self.records[idx]
        pair = {}
        for side, answer in (("chosen", chosen), ("rejected", rejected)):
            encoded = PiscesLxToolsDataSampleRecord.encode_pair(
                self.tokenizer, prompt, answer, self.max_len
            )
            pair[f"{side}_input_ids"] = encoded["input_ids"]
            pair[f"{side}_labels"] = encoded["labels"]
            pair[f"{side}_attention_mask"] = encoded["attention_mask"]
        return pair


class PiscesLxToolsDataClassificationDataset(torch.utils.data.Dataset):
    """Real labelled text records: input ids plus a scalar class label."""

    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.records = []
        for sample in samples:
            text = PiscesLxToolsDataSampleRecord.field(
                sample, PiscesLxToolsDataSampleRecord.PROMPT_FIELDS
            )
            label = next((sample[name] for name in PiscesLxToolsDataSampleRecord.LABEL_FIELDS
                          if name in sample), None)
            if not text or label is None:
                continue
            self.records.append((text, int(label)))
        if not self.records:
            raise ValueError("no labelled records: samples lack text and label")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label = self.records[idx]
        ids = PiscesLxToolsDataSampleRecord.encode_ids(self.tokenizer, text, self.max_len)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "labels": torch.tensor([label], dtype=torch.long)}
