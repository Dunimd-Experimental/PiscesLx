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



import os
import gc
import json
import hashlib
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Callable, Iterator
from datasets import load_from_disk, Dataset, concatenate_datasets
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading
from pathlib import Path

from .rules import PiscesLxToolsDataStreamCleaner as StreamCleaner, AUTO_FIELDS
from utils.dc import PiscesLxLogger
from utils.paths import get_cache_dir, get_log_file
from .quality import PiscesLxToolsDataQualityController

_LOG = PiscesLxLogger("PiscesLx.Tools.Data", file_path=get_log_file("PiscesLx.Tools.Data"), enable_file=True)


class CleaningMode(Enum):
    LOCAL = "local"
    DISTRIBUTED_RAY = "distributed_ray"
    DISTRIBUTED_DASK = "distributed_dask"


@dataclass
class PiscesLxDataCleaningConfig:
    batch_size: int = 1000
    max_workers: int = 4
    quality_threshold: float = 0.5
    min_text_length: int = 1
    max_text_length: int = 65536
    enable_quality_scoring: bool = True
    enable_incremental: bool = False
    checkpoint_interval: int = 5000
    mode: CleaningMode = CleaningMode.LOCAL
    preserve_metadata: bool = True
    dedup_enabled: bool = False
    dedup_method: str = "minhash"
    dedup_threshold: float = 0.8


class PiscesLxDataIncrementalState:
    def __init__(self, state_path: str):
        self.state_path = state_path
        self.processed_files: Dict[str, str] = {}
        self.processed_samples: int = 0
        self.last_checkpoint: int = 0
        self._lock = threading.Lock()
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.processed_files = data.get("processed_files", {})
                    self.processed_samples = data.get("processed_samples", 0)
                    self.last_checkpoint = data.get("last_checkpoint", 0)
            except Exception as e:
                _LOG.warning(f"Failed to load incremental state: {e}")

    def save_state(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "processed_files": self.processed_files,
                    "processed_samples": self.processed_samples,
                    "last_checkpoint": self.last_checkpoint
                }, f, indent=2)

    def is_processed(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        file_hash = self._compute_file_hash(file_path)
        with self._lock:
            return self.processed_files.get(file_path) == file_hash

    def mark_processed(self, file_path: str, sample_count: int = 0):
        file_hash = self._compute_file_hash(file_path)
        with self._lock:
            self.processed_files[file_path] = file_hash
            self.processed_samples += sample_count

    def _compute_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return os.path.getmtime(file_path).__str__()


class PiscesLxDataQualityAwareCleaner:
    def __init__(self, config: Optional[PiscesLxDataCleaningConfig] = None):
        self.config = config or PiscesLxDataCleaningConfig()
        self.quality_controller = PiscesLxToolsDataQualityController(
            quality_threshold=self.config.quality_threshold
        )
        self._cleaning_stats: Dict[str, Any] = {
            "total_processed": 0,
            "high_quality": 0,
            "medium_quality": 0,
            "low_quality": 0,
            "filtered_out": 0
        }

    def clean_with_quality(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        if not text or not isinstance(text, str):
            return "", 0.0, {"reason": "empty_or_invalid"}

        text = self._normalize_text(text)

        if len(text) < self.config.min_text_length:
            return "", 0.0, {"reason": "too_short", "length": len(text)}

        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]

        quality_score = self.quality_controller.calculate_text_quality_score(text)

        quality_metrics = {
            "length": len(text),
            "quality_score": quality_score,
            "word_count": len(text.split()),
            "char_diversity": len(set(text.lower())) / max(len(text), 1)
        }

        if quality_score < self.config.quality_threshold:
            self._cleaning_stats["filtered_out"] += 1
            return text, quality_score, {**quality_metrics, "reason": "low_quality"}

        if quality_score >= 0.8:
            self._cleaning_stats["high_quality"] += 1
        elif quality_score >= 0.5:
            self._cleaning_stats["medium_quality"] += 1
        else:
            self._cleaning_stats["low_quality"] += 1

        self._cleaning_stats["total_processed"] += 1

        return text, quality_score, quality_metrics

    def _normalize_text(self, text: str) -> str:
        import re
        text = str(text).strip()
        if not text:
            return ""

        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.!?])\1+", r"\1", text)
        text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)

        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        return self._cleaning_stats.copy()


class DatasetCleaner:
    _incremental_states: Dict[str, PiscesLxDataIncrementalState] = {}

    @classmethod
    def get_incremental_state(cls, state_path: str) -> PiscesLxDataIncrementalState:
        if state_path not in cls._incremental_states:
            cls._incremental_states[state_path] = PiscesLxDataIncrementalState(state_path)
        return cls._incremental_states[state_path]

    @staticmethod
    def process_dataset(
        input_path: str,
        output_path: str,
        text_field: str = "text",
        config: Optional[PiscesLxDataCleaningConfig] = None,
        **clean_kwargs: Any
    ) -> Tuple[int, int]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset path does not exist: {input_path}")

        config = config or PiscesLxDataCleaningConfig(
            min_text_length=int(clean_kwargs.get("min_length", 1)),
            quality_threshold=float(clean_kwargs.get("quality_threshold", 0.5))
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        original_size = 0
        if os.path.isdir(input_path):
            dataset = load_from_disk(input_path)
            df = dataset.to_pandas()
            original_size = len(dataset)
        elif input_path.endswith(".json"):
            if os.path.basename(input_path) in ["dataset_info.json", "state.json"]:
                raise ValueError(f"Skipping system file: {input_path}")
            df = pd.read_json(input_path)
            original_size = len(df)
        elif input_path.endswith(".jsonl"):
            df = pd.read_json(input_path, lines=True)
            original_size = len(df)
        elif input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
            original_size = len(df)
        elif input_path.endswith(".parquet"):
            df = pd.read_parquet(input_path)
            original_size = len(df)
        else:
            raise ValueError("Unsupported file format. Supported: .arrow dir, .json, .jsonl, .csv, .parquet")

        if text_field not in df.columns:
            detected = None
            from .. import TEXT_FIELD_KEYS
            for k in TEXT_FIELD_KEYS:
                if k in df.columns:
                    detected = k
                    break
            if detected:
                text_field = detected
            else:
                string_cols = df.select_dtypes(include=["object"]).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                else:
                    raise ValueError(f"No text field found. Columns: {list(df.columns)}")

        if text_field in ["conversations", "messages", "conversation", "code"]:
            df[text_field] = df[text_field].apply(DatasetCleaner._extract_text_from_complex_format)

        quality_cleaner = PiscesLxDataQualityAwareCleaner(config)

        def _clean_with_quality(text):
            cleaned, score, metrics = quality_cleaner.clean_with_quality(str(text) if text else "")
            return cleaned, score

        results = df[text_field].apply(_clean_with_quality)
        df[text_field] = results.apply(lambda x: x[0])
        if config.enable_quality_scoring:
            df["quality_score"] = results.apply(lambda x: x[1])

        min_length = config.min_text_length
        mask = df[text_field].astype(str).str.strip().str.len() >= min_length
        if config.enable_quality_scoring:
            mask = mask & (df["quality_score"] >= config.quality_threshold)
        df = df[mask]

        df = df.dropna(how="all")
        cleaned = Dataset.from_pandas(df, preserve_index=False)
        cleaned.save_to_disk(output_path)

        return len(df), original_size

    @staticmethod
    def _extract_text_from_complex_format(data) -> str:
        if isinstance(data, str):
            return data.strip()
        elif isinstance(data, list):
            texts: List[str] = []
            for item in data:
                if isinstance(item, dict):
                    for key in ["content", "text", "value", "human", "assistant", "user", "bot", "output", "response"]:
                        if key in item and item[key]:
                            texts.append(str(item[key]).strip())
                            break
                elif isinstance(item, str):
                    texts.append(item.strip())
            return " ".join(texts)
        elif isinstance(data, dict):
            texts: List[str] = []
            for key in ["content", "text", "value", "human", "assistant", "user", "bot", "output", "response"]:
                if key in data and data[key]:
                    texts.append(str(data[key]).strip())
            return " ".join(texts)
        else:
            return str(data).strip()

    @staticmethod
    def process_distributed(
        input_paths: List[str],
        output_dir: str,
        config: Optional[PiscesLxDataCleaningConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Tuple[int, int]]:
        config = config or PiscesLxDataCleaningConfig()
        results: Dict[str, Tuple[int, int]] = {}

        if config.mode == CleaningMode.DISTRIBUTED_RAY:
            results = DatasetCleaner._process_with_ray(input_paths, output_dir, config, progress_callback)
        elif config.mode == CleaningMode.DISTRIBUTED_DASK:
            results = DatasetCleaner._process_with_dask(input_paths, output_dir, config, progress_callback)
        else:
            results = DatasetCleaner._process_with_multiprocessing(input_paths, output_dir, config, progress_callback)

        return results

    @staticmethod
    def _process_with_ray(
        input_paths: List[str],
        output_dir: str,
        config: PiscesLxDataCleaningConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Tuple[int, int]]:
        try:
            import ray
        except ImportError:
            _LOG.warning("Ray not installed, falling back to multiprocessing")
            return DatasetCleaner._process_with_multiprocessing(input_paths, output_dir, config, progress_callback)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        @ray.remote
        def clean_remote(input_path: str, output_path: str, config_dict: dict):
            cfg = PiscesLxDataCleaningConfig(**config_dict)
            try:
                return DatasetCleaner.process_dataset(input_path, output_path, config=cfg)
            except Exception as e:
                _LOG.error(f"Ray worker failed: {e}")
                return (0, 0)

        config_dict = {
            "batch_size": config.batch_size,
            "max_workers": config.max_workers,
            "quality_threshold": config.quality_threshold,
            "min_text_length": config.min_text_length,
            "max_text_length": config.max_text_length,
            "enable_quality_scoring": config.enable_quality_scoring,
            "enable_incremental": config.enable_incremental,
            "checkpoint_interval": config.checkpoint_interval,
            "mode": config.mode.value,
            "preserve_metadata": config.preserve_metadata,
            "dedup_enabled": config.dedup_enabled,
            "dedup_method": config.dedup_method,
            "dedup_threshold": config.dedup_threshold
        }

        futures = []
        for input_path in input_paths:
            name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, f"{name}_clean")
            futures.append((input_path, clean_remote.remote(input_path, output_path, config_dict)))

        results = {}
        total = len(futures)
        for i, (input_path, future) in enumerate(futures):
            result = ray.get(future)
            results[input_path] = result
            if progress_callback:
                progress_callback(i + 1, total)

        return results

    @staticmethod
    def _process_with_dask(
        input_paths: List[str],
        output_dir: str,
        config: PiscesLxDataCleaningConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Tuple[int, int]]:
        try:
            import dask
            import dask.bag as db
            from dask.distributed import Client
        except ImportError:
            _LOG.warning("Dask not installed, falling back to multiprocessing")
            return DatasetCleaner._process_with_multiprocessing(input_paths, output_dir, config, progress_callback)

        results = {}
        total = len(input_paths)

        for i, input_path in enumerate(input_paths):
            try:
                name = os.path.basename(input_path)
                output_path = os.path.join(output_dir, f"{name}_clean")
                result = DatasetCleaner.process_dataset(input_path, output_path, config=config)
                results[input_path] = result
                if progress_callback:
                    progress_callback(i + 1, total)
            except Exception as e:
                _LOG.error(f"Dask processing failed for {input_path}: {e}")
                results[input_path] = (0, 0)

        return results

    @staticmethod
    def _process_with_multiprocessing(
        input_paths: List[str],
        output_dir: str,
        config: PiscesLxDataCleaningConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Tuple[int, int]]:
        results = {}
        total = len(input_paths)
        workers = min(config.max_workers, mp.cpu_count(), len(input_paths))

        if workers <= 1 or len(input_paths) <= 1:
            for i, input_path in enumerate(input_paths):
                name = os.path.basename(input_path)
                output_path = os.path.join(output_dir, f"{name}_clean")
                try:
                    result = DatasetCleaner.process_dataset(input_path, output_path, config=config)
                    results[input_path] = result
                except Exception as e:
                    _LOG.error(f"Processing failed for {input_path}: {e}")
                    results[input_path] = (0, 0)
                if progress_callback:
                    progress_callback(i + 1, total)
        else:
            args = []
            for input_path in input_paths:
                name = os.path.basename(input_path)
                output_path = os.path.join(output_dir, f"{name}_clean")
                args.append((input_path, output_path, config))

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(DatasetCleaner._process_single_wrapper, arg): arg[0] for arg in args}
                for i, future in enumerate(as_completed(futures)):
                    input_path = futures[future]
                    try:
                        results[input_path] = future.result()
                    except Exception as e:
                        _LOG.error(f"Worker failed for {input_path}: {e}")
                        results[input_path] = (0, 0)
                    if progress_callback:
                        progress_callback(i + 1, total)

        return results

    @staticmethod
    def _process_single_wrapper(args):
        input_path, output_path, config = args
        return DatasetCleaner.process_dataset(input_path, output_path, config=config)

    @staticmethod
    def process_incremental(
        input_dir: str,
        output_dir: str,
        state_file: Optional[str] = None,
        config: Optional[PiscesLxDataCleaningConfig] = None
    ) -> Dict[str, Tuple[int, int]]:
        config = config or PiscesLxDataCleaningConfig(enable_incremental=True)

        state_path = state_file or os.path.join(output_dir, ".cleaning_state.json")
        state = DatasetCleaner.get_incremental_state(state_path)

        todo_paths = []
        for name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, name)
            if os.path.isdir(input_path) and not name.endswith("_clean"):
                if not state.is_processed(input_path):
                    todo_paths.append(input_path)

        if not todo_paths:
            _LOG.info("No new files to process (incremental mode)")
            return {}

        results = DatasetCleaner.process_distributed(todo_paths, output_dir, config)

        for input_path, (cleaned, total) in results.items():
            state.mark_processed(input_path, cleaned)

        state.save_state()

        return results

    @staticmethod
    def _process_chunk_with_quality(
        chunk,
        text_field: str,
        multimodal_fields: Dict[str, str],
        enable_quality_scoring: bool,
        config: Optional[PiscesLxDataCleaningConfig] = None,
        **kwargs: Any
    ):
        config = config or PiscesLxDataCleaningConfig()
        quality_cleaner = PiscesLxDataQualityAwareCleaner(config)

        cleaned_rows: List[Dict[str, Any]] = []
        text_scores: List[float] = []
        media_scores: List[float] = []

        for sample in chunk:
            try:
                row = dict(sample)

                if text_field in row:
                    v = row[text_field]
                    if isinstance(v, (list, dict)):
                        v = DatasetCleaner._extract_text_from_complex_format(v)

                    cleaned_text, score, metrics = quality_cleaner.clean_with_quality(str(v) if v else "")
                    row[text_field] = cleaned_text

                    if enable_quality_scoring:
                        row["text_quality_score"] = score
                        text_scores.append(score)
                    else:
                        text_scores.append(1.0)

                sum_q = 0.0
                cnt_q = 0
                for col, mtype in multimodal_fields.items():
                    if col in row and row[col]:
                        cleaned_path = StreamCleaner().clean_media(str(row[col]), mtype)
                        if cleaned_path:
                            row[col] = cleaned_path
                            if enable_quality_scoring:
                                q = StreamCleaner.get_media_quality_score(cleaned_path, mtype)
                                row[f"{col}_quality_score"] = q
                                sum_q += q
                                cnt_q += 1
                        else:
                            row.pop(col, None)

                media_scores.append((sum_q / cnt_q) if cnt_q else 1.0)
                cleaned_rows.append(row)
            except Exception:
                continue

        return cleaned_rows, text_scores, media_scores

    @staticmethod
    def merge_and_clean(
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_len: int = 1,
        max_len: int = 1024,
        workers: Optional[int] = None,
        rules=None,
        config: Optional[PiscesLxDataCleaningConfig] = None
    ):
        input_dir = input_dir or get_cache_dir("data_cache")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        config = config or PiscesLxDataCleaningConfig(
            min_text_length=min_len,
            max_text_length=max_len * 100,
            max_workers=workers or min(4, mp.cpu_count())
        )

        raw_paths = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                     if os.path.isdir(os.path.join(input_dir, d))]
        if not raw_paths:
            return None

        from multiprocessing import Pool

        def _worker(ds_path):
            try:
                ds = load_from_disk(ds_path)
                cleaner = _StreamCleaner(min_len=min_len, max_len=max_len)
                chunk = []
                step = 10000
                for s in range(0, len(ds), step):
                    sub = ds.select(range(s, min(s + step, len(ds))))
                    df = sub.to_pandas()
                    if "text" in df.columns:
                        df["text"] = df["text"].apply(cleaner.clean_text)
                        df = df[df["text"].astype(str).str.strip() != ""]
                    for mtype, cands in AUTO_FIELDS.items():
                        col = next((c for c in cands if c in df.columns), None)
                        if col:
                            df[col] = df[col].apply(
                                lambda x: cleaner.clean_media(str(x), mtype) if pd.notna(x) else None
                            )
                    if len(df) > 0:
                        chunk.append(Dataset.from_pandas(df))
                    del df, sub
                    gc.collect()
                return concatenate_datasets(chunk) if chunk else None
            except Exception as e:
                _LOG.error(f"Worker failed: {ds_path} {e}")
                return None

        with Pool(processes=config.max_workers) as pool:
            results = list(pool.imap(_worker, raw_paths))

        valid = [r for r in results if r is not None]
        if not valid:
            return None

        merged = concatenate_datasets(valid)
        if "source" in merged.column_names:
            merged = merged.remove_columns(["source"])

        _LOG.info(f"Merged cleaned datasets: {len(merged)} rows")
        return merged

    @staticmethod
    def is_download_complete(dataset_path: str) -> bool:
        markers = [".download_complete", ".finished", "download_status.txt", "completed.flag"]
        for m in markers:
            if os.path.exists(os.path.join(dataset_path, m)):
                return True
        try:
            ds = load_from_disk(dataset_path)
            if len(ds) > 0:
                for _ in ds.take(1):
                    pass
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def auto_clean(
        input_dir: str,
        output_dir: str = "data_clean",
        media_fields: Optional[Dict[str, str]] = None,
        workers: Optional[int] = None,
        config: Optional[PiscesLxDataCleaningConfig] = None,
        **clean_kwargs: Any
    ) -> bool:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        config = config or PiscesLxDataCleaningConfig(
            max_workers=workers or min(4, mp.cpu_count()),
            min_text_length=int(clean_kwargs.get("min_length", 1)),
            quality_threshold=float(clean_kwargs.get("quality_threshold", 0.5))
        )

        todo: List[Tuple[str, str, str]] = []
        for name in os.listdir(input_dir):
            in_p = os.path.join(input_dir, name)
            if os.path.isdir(in_p) and not name.endswith("_clean"):
                if not DatasetCleaner.is_download_complete(in_p):
                    _LOG.debug(f"Dataset {name} download not complete, skip")
                    continue
                out_p = os.path.join(output_dir, f"{name}_clean")
                if not os.path.exists(out_p):
                    todo.append((name, in_p, out_p))
                else:
                    _LOG.info(f"Cleaned dataset exists: {out_p}, skip")

        if not todo:
            _LOG.info("No datasets to clean")
            return True

        if config.max_workers <= 1 or len(todo) <= 1:
            for name, in_p, out_p in todo:
                DatasetCleaner._process_single_dataset(name, in_p, out_p, media_fields, config=config, **clean_kwargs)
        else:
            _LOG.debug(f"Using {config.max_workers} processes to clean {len(todo)} datasets...")
            args = [(n, i, o, media_fields, config, clean_kwargs) for (n, i, o) in todo]
            with ProcessPoolExecutor(max_workers=config.max_workers) as ex:
                fut = {ex.submit(DatasetCleaner._process_single_dataset_wrapper, a): a[0] for a in args}
                for f in as_completed(fut):
                    name = fut[f]
                    try:
                        cleaned, total = f.result()
                        if cleaned == 0:
                            _LOG.debug(f"No valid samples left after cleaning {name} (original {total})")
                        else:
                            _LOG.info(f"Cleaning successful: {name} -> {name}_clean | {cleaned}/{total}")
                    except Exception as e:
                        _LOG.error(f"Error cleaning {name}: {e}")

        return True

    @staticmethod
    def _process_single_dataset_wrapper(args):
        n, i, o, m, cfg, kw = args
        return DatasetCleaner._process_single_dataset(n, i, o, m, config=cfg, **kw)

    @staticmethod
    def _process_single_dataset(
        dataset_name,
        input_path,
        output_path,
        media_fields=None,
        config: Optional[PiscesLxDataCleaningConfig] = None,
        **clean_kwargs
    ):
        config = config or PiscesLxDataCleaningConfig()

        try:
            if not os.path.exists(input_path):
                _LOG.debug(f"Dataset does not exist: {input_path}")
                return (0, 0)

            try:
                dataset = load_from_disk(input_path)
                if len(dataset) == 0:
                    _LOG.debug(f"Dataset {dataset_name} is empty, skip")
                    return (0, 0)
                _LOG.debug(f"Processing dataset: {dataset_name} ({len(dataset)} rows)")
            except Exception as e:
                _LOG.error(f"Failed to load dataset {dataset_name}: {e}")
                return (0, 0)

            if media_fields:
                cleaned, total = DatasetCleaner.process_multimodal_dataset(
                    input_path, output_path, media_fields=media_fields, config=config, **clean_kwargs
                )
            else:
                cleaned, total = DatasetCleaner.process_dataset(
                    input_path, output_path, config=config, **clean_kwargs
                )
            return (cleaned, total)
        except Exception as e:
            _LOG.error(f"Error cleaning {dataset_name}: {e}")
            return (0, 0)

    @staticmethod
    def process_multimodal_dataset(
        input_path: str,
        output_path: str,
        text_field: str = "text",
        media_fields: Optional[Dict[str, str]] = None,
        quality_threshold: float = 0.5,
        enable_quality_scoring: bool = True,
        chunk_size: int = 2000,
        config: Optional[PiscesLxDataCleaningConfig] = None,
        **kwargs: Any
    ):
        config = config or PiscesLxDataCleaningConfig(
            quality_threshold=quality_threshold,
            enable_quality_scoring=enable_quality_scoring,
            batch_size=chunk_size
        )

        if not os.path.exists(input_path):
            return (0, 0)

        dataset = load_from_disk(input_path) if os.path.isdir(input_path) else None
        if dataset is None:
            return (0, 0)

        total = len(dataset)
        multimodal_fields = media_fields or _StreamCleaner.find_multimodal_fields_from_dataset(dataset)

        cleaned_rows: List[Dict[str, Any]] = []
        text_scores: List[float] = []
        media_scores: List[float] = []

        for i in range(0, total, config.batch_size):
            chunk = dataset.select(range(i, min(i + config.batch_size, total)))
            rows, ts, ms = DatasetCleaner._process_chunk_with_quality(
                chunk, text_field, multimodal_fields, config.enable_quality_scoring, config, **kwargs
            )
            cleaned_rows.extend(rows)
            text_scores.extend(ts)
            media_scores.extend(ms)
            gc.collect()
            _LOG.debug(f"Processed chunk {i // config.batch_size + 1}/{(total - 1) // config.batch_size + 1}")

        if config.enable_quality_scoring:
            idx = [i for i, s in enumerate(text_scores) if s >= config.quality_threshold]
            cleaned_rows = [cleaned_rows[i] for i in idx]

        if cleaned_rows:
            df = pd.DataFrame(cleaned_rows)
            ds = Dataset.from_pandas(df)
            ds.save_to_disk(output_path)
            return (len(cleaned_rows), total)

        return (0, total)


class _StreamCleaner:
    def __init__(self, min_len: int = 1, max_len: int = 1024):
        self.min_len = min_len
        self.max_len = max_len

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        import re
        t = str(text).strip()
        if not t:
            return ""
        try:
            t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", t)
            t = re.sub(r"[^\w\s\.,!?;:-\(\)\[\]\{\}\"'@#$%&*+=<>/?\\|`~]", "", t)
            t = re.sub(r"\s+", " ", t)
            t = re.sub(r"([.!?])\1+", r"\1", t)
            t = re.sub(r"(.)\1{3,}", r"\1\1", t)
            t = t.strip()
            return t
        except Exception:
            t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

    def clean_media(self, path: str, media_type: str) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        return path

    @staticmethod
    def find_multimodal_fields_from_dataset(dataset) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        cols = dataset.column_names if hasattr(dataset, "column_names") else []

        for mtype, cands in AUTO_FIELDS.items():
            for col in cands:
                if col in cols:
                    fields[col] = mtype
                    break

        return fields

    @staticmethod
    def get_media_quality_score(path: str, media_type: str) -> float:
        if not path or not os.path.exists(path):
            return 0.0

        try:
            file_size = os.path.getsize(path)
            if file_size == 0:
                return 0.0

            if media_type == "image":
                if file_size < 1024:
                    return 0.3
                elif file_size < 10240:
                    return 0.6
                else:
                    return 0.9
            elif media_type == "audio":
                if file_size < 10240:
                    return 0.4
                elif file_size < 102400:
                    return 0.7
                else:
                    return 0.9
            else:
                return 0.5
        except Exception:
            return 0.5
