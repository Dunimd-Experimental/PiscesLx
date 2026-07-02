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
import time
import shutil
import json
import hashlib
import random
from tqdm import tqdm
import multiprocessing
from typing import Any, Tuple, Optional, Set, List, Dict, Callable
from dataclasses import dataclass, field
from functools import wraps
from utils.dc import PiscesLxLogger
from .caches import PiscesLxToolsDataDownloadCache
from datasets import load_from_disk, Dataset
from tools.data.clean import DatasetCleaner
from .config import PiscesLxToolsDataConfigLoader, PiscesLxToolsDataDownloadConfig, PiscesLxToolsDatasetItem
from .sources import PiscesLxToolsDataSourceRouter

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Data", file_path=get_log_file("PiscesLx.Tools.Data"), enable_file=True)


@dataclass
class PiscesLxDataDownloadState:
    dataset_name: str
    save_name: str
    status: str = "pending"
    attempt: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    downloaded_bytes: int = 0
    total_bytes: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source: Optional[str] = None
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "save_name": self.save_name,
            "status": self.status,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "source": self.source,
            "checkpoint_path": self.checkpoint_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PiscesLxDataDownloadState":
        return cls(**data)


class PiscesLxDataRetryStrategy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def should_retry(self, attempt: int, error: Optional[Exception] = None) -> bool:
        if attempt >= self.max_retries:
            return False

        if error is not None:
            retryable_errors = (
                ConnectionError,
                TimeoutError,
                OSError,
            )
            if isinstance(error, retryable_errors):
                return True

            error_str = (str(error) or "").lower()

            retryable_keywords = [
                "timeout", "connection", "network", "rate limit",
                "too many requests", "service unavailable", "gateway",
            ]
            if any(kw in error_str for kw in retryable_keywords):
                return True

            # Deterministic failures: the environment is broken in a way that
            # retrying without user intervention cannot fix. Examples include
            # missing packages and incompatible package versions. Retrying
            # just burns time and floods the log.
            fatal_keywords = [
                "importerror", "modulenotfounderror",
                "pretrainedmodel",
                "failed to import `modelscope.msdatasets`",
                "package is not installed",
                "is not importable",
                "syntaxerror",
            ]
            if any(kw in error_str for kw in fatal_keywords):
                return False

        return True


class PiscesLxDataResumeManager:
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self._states: Dict[str, PiscesLxDataDownloadState] = {}
        os.makedirs(state_dir, exist_ok=True)

    def _get_state_path(self, save_name: str) -> str:
        return os.path.join(self.state_dir, f"{save_name}.state.json")

    def load_state(self, save_name: str) -> Optional[PiscesLxDataDownloadState]:
        if save_name in self._states:
            return self._states[save_name]

        state_path = self._get_state_path(save_name)
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    state = PiscesLxDataDownloadState.from_dict(data)
                    self._states[save_name] = state
                    return state
            except Exception:
                pass

        return None

    def save_state(self, state: PiscesLxDataDownloadState):
        self._states[state.save_name] = state
        state_path = self._get_state_path(state.save_name)

        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
        except Exception:
            pass

    def clear_state(self, save_name: str):
        if save_name in self._states:
            del self._states[save_name]

        state_path = self._get_state_path(save_name)
        if os.path.exists(state_path):
            try:
                os.remove(state_path)
            except Exception:
                pass

    def is_completed(self, save_name: str) -> bool:
        state = self.load_state(save_name)
        return state is not None and state.status == "completed"

    def get_pending_attempt(self, save_name: str) -> int:
        state = self.load_state(save_name)
        if state is not None and state.status == "failed":
            return state.attempt
        return 0


class PiscesLxToolsDataDatasetDownload:
    @staticmethod
    def save_dataset(ds: Any, data_dir: str, name: str, max_samples: Optional[int] = None) -> Optional[str]:
        try:
            if not hasattr(ds, "save_to_disk"):
                _LOG.error(
                    f"Cannot save dataset {name}: expected a HuggingFace Dataset, got {type(ds).__name__}"
                )
                return None

            if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
                _LOG.info(f"Limiting dataset {name} from {len(ds)} to {max_samples} samples")
                ds = ds.select(range(max_samples))

            save_path = os.path.join(data_dir, name)
            _LOG.info(f"Saving dataset '{name}' to: {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ds.save_to_disk(save_path)
            _LOG.info(f"Successfully saved dataset '{name}' to: {save_path}")
            return save_path
        except Exception as e:
            _LOG.error(f"Failed to save dataset {name}: {str(e)}")
            return None

    @staticmethod
    def download_worker(task: Tuple) -> Optional[str]:
        from .sources import PiscesLxToolsDataSourceRouter
        from .caches import PiscesLxToolsDataDownloadCache

        dataset_name, save_name, description, preferred_sources, data_dir, max_samples, retry_config = task

        cache = PiscesLxToolsDataDownloadCache()
        cache.setup_env()

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_HTTP_TIMEOUT"] = "300"

        PiscesLxToolsDataSourceRouter.setup_hf_mirror()

        logger = PiscesLxLogger("PiscesLx.Tools.Data.Download", file_path=get_log_file("PiscesLx.Tools.Data.Download"), enable_file=True)
        logger.info(f"Starting download: {dataset_name} -> {save_name}")

        temp_dir = cache.get_temp_dir()
        state_dir = os.path.join(temp_dir, "download_states")
        resume_manager = PiscesLxDataResumeManager(state_dir)

        existing_state = resume_manager.load_state(save_name)
        if existing_state and existing_state.status == "completed":
            logger.info(f"Dataset {save_name} already completed, skipping")
            return save_name

        state = PiscesLxDataDownloadState(
            dataset_name=dataset_name,
            save_name=save_name,
            status="in_progress",
            source=preferred_sources[0] if preferred_sources else "modelscope"
        )

        max_retries = retry_config.get("max_retries", 3) if retry_config else 3
        base_delay = retry_config.get("base_delay", 2.0) if retry_config else 2.0
        max_delay = retry_config.get("max_delay", 120.0) if retry_config else 120.0

        retry_strategy = PiscesLxDataRetryStrategy(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay
        )

        for attempt in range(max_retries):
            state.attempt = attempt
            state.start_time = time.time()

            try:
                logger.debug(f"Downloading {dataset_name} (attempt {attempt + 1}/{max_retries})")

                router = PiscesLxToolsDataSourceRouter()
                strict_sources: List[str] = [preferred_sources[0]] if preferred_sources else ["modelscope"]
                src = strict_sources[0].strip().lower()

                splits = PiscesLxToolsDataSourceRouter.detect_available_splits(dataset_name, src)
                methods: List[Tuple[dict, str]] = []

                if "__direct__" in splits or not splits:
                    methods.append(({}, "direct"))

                for sp in splits:
                    if sp == "__direct__":
                        continue
                    methods.append(({"split": sp}, f"split={sp}"))

                last_err: Optional[str] = None
                ds = None

                for kwargs, desc in methods:
                    try:
                        logger.debug(f"Trying method {desc}")
                        tmp = router.load(dataset_name, kwargs, preferred_sources=strict_sources)
                        if tmp is not None and (not hasattr(tmp, "__len__") or len(tmp) > 0):
                            ds = tmp
                            logger.debug(f"Successfully loaded with method {desc}")
                            break
                        last_err = router.last_error or last_err
                    except Exception as e:
                        last_err = str(e)
                        logger.debug(f"Method {desc} failed: {str(e)}")
                        continue

                if ds is None:
                    logger.error(f"Failed to load dataset {dataset_name} after all methods. Last error: {last_err}")
                    state.last_error = last_err

                    # Wrap the string in a generic exception only if it's not
                    # empty/None; the retry strategy now classifies fatal
                    # import/version errors and will refuse to retry them.
                    err_for_retry: Optional[Exception] = None
                    if last_err:
                        err_for_retry = Exception(last_err)

                    if attempt < max_retries - 1 and retry_strategy.should_retry(attempt, err_for_retry):
                        delay = retry_strategy.get_delay(attempt)
                        logger.info(f"Retrying {dataset_name} in {delay:.1f} seconds...")
                        resume_manager.save_state(state)
                        time.sleep(delay)
                        continue
                    state.status = "failed"
                    state.end_time = time.time()
                    resume_manager.save_state(state)
                    return None

                saved_path = PiscesLxToolsDataDatasetDownload.save_dataset(ds, data_dir, save_name, max_samples)
                if saved_path:
                    logger.info(f"Successfully saved dataset {dataset_name} -> {save_name}")
                    state.status = "completed"
                    state.end_time = time.time()
                    resume_manager.save_state(state)
                    return save_name
                else:
                    logger.error(f"Failed to save dataset {dataset_name} to {save_name}")
                    state.last_error = "Save failed"
                    state.status = "failed"
                    state.end_time = time.time()
                    resume_manager.save_state(state)
                    return None

            except Exception as e:
                logger.error(f"Exception in download_worker for {dataset_name}: {str(e)}")
                state.last_error = str(e)
                state.end_time = time.time()

                if attempt < max_retries - 1 and retry_strategy.should_retry(attempt, e):
                    delay = retry_strategy.get_delay(attempt)
                    logger.info(f"Retrying after exception for {dataset_name} in {delay:.1f} seconds...")
                    resume_manager.save_state(state)
                    time.sleep(delay)
                    continue

                state.status = "failed"
                resume_manager.save_state(state)
                return None

        state.status = "failed"
        state.end_time = time.time()
        resume_manager.save_state(state)
        return None

    def __init__(self) -> None:
        try:
            import logging
            logging.getLogger("modelscope").setLevel(logging.ERROR)
        except Exception:
            pass

        self._cache = PiscesLxToolsDataDownloadCache()
        self._cache.setup_env()
        self._router = PiscesLxToolsDataSourceRouter()
        self._DATA = self._cache.get_data_dir()
        self._DATATEMP = self._cache.get_temp_dir()

        state_dir = os.path.join(self._DATATEMP, "download_states")
        self._resume_manager = PiscesLxDataResumeManager(state_dir)

        self._retry_config = {
            "max_retries": 5,
            "base_delay": 5.0,
            "max_delay": 300.0
        }

    def download(self, config_path: str | int = "configs/dataset.yaml", max_samples_per_dataset: Optional[int] = None):
        cfg = self._load_config(config_path, max_samples_per_dataset)
        self._run_download(cfg)

    def optimize(self, max_keep=None):
        for entry in os.listdir(self._DATA):
            raw_dir = os.path.join(self._DATA, entry)
            if not os.path.isdir(raw_dir):
                continue
            try:
                ds = load_from_disk(raw_dir)
                original_len = len(ds)
                if original_len == 0:
                    continue

                df = ds.to_pandas()
                text_field = None
                from tools.data.__init__ import TEXT_FIELD_KEYS
                for field in TEXT_FIELD_KEYS:
                    if field in df.columns:
                        text_field = field
                        break
                if not text_field:
                    string_cols = df.select_dtypes(include=["object"]).columns
                    if len(string_cols) > 0:
                        text_field = string_cols[0]
                    else:
                        continue

                import re
                def clean_text_simple(text):
                    if not isinstance(text, str):
                        return ""
                    text = str(text).strip()
                    if not text:
                        return ""
                    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text

                df[text_field] = df[text_field].apply(clean_text_simple)
                mask = df[text_field].astype(str).str.strip().str.len() >= 1
                df_cleaned = df[mask]
                if len(df_cleaned) == 0:
                    continue

                new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
                new_ds.save_to_disk(raw_dir)
            except Exception as e:
                continue

    def _load_config(self, config_path: str | int, max_samples_override: Optional[int]) -> PiscesLxToolsDataDownloadConfig:
        if isinstance(config_path, (int, float)) and max_samples_override is None:
            max_samples_override = int(config_path)
            config_path = "configs/dataset.yaml"
        loader = PiscesLxToolsDataConfigLoader(str(config_path))
        cfg = loader.load()
        if isinstance(max_samples_override, int) and max_samples_override > 0:
            cfg.max_samples_per_dataset = max_samples_override
        return cfg

    @staticmethod
    def _norm_sources(srcs: List[str] | None) -> List[str]:
        if not srcs:
            return ["modelscope", "huggingface"]

        norm: List[str] = []
        for s in srcs:
            s_lower = (s or "").strip().lower()
            if s_lower in ("hf", "huggingface", "baobaolian"):
                norm.append("huggingface")
            elif s_lower in ("ms", "modelscope", "motta", "mota"):
                norm.append("modelscope")
            else:
                norm.append(s_lower)

        seen = set()
        out: List[str] = []
        for s in norm:
            if s not in seen:
                seen.add(s)
                out.append(s)

        return out or ["modelscope", "huggingface"]

    def _run_download(self, cfg: PiscesLxToolsDataDownloadConfig):
        logger = PiscesLxLogger("PiscesLx.Tools.Data.Download", file_path=get_log_file("PiscesLx.Tools.Data.Download"), enable_file=True)
        logger.info(f"Starting download with config: {cfg}")
        logger.info(f"Using data directory: {self._DATA}")
        logger.info(f"Using cache directory: {self._DATATEMP}")
        logger.info(f"Datasets will be saved to: {self._DATA} (data_cache directory)")
        logger.info(f"Temporary files will use: {self._DATATEMP} (datatmp directory)")

        downloaded: Set[str] = set()
        for item in cfg.datasets:
            p = os.path.join(self._DATA, item.save)
            if os.path.exists(p):
                state = self._resume_manager.load_state(item.save)
                if state and state.status == "completed":
                    downloaded.add(item.save)
                else:
                    logger.info(f"Found incomplete download for {item.save}, will resume")

        seen_names: Set[str] = set()
        def preferred_sources_for(d: PiscesLxToolsDatasetItem) -> List[str]:
            # Use the dataclass's own normalizer so aliases ("ModelScope",
            # "HF", "baobaolian", ...) are mapped to canonical "modelscope" /
            # "huggingface" and case is ignored. This is what `source` is
            # *supposed* to be matched against.
            return d.normalize_source_preference(cfg.source_preference)

        to_download: List[Tuple[str, str, str, List[str]]] = []
        for d in cfg.datasets:
            if d.save not in downloaded and d.name not in seen_names:
                state = self._resume_manager.load_state(d.save)
                if state and state.status == "completed":
                    logger.info(f"Skipping {d.save} (already completed in data_cache)")
                    continue
                to_download.append((d.name, d.save, d.desc, self._norm_sources(preferred_sources_for(d))))
                seen_names.add(d.name)
            elif d.name in seen_names:
                continue

        max_samples_per_dataset = getattr(cfg, 'max_samples_per_dataset', None)
        total = len(cfg.datasets)
        if not to_download:
            return

        cpu_cores = multiprocessing.cpu_count()
        total_memory_gb = self._get_total_memory_gb()
        max_concurrent = self._calculate_max_concurrent(cpu_cores, total_memory_gb)

        logger.info(f"CPU cores: {cpu_cores}, Total memory: {total_memory_gb:.1f}GB, Max concurrent: {max_concurrent}")

        success_count = 0
        successfully_downloaded: Set[str] = set()

        tasks = []
        for n, s, d, prefs in to_download:
            state = self._resume_manager.load_state(s)
            if state and state.attempt > 0:
                logger.info(f"Resuming {s} from attempt {state.attempt}")

            tasks.append((n, s, d, prefs, self._DATA, max_samples_per_dataset, self._retry_config))

        total_datasets = len(cfg.datasets)
        skipped_datasets = total_datasets - len(to_download)
        if skipped_datasets > 0:
            logger.info(f"Skipping {skipped_datasets} duplicate/already downloaded datasets")
        if len(to_download) == 0:
            logger.info("All datasets already downloaded or skipped")
            return

        model_file = os.path.join(self._DATA, "model.txt")

        def update_model_file(save_name: str):
            try:
                with open(model_file, "a", encoding="utf-8") as f:
                    f.write(f"{save_name}\n")
                logger.info(f"Recorded {save_name} to model.txt")
            except Exception as e:
                logger.debug(f"Failed to update model.txt for {save_name}: {e}")

        logger.info(f"Using multiprocessing with {max_concurrent} concurrent workers")

        successfully_downloaded: Set[str] = set()

        try:
            with multiprocessing.Pool(processes=max_concurrent) as pool:
                for result in pool.imap_unordered(PiscesLxToolsDataDatasetDownload.download_worker, tasks):
                    if result:
                        successfully_downloaded.add(result)
                        update_model_file(result)
                        logger.info(f"Progress: {len(successfully_downloaded)}/{len(tasks)} datasets downloaded")
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}, falling back to sequential")
            for task in tqdm(tasks, total=len(tasks), desc=f"Downloading {len(tasks)} datasets"):
                result = PiscesLxToolsDataDatasetDownload.download_worker(task)
                if result:
                    successfully_downloaded.add(result)
                    update_model_file(result)

        if getattr(cfg, 'post_download_clean', True) and successfully_downloaded:
            try:
                DatasetCleaner.auto_clean(
                    input_dir=self._DATA,
                    output_dir=self._DATA,
                    min_length=1,
                    text_field=None,
                    workers=None
                )
            except Exception as e:
                try:
                    DatasetCleaner.auto_clean(
                        input_dir=self._DATA,
                        output_dir=self._DATA,
                        min_length=1,
                        text_field=None
                    )
                except Exception as e2:
                    logger.error(f"Exception in unified cleaning: {str(e)} -> {str(e2)}")

            try:
                self._cleanup_caches()
            except Exception as e:
                logger.debug(f"Cache cleanup skipped: {e}")
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if successfully_downloaded:
            try:
                model_file = os.path.join(self._DATA, "model.txt")
                existing = set()
                if os.path.exists(model_file):
                    with open(model_file, "r", encoding="utf-8") as f:
                        existing = set(line.strip() for line in f if line.strip())

                all_names = existing | successfully_downloaded
                with open(model_file, "w", encoding="utf-8") as f:
                    for name in sorted(all_names):
                        f.write(f"{name}\n")
                logger.info(f"model.txt updated with {len(all_names)} datasets ({len(successfully_downloaded)} new)")
            except Exception as e:
                logger.error(f"Exception in updating model.txt: {str(e)}")

    def _get_total_memory_gb(self) -> float:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)

    def _calculate_max_concurrent(self, cpu_cores: int, memory_gb: float) -> int:
        base_workers = max(1, cpu_cores - 1)

        if memory_gb <= 8:
            return 1
        elif memory_gb <= 16:
            return min(base_workers, 2)
        elif memory_gb <= 32:
            return min(base_workers, 3)
        elif memory_gb <= 64:
            return min(base_workers, 4)
        else:
            return min(base_workers, 6)

    def _cleanup_caches(self) -> None:
        logger = PiscesLxLogger("PiscesLx.Tools.Data.Download", file_path=get_log_file("PiscesLx.Tools.Data.Download"), enable_file=True)
        try:
            self._cache.cleanup_cache()
            logger.debug("Temporary caches cleaned")
        except Exception as e:
            logger.debug(f"Cache cleanup failed: {e}")

    def download_single(
        self,
        dataset_name: str,
        save_name: Optional[str] = None,
        source: str = "modelscope",
        max_samples: Optional[int] = None
    ) -> bool:
        save_name = save_name or dataset_name.replace("/", "_")

        save_path = os.path.join(self._DATA, save_name)
        if os.path.exists(save_path):
            _LOG.info(f"Dataset already exists: {save_path}")
            return True

        task = (dataset_name, save_name, "", [source], self._DATA, max_samples, self._retry_config)
        result = self.download_worker(task)

        return result is not None

    def download_multi_source(
        self,
        dataset_name: str,
        sources: List[str],
        save_name: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> bool:
        save_name = save_name or dataset_name.replace("/", "_")

        for source in sources:
            if self.download_single(dataset_name, save_name, source, max_samples):
                return True

        return False
