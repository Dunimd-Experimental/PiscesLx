#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import gc
import time
import shutil
from tqdm import tqdm
import multiprocessing
from typing import Any, Tuple
from utils import PiscesLxCoreLog
from .caches import DownloadCacheContext
from datasets import load_from_disk, Dataset
from typing import Optional, Set, List, Tuple
from tools.data.clean import DatasetCleaner
from .config import ConfigLoader, DownloadConfig, DatasetItem
from .sources import SourceRouter, to_hf_if_needed, detect_available_splits

logger = PiscesLxCoreLog("PiscesLx.Tools.DataDownload")

def save_dataset(ds: Any, data_dir: str, name: str, max_samples: Optional[int] = None) -> bool:
    """
    Save a dataset to the specified directory with optional sample limitation.

    Args:
        ds (Any): The dataset to be saved.
        data_dir (str): The directory where the dataset will be saved.
        name (str): The name of the dataset.
        max_samples (Optional[int]): The maximum number of samples to save. If None, all samples will be saved.

    Returns:
        bool: True if the dataset is saved successfully, False otherwise.
    """
    import os
    
    try:
        # Ensure dataset is in HuggingFace format if needed
        try:
            from .sources import to_hf_if_needed
            ds = to_hf_if_needed(ds)
        except Exception:
            logger.debug("Dataset is already in HuggingFace format or conversion failed")
        
        # Apply max_samples limit if specified and valid
        if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
            logger.info(f"Limiting dataset {name} from {len(ds)} to {max_samples} samples")
            ds = ds.select(range(max_samples))
        
        save_path = os.path.join(data_dir, name)
        logger.debug(f"Saving dataset to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ds.save_to_disk(save_path)
        logger.info(f"Successfully saved dataset to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save dataset {name}: {str(e)}")
        return False

def download_worker(task: Tuple[str, str, str, list[str], str, Optional[int]]) -> Optional[str]:
    """
    Module-level worker function used by multiprocessing.Pool to download and save a dataset.

    Args:
        task (Tuple[str, str, str, list[str], str, Optional[int]]): A tuple containing task information, 
            including dataset name, save name, description, preferred sources, data directory, and max samples.

    Returns:
        Optional[str]: The save name of the dataset if downloaded and saved successfully, None otherwise.
    """
    from .sources import SourceRouter, setup_hf_mirror
    from .caches import DownloadCacheContext
    
    dataset_name, save_name, description, preferred_sources, data_dir, max_samples = task
    
    # Set up cache environment variables in the child process
    cache = DownloadCacheContext()
    cache.setup_env()
    
    # Set up HuggingFace mirror if needed in child process
    setup_hf_mirror()
    
    logger = PiscesLxCoreLog("pisceslx.data.download")
    logger.info(f"Starting download: {dataset_name} -> {save_name}")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Downloading {dataset_name} (attempt {attempt + 1}/{max_retries})")
            
            router = SourceRouter()
            # Strictly respect the first preferred source, do not perform cross-hub fallback here
            strict_sources: List[str] = [preferred_sources[0]] if preferred_sources else ["modelscope"]
            src = strict_sources[0].strip().lower()
            # Build loading methods based on detected splits from the chosen source
            splits = detect_available_splits(dataset_name, src)
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
                except Exception as e:
                    last_err = str(e)
                    logger.debug(f"Method {desc} failed: {str(e)}")
                    continue
            
            if ds is None:
                logger.error(f"Failed to load dataset {dataset_name} after all methods")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {dataset_name} in 5 seconds...")
                    import time
                    time.sleep(5)
                    continue
                return None
                
            if save_dataset(ds, data_dir, save_name, max_samples):  # Apply max_samples limit
                logger.info(f"Successfully saved dataset {dataset_name} -> {save_name}")
                return save_name
            else:
                logger.error(f"Failed to save dataset {dataset_name} to {save_name}")
                return None
                
        except Exception as e:
            logger.error(f"Exception in download_worker for {dataset_name}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying after exception for {dataset_name}...")
                import time
                time.sleep(5)
                continue
            return None

class DatasetDownload:
    def __init__(self) -> None:
        """
        Initialize the dataset download tool.
        Set up logging, cache context, source router, and data directories.
        """
        # Reduce third-party log noise via our logging helper
        try:
            from utils.log.core import set_external_logger_level
            set_external_logger_level("modelscope", "ERROR")
        except Exception:
            pass

        self._cache = DownloadCacheContext()
        self._cache.setup_env()
        self._router = SourceRouter()
        self._DATA = self._cache.get_data_dir()
        self._DATATEMP = self._cache.get_temp_dir()

    def download(self, config_path: str | int = "configs/dataset.json", max_samples_per_dataset: Optional[int] = None):
        """
        Download datasets based on the specified configuration.

        Args:
            config_path (str | int): Path to the configuration file or the maximum number of samples. 
                                    If an integer is provided, it is treated as max_samples_per_dataset.
            max_samples_per_dataset (Optional[int]): Maximum number of samples per dataset.
        """
        cfg = self._load_config(config_path, max_samples_per_dataset)
        self._run_download(cfg)

    def optimize(self, max_keep=None):
        """
        Perform in-place cleaning on the downloaded datasets.

        Args:
            max_keep: This parameter is currently unused in the implementation.
        """
        # Iterate over all entries in the data directory
        for entry in os.listdir(self._DATA):
            raw_dir = os.path.join(self._DATA, entry)
            # Skip non-directory entries
            if not os.path.isdir(raw_dir):
                continue
            try:
                ds = load_from_disk(raw_dir)
                original_len = len(ds)
                # Skip empty datasets
                if original_len == 0:
                    continue

                df = ds.to_pandas()
                # Identify the text field
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

                # Define a simple text cleaning function
                import re
                def clean_text_simple(text):
                    """
                    Perform simple text cleaning to remove control characters and extra whitespace.

                    Args:
                        text: Input text to be cleaned.

                    Returns:
                        str: Cleaned text.
                    """
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
                # Skip if no valid data after cleaning
                if len(df_cleaned) == 0:
                    continue

                new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
                new_ds.save_to_disk(raw_dir)
            except Exception as e:
                continue

    def _load_config(self, config_path: str | int, max_samples_override: Optional[int]) -> DownloadConfig:
        """
        Load the download configuration and override the maximum number of samples per dataset if specified.

        Args:
            config_path (str | int): Path to the configuration file or the maximum number of samples.
            max_samples_override (Optional[int]): Maximum number of samples per dataset to override.

        Returns:
            DownloadConfig: Loaded download configuration object.
        """
        if isinstance(config_path, (int, float)) and max_samples_override is None:
            max_samples_override = int(config_path)
            config_path = "configs/dataset.json"
        loader = ConfigLoader(str(config_path))
        cfg = loader.load()
        if isinstance(max_samples_override, int) and max_samples_override > 0:
            cfg.max_samples_per_dataset = max_samples_override
        return cfg

    @staticmethod
    def _norm_sources(srcs: List[str] | None) -> List[str]:
        """
        Normalize source names with support for aliases and de-duplicate while preserving order.

        Args:
            srcs (List[str] | None): List of source names.

        Returns:
            List[str]: Normalized list of source names.
        """
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

    def _run_download(self, cfg: DownloadConfig):
        """
        Execute the dataset download process based on the given configuration.

        Args:
            cfg (DownloadConfig): Download configuration object.
        """
        # Logger for this run
        logger = PiscesLxCoreLog("pisceslx.data.download")

        # Collect the names of already downloaded datasets
        downloaded: Set[str] = set()
        for item in cfg.datasets:
            p = os.path.join(self._DATA, item.save)
            if os.path.exists(p):
                downloaded.add(item.save)

        # Generate download tasks and deduplicate by dataset name
        seen_names: Set[str] = set()
        def preferred_sources_for(d: DatasetItem) -> List[str]:
            if getattr(d, "source", None):
                return [d.source]
            if getattr(d, "source_preference", None):
                return d.source_preference
            return cfg.source_preference

        to_download: List[Tuple[str, str, str, List[str]]] = []
        for d in cfg.datasets:
            if d.save not in downloaded and d.name not in seen_names:
                to_download.append((d.name, d.save, d.desc, self._norm_sources(preferred_sources_for(d))))
                seen_names.add(d.name)
            elif d.name in seen_names:
                continue  # Skip duplicate dataset names
        
        # Store max_samples_per_dataset for worker processes
        max_samples_per_dataset = getattr(cfg, 'max_samples_per_dataset', None)
        total = len(cfg.datasets)
        if not to_download:
            return

        # Download datasets in parallel
        cpu_cores = multiprocessing.cpu_count()
        workers = max(1, cpu_cores - 1) if cpu_cores < 8 else min(cpu_cores, 8)

        success_count = 0
        successfully_downloaded: Set[str] = set()
        # Build picklable tasks: (dataset_name, save_name, desc, preferred_sources, data_dir, max_samples)
        tasks = [(n, s, d, prefs, self._DATA, max_samples_per_dataset) for (n, s, d, prefs) in to_download]

        # Show real download statistics
        total_datasets = len(cfg.datasets)
        skipped_datasets = total_datasets - len(to_download)
        if skipped_datasets > 0:
            logger.info(f"Skipping {skipped_datasets} duplicate/already downloaded datasets")
        if len(to_download) == 0:
            logger.info("All datasets already downloaded or skipped")
            return

        # Run pool with Windows-safe fallback to sequential execution
        try:
            with multiprocessing.Pool(processes=workers) as pool:
                results = list(tqdm(pool.imap_unordered(download_worker, tasks), total=len(tasks), desc=f"Downloading {len(tasks)} unique datasets"))
        except Exception:
            results = []
            for t in tqdm(tasks, total=len(tasks), desc=f"Downloading {len(tasks)} unique datasets (sequential)"):
                results.append(download_worker(t))

        for save_name in results:
            if save_name:
                success_count += 1
                successfully_downloaded.add(save_name)

        # Perform unified cleaning on downloaded datasets
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

            # Clean up caches
            try:
                self._cleanup_caches()
            except Exception as e:
                logger.debug(f"Cache cleanup skipped: {e}")
            gc.collect()

            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Generate model.txt file
            try:
                model_file = os.path.join(self._DATA, "model.txt")
                with open(model_file, "w", encoding="utf-8") as f:
                    for name in sorted(successfully_downloaded):
                        f.write(f"{name}\n")
            except Exception as e:
                logger.error(f"Exception in generating model.txt: {str(e)}")

    def _cleanup_caches(self) -> None:
        """Clean temporary caches after a run."""
        logger = PiscesLxCoreLog("pisceslx.data.download")
        try:
            # Delegate to cache context cleanup
            self._cache.cleanup_cache()
            logger.debug("Temporary caches cleaned")
        except Exception as e:
            # Keep it low-noise; not fatal
            logger.debug(f"Cache cleanup failed: {e}")

