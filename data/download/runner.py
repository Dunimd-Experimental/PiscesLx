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
import gc
import shutil
from tqdm import tqdm
import multiprocessing
from typing import Any, Tuple
from data.clean import DatasetCleaner
from .caches import DownloadCacheContext
from datasets import load_from_disk, Dataset
from typing import Optional, Set, List, Tuple
from .sources import SourceRouter, to_hf_if_needed
from .config import ConfigLoader, DownloadConfig, DatasetItem

def save_dataset(ds: Any, data_dir: str, name: str, max_samples: Optional[int] = None) -> bool:
    import os
    try:
        # Ensure dataset is in HuggingFace format if needed
        try:
            from .sources import to_hf_if_needed
            ds = to_hf_if_needed(ds)
        except Exception:
            pass
        
        # Apply max_samples limit if specified and valid
        if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        
        save_path = os.path.join(data_dir, name)
        pass
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ds.save_to_disk(save_path)
        pass
        return True
    except Exception as e:
        pass
        return False

def download_worker(task: Tuple[str, str, str, list[str], str, Optional[int]]) -> Optional[str]:
    """
    Module-level worker function used by multiprocessing.Pool.
    Accepts only picklable arguments.
    """
    # logs removed
    from .sources import SourceRouter
    dataset_name, save_name, description, preferred_sources, data_dir, max_samples = task
    # logs removed
    pass
    try:
        router = SourceRouter()
        methods = [
            ({}, "direct"),
            ({"split": "train"}, "split=train"),
            ({"split": "validation"}, "split=validation"),
            ({"split": "test"}, "split=test"),
            ({"split": "default"}, "split=default"),
        ]
        last_err: Optional[str] = None
        ds = None
        for kwargs, desc in methods:
            try:
                pass
                tmp = router.load(dataset_name, kwargs, preferred_sources=preferred_sources)
                if tmp is not None and (not hasattr(tmp, "__len__") or len(tmp) > 0):
                    ds = tmp
                    pass
                    break
            except Exception as e:
                last_err = str(e)
                pass
                continue
        if ds is None:
            pass
            return None
        if save_dataset(ds, data_dir, save_name, max_samples):  # Apply max_samples limit
            pass
            return save_name
        else:
            pass
            return None
    except Exception as e:
        pass
        return None

class PiscesLxToolsDatasetDownload:
    def __init__(self) -> None:
        """
        Initialize the dataset download tool.
        Set up logging, cache context, source router, and data directories.
        """
        
        self._cache = DownloadCacheContext()
        self._cache.setup_env()
        self._router = SourceRouter()
        self._DATA = self._cache.get_data_dir()
        self._DATATEMP = self._cache.get_temp_dir()

    def download(self, config_path: str | int = "configs/model.json", max_samples_per_dataset: Optional[int] = None):
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
                pass
                ds = load_from_disk(raw_dir)
                original_len = len(ds)
                # Skip empty datasets
                if original_len == 0:
                    pass
                    continue

                df = ds.to_pandas()
                # Identify the text field
                text_field = None
                from data.__init__ import TEXT_FIELD_KEYS
                for field in TEXT_FIELD_KEYS:
                    if field in df.columns:
                        text_field = field
                        break
                if not text_field:
                    string_cols = df.select_dtypes(include=["object"]).columns
                    if len(string_cols) > 0:
                        text_field = string_cols[0]
                        pass
                    else:
                        pass
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
                    pass
                    continue

                new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
                new_ds.save_to_disk(raw_dir)
                pass
            except Exception as e:
                pass
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
            config_path = "configs/model.json"
        loader = ConfigLoader(str(config_path))
        cfg = loader.load()
        if isinstance(max_samples_override, int) and max_samples_override > 0:
            cfg.max_samples_per_dataset = max_samples_override
        return cfg

    def _run_download(self, cfg: DownloadConfig):
        """
        Execute the dataset download process based on the given configuration.

        Args:
            cfg (DownloadConfig): Download configuration object.
        """
        # Collect the names of already downloaded datasets
        downloaded: Set[str] = set()
        for item in cfg.datasets:
            p = os.path.join(self._DATA, item.save)
            if os.path.exists(p):
                downloaded.add(item.save)

        # Generate download tasks
        def preferred_sources_for(d: DatasetItem) -> List[str]:
            if getattr(d, "source", None):
                return [d.source]
            if getattr(d, "source_preference", None):
                return d.source_preference
            return cfg.source_preference

        to_download: List[Tuple[str, str, str, List[str]]] = [
            (d.name, d.save, d.desc, self._norm_sources(preferred_sources_for(d)))
            for d in cfg.datasets if d.save not in downloaded
        ]
        
        # Store max_samples_per_dataset for worker processes
        max_samples_per_dataset = getattr(cfg, 'max_samples_per_dataset', None)
        total = len(cfg.datasets)
        if not to_download:
            return
            return

        pass
        pass

        # Download datasets in parallel
        cpu_cores = multiprocessing.cpu_count()
        workers = max(1, cpu_cores - 1) if cpu_cores < 8 else min(cpu_cores, 8)

        success_count = 0
        successfully_downloaded: Set[str] = set()
        # Build picklable tasks: (dataset_name, save_name, desc, preferred_sources, data_dir, max_samples)
        tasks = [(n, s, d, prefs, self._DATA, max_samples_per_dataset) for (n, s, d, prefs) in to_download]
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(download_worker, tasks), total=len(tasks), desc="Downloading datasets"))
            for save_name in results:
                if save_name:
                    success_count += 1
                    successfully_downloaded.add(save_name)

        # Perform unified cleaning on downloaded datasets
        if cfg.post_download_clean and successfully_downloaded:
            pass
            try:
                DatasetCleaner.auto_clean(
                    input_dir=self._DATA,
                    output_dir=self._DATA,
                    min_length=1,
                    text_field=None,
                    workers=None
                )
                pass
            except Exception as e:
                pass
                try:
                    DatasetCleaner.auto_clean(
                        input_dir=self._DATA,
                        output_dir=self._DATA,
                        min_length=1,
                        text_field=None
                    )
                    pass
                except Exception as e2:
                    pass

            # Clean up caches
            self._cleanup_caches()
            gc.collect()
            pass

            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    pass
            except Exception:
                pass

            pass

            # Generate model.txt file
            if successfully_downloaded:
                model_file = os.path.join(self._DATA, "model.txt")
                try:
                    with open(model_file, "w", encoding="utf-8") as f:
                        for name in sorted(successfully_downloaded):
                            f.write(f"{name}\n")
                    pass
                except Exception as e:
                    pass

    def _download_worker(self, args: Tuple[str, str, str, List[str]]) -> Optional[str]:
        """
        Worker function responsible for downloading a single dataset.

        Args:
            args (Tuple[str, str, str, List[str]]): A tuple containing dataset name, save name, description, and preferred sources.

        Returns:
            Optional[str]: Save name if the download is successful, None otherwise.
        """
        dataset_name, save_name, description, preferred_sources = args
        # logs removed
        
        try:
            # Try loading the dataset from different sources
            ds = self._load_with_methods(dataset_name, preferred_sources)
            if ds is None:
                # logs removed
                return None
            
            # Save the loaded dataset
            self._save(ds, save_name, description)
            # logs removed
            return save_name
            
        except Exception as e:
            # logs removed
            return None

    def _save(self, ds, name: str, description: str = "") -> bool:
        """
        Save the dataset to the final location in the data directory.

        Args:
            ds: Dataset object to be saved.
            name (str): Name of the dataset.
            description (str): Description of the dataset.

        Returns:
            bool: True if the dataset is saved successfully, False otherwise.
        """
        try:
            save_path = os.path.join(self._DATA, name)
            # logs removed
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the dataset to the specified path
            ds.save_to_disk(save_path)
            # logs removed
            return True
            
        except Exception as e:
            # logs removed
            return False

    def _cleanup_caches(self):
        """
        Clean up temporary cache directories while preserving the final data.
        """
        pass
        
        # List of ModelScope specific cache directories
        modelscope_dirs = [
            os.path.join(self._cache.MODELSCOPE_CACHE_DIR, "datasets"),
            os.path.join(self._cache.MODELSCOPE_CACHE_DIR, "hub"),
            str(self._cache.get_modelscope_cache_dir()),
        ]
        
        # List of HuggingFace specific cache directories
        huggingface_dirs = [
            os.path.join(self._cache.HUGGINGFACE_CACHE_DIR, "hf_datasets"),
            os.path.join(self._cache.HUGGINGFACE_CACHE_DIR, "hf_hub"),
            os.path.join(self._cache.HUGGINGFACE_CACHE_DIR, "transformers"),
            str(self._cache.get_huggingface_cache_dir()),
        ]
        
        # List of general temporary directories (excluding the main datatmp)
        temp_dirs = [
            os.path.join(self._DATA, ".cache"),
            os.path.join(self._DATA, "tmp"),
            os.path.join(self._DATA, "temp"),
            os.path.join(self._DATA, "cache"),
            os.path.join(self._DATA, "downloads"),
            os.path.join(os.path.dirname(__file__), "..", "modelscope"),
        ]
        
        # Combine all directories to be cleaned
        all_cache_dirs = modelscope_dirs + huggingface_dirs + temp_dirs
        
        # Remove duplicate directories
        unique_dirs = []
        seen = set()
        for d in all_cache_dirs:
            if d and d not in seen and os.path.exists(d):
                seen.add(d)
                unique_dirs.append(d)
        
        cleaned_count = 0
        for dir_path in unique_dirs:
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    pass
                    cleaned_count += 1
            except Exception as e:
                pass
        
        pass
        
        # Keep main datatmp and data_cache directories intact
        pass

    @staticmethod
    def _norm_sources(srcs: List[str] | None) -> List[str]:
        """
        Normalize source names with support for various identifiers including Chinese.

        Args:
            srcs (List[str] | None): List of source names.

        Returns:
            List[str]: Normalized list of source names.
        """
        if not srcs:
            return ["modelscope", "huggingface"]
        
        norm = []
        for s in srcs:
            s_lower = (s or "").strip().lower()
            if s_lower in ("hf", "huggingface", "BaoBaoLian"):
                norm.append("huggingface")
            elif s_lower in ("ms", "modelscope", "MoTa"):
                norm.append("modelscope")
        
        # Remove duplicates while preserving order
        seen = set()
        out = []
        for s in norm:
            if s not in seen:
                seen.add(s)
                out.append(s)
        
        return out or ["modelscope", "huggingface"]

    def _load_with_methods(self, dataset_name: str, preferred_sources: List[str]):
        """
        Attempt to load a dataset using multiple methods and splits.

        Args:
            dataset_name (str): Name of the dataset to load.
            preferred_sources (List[str]): List of preferred sources to load the dataset from.

        Returns:
            Dataset: Loaded dataset if successful.

        Raises:
            RuntimeError: If all attempts to load the dataset fail.
        """
        # logs removed
        
        methods = [
            ({}, "direct"),
            ({"split": "train"}, "split=train"),
            ({"split": "validation"}, "split=validation"),
            ({"split": "test"}, "split=test"),
            ({"split": "default"}, "split=default"),
        ]
        
        last_err: Optional[str] = None
        
        for kwargs, method_desc in methods:
            try:
                pass
                ds = self._router.load(dataset_name, kwargs, preferred_sources=preferred_sources)
                
                if ds is not None and len(ds) > 0:
                    pass
                    return ds
                    
            except Exception as e:
                last_err = str(e)
                pass
                continue
        
        # If all methods failed, raise an error with the last error message
        raise RuntimeError(f"Failed to load dataset {dataset_name} from all sources. Last error: {last_err}")
        raise RuntimeError(f"Failed to load dataset {dataset_name} from all sources. Last error: {last_err}")