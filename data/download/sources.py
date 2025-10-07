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
from utils import PiscesLxCoreLog
from typing import Any, Dict, List, Optional

_logger = PiscesLxCoreLog("pisceslx.data.download.sources")
# Verbose switch: set PISCESLX_DOWNLOAD_VERBOSE=1 to see detailed debug logs
_VERBOSE = (os.getenv("PISCESLX_DOWNLOAD_VERBOSE", "0") == "1")

class SourceRouter:
    """
    A router class responsible for loading datasets from different sources.
    """
    
    def __init__(self):
        """Initialize the source router. Log initialization if verbose mode is enabled."""
        if _VERBOSE:
            _logger.debug("SourceRouter initialized")
    
    def load(self, dataset_name: str, kwargs: Dict[str, Any] = None, 
             preferred_sources: List[str] = None) -> Optional[Any]:
        """
        Attempt to load a dataset from the specified sources in order of preference.

        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Dict[str, Any], optional): Additional arguments for dataset loading. Defaults to None.
            preferred_sources (List[str], optional): List of preferred sources in order. 
                Defaults to ["modelscope", "huggingface"].

        Returns:
            Optional[Any]: The loaded dataset object if successful, None otherwise.
        """
        if kwargs is None:
            kwargs = {}
        if preferred_sources is None:
            preferred_sources = ["modelscope", "huggingface"]
            
        # Try each source in order of preference
        for source in preferred_sources:
            try:
                if source == "modelscope":
                    return self._load_from_modelscope(dataset_name, kwargs)
                elif source == "huggingface":
                    return self._load_from_huggingface(dataset_name, kwargs)
            except Exception as e:
                if _VERBOSE:
                    _logger.debug(f"Router load error: source={source} dataset={dataset_name} kwargs={kwargs}: {e}")
                continue
                
        return None
    
    def _load_from_modelscope(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Load a dataset from the ModelScope platform.

        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Dict[str, Any]): Additional arguments for dataset loading.

        Returns:
            Optional[Any]: The loaded dataset object if successful, None otherwise.
        """
        try:
            # Import MsDataset from modelscope.msdatasets
            from modelscope.msdatasets import MsDataset  # type: ignore
            return MsDataset.load(dataset_name, **kwargs)
        except Exception as e:
            if _VERBOSE:
                _logger.debug(f"ModelScope load failed for {dataset_name} with kwargs={kwargs}: {e}")
            return None
    
    def _load_from_huggingface(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Load a dataset from the HuggingFace platform.

        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Dict[str, Any]): Additional arguments for dataset loading.

        Returns:
            Optional[Any]: The loaded dataset object if successful, None otherwise.
        """
        try:
            from datasets import load_dataset
            return load_dataset(dataset_name, **kwargs)
        except Exception as e:
            if _VERBOSE:
                _logger.debug(f"HuggingFace load failed for {dataset_name} with kwargs={kwargs}: {e}")
            return None


def detect_available_splits(dataset_name: str, source: str | None = None) -> list[str]:
    """
    Detect available splits for a dataset on a specific source without brute-force attempts.

    Args:
        dataset_name (str): The name of the dataset repository.
        source (str | None, optional): The source platform, either "modelscope" or "huggingface". 
            If None, defaults to "modelscope".

    Returns:
        list[str]: A list of available split names. If empty, the caller should try direct load without a split.
            If only direct load works, returns ["__direct__"].
    """
    # List of common split names to probe
    candidates = [
        "train", "train_full", "train_all",
        "validation", "valid", "dev",
        "test", "eval", "test_all",
    ]

    src = (source or "modelscope").strip().lower()
    available: list[str] = []

    # Try each candidate split
    for split in candidates:
        try:
            if src == "modelscope":
                from modelscope.msdatasets import MsDataset  # type: ignore
                _ = MsDataset.load(dataset_name, split=split, trust_remote_code=True)
            elif src == "huggingface":
                from datasets import load_dataset  # type: ignore
                _ = load_dataset(dataset_name, split=split, trust_remote_code=True)
            else:
                continue
            available.append(split)
        except Exception as e:
            if _VERBOSE:
                _logger.debug(f"Split probe failed: source={src} dataset={dataset_name} split={split}: {e}")
            continue

    if not available:
        # Try to load the dataset directly without specifying a split
        try:
            if src == "modelscope":
                from modelscope.msdatasets import MsDataset  # type: ignore
                _ = MsDataset.load(dataset_name, trust_remote_code=True)
            elif src == "huggingface":
                from datasets import load_dataset  # type: ignore
                _ = load_dataset(dataset_name, trust_remote_code=True)
            available.append("__direct__")
        except Exception as e:
            # No available splits or direct load
            if _VERBOSE:
                _logger.debug(f"Direct probe failed: source={src} dataset={dataset_name}: {e}")

    return available

def to_hf_if_needed(ds: Any) -> Any:
    """
    Convert a dataset to HuggingFace format if necessary.

    - If the dataset is already a HuggingFace dataset (has save_to_disk method), return it as-is.
    - If the dataset is a ModelScope dataset with to_hf_dataset method, convert it and return.
    - Otherwise, return the original object.

    Args:
        ds (Any): The dataset object to potentially convert.

    Returns:
        Any: The original or converted dataset object.
    """
    try:
        # Check if it's already a HuggingFace dataset
        if hasattr(ds, "save_to_disk"):
            return ds
        # Try to convert if it's a ModelScope dataset
        if hasattr(ds, "to_hf_dataset"):
            try:
                return ds.to_hf_dataset()  # type: ignore[attr-defined]
            except Exception as e:
                if _VERBOSE:
                    _logger.debug(f"to_hf_if_needed conversion failed: {e}")
    except Exception as e:
        if _VERBOSE:
            _logger.debug(f"to_hf_if_needed conversion failed: {e}")
    return ds