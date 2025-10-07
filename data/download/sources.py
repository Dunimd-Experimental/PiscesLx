"""
Dataset source router and utilities for PiscesL1 data download module.

This module provides source routing capabilities for downloading datasets from
various sources including ModelScope and HuggingFace.
"""

import os
from typing import Any, Dict, List, Optional
from utils import PiscesLxCoreLog

# module-level logger for this source module
_logger = PiscesLxCoreLog("pisceslx.data.download.sources")


class SourceRouter:
    """
    Router for loading datasets from different sources.
    """
    
    def __init__(self):
        """Initialize the source router."""
        _logger.debug("SourceRouter initialized")
    
    def load(self, dataset_name: str, kwargs: Dict[str, Any] = None, 
             preferred_sources: List[str] = None) -> Optional[Any]:
        """
        Load a dataset from the specified sources.
        
        Args:
            dataset_name: Name of the dataset to load
            kwargs: Additional arguments for dataset loading
            preferred_sources: List of preferred sources in order
            
        Returns:
            Loaded dataset object or None if failed
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
                _logger.debug(f"Router load error: source={source} dataset={dataset_name} kwargs={kwargs}: {e}")
                continue
                
        return None
    
    def _load_from_modelscope(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Load dataset from ModelScope."""
        try:
            # Correct import path for MsDataset
            from modelscope.msdatasets import MsDataset  # type: ignore
            return MsDataset.load(dataset_name, **kwargs)
        except Exception as e:
            _logger.debug(f"ModelScope load failed for {dataset_name} with kwargs={kwargs}: {e}")
            return None
    
    def _load_from_huggingface(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            return load_dataset(dataset_name, **kwargs)
        except Exception as e:
            _logger.debug(f"HuggingFace load failed for {dataset_name} with kwargs={kwargs}: {e}")
            return None


def detect_available_splits(dataset_name: str, source: str | None = None) -> list[str]:
    """
    Detect available splits for a dataset on a specific source without brute-force spam.

    Args:
        dataset_name: The dataset repository name (as configured).
        source: "modelscope" or "huggingface". If None, defaults to "modelscope".

    Returns:
        A list of available split names. If empty, caller should try direct load without split.
        If only direct load works, returns ["__direct__"].
    """
    candidates = [
        "train", "train_full", "train_all",
        "validation", "valid", "dev",
        "test", "eval", "test_all",
    ]

    src = (source or "modelscope").strip().lower()
    available: list[str] = []

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
            _logger.debug(f"Split probe failed: source={src} dataset={dataset_name} split={split}: {e}")
            continue

    if not available:
        # Probe direct/no-split
        try:
            if src == "modelscope":
                from modelscope.msdatasets import MsDataset  # type: ignore
                _ = MsDataset.load(dataset_name, trust_remote_code=True)
            elif src == "huggingface":
                from datasets import load_dataset  # type: ignore
                _ = load_dataset(dataset_name, trust_remote_code=True)
            available.append("__direct__")
        except Exception as e:
            # none available
            _logger.debug(f"Direct probe failed: source={src} dataset={dataset_name}: {e}")

    return available

def to_hf_if_needed(ds: Any) -> Any:
    """
    Convert dataset to HuggingFace format if needed.

    - If it's already an HF dataset (has save_to_disk), return as-is.
    - If it's a ModelScope dataset exposing to_hf_dataset(), convert and return.
    - Otherwise, return original object.
    """
    try:
        if hasattr(ds, "save_to_disk"):
            return ds
        # ModelScope dataset conversion helper
        if hasattr(ds, "to_hf_dataset"):
            try:
                return ds.to_hf_dataset()  # type: ignore[attr-defined]
            except Exception as e:
                _logger.debug(f"to_hf_if_needed conversion failed: {e}")
    except Exception as e:
        _logger.debug(f"to_hf_if_needed conversion failed: {e}")
    return ds