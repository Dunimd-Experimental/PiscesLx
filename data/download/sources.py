"""
Dataset source router and utilities for PiscesL1 data download module.

This module provides source routing capabilities for downloading datasets from
various sources including ModelScope and HuggingFace.
"""

import os
from typing import Any, Dict, List, Optional


class SourceRouter:
    """
    Router for loading datasets from different sources.
    """
    
    def __init__(self):
        """Initialize the source router."""
        pass
    
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
            except Exception:
                continue
                
        return None
    
    def _load_from_modelscope(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Load dataset from ModelScope."""
        try:
            from modelscope import MsDataset
            return MsDataset.load(dataset_name, **kwargs)
        except Exception:
            return None
    
    def _load_from_huggingface(self, dataset_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            return load_dataset(dataset_name, **kwargs)
        except Exception:
            return None


def to_hf_if_needed(ds: Any) -> Any:
    """
    Convert dataset to HuggingFace format if needed.
    
    Args:
        ds: Input dataset object
        
    Returns:
        Dataset in HuggingFace format
    """
    # If it's already in HuggingFace format, return as-is
    if hasattr(ds, 'save_to_disk'):
        return ds
    
    # Try to convert from ModelScope format
    try:
        from datasets import Dataset
        if hasattr(ds, 'to_hf_dataset'):
            return ds.to_hf_dataset()
    except Exception:
        pass
    
    # Return original if conversion fails
    return ds