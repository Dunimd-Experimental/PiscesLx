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

from utils import PiscesLxCoreLog
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class DataSource(ABC):
    """Abstract base class defining the interface for all dataset sources."""
    
    @abstractmethod
    def load(self, dataset_name: str, kwargs: Dict[str, Any] | None = None) -> Any:
        """
        Load a dataset from the source.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Optional[Dict[str, Any]]): Additional arguments for dataset loading.
            
        Returns:
            Any: The loaded dataset object.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_source_name(self) -> str:
        """
        Get the name of this data source.
        
        Returns:
            str: The name of the data source.
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

class ModelScopeSource(DataSource):
    """Data source implementation for loading datasets from ModelScope."""
    
    def __init__(self) -> None:
        """Initialize the ModelScopeSource instance."""
        self._log = PiscesLxCoreLog("PiscesLx.DataDownload.Source.ModelScope")
        self._source_name = "modelscope"

    def get_source_name(self) -> str:
        """
        Get the name of this data source.
        
        Returns:
            str: The name of the data source, which is 'modelscope'.
        """
        return self._source_name

    def load(self, dataset_name: str, kwargs: Dict[str, Any] | None = None) -> Any:
        """
        Load a dataset from ModelScope.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Optional[Dict[str, Any]]): Additional arguments for dataset loading.
            
        Returns:
            Any: The loaded dataset object from ModelScope.
            
        Raises:
            RuntimeError: If the ModelScope library cannot be imported, 
                         the dataset is empty, or loading fails.
        """
        kwargs = kwargs or {}
        try:
            # Import MsDataset on demand
            from modelscope.msdatasets import MsDataset
            self._log.info(f"Loading dataset from ModelScope: {dataset_name}")
            
            ds = MsDataset.load(dataset_name, **kwargs)
            
            if ds is None or (hasattr(ds, "__len__") and len(ds) == 0):
                raise RuntimeError(f"ModelScope returned an empty dataset for {dataset_name}")
            
            self._log.success(f"Successfully loaded dataset from ModelScope: {dataset_name} ({len(ds) if hasattr(ds, '__len__') else 'unknown'} samples)")
            return ds
            
        except ImportError as e:
            self._log.error(f"ModelScope library is not available: {e}")
            raise RuntimeError(f"Failed to import the ModelScope library: {e}")
        except Exception as e:
            self._log.error(f"Failed to load dataset from ModelScope {dataset_name}: {e}")
            raise RuntimeError(f"ModelScope dataset loading failed for {dataset_name}: {e}")

class HuggingFaceSource(DataSource):
    """Data source implementation for loading datasets from HuggingFace."""
    
    def __init__(self) -> None:
        """Initialize the HuggingFaceSource instance."""
        self._log = PiscesLxCoreLog("PiscesLx.DataDownload.Source.HuggingFace")
        self._source_name = "huggingface"

    def get_source_name(self) -> str:
        """
        Get the name of this data source.
        
        Returns:
            str: The name of the data source, which is 'huggingface'.
        """
        return self._source_name

    def load(self, dataset_name: str, kwargs: Dict[str, Any] | None = None) -> Any:
        """
        Load a dataset from HuggingFace.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Optional[Dict[str, Any]]): Additional arguments for dataset loading.
            
        Returns:
            Any: The loaded dataset object from HuggingFace.
            
        Raises:
            RuntimeError: If the HuggingFace datasets library cannot be imported, 
                         the dataset is empty, or loading fails.
        """
        kwargs = kwargs or {}
        try:
            # Import load_dataset on demand
            from datasets import load_dataset
            self._log.info(f"Loading dataset from HuggingFace: {dataset_name}")
            
            split = kwargs.get("split")
            if split and split != "default":
                ds = load_dataset(dataset_name, split=split)
            else:
                ds = load_dataset(dataset_name)
            
            if ds is None or (hasattr(ds, "__len__") and len(ds) == 0):
                raise RuntimeError(f"HuggingFace returned an empty dataset for {dataset_name}")
            
            self._log.success(f"Successfully loaded dataset from HuggingFace: {dataset_name} ({len(ds) if hasattr(ds, '__len__') else 'unknown'} samples)")
            return ds
            
        except ImportError as e:
            self._log.error(f"HuggingFace datasets library is not available: {e}")
            raise RuntimeError(f"Failed to import the HuggingFace datasets library: {e}")
        except Exception as e:
            self._log.error(f"Failed to load dataset from HuggingFace {dataset_name}: {e}")
            raise RuntimeError(f"HuggingFace dataset loading failed for {dataset_name}: {e}")

class SourceRouter:
    """
    An intelligent router for dataset sources.
    Routes dataset loading requests to appropriate sources based on preferences and availability.
    
    Supported source identifiers:
      - "modelscope", "ms", "Magic Tower"
      - "huggingface", "hf", "Hugging Face"
    """
    
    def __init__(self) -> None:
        """Initialize the SourceRouter instance."""
        self._log = PiscesLxCoreLog("PiscesLx.DataDownload.SourceRouter")
        self._ms = ModelScopeSource()
        self._hf = HuggingFaceSource()
        self._sources = {
            "modelscope": self._ms,
            "huggingface": self._hf,
        }

    @staticmethod
    def _norm_sources(srcs: List[str] | None) -> List[str]:
        """
        Normalize source names to standard format.
        
        Args:
            srcs (Optional[List[str]]): List of source names to normalize.
            
        Returns:
            List[str]: List of normalized source names. If input is None or empty, 
                      returns ["modelscope", "huggingface"].
        """
        if not srcs:
            return ["modelscope", "huggingface"]
        
        norm = []
        for s in srcs:
            s_lower = (s or "").strip().lower()
            if s_lower in ("hf", "huggingface", "Hugging Face"):
                norm.append("huggingface")
            elif s_lower in ("ms", "modelscope", "Magic Tower"):
                norm.append("modelscope")
        
        # Remove duplicates while preserving order
        seen = set()
        out = []
        for s in norm:
            if s not in seen:
                seen.add(s)
                out.append(s)
        
        return out or ["modelscope", "huggingface"]

    def load(self, dataset_name: str, kwargs: Dict[str, Any] | None = None, preferred_sources: List[str] | None = None) -> Any:
        """
        Load a dataset with intelligent source routing.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            kwargs (Optional[Dict[str, Any]]): Additional arguments for dataset loading.
            preferred_sources (Optional[List[str]]): List of preferred sources in order of preference.
            
        Returns:
            Any: The loaded dataset object.
            
        Raises:
            RuntimeError: If all sources fail to load the dataset.
        """
        order = self._norm_sources(preferred_sources)
        last_err: Optional[str] = None
        
        self._log.info(f"Attempting to load dataset '{dataset_name}' with preferred sources: {order}")
        
        for src_name in order:
            try:
                source = self._sources.get(src_name)
                if source:
                    self._log.info(f"Trying source: {src_name}")
                    result = source.load(dataset_name, kwargs)
                    self._log.success(f"Successfully loaded dataset '{dataset_name}' from {src_name}")
                    return result
                else:
                    self._log.warning(f"Unknown source: {src_name}")
                    
            except Exception as e:
                last_err = str(e)
                self._log.warning(f"Source {src_name} failed for dataset '{dataset_name}': {e}")
                continue
        
        # All sources failed
        error_msg = f"Failed to load dataset '{dataset_name}' from all sources. Last error: {last_err}"
        self._log.error(error_msg)
        raise RuntimeError(error_msg)

def to_hf_if_needed(ds: Any) -> Any:
    """
    Convert various dataset types to HuggingFace Dataset format if necessary.
    
    Args:
        ds (Any): Input dataset object.
        
    Returns:
        Any: HuggingFace Dataset object or the original object if already in the correct format.
    """
    if hasattr(ds, "to_hf_dataset"):
        return ds.to_hf_dataset()
    if hasattr(ds, "data") and hasattr(ds, "info"):
        return ds
    if hasattr(ds, "__iter__") and not hasattr(ds, "save_to_disk"):
        try:
            from datasets import Dataset
            if hasattr(ds, "to_pandas"):
                return Dataset.from_pandas(ds.to_pandas())
        except Exception:
            pass
    return ds

__all__ = ["DataSource", "ModelScopeSource", "HuggingFaceSource", "SourceRouter", "to_hf_if_needed"]