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

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from utils import PiscesLxCoreLog

_log = PiscesLxCoreLog("PiscesLx.DataDownload.Config")

@dataclass
class DatasetItem:
    """
    Represents a configuration item for a dataset, including source preferences.
    """
    name: str  # Name of the dataset
    save: str  # Save path of the dataset
    desc: str  # Description of the dataset
    source: Optional[str] = None  # Specify a specific source for the dataset
    source_preference: Optional[List[str]] = None  # Ordered list of preferred sources

    def normalize_source_preference(self, default_preference: List[str]) -> List[str]:
        """
        Normalize the source preference list, converting aliases for "HuggingFace" and "ModelScope" to standard names.
        Remove duplicates while preserving the order of sources.

        Args:
            default_preference (List[str]): The default source preference list to return if no valid preference is provided.

        Returns:
            List[str]: A normalized and deduplicated source preference list.
        """
        # If a specific source is specified, normalize it
        if self.source:
            source_lower = self.source.strip().lower()
            if source_lower in ("hf", "huggingface", "baobaolian"):
                return ["huggingface"]
            elif source_lower in ("ms", "modelscope", "motta"):
                return ["modelscope"]
            else:
                return [source_lower]
        
        # If a custom preference list is provided, normalize it
        if self.source_preference:
            normalized = []
            for src in self.source_preference:
                src_lower = src.strip().lower()
                if src_lower in ("hf", "huggingface", "baobaolian"):
                    normalized.append("huggingface")
                elif src_lower in ("ms", "modelscope", "motta"):
                    normalized.append("modelscope")
                else:
                    normalized.append(src_lower)
            
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for src in normalized:
                if src not in seen:
                    seen.add(src)
                    unique.append(src)
            return unique if unique else default_preference
        
        return default_preference

@dataclass
class DownloadConfig:
    """
    Represents the main configuration for dataset download.
    """
    max_samples_per_dataset: int = 50000  # Maximum number of samples per dataset
    post_download_clean: bool = True  # Whether to perform cleanup after download
    source_preference: List[str] = None  # Default source preference list
    datasets: List[DatasetItem] = None  # List of dataset configuration items
    
    def __post_init__(self):
        """
        Initialize default values for optional attributes after the dataclass is created.
        """
        if self.source_preference is None:
            self.source_preference = ["modelscope", "huggingface"]
        if self.datasets is None:
            self.datasets = []

class ConfigLoader:
    """
    Loads and validates the dataset download configuration from a JSON file.
    """
    
    def __init__(self, path: str = "configs/model.json") -> None:
        """
        Initialize the configuration loader.

        Args:
            path (str, optional): Path to the configuration JSON file. Defaults to "configs/model.json".

        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")

    def load(self) -> DownloadConfig:
        """
        Load and validate the download configuration from the JSON file.

        Returns:
            DownloadConfig: A configured DownloadConfig object.

        Raises:
            ValueError: If the JSON content in the configuration file is invalid.
            RuntimeError: If there is an error loading the configuration file.
        """
        _log.info(f"Loading download configuration from: {self.path}")
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {self.path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file {self.path}: {e}")

        defaults = raw.get("defaults", {})
        datasets_raw = raw.get("datasets", [])
        
        if not datasets_raw:
            _log.warning("No datasets found in configuration")
        
        items: List[DatasetItem] = []
        for i, d in enumerate(datasets_raw):
            try:
                item = DatasetItem(
                    name=d["name"],
                    save=d["save"],
                    desc=d.get("desc", d["save"]),
                    source=d.get("source"),
                    source_preference=d.get("source_preference"),
                )
                items.append(item)
                _log.debug(f"Loaded dataset config {i+1}: {item.name} -> {item.save}")
            except KeyError as e:
                _log.error(f"Invalid dataset configuration at index {i}: missing {e}")
                continue
            except Exception as e:
                _log.error(f"Error loading dataset config at index {i}: {e}")
                continue

        # Normalize default source preference
        default_source_pref = defaults.get("source_preference", ["modelscope", "huggingface"])
        normalized_default_pref = []
        for src in default_source_pref:
            src_lower = src.strip().lower()
            if src_lower in ("hf", "huggingface", "baobaolian"):
                normalized_default_pref.append("huggingface")
            elif src_lower in ("ms", "modelscope", "motta"):
                normalized_default_pref.append("modelscope")
            else:
                normalized_default_pref.append(src_lower)

        cfg = DownloadConfig(
            max_samples_per_dataset=defaults.get("max_samples_per_dataset", 50000),
            post_download_clean=defaults.get("post_download_clean", True),
            source_preference=normalized_default_pref,
            datasets=items,
        )
        
        _log.success(f"Successfully loaded download configuration with {len(items)} datasets")
        _log.info(f"Default source preference: {cfg.source_preference}")
        
        return cfg

__all__ = ["DatasetItem", "DownloadConfig", "ConfigLoader"]