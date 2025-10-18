#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import os
import logging
import warnings
from pathlib import Path
from utils import PiscesLxCoreCacheManagerFacade

class DownloadCacheContext:
    """
    Manages the download cache and configures the environment for data sources.

    Attributes:
        _cache (PiscesLxCoreCacheManagerFacade): An instance of the cache manager.
        _datatemp_dir (Path): Path to the temporary cache directory.
        _data_dir (Path): Path to the dataset save directory.
        _initialized (bool): A flag indicating whether the environment has been set up.
    """
    
    def __init__(self) -> None:
        """Initialize the download cache context.
        
        Retrieves a cache manager instance and initializes paths for cache directories.
        Sets the initialization flag to False.
        """
        self._cache = PiscesLxCoreCacheManagerFacade.get_instance()
        
        # Retrieve and initialize cache directory paths
        self._datatemp_dir = Path(self._cache.get_cache_dir("datatmp"))
        self._data_dir = Path(self._cache.get_cache_dir("data_cache"))
        
        self._initialized = False

    def setup_env(self) -> None:
        """Set up environment variables for all data sources.
        
        Creates necessary directories if they don't exist, configures cache directories 
        for ModelScope and HuggingFace, disables tokenizer parallelism, and reduces 
        log noise from third - party libraries.
        """
        if self._initialized:
            return
        
        # Create cache directories if they do not exist
        self._datatemp_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure ModelScope cache environment variables
        os.environ["MODELSCOPE_CACHE"] = str(self._datatemp_dir)
        os.environ["MODELSCOPE_HUB_CACHE"] = str(self._datatemp_dir)
        os.environ["MODELSCOPE_DATASETS_CACHE"] = str(self._datatemp_dir / "datasets")
        
        # Configure HuggingFace cache environment variables
        os.environ["HF_DATASETS_CACHE"] = str(self._datatemp_dir / "hf_datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self._datatemp_dir / "hf_hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self._datatemp_dir / "transformers")
        
        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Reduce the log level of third-party libraries
        logging.getLogger("modelscope").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        
        # Ignore user warnings from third-party libraries
        warnings.filterwarnings("ignore", category=UserWarning, module="modelscope")
        warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
        
        self._initialized = True

    @property
    def MODELSCOPE_CACHE_DIR(self) -> str:
        """Get the path to the ModelScope cache directory.

        Returns:
            str: The path to the ModelScope cache directory as a string.
        """
        return str(self._datatemp_dir)

    @property
    def HUGGINGFACE_CACHE_DIR(self) -> str:
        """Get the path to the HuggingFace cache directory.

        Returns:
            str: The path to the HuggingFace cache directory as a string.
        """
        return str(self._datatemp_dir)

    @property
    def DATA_CACHE_DIR(self) -> str:
        """Get the path to the dataset save directory.

        Returns:
            str: The path to the dataset save directory as a string.
        """
        return str(self._data_dir)

    def get_data_dir(self) -> str:
        """Get the path to the dataset save directory.

        Returns:
            str: The path to the dataset save directory as a string.
        """
        return str(self._data_dir)

    def get_temp_dir(self) -> str:
        """Get the path to the temporary/cache directory.

        Returns:
            str: The path to the temporary/cache directory as a string.
        """
        return str(self._datatemp_dir)

    def get_modelscope_cache_dir(self) -> Path:
        """Get the path to the ModelScope specific cache directory.

        Returns:
            Path: The path to the ModelScope specific cache directory.
        """
        return self._datatemp_dir / "modelscope"

    def get_huggingface_cache_dir(self) -> Path:
        """Get the path to the HuggingFace specific cache directory.

        Returns:
            Path: The path to the HuggingFace specific cache directory.
        """
        return self._datatemp_dir / "huggingface"

    def cleanup_cache(self) -> None:
        """Clean up temporary cache files.
        
        Removes the temporary cache directory if it exists and then recreates it.
        """
        try:
            import shutil
            if self._datatemp_dir.exists():
                shutil.rmtree(self._datatemp_dir)
                self._datatemp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

__all__ = ["DownloadCacheContext"]