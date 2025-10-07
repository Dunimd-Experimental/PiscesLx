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
import logging
import warnings
from pathlib import Path
from utils import PiscesLxCoreLog, PiscesLxCoreCacheManagerFacade

class DownloadCacheContext:
    """
    Manages download cache and environment configuration.

    Attributes:
        _log (PiscesLxCoreLog): Logger instance for cache-related operations.
        _cache (PiscesLxCoreCacheManagerFacade): Cache manager instance.
        _datatemp_dir (Path): Path to the temporary cache directory.
        _data_dir (Path): Path to the dataset save directory.
        _initialized (bool): Flag indicating whether the environment is set up.
    """
    
    def __init__(self) -> None:
        """Initialize the download cache context.
        
        Sets up the logger, cache manager, and initializes cache directory paths.
        Logs the initialization information.
        """
        self._log = PiscesLxCoreLog("PiscesLx.DataDownload.Cache")
        self._cache = PiscesLxCoreCacheManagerFacade.get_instance()
        
        # Get and ensure consistent cache paths
        self._datatemp_dir = Path(self._cache.get_cache_dir("datatmp"))
        self._data_dir = Path(self._cache.get_cache_dir("data_cache"))
        
        self._initialized = False
        self._log.info(f"Cache context initialized - Temp: {self._datatemp_dir}, Data: {self._data_dir}")

    def setup_env(self) -> None:
        """Set up environment variables for all data sources.
        
        Creates necessary directories if they don't exist.
        Configures cache directories for ModelScope and HuggingFace.
        Sets tokenizer parallelism and reduces third-party log noise.
        """
        if self._initialized:
            return
        
        # Create cache directories if they don't exist
        self._datatemp_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure ModelScope cache environment
        os.environ["MODELSCOPE_CACHE"] = str(self._datatemp_dir)
        os.environ["MODELSCOPE_HUB_CACHE"] = str(self._datatemp_dir)
        os.environ["MODELSCOPE_DATASETS_CACHE"] = str(self._datatemp_dir / "datasets")
        
        # Configure HuggingFace cache environment
        os.environ["HF_DATASETS_CACHE"] = str(self._datatemp_dir / "hf_datasets")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self._datatemp_dir / "hf_hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self._datatemp_dir / "transformers")
        
        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Reduce log level of third-party libraries
        logging.getLogger("modelscope").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        
        # Ignore user warnings from third-party libraries
        warnings.filterwarnings("ignore", category=UserWarning, module="modelscope")
        warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
        
        self._initialized = True
        self._log.success(f"Download cache environment configured successfully")
        self._log.info(f"ModelScope cache: {self._datatemp_dir}")
        self._log.info(f"HuggingFace cache: {self._datatemp_dir}")
        self._log.info(f"Dataset save directory: {self._data_dir}")

    @property
    def MODELSCOPE_CACHE_DIR(self) -> str:
        """Get the ModelScope cache directory path.

        Returns:
            str: Path to the ModelScope cache directory.
        """
        return str(self._datatemp_dir)

    @property
    def HUGGINGFACE_CACHE_DIR(self) -> str:
        """Get the HuggingFace cache directory path.

        Returns:
            str: Path to the HuggingFace cache directory.
        """
        return str(self._datatemp_dir)

    @property
    def DATA_CACHE_DIR(self) -> str:
        """Get the dataset save directory path.

        Returns:
            str: Path to the dataset save directory.
        """
        return str(self._data_dir)

    def get_data_dir(self) -> str:
        """Get the dataset save directory.

        Returns:
            str: Path to the dataset save directory.
        """
        return str(self._data_dir)

    def get_temp_dir(self) -> str:
        """Get the temporary/cache directory.

        Returns:
            str: Path to the temporary/cache directory.
        """
        return str(self._datatemp_dir)

    def get_modelscope_cache_dir(self) -> Path:
        """Get the ModelScope specific cache directory.

        Returns:
            Path: Path to the ModelScope specific cache directory.
        """
        return self._datatemp_dir / "modelscope"

    def get_huggingface_cache_dir(self) -> Path:
        """Get the HuggingFace specific cache directory.

        Returns:
            Path: Path to the HuggingFace specific cache directory.
        """
        return self._datatemp_dir / "huggingface"

    def cleanup_cache(self) -> None:
        """Clean up temporary cache files.
        
        Removes the temporary cache directory and recreates it.
        Logs the success or failure of the operation.
        """
        try:
            import shutil
            if self._datatemp_dir.exists():
                shutil.rmtree(self._datatemp_dir)
                self._datatemp_dir.mkdir(parents=True, exist_ok=True)
                self._log.success(f"Cache directory cleaned: {self._datatemp_dir}")
        except Exception as e:
            self._log.warning(f"Failed to cleanup cache: {e}")

__all__ = ["DownloadCacheContext"]