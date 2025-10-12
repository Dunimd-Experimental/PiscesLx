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
from .rules import StreamCleaner
from .pipeline import DatasetCleaner
from .quality import DataQualityController
from typing import Optional, Dict, Any, Tuple, List
from utils import PiscesLxCoreLog, PiscesLxCoreCacheManagerFacade

_log = PiscesLxCoreLog("PiscesLx.DataClean")

class PiscesLxToolsDatasetClean:
    """
    A unified entry class for PiscesLx dataset cleaning tools.
    It encapsulates the original functional cleaning logic in an object-oriented manner
    and supports fast cleaning, automatic cleaning, single dataset processing, and quality analysis.
    """

    def __init__(self) -> None:
        """
        Initialize the dataset cleaning tool.
        Get the cache manager instance and set the data cache directory.
        """
        self.cache = PiscesLxCoreCacheManagerFacade.get_instance()
        self.data_cache_dir = self.cache.get_cache_dir("data_cache")

    def run(
        self,
        mode: str = "auto",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Unified entry point for dataset cleaning operations.

        Args:
            mode (str, optional): Cleaning mode. Options are "auto", "fast", "one", "analyze". Defaults to "auto".
            input_dir (Optional[str], optional): Input directory path. Defaults to None.
            output_dir (Optional[str], optional): Output directory path. Defaults to None.
            **kwargs (Any): Additional arguments for different cleaning modes.

        Returns:
            Any: The return value varies depending on the cleaning mode.

        Raises:
            ValueError: If the specified mode is not supported.
        """
        mode = (mode or "auto").lower()
        if mode == "auto":
            return self.auto_clean(input_dir=input_dir, output_dir=output_dir, **kwargs)
        elif mode == "fast":
            return self.fast_clean(input_dir=input_dir, output_dir=output_dir, **kwargs)
        elif mode == "one":
            ip = kwargs.get("input_path")
            op = kwargs.get("output_path")
            tf = kwargs.get("text_field", "text")
            return self.process_one(ip, op, text_field=tf, **kwargs)
        elif mode == "analyze":
            ds = kwargs.get("dataset_path")
            return self.analyze(ds)
        else:
            raise ValueError(f"Unsupported clean mode: {mode}")

    def fast_clean(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_len: int = 1,
        max_len: int = 1024,
        workers: Optional[int] = None,
        enable_multiprocessing: bool = True
    ) -> Any:
        """
        Perform fast cleaning and merging of datasets.

        Args:
            input_dir (Optional[str], optional): Input directory path. Defaults to the data cache directory.
            output_dir (Optional[str], optional): Output directory path. Defaults to None.
            min_len (int, optional): Minimum length of data. Defaults to 1.
            max_len (int, optional): Maximum length of data. Defaults to 1024.
            workers (Optional[int], optional): Number of worker processes. Defaults to None.
            enable_multiprocessing (bool, optional): Whether to enable multiprocessing. Defaults to True.

        Returns:
            Any: The result of fast cleaning.
        """
        input_dir = input_dir or self.data_cache_dir
        _log.debug(f"Fast clean start | input_dir={input_dir} output_dir={output_dir}")
        try:
            return DatasetCleaner.fast_clean(
                input_dir=input_dir,
                output_dir=output_dir,
                min_len=min_len,
                max_len=max_len,
                workers=workers,
                enable_multiprocessing=enable_multiprocessing,
            )
        finally:
            _log.success("Fast clean finished")

    def auto_clean(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        media_fields: Optional[Dict[str, Any]] = None,
        workers: Optional[int] = None,
        **clean_kwargs: Any
    ) -> bool:
        """
        Automatically clean all datasets in the specified directory.

        Args:
            input_dir (Optional[str], optional): Input directory path. Defaults to the data cache directory.
            output_dir (Optional[str], optional): Output directory path. Defaults to a subdirectory in the data cache directory.
            media_fields (Optional[Dict[str, Any]], optional): Media fields configuration. Defaults to None.
            workers (Optional[int], optional): Number of worker processes. Defaults to None.
            **clean_kwargs (Any): Additional cleaning arguments.

        Returns:
            bool: True if the cleaning is successful, False otherwise.
        """
        input_dir = input_dir or self.data_cache_dir
        output_dir = output_dir or os.path.join(self.data_cache_dir, "data_clean")
        _log.debug(f"Auto clean start | input_dir={input_dir} output_dir={output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        ok = DatasetCleaner.auto_clean(
            input_dir=input_dir,
            output_dir=output_dir,
            media_fields=media_fields,
            workers=workers,
            **clean_kwargs,
        )
        _log.success("Auto clean finished")
        return ok

    def process_one(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        **clean_kwargs: Any
    ) -> Tuple[int, int]:
        """
        Process a single dataset.

        Args:
            input_path (str): Path to the input dataset.
            output_path (str): Path to the output dataset.
            text_field (str, optional): Name of the text field. Defaults to "text".
            **clean_kwargs (Any): Additional cleaning arguments.

        Returns:
            Tuple[int, int]: A tuple containing the number of retained records and the total number of records.

        Raises:
            ValueError: If input_path or output_path is not provided.
        """
        if not input_path or not output_path:
            raise ValueError("process_one requires input_path and output_path")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _log.debug(f"Process dataset | input={input_path} output={output_path} text_field={text_field}")
        retained, total = DatasetCleaner.process_dataset(
            input_path=input_path,
            output_path=output_path,
            text_field=text_field,
            **clean_kwargs,
        )
        _log.success(f"Process finished | retained={retained}/{total}")
        return retained, total

    def analyze(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze the quality of a single dataset.

        Args:
            dataset_path (str): Path to the dataset to be analyzed.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis statistics.

        Raises:
            ValueError: If dataset_path is not provided.
        """
        if not dataset_path:
            raise ValueError("analyze requires dataset_path")
        _log.debug(f"Analyze dataset quality | path={dataset_path}")
        controller = DataQualityController(
            quality_threshold=0.7, diversity_threshold=0.5, min_samples_per_domain=100
        )
        stats = controller.analyze_dataset_quality(dataset_path)
        if "error" in stats:
            _log.error(f"Analyze failed: {stats['error']}")
        else:
            _log.success("Analyze finished")
        return stats