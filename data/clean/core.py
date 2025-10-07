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
from typing import Optional, Dict, Any, Tuple, List

from utils import PiscesLxCoreLog, PiscesLxCoreCacheManagerFacade

# 使用全新 OOP 模块实现，彻底移除 legacy 依赖
from .pipeline import DatasetCleaner
from .quality import DataQualityController
from .rules import StreamCleaner

_log = PiscesLxCoreLog("PiscesLx.DataClean")


class PiscesLxToolsDatasetClean:
    """
    PiscesLx 数据清理工具（对外唯一入口类）
    - 面向对象封装原有函数式清理逻辑
    - 支持快速清理、自动清理、单数据集处理与质量分析
    - 使用统一日志系统，不做环境探测
    """

    def __init__(self) -> None:
        self.cache = PiscesLxCoreCacheManagerFacade.get_instance()
        self.data_cache_dir = self.cache.get_cache_dir("data_cache")

    # 统一入口
    def run(
        self,
        mode: str = "auto",
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        统一入口：
        - mode="auto": 自动清理目录下所有数据集（并发）
        - mode="fast": 快速清理并合并
        - mode="one": 处理单个数据集（需提供 input_path/output_path）
        - mode="analyze": 分析单数据集质量（需提供 dataset_path）
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

    # 快速清理并合并
    def fast_clean(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_len: int = 1,
        max_len: int = 1024,
        workers: Optional[int] = None,
        enable_multiprocessing: bool = True
    ) -> Any:
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

    # 自动清理目录下所有数据集
    def auto_clean(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        media_fields: Optional[Dict[str, Any]] = None,
        workers: Optional[int] = None,
        **clean_kwargs: Any
    ) -> bool:
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

    # 单数据集处理
    def process_one(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text",
        **clean_kwargs: Any
    ) -> Tuple[int, int]:
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

    # 数据集质量分析
    def analyze(self, dataset_path: str) -> Dict[str, Any]:
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