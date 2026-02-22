#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
from typing import Dict, List, Any, Optional

from evalscope.models import ModelConfig
from evalscope.metrics import MetricConfig
from evalscope.datasets import DatasetConfig
from evalscope import TaskConfig
from evalscope.constants import EvalType

from .config import PiscesLxToolsBenchmarkConfig


class PiscesLxToolsModelConfigBuilder:
    """Builder for model configurations"""

    @staticmethod
    def build(config: PiscesLxToolsBenchmarkConfig) -> ModelConfig:
        return ModelConfig(
            model_id=config.model_path,
            model_name=config.model_name,
            device=config.device,
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
            batch_size=config.batch_size,
            use_cache=config.use_cache,
        )


class PiscesLxToolsDatasetConfigBuilder:
    """Builder for dataset configurations"""

    @staticmethod
    def build(config: PiscesLxToolsBenchmarkConfig) -> List[DatasetConfig]:
        dataset_configs: List[DatasetConfig] = []
        for dataset_name in config.datasets:
            dataset_config = DatasetConfig(
                dataset_id=dataset_name,
                subset="default",
                split="test",
                limit=None,
                cache_dir=os.path.join(config.output_dir, "cache", dataset_name),
            )
            dataset_configs.append(dataset_config)
        return dataset_configs


class PiscesLxToolsMetricConfigBuilder:
    """Builder for metric configurations"""

    @staticmethod
    def build(config: PiscesLxToolsBenchmarkConfig) -> List[MetricConfig]:
        metric_configs: List[MetricConfig] = []
        for metric_name in config.metrics:
            metric_configs.append(MetricConfig(metric_id=metric_name, params={}))
        return metric_configs


class PiscesLxToolsTaskConfigBuilder:
    """Builder for task configurations"""

    @staticmethod
    def build(config: PiscesLxToolsBenchmarkConfig) -> TaskConfig:
        model_config = PiscesLxToolsModelConfigBuilder.build(config)
        dataset_configs = PiscesLxToolsDatasetConfigBuilder.build(config)
        metric_configs = PiscesLxToolsMetricConfigBuilder.build(config)

        optional_kwargs: Dict[str, Any] = {}
        if config.eval_type is not None:
            if hasattr(EvalType, config.eval_type.upper()):
                optional_kwargs["eval_type"] = getattr(EvalType, config.eval_type.upper())
            else:
                optional_kwargs["eval_type"] = config.eval_type
        if config.api_url:
            optional_kwargs["api_url"] = config.api_url
        if config.generation_config is not None:
            optional_kwargs["generation_config"] = config.generation_config
        if config.eval_batch_size is not None:
            optional_kwargs["eval_batch_size"] = config.eval_batch_size
        if config.timeout is not None:
            optional_kwargs["timeout"] = config.timeout

        return TaskConfig(
            model_config=model_config,
            dataset_configs=dataset_configs,
            metric_configs=metric_configs,
            output_dir=config.output_dir,
            save_predictions=config.save_predictions,
            debug=config.debug,
            **optional_kwargs,
        )
