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

from dataclasses import dataclass
from dataclasses import asdict  # re-export convenience
from pathlib import Path
from typing import Dict, List, Optional, Any
from utils.paths import get_work_dir

# Comprehensive benchmark presets aligned with flagship models (DeepSeek-V3.2, Llama 4, Qwen3.5, etc.)
MODALITY_DATASETS: Dict[str, List[str]] = {
    "text": [
        # Knowledge & Reasoning
        "mmlu", "mmlu_pro", "ceval", "cmmlu", "agieval", "arc", "hellaswag",
        "gpqa", "musr", "bbh", "drop", "winogrande",
        # Math & Logic
        "gsm8k", "math", "aime", "amc", "olympiadbench",
        # Code
        "humaneval", "mbpp", "humaneval_plus", "multipl_e", "ds1000",
        # Commonsense & Social
        "piqa", "siqa", "openbookqa", "commonsenseqa", "strategyqa",
        # Truthfulness & Safety
        "truthfulqa", "toxigen", "bbq",
        # Instruction Following
        "ifeval", "mt_bench", "alpaca_eval", "arena_hard",
    ],
    "image": [
        # General VQA
        "mmbench", "vqav2", "gqa", "ok_vqa", "textvqa",
        # Chart & Document
        "chartqa", "docvqa", "infographicvqa",
        # Science & Knowledge
        "ai2d", "scienceqa", "mmmu", "ocrbench",
        # Fine-grained
        "refcoco", "refcoco_plus", "refcocog",
    ],
    "audio": [
        "librispeech_asr", "common_voice", "fleurs", "gigaspeech",
        "librispeech_asr_clean", "librispeech_asr_other",
    ],
    "video": [
        "mvbench", "activitynet_qa", "videoqa", "msrvtt_qa", "msvd_qa",
    ],
    "doc": [
        "docvqa", "infographicvqa", "chartqa", "deepform", "kleister_nda",
    ],
    "code": [
        "humaneval", "mbpp", "humaneval_plus", "multipl_e", "ds1000",
        "livecodebench", "codecontests", "apps",
    ],
    "math": [
        "gsm8k", "math", "aime", "amc", "olympiadbench",
        "minif2f", "proofnet", "lean_dojo",
    ],
    "safety": [
        "truthfulqa", "toxigen", "bbq", "realtoxicityprompts",
        "adv_glue", "anli", "halueval",
    ],
    "agent": [
        "agentbench", "webshop", "webarena", "osworld", "swebench",
        "toolbench", "apibench", "gsm8k_tool",
    ],
    "long_context": [
        "longbench", "l_eval", "needle_in_haystack", "passkey_retrieval",
        "kv_retrieval", "longbook_summarization", "longbook_qa",
    ],
}


@dataclass
class PiscesLxToolsBenchmarkConfig:
    """Benchmark configuration data class"""
    model_path: str
    model_name: Optional[str] = None
    datasets: List[str] = None
    metrics: List[str] = None
    batch_size: int = 8
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    output_dir: str = ".pisceslx/benchmark"
    use_cache: bool = True
    save_predictions: bool = True
    debug: bool = False

    # New fields for modality and service evaluation
    modality: str = "text"  # one of: text, image, audio, video, doc
    eval_type: str = "LOCAL"  # or "SERVICE"
    api_url: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    eval_batch_size: Optional[int] = None
    timeout: Optional[int] = None  # milliseconds

    def __post_init__(self):
        if self.datasets is None:
            if self.modality in MODALITY_DATASETS:
                self.datasets = MODALITY_DATASETS[self.modality]
            else:
                self.datasets = ["mmlu", "ceval", "gsm8k", "arc", "hellaswag"]
        if self.metrics is None:
            self.metrics = ["accuracy", "f1", "precision", "recall"]
        if self.model_name is None:
            self.model_name = Path(self.model_path).name
