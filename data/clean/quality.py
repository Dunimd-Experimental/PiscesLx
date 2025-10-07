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
import re
from collections import Counter
from typing import Dict, Any, List

import pandas as pd
from datasets import load_from_disk, Dataset


# logs removed


def calculate_text_quality_score(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    text = text.strip()
    if not text:
        return 0.0
    try:
        length_score = min(len(text) / 1000, 1.0)
        unique_chars = len(set(text.lower()))
        char_diversity = min(unique_chars / 26, 1.0)
        words = re.findall(r"\\b\\w+\\b", text.lower())
        unique_words = len(set(words))
        word_diversity = min(unique_words / len(words), 1.0) if words else 0.0
        sentences = re.split(r"[.!?]+", text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        structure_score = min(len(valid_sentences) / len(sentences), 1.0) if sentences else 0.0
        punct_count = len(re.findall(r"[.!?,:;]", text))
        punct_score = min(punct_count / (len(text) / 100), 1.0)
        word_counts = Counter(words)
        repetition_penalty = 1.0 - min((word_counts.most_common(1)[0][1] / len(words)) if words else 0.0, 0.5)
        score = (
            length_score * 0.2 +
            char_diversity * 0.15 +
            word_diversity * 0.25 +
            structure_score * 0.25 +
            punct_score * 0.1 +
            repetition_penalty * 0.05
        )
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.5


class DataQualityController:
    def __init__(self, quality_threshold: float = 0.7, diversity_threshold: float = 0.5, min_samples_per_domain: int = 100):
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.min_samples_per_domain = min_samples_per_domain
        self.quality_stats: Dict[str, Any] = {}
        self.domain_weights: Dict[str, float] = {}

    def analyze_dataset_quality(self, dataset_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(dataset_path):
                return {"error": "Dataset path does not exist"}
            if os.path.isdir(dataset_path):
                dataset = load_from_disk(dataset_path)
                df = dataset.to_pandas()
            elif dataset_path.endswith(".json"):
                df = pd.read_json(dataset_path)
            elif dataset_path.endswith(".jsonl"):
                df = pd.read_json(dataset_path, lines=True)
            elif dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith(".parquet"):
                df = pd.read_parquet(dataset_path)
            else:
                return {"error": "Unsupported file format. Supported: .arrow dir, .json, .jsonl, .csv, .parquet"}

            total = len(df)
            if total == 0:
                return {"error": "Empty dataset"}

            # 文本字段探测
            text_field = None
            # 使用统一的TEXT_FIELD_KEYS进行字段探测
            from .. import TEXT_FIELD_KEYS
            for k in TEXT_FIELD_KEYS:
                if k in df.columns:
                    text_field = k
                    break
            if not text_field:
                string_cols = df.select_dtypes(include=["object"]).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                else:
                    return {"error": "No text field found"}

            series = df[text_field].astype(str)
            lengths = series.str.len()
            qual = series.apply(calculate_text_quality_score)

            domain_keywords = {
                "code": ["function", "class", "def", "import", "return", "{", "}", ";"],
                "math": ["equation", "formula", "calculate", "solve", "math", "algebra"],
                "science": ["experiment", "theory", "research", "study", "analysis"],
                "medical": ["patient", "treatment", "diagnosis", "symptom", "medicine"],
                "finance": ["investment", "market", "stock", "trading", "financial"],
            }
            domain_scores: Dict[str, float] = {}
            for dom, keys in domain_keywords.items():
                matches = series.apply(lambda x: sum(1 for kw in keys if kw.lower() in x.lower()) / len(keys))
                domain_scores[dom] = float(matches.mean())

            high = int((qual >= self.quality_threshold).sum())
            med = int(((qual >= 0.5) & (qual < self.quality_threshold)).sum())
            low = int((qual < 0.5).sum())

            stats = {
                "total_samples": total,
                "text_field": text_field,
                "avg_text_length": float(lengths.mean()),
                "median_text_length": float(lengths.median()),
                "std_text_length": float(lengths.std()),
                "avg_quality_score": float(qual.mean()),
                "median_quality_score": float(qual.median()),
                "quality_score_std": float(qual.std()),
                "quality_distribution": {
                    "high": high / total,
                    "medium": med / total,
                    "low": low / total,
                },
                "domain_scores": domain_scores,
            }
            self.quality_stats[dataset_path] = stats
            return stats
        except Exception as e:
            return {"error": str(e)}
            return {"error": str(e)}