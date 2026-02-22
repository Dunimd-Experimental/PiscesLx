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
import re
import pandas as pd
from collections import Counter
from typing import Dict, Any, List
from datasets import load_from_disk, Dataset

class PiscesLxToolsDataQualityController:
    def __init__(self, quality_threshold: float = 0.7, diversity_threshold: float = 0.5, min_samples_per_domain: int = 100):
        """
        Initialize the DataQualityController.

        Args:
            quality_threshold (float, optional): Threshold for high-quality samples. Defaults to 0.7.
            diversity_threshold (float, optional): Threshold for diversity. Defaults to 0.5.
            min_samples_per_domain (int, optional): Minimum samples per domain. Defaults to 100.
        """
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.min_samples_per_domain = min_samples_per_domain
        self.quality_stats: Dict[str, Any] = {}
        self.domain_weights: Dict[str, float] = {}

    @staticmethod
    def calculate_text_quality_score(text: str) -> float:
        """
        Calculate the quality score of a given text based on multiple criteria.

        Args:
            text (str): The input text to calculate the quality score for.

        Returns:
            float: The calculated quality score, ranging from 0.0 to 1.0.
                   Returns 0.0 if the input is invalid or empty, and 0.5 if an exception occurs.
        """
        if not text or not isinstance(text, str):
            return 0.0
        text = text.strip()
        if not text:
            return 0.0
        try:
            # Calculate length score: longer text gets higher score, capped at 1.0
            length_score = min(len(text) / 1000, 1.0)
            # Calculate character diversity score based on unique lowercase characters
            unique_chars = len(set(text.lower()))
            char_diversity = min(unique_chars / 26, 1.0)
            # Bug fix: correct regex pattern from r"\\b\\w+\\b" to r"\b\w+\b"
            words = re.findall(r"\b\w+\b", text.lower())
            unique_words = len(set(words))
            # Calculate word diversity score
            word_diversity = min(unique_words / len(words), 1.0) if words else 0.0
            # Split text into sentences
            sentences = re.split(r"[.!?]+", text)
            # Filter valid sentences with at least 3 words
            valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            # Calculate sentence structure score
            structure_score = min(len(valid_sentences) / len(sentences), 1.0) if sentences else 0.0
            # Count punctuation marks
            punct_count = len(re.findall(r"[.!?,:;]", text))
            # Calculate punctuation score
            punct_score = min(punct_count / (len(text) / 100), 1.0)
            # Count word occurrences
            word_counts = Counter(words)
            # Calculate repetition penalty
            repetition_penalty = 1.0 - min((word_counts.most_common(1)[0][1] / len(words)) if words else 0.0, 0.5)
            # Combine all scores to get the final quality score
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

    def analyze_dataset_quality(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze the quality of a dataset from the given path.

        Args:
            dataset_path (str): Path to the dataset, which can be a directory or a file.

        Returns:
            Dict[str, Any]: A dictionary containing quality statistics of the dataset.
                            Returns an error message if an issue occurs.
        """
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

            # Detect text field
            text_field = None
            # Use unified TEXT_FIELD_KEYS for field detection
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
            qual = series.apply(self.calculate_text_quality_score)

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
