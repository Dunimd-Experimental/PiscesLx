#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025-2026 Wenze Wei. All Rights Reserved.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

import os
import re
import math
import pandas as pd
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Set
from datasets import load_from_disk, Dataset
from dataclasses import dataclass, field
from enum import Enum


class LanguageType(Enum):
    UNKNOWN = "unknown"
    ENGLISH = "english"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    FRENCH = "french"
    GERMAN = "german"
    SPANISH = "spanish"
    RUSSIAN = "russian"
    ARABIC = "arabic"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class PiscesLxDataQualityMetrics:
    text_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    char_diversity: float = 0.0
    word_diversity: float = 0.0
    quality_score: float = 0.0
    perplexity: Optional[float] = None
    language: str = "unknown"
    domain: str = "general"
    readability_score: float = 0.0
    repetition_ratio: float = 0.0
    special_char_ratio: float = 0.0
    digit_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    ngram_overlap_score: float = 0.0


class PiscesLxDataLanguageDetector:
    _language_patterns: Dict[str, re.Pattern] = {}

    _CHAR_RANGES = {
        LanguageType.CHINESE: [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
        LanguageType.JAPANESE: [(0x3040, 0x309F), (0x30A0, 0x30FF)],
        LanguageType.KOREAN: [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
        LanguageType.ARABIC: [(0x0600, 0x06FF), (0x0750, 0x077F)],
        LanguageType.RUSSIAN: [(0x0400, 0x04FF)],
    }

    _CODE_KEYWORDS = {
        "def ", "class ", "import ", "from ", "return ", "function ",
        "const ", "let ", "var ", "public ", "private ", "void ",
        "int ", "string ", "bool ", "if (", "for (", "while (",
        "#include", "namespace ", "package ", "func ", "fn ",
        "async ", "await ", "yield ", "lambda ", "->", "=>",
        "print(", "console.log", "System.out", "printf",
    }

    @classmethod
    def detect(cls, text: str) -> LanguageType:
        if not text or not isinstance(text, str):
            return LanguageType.UNKNOWN

        text = text.strip()
        if not text:
            return LanguageType.UNKNOWN

        code_score = cls._detect_code(text)
        if code_score > 0.3:
            return LanguageType.CODE

        char_counts: Dict[LanguageType, int] = {}
        total_alpha = 0

        for char in text:
            code_point = ord(char)
            for lang_type, ranges in cls._CHAR_RANGES.items():
                for start, end in ranges:
                    if start <= code_point <= end:
                        char_counts[lang_type] = char_counts.get(lang_type, 0) + 1
                        break

            if char.isalpha():
                total_alpha += 1

        if total_alpha == 0:
            return LanguageType.UNKNOWN

        detected_langs = []
        for lang_type, count in char_counts.items():
            ratio = count / max(total_alpha, 1)
            if ratio > 0.1:
                detected_langs.append((lang_type, ratio))

        if len(detected_langs) > 1:
            return LanguageType.MIXED

        if detected_langs:
            return detected_langs[0][0]

        latin_ratio = sum(1 for c in text if c.isalpha() and ord(c) < 128) / max(total_alpha, 1)
        if latin_ratio > 0.8:
            return cls._detect_latin_language(text)

        return LanguageType.ENGLISH

    @classmethod
    def _detect_code(cls, text: str) -> float:
        text_lower = text.lower()
        matches = sum(1 for kw in cls._CODE_KEYWORDS if kw.lower() in text_lower)
        return matches / max(len(cls._CODE_KEYWORDS), 1)

    @classmethod
    def _detect_latin_language(cls, text: str) -> LanguageType:
        text_lower = text.lower()

        french_words = {"le", "la", "les", "de", "du", "des", "et", "est", "en", "que", "qui", "dans", "pour"}
        german_words = {"der", "die", "das", "und", "ist", "von", "zu", "den", "mit", "sich", "auf", "nicht"}
        spanish_words = {"el", "la", "los", "las", "de", "en", "que", "y", "es", "por", "con", "para", "como"}

        words = set(text_lower.split())

        french_overlap = len(words & french_words) / max(len(french_words), 1)
        german_overlap = len(words & german_words) / max(len(german_words), 1)
        spanish_overlap = len(words & spanish_words) / max(len(spanish_words), 1)

        max_overlap = max(french_overlap, german_overlap, spanish_overlap)

        if max_overlap < 0.05:
            return LanguageType.ENGLISH

        if french_overlap == max_overlap:
            return LanguageType.FRENCH
        elif german_overlap == max_overlap:
            return LanguageType.GERMAN
        else:
            return LanguageType.SPANISH


class PiscesLxDataDomainClassifier:
    _DOMAIN_KEYWORDS = {
        "code": {
            "function", "class", "method", "variable", "array", "object",
            "string", "integer", "boolean", "loop", "condition", "algorithm",
            "api", "database", "server", "client", "request", "response",
            "debug", "compile", "runtime", "library", "framework", "module",
            "def ", "import ", "return ", "const ", "let ", "var ",
        },
        "math": {
            "equation", "formula", "calculate", "solve", "theorem", "proof",
            "integral", "derivative", "matrix", "vector", "polynomial",
            "algebra", "geometry", "calculus", "probability", "statistics",
            "function", "variable", "coefficient", "exponential", "logarithm",
        },
        "science": {
            "experiment", "hypothesis", "theory", "research", "study",
            "analysis", "observation", "data", "result", "conclusion",
            "molecule", "atom", "cell", "organism", "species", "evolution",
            "physics", "chemistry", "biology", "experiment", "laboratory",
        },
        "medical": {
            "patient", "treatment", "diagnosis", "symptom", "medicine",
            "doctor", "hospital", "disease", "therapy", "drug", "dosage",
            "clinical", "surgery", "prescription", "health", "medical",
            "vaccine", "virus", "bacteria", "infection", "chronic",
        },
        "finance": {
            "investment", "market", "stock", "trading", "financial",
            "portfolio", "asset", "equity", "bond", "dividend", "interest",
            "bank", "loan", "credit", "debt", "revenue", "profit", "loss",
            "currency", "exchange", "inflation", "gdp", "economic",
        },
        "legal": {
            "law", "court", "judge", "attorney", "lawyer", "contract",
            "lawsuit", "verdict", "defendant", "plaintiff", "legal",
            "rights", "violation", "penalty", "regulation", "compliance",
            "jurisdiction", "statute", "amendment", "constitution",
        },
        "education": {
            "student", "teacher", "school", "university", "education",
            "learning", "curriculum", "exam", "grade", "course", "lesson",
            "homework", "assignment", "lecture", "professor", "academic",
            "degree", "diploma", "scholarship", "enrollment",
        },
        "technology": {
            "software", "hardware", "computer", "internet", "network",
            "digital", "technology", "device", "application", "platform",
            "cloud", "data", "security", "encryption", "algorithm",
            "artificial", "intelligence", "machine", "learning", "automation",
        },
        "news": {
            "breaking", "report", "journalist", "headline", "article",
            "news", "update", "coverage", "broadcast", "media", "press",
            "announced", "confirmed", "according", "sources", "official",
            "statement", "spokesperson", "development", "incident",
        },
        "social": {
            "friend", "family", "relationship", "social", "community",
            "people", "person", "life", "love", "happiness", "emotion",
            "feeling", "experience", "memory", "story", "journey",
            "together", "share", "connect", "support", "care",
        },
    }

    @classmethod
    def classify(cls, text: str) -> Tuple[str, float]:
        if not text or not isinstance(text, str):
            return ("general", 0.0)

        text_lower = text.lower()
        words = set(text_lower.split())

        scores: Dict[str, float] = {}
        for domain, keywords in cls._DOMAIN_KEYWORDS.items():
            overlap = len(words & keywords)
            coverage = overlap / max(len(keywords), 1)
            density = overlap / max(len(words), 1)
            scores[domain] = coverage * 0.6 + density * 0.4

        if not scores:
            return ("general", 0.0)

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score < 0.01:
            return ("general", best_score)

        return (best_domain, best_score)

    @classmethod
    def get_multi_domain_scores(cls, text: str) -> Dict[str, float]:
        if not text or not isinstance(text, str):
            return {}

        text_lower = text.lower()
        words = set(text_lower.split())

        scores: Dict[str, float] = {}
        for domain, keywords in cls._DOMAIN_KEYWORDS.items():
            overlap = len(words & keywords)
            coverage = overlap / max(len(keywords), 1)
            density = overlap / max(len(words), 1)
            scores[domain] = coverage * 0.6 + density * 0.4

        return scores


class PiscesLxDataPerplexityCalculator:
    def __init__(self, model_name: str = "gpt2"):
        self._model = None
        self._tokenizer = None
        self._model_name = model_name
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._model.eval()

            if torch.cuda.is_available():
                self._model = self._model.cuda()

            self._initialized = True
        except Exception:
            self._initialized = False

    def calculate(self, text: str) -> Optional[float]:
        if not text or not isinstance(text):
            return None

        self._lazy_init()

        if not self._initialized or self._model is None:
            return self._estimate_perplexity(text)

        try:
            import torch

            encodings = self._tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                encodings = {k: v.cuda() for k, v in encodings.items()}

            max_length = self._model.config.n_positions
            stride = 512

            nlls = []
            for i in range(0, encodings["input_ids"].size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, encodings["input_ids"].size(1))
                trg_len = end_loc - i

                input_ids = encodings["input_ids"][:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self._model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)

            if nlls:
                ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
                return float(ppl.item())

        except Exception:
            pass

        return self._estimate_perplexity(text)

    def _estimate_perplexity(self, text: str) -> float:
        if not text:
            return float("inf")

        words = text.split()
        if not words:
            return float("inf")

        word_counts = Counter(words)
        total_words = len(words)

        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)

        perplexity = 2 ** entropy

        bigrams = list(zip(words[:-1], words[1:]))
        if bigrams:
            bigram_counts = Counter(bigrams)
            bigram_entropy = 0.0
            for count in bigram_counts.values():
                prob = count / len(bigrams)
                bigram_entropy -= prob * math.log2(prob)

            bigram_ppl = 2 ** bigram_entropy
            perplexity = (perplexity + bigram_ppl) / 2

        return perplexity


class PiscesLxDataReadabilityScorer:
    @staticmethod
    def calculate(text: str, language: str = "english") -> float:
        if not text or not isinstance(text, str):
            return 0.0

        text = text.strip()
        if not text:
            return 0.0

        if language in ["chinese", "japanese", "korean"]:
            return PiscesLxDataReadabilityScorer._calculate_asian_readability(text)

        return PiscesLxDataReadabilityScorer._calculate_english_readability(text)

    @staticmethod
    def _calculate_english_readability(text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        syllables = sum(PiscesLxDataReadabilityScorer._count_syllables(word) for word in words)

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)

        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

        return max(0.0, min(100.0, flesch_score)) / 100.0

    @staticmethod
    def _calculate_asian_readability(text: str) -> float:
        char_count = len(text)
        if char_count == 0:
            return 0.0

        punct_count = len(re.findall(r"[。！？、，；：""''（）【】]", text))

        punct_ratio = punct_count / char_count

        readability = 1.0 - min(punct_ratio * 5, 0.5)

        return max(0.0, min(1.0, readability))

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"

        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        if word.endswith("e"):
            count -= 1

        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(1, count)


class PiscesLxToolsDataQualityController:
    def __init__(
        self,
        quality_threshold: float = 0.7,
        diversity_threshold: float = 0.5,
        min_samples_per_domain: int = 100,
        enable_perplexity: bool = False,
        perplexity_model: str = "gpt2"
    ):
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.min_samples_per_domain = min_samples_per_domain
        self.quality_stats: Dict[str, Any] = {}
        self.domain_weights: Dict[str, float] = {}

        self._language_detector = PiscesLxDataLanguageDetector()
        self._domain_classifier = PiscesLxDataDomainClassifier()
        self._readability_scorer = PiscesLxDataReadabilityScorer()
        self._perplexity_calculator = PiscesLxDataPerplexityCalculator(perplexity_model) if enable_perplexity else None

    @staticmethod
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

            words = re.findall(r"\b\w+\b", text.lower())
            unique_words = len(set(words))
            word_diversity = min(unique_words / len(words), 1.0) if words else 0.0

            sentences = re.split(r"[.!?]+", text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            structure_score = min(len(valid_sentences) / len(sentences), 1.0) if sentences else 0.0

            punct_count = len(re.findall(r"[.!?,:;]", text))
            punct_score = min(punct_count / (len(text) / 100), 1.0)

            word_counts = Counter(words)
            repetition_penalty = 1.0 - min(
                (word_counts.most_common(1)[0][1] / len(words)) if words else 0.0,
                0.5
            )

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

    def calculate_comprehensive_metrics(self, text: str) -> PiscesLxDataQualityMetrics:
        if not text or not isinstance(text, str):
            return PiscesLxDataQualityMetrics()

        text = text.strip()
        if not text:
            return PiscesLxDataQualityMetrics()

        metrics = PiscesLxDataQualityMetrics()

        metrics.text_length = len(text)

        words = re.findall(r"\b\w+\b", text.lower())
        metrics.word_count = len(words)

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        metrics.sentence_count = len(sentences)

        unique_chars = len(set(text.lower()))
        metrics.char_diversity = unique_chars / max(len(text), 1)

        unique_words = len(set(words))
        metrics.word_diversity = unique_words / max(len(words), 1)

        metrics.quality_score = self.calculate_text_quality_score(text)

        language_type = self._language_detector.detect(text)
        metrics.language = language_type.value

        domain, domain_score = self._domain_classifier.classify(text)
        metrics.domain = domain

        metrics.readability_score = self._readability_scorer.calculate(text, metrics.language)

        if words:
            word_counts = Counter(words)
            max_count = word_counts.most_common(1)[0][1] if word_counts else 0
            metrics.repetition_ratio = max_count / len(words)

        special_chars = len(re.findall(r"[^\w\s]", text))
        metrics.special_char_ratio = special_chars / max(len(text), 1)

        digits = len(re.findall(r"\d", text))
        metrics.digit_ratio = digits / max(len(text), 1)

        uppercase = len(re.findall(r"[A-Z]", text))
        alpha_chars = len(re.findall(r"[A-Za-z]", text))
        metrics.uppercase_ratio = uppercase / max(alpha_chars, 1)

        if words:
            metrics.avg_word_length = sum(len(w) for w in words) / len(words)

        if sentences and words:
            metrics.avg_sentence_length = len(words) / len(sentences)

        metrics.ngram_overlap_score = self._calculate_ngram_overlap(text)

        if self._perplexity_calculator:
            metrics.perplexity = self._perplexity_calculator.calculate(text)

        return metrics

    def _calculate_ngram_overlap(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 4:
            return 0.0

        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        if not trigrams:
            return 0.0

        trigram_counts = Counter(trigrams)
        repeated = sum(1 for c in trigram_counts.values() if c > 1)

        return repeated / len(trigrams)

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

            text_field = None
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

            languages: Dict[str, int] = {}
            domains: Dict[str, int] = {}

            for text in series.head(min(1000, total)):
                lang = self._language_detector.detect(str(text)).value
                languages[lang] = languages.get(lang, 0) + 1

                domain, _ = self._domain_classifier.classify(str(text))
                domains[domain] = domains.get(domain, 0) + 1

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
                "language_distribution": {k: v / sum(languages.values()) for k, v in languages.items()},
                "domain_distribution": {k: v / sum(domains.values()) for k, v in domains.items()},
            }

            self.quality_stats[dataset_path] = stats
            return stats
        except Exception as e:
            return {"error": str(e)}

    def get_quality_report(self, dataset_path: str) -> Dict[str, Any]:
        if dataset_path not in self.quality_stats:
            self.analyze_dataset_quality(dataset_path)

        stats = self.quality_stats.get(dataset_path, {})
        if "error" in stats:
            return stats

        report = {
            "dataset_path": dataset_path,
            "overall_quality": stats.get("avg_quality_score", 0),
            "quality_grade": self._get_quality_grade(stats.get("avg_quality_score", 0)),
            "statistics": stats,
            "recommendations": self._generate_recommendations(stats),
        }

        return report

    def _get_quality_grade(self, score: float) -> str:
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Fair)"
        elif score >= 0.5:
            return "D (Poor)"
        else:
            return "F (Very Poor)"

    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        recommendations = []

        avg_quality = stats.get("avg_quality_score", 0)
        if avg_quality < 0.5:
            recommendations.append("Consider filtering low-quality samples (quality_score < 0.5)")

        quality_dist = stats.get("quality_distribution", {})
        low_ratio = quality_dist.get("low", 0)
        if low_ratio > 0.3:
            recommendations.append(f"High ratio of low-quality samples ({low_ratio:.1%}), consider stricter filtering")

        avg_length = stats.get("avg_text_length", 0)
        if avg_length < 100:
            recommendations.append("Average text length is very short, may affect model training quality")

        lang_dist = stats.get("language_distribution", {})
        if len(lang_dist) > 3:
            recommendations.append("Multiple languages detected, consider language-specific processing")

        domain_dist = stats.get("domain_distribution", {})
        if "general" in domain_dist and domain_dist["general"] > 0.5:
            recommendations.append("High proportion of general domain data, consider adding domain-specific datasets")

        return recommendations
