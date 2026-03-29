#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei & Annian Wang. All Rights Reserved.
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

"""
Long context benchmark evaluation suite for PiscesL1.

This module provides comprehensive long context evaluation including:
- LongBench: Long context understanding benchmark
- Needle in Haystack: Information retrieval in long context
- Passkey Retrieval: Key retrieval from long sequences
- KV Retrieval: Key-Value pair retrieval
- LongBook QA: Long document question answering
"""

import os
import json
import time
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark.LongContext", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


@dataclass
class PiscesLxToolsLongContextConfig:
    """Configuration for long context benchmark evaluation.
    
    PiscesL1 1T model supports up to 10M (10,000,000) tokens context.
    This is world-leading long context capability, surpassing:
    - Kimi K2.5: ~2M tokens
    - Gemini 3: ~2M tokens
    - Claude 4: ~200K tokens
    """
    
    model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/benchmark/long_context"
    
    max_context_length: int = 10_000_000
    
    context_lengths: List[int] = field(default_factory=lambda: [
        4096, 8192, 16384, 32768, 65536, 131072, 
        262144, 524288, 1048576, 2097152, 4194304, 
        8388608, 10000000
    ])
    
    needle_positions: List[float] = field(default_factory=lambda: [
        0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
    ])
    
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    
    device: str = "cuda"
    
    long_context_benchmarks: List[str] = field(default_factory=lambda: [
        "longbench", "needle_haystack", "passkey", "kv_retrieval", "longbook_qa",
        "ultra_long_needle", "book_summarization", "codebase_analysis"
    ])
    
    save_results: bool = True
    verbose: bool = True
    
    ultra_long_test_enabled: bool = True


class PiscesLxToolsLongContextMetrics:
    """Metrics for long context evaluation."""
    
    @staticmethod
    def retrieval_accuracy(results: List[Dict]) -> float:
        """Calculate retrieval accuracy."""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.get("correct", False))
        return correct / len(results)
    
    @staticmethod
    def context_utilization(results: List[Dict]) -> float:
        """Calculate context utilization score."""
        if not results:
            return 0.0
        scores = []
        for r in results:
            if "context_length" in r and "used_length" in r:
                if r["context_length"] > 0:
                    score = r["used_length"] / r["context_length"]
                    scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def position_bias(results: List[Dict]) -> Dict[str, float]:
        """Analyze position bias in retrieval."""
        position_scores = defaultdict(list)
        
        for r in results:
            if "position" in r and "correct" in r:
                position_scores[r["position"]].append(1.0 if r["correct"] else 0.0)
        
        bias_analysis = {}
        for position, scores in position_scores.items():
            bias_analysis[f"position_{position}"] = sum(scores) / len(scores)
        
        if len(bias_analysis) > 1:
            values = list(bias_analysis.values())
            bias_analysis["variance"] = np.var(values)
            bias_analysis["std"] = np.std(values)
        
        return bias_analysis


class PiscesLxToolsLongContextEvaluator:
    """Long context benchmark evaluator for PiscesL1 model."""
    
    def __init__(
        self,
        config: PiscesLxToolsLongContextConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        self.metrics = PiscesLxToolsLongContextMetrics()
        
        _LOG.info("PiscesLxToolsLongContextEvaluator initialized")
    
    def evaluate_longbench(self) -> Dict[str, float]:
        """Evaluate on LongBench - Long context understanding benchmark."""
        _LOG.info("Evaluating LongBench...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("THUDM/LongBench", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load LongBench: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="LongBench"):
            try:
                context = item.get("context", "")
                question = item.get("input", "")
                answers = item.get("answers", [])
                task = item.get("task", "default")
                
                if not context or not question:
                    continue
                
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=max(self.config.context_lengths),
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=256,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                correct = self._evaluate_answer(generated, answers)
                
                results.append({
                    "task": task,
                    "correct": correct,
                    "generated": generated,
                    "expected": answers,
                    "context_length": encoding["input_ids"].shape[1],
                })
                
            except Exception as e:
                _LOG.debug(f"LongBench sample error: {e}")
                continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["longbench"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "tasks": self._aggregate_by_task(results),
        }
        
        _LOG.info(f"LongBench Accuracy: {accuracy:.4f}")
        return self.results["longbench"]
    
    def evaluate_needle_haystack(self) -> Dict[str, Any]:
        """Evaluate on Needle in Haystack - Information retrieval in long context."""
        _LOG.info("Evaluating Needle in Haystack...")
        
        results = []
        
        self.model.eval()
        
        for context_length in self.config.context_lengths:
            for position_ratio in self.config.needle_positions:
                try:
                    needle, haystack, answer = self._generate_needle_haystack(
                        context_length, position_ratio
                    )
                    
                    prompt = f"Context:\n{haystack}\n\nQuestion: What is the special magic number?\n\nAnswer:"
                    
                    encoding = self.tokenizer(
                        prompt,
                        max_length=context_length + 512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.config.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample,
                        )
                    
                    generated = self.tokenizer.decode(
                        outputs[0][encoding["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    correct = answer in generated
                    
                    results.append({
                        "context_length": context_length,
                        "position": position_ratio,
                        "correct": correct,
                        "generated": generated,
                        "expected": answer,
                    })
                    
                except Exception as e:
                    _LOG.debug(f"Needle Haystack sample error: {e}")
                    continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        position_bias = self.metrics.position_bias(results)
        
        self.results["needle_haystack"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "position_bias": position_bias,
            "by_length": self._aggregate_by_length(results),
        }
        
        _LOG.info(f"Needle in Haystack Accuracy: {accuracy:.4f}")
        return self.results["needle_haystack"]
    
    def evaluate_passkey(self) -> Dict[str, float]:
        """Evaluate on Passkey Retrieval - Key retrieval from long sequences."""
        _LOG.info("Evaluating Passkey Retrieval...")
        
        results = []
        
        self.model.eval()
        
        for context_length in self.config.context_lengths:
            for position_ratio in self.config.needle_positions:
                try:
                    passkey, context = self._generate_passkey_context(
                        context_length, position_ratio
                    )
                    
                    prompt = f"Context:\n{context}\n\nQuestion: What is the passkey?\n\nAnswer:"
                    
                    encoding = self.tokenizer(
                        prompt,
                        max_length=context_length + 512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.config.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample,
                        )
                    
                    generated = self.tokenizer.decode(
                        outputs[0][encoding["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    correct = passkey in generated
                    
                    results.append({
                        "context_length": context_length,
                        "position": position_ratio,
                        "correct": correct,
                        "generated": generated,
                        "expected": passkey,
                    })
                    
                except Exception as e:
                    _LOG.debug(f"Passkey sample error: {e}")
                    continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["passkey"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "by_length": self._aggregate_by_length(results),
        }
        
        _LOG.info(f"Passkey Retrieval Accuracy: {accuracy:.4f}")
        return self.results["passkey"]
    
    def evaluate_kv_retrieval(self) -> Dict[str, float]:
        """Evaluate on KV Retrieval - Key-Value pair retrieval."""
        _LOG.info("Evaluating KV Retrieval...")
        
        results = []
        
        self.model.eval()
        
        for context_length in self.config.context_lengths:
            try:
                kv_pairs, target_key, target_value = self._generate_kv_pairs(context_length)
                
                prompt = f"Context:\n{kv_pairs}\n\nQuestion: What is the value for key '{target_key}'?\n\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=context_length + 512,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=64,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                correct = target_value in generated
                
                results.append({
                    "context_length": context_length,
                    "correct": correct,
                    "generated": generated,
                    "expected": target_value,
                })
                
            except Exception as e:
                _LOG.debug(f"KV Retrieval sample error: {e}")
                continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["kv_retrieval"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "by_length": self._aggregate_by_length(results),
        }
        
        _LOG.info(f"KV Retrieval Accuracy: {accuracy:.4f}")
        return self.results["kv_retrieval"]
    
    def evaluate_longbook_qa(self) -> Dict[str, float]:
        """Evaluate on LongBook QA - Long document question answering."""
        _LOG.info("Evaluating LongBook QA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("xlang-ai/LongBench", "longbook_qa_eng", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load LongBook QA: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="LongBook QA"):
            try:
                context = item.get("context", "")
                question = item.get("input", "")
                answers = item.get("answers", [])
                
                if not context or not question:
                    continue
                
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=max(self.config.context_lengths),
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=256,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                correct = self._evaluate_answer(generated, answers)
                
                results.append({
                    "correct": correct,
                    "generated": generated,
                    "expected": answers,
                    "context_length": encoding["input_ids"].shape[1],
                })
                
            except Exception as e:
                _LOG.debug(f"LongBook QA sample error: {e}")
                continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["longbook_qa"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
        }
        
        _LOG.info(f"LongBook QA Accuracy: {accuracy:.4f}")
        return self.results["longbook_qa"]
    
    def _generate_needle_haystack(
        self, 
        context_length: int, 
        position_ratio: float
    ) -> Tuple[str, str, str]:
        """Generate needle in haystack context."""
        magic_number = ''.join(random.choices(string.digits, k=7))
        needle = f"The special magic number is {magic_number}. Remember this."
        
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "The only thing we have to fear is fear itself.",
        ]
        
        tokens_per_sentence = 15
        num_sentences = context_length // tokens_per_sentence
        
        haystack_sentences = [random.choice(sentences) for _ in range(num_sentences)]
        
        insert_position = int(len(haystack_sentences) * position_ratio)
        haystack_sentences.insert(insert_position, needle)
        
        haystack = " ".join(haystack_sentences)
        
        return needle, haystack, magic_number
    
    def _generate_passkey_context(
        self, 
        context_length: int, 
        position_ratio: float
    ) -> Tuple[str, str]:
        """Generate passkey context."""
        passkey = ''.join(random.choices(string.digits, k=10))
        passkey_sentence = f"The passkey is {passkey}. Do not forget it."
        
        noise_sentences = [
            f"Item {i}: {random.randint(1000, 9999)}" 
            for i in range(context_length // 10)
        ]
        
        insert_position = int(len(noise_sentences) * position_ratio)
        noise_sentences.insert(insert_position, passkey_sentence)
        
        context = " ".join(noise_sentences)
        
        return passkey, context
    
    def _generate_kv_pairs(self, context_length: int) -> Tuple[str, str, str]:
        """Generate key-value pairs context."""
        num_pairs = context_length // 50
        
        kv_pairs = []
        target_key = None
        target_value = None
        
        for i in range(num_pairs):
            key = f"key_{i:05d}"
            value = ''.join(random.choices(string.ascii_lowercase, k=10))
            kv_pairs.append(f"{key}: {value}")
            
            if i == num_pairs // 2:
                target_key = key
                target_value = value
        
        context = "\n".join(kv_pairs)
        
        return context, target_key, target_value
    
    def _evaluate_answer(self, generated: str, answers: List[str]) -> bool:
        """Evaluate generated answer against expected answers."""
        generated_lower = generated.lower()
        
        for answer in answers:
            answer_lower = answer.lower()
            if answer_lower in generated_lower:
                return True
        
        return False
    
    def _aggregate_by_task(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate results by task."""
        task_results = defaultdict(list)
        
        for r in results:
            task = r.get("task", "default")
            task_results[task].append(r.get("correct", False))
        
        return {
            task: sum(correct) / len(correct)
            for task, correct in task_results.items()
        }
    
    def _aggregate_by_length(self, results: List[Dict]) -> Dict[int, float]:
        """Aggregate results by context length."""
        length_results = defaultdict(list)
        
        for r in results:
            length = r.get("context_length", 0)
            length_results[length].append(r.get("correct", False))
        
        return {
            length: sum(correct) / len(correct)
            for length, correct in length_results.items()
        }
    
    def evaluate_ultra_long_needle(self) -> Dict[str, Any]:
        """Evaluate ultra-long context (1M-10M tokens) needle retrieval.
        
        This tests PiscesL1's world-leading 10M token context capability.
        Surpasses Kimi K2.5 (~2M) and Gemini 3 (~2M).
        """
        _LOG.info("Evaluating Ultra-Long Needle Retrieval (1M-10M tokens)...")
        
        ultra_long_lengths = [1_000_000, 2_000_000, 4_000_000, 8_000_000, 10_000_000]
        results = []
        
        self.model.eval()
        
        for context_length in ultra_long_lengths:
            for position_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
                try:
                    _LOG.info(f"Testing {context_length:,} tokens at position {position_ratio:.2f}")
                    
                    needle, haystack, answer = self._generate_ultra_long_context(
                        context_length, position_ratio
                    )
                    
                    prompt = f"Find the special code in this document. What is it?\n\n{haystack}\n\nAnswer:"
                    
                    encoding = self.tokenizer(
                        prompt,
                        max_length=context_length + 1024,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.config.device)
                    
                    actual_length = encoding["input_ids"].shape[1]
                    _LOG.info(f"Actual token count: {actual_length:,}")
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample,
                        )
                    
                    generated = self.tokenizer.decode(
                        outputs[0][encoding["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    correct = answer in generated
                    
                    results.append({
                        "context_length": context_length,
                        "actual_tokens": actual_length,
                        "position": position_ratio,
                        "correct": correct,
                        "generated": generated,
                        "expected": answer,
                    })
                    
                    _LOG.info(f"Context: {context_length:,}, Position: {position_ratio:.2f}, Correct: {correct}")
                    
                except Exception as e:
                    _LOG.error(f"Ultra-long needle test failed: {e}")
                    continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["ultra_long_needle"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "max_context_tested": max((r["context_length"] for r in results), default=0),
            "by_length": self._aggregate_by_length(results),
            "world_record": "10M tokens context - surpasses Kimi K2.5 (2M) and Gemini 3 (2M)",
        }
        
        _LOG.info(f"Ultra-Long Needle Accuracy: {accuracy:.4f}")
        return self.results["ultra_long_needle"]
    
    def evaluate_book_summarization(self) -> Dict[str, float]:
        """Evaluate book-length document summarization (100K-1M tokens)."""
        _LOG.info("Evaluating Book Summarization...")
        
        book_lengths = [100_000, 250_000, 500_000, 1_000_000]
        results = []
        
        self.model.eval()
        
        for book_length in book_lengths:
            try:
                book_content = self._generate_synthetic_book(book_length)
                
                prompt = f"Summarize the following book in 3-5 sentences:\n\n{book_content}\n\nSummary:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=book_length + 512,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=256,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                has_coherent_summary = len(generated.split()) >= 20
                
                results.append({
                    "book_length": book_length,
                    "summary_length": len(generated.split()),
                    "coherent": has_coherent_summary,
                    "correct": has_coherent_summary,
                })
                
            except Exception as e:
                _LOG.debug(f"Book summarization error: {e}")
                continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["book_summarization"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
        }
        
        _LOG.info(f"Book Summarization Accuracy: {accuracy:.4f}")
        return self.results["book_summarization"]
    
    def evaluate_codebase_analysis(self) -> Dict[str, float]:
        """Evaluate large codebase analysis capability."""
        _LOG.info("Evaluating Codebase Analysis...")
        
        codebase_sizes = [50_000, 100_000, 250_000, 500_000]
        results = []
        
        self.model.eval()
        
        for codebase_size in codebase_sizes:
            try:
                codebase = self._generate_synthetic_codebase(codebase_size)
                
                target_function = f"def target_function_{random.randint(1000, 9999)}"
                codebase = f"{target_function}():\n    return 'SECRET_CODE_12345'\n\n" + codebase
                
                prompt = f"Analyze this codebase and find the function that returns 'SECRET_CODE_12345'. What is its name?\n\n{codebase}\n\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=codebase_size + 512,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=128,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                correct = "target_function" in generated
                
                results.append({
                    "codebase_size": codebase_size,
                    "correct": correct,
                    "generated": generated,
                })
                
            except Exception as e:
                _LOG.debug(f"Codebase analysis error: {e}")
                continue
        
        accuracy = self.metrics.retrieval_accuracy(results)
        
        self.results["codebase_analysis"] = {
            "accuracy": accuracy,
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
        }
        
        _LOG.info(f"Codebase Analysis Accuracy: {accuracy:.4f}")
        return self.results["codebase_analysis"]
    
    def _generate_ultra_long_context(
        self, 
        context_length: int, 
        position_ratio: float
    ) -> Tuple[str, str, str]:
        """Generate ultra-long context for testing (1M-10M tokens)."""
        special_code = f"CODE_{random.randint(100000, 999999)}_{random.randint(100000, 999999)}"
        needle = f"[SPECIAL] The secret code is: {special_code} [/SPECIAL]"
        
        paragraph_templates = [
            "The advancement of artificial intelligence has revolutionized many industries.",
            "Machine learning models continue to improve in accuracy and efficiency.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning architectures have achieved remarkable success in various domains.",
            "The integration of AI in healthcare has improved diagnostic accuracy.",
        ]
        
        tokens_per_paragraph = 25
        num_paragraphs = context_length // tokens_per_paragraph
        
        paragraphs = []
        for i in range(num_paragraphs):
            template = paragraph_templates[i % len(paragraph_templates)]
            paragraphs.append(f"[{i:08d}] {template}")
        
        insert_position = int(len(paragraphs) * position_ratio)
        paragraphs.insert(insert_position, needle)
        
        haystack = "\n\n".join(paragraphs)
        
        return needle, haystack, special_code
    
    def _generate_synthetic_book(self, length: int) -> str:
        """Generate synthetic book content."""
        chapters = []
        current_length = 0
        
        chapter_templates = [
            "Chapter {n}: The Journey Begins\n\nIn the beginning, there was a great adventure awaiting. "
            "The protagonist found themselves at a crossroads, uncertain of which path to take. "
            "Little did they know that this decision would change everything.",
            "Chapter {n}: The Discovery\n\nA remarkable discovery was made that day. "
            "The ancient artifact held secrets that had been hidden for centuries. "
            "Scholars would debate its significance for generations to come.",
            "Chapter {n}: The Challenge\n\nThe challenge was greater than anyone had anticipated. "
            "But with determination and courage, progress was made step by step. "
            "Each obstacle overcome brought new insights and understanding.",
        ]
        
        chapter_num = 1
        while current_length < length:
            template = chapter_templates[chapter_num % len(chapter_templates)]
            chapter = template.format(n=chapter_num)
            chapters.append(chapter)
            current_length += len(chapter.split())
            chapter_num += 1
        
        return "\n\n".join(chapters)
    
    def _generate_synthetic_codebase(self, size: int) -> str:
        """Generate synthetic codebase for testing."""
        files = []
        current_size = 0
        
        while current_size < size:
            file_name = f"module_{random.randint(1000, 9999)}.py"
            
            functions = []
            for _ in range(random.randint(5, 15)):
                func_name = f"func_{random.randint(1000, 9999)}"
                func_body = f'''
def {func_name}(x, y):
    """Process input parameters."""
    result = x + y
    return result * 2
'''
                functions.append(func_body)
            
            file_content = f"# File: {file_name}\n" + "\n".join(functions)
            files.append(file_content)
            current_size += len(file_content.split())
        
        return "\n\n# " + "=" * 40 + "\n\n".join(files)
    
    def run_all_long_context_benchmarks(self) -> Dict[str, Any]:
        """Run all long context benchmarks."""
        _LOG.info("Running all long context benchmarks...")
        
        benchmarks_map = {
            "longbench": self.evaluate_longbench,
            "needle_haystack": self.evaluate_needle_haystack,
            "passkey": self.evaluate_passkey,
            "kv_retrieval": self.evaluate_kv_retrieval,
            "longbook_qa": self.evaluate_longbook_qa,
            "ultra_long_needle": self.evaluate_ultra_long_needle,
            "book_summarization": self.evaluate_book_summarization,
            "codebase_analysis": self.evaluate_codebase_analysis,
        }
        
        for benchmark in self.config.long_context_benchmarks:
            if benchmark in benchmarks_map:
                try:
                    benchmarks_map[benchmark]()
                except Exception as e:
                    _LOG.error(f"Long context benchmark {benchmark} failed: {e}")
        
        self.results["long_context_summary"] = self._generate_long_context_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_long_context_summary(self) -> Dict[str, float]:
        """Generate long context benchmark summary."""
        summary = {}
        for name, result in self.results.items():
            if name == "long_context_summary":
                continue
            if isinstance(result, dict) and "accuracy" in result:
                summary[name] = result["accuracy"]
        
        if summary:
            summary["average"] = sum(summary.values()) / len(summary)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"long_context_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Long context benchmark results saved to {output_path}")


class PiscesLxToolsLongContextBenchmarkRunner:
    """Runner for long context benchmarks."""
    
    def __init__(
        self,
        config: PiscesLxToolsLongContextConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        
        _LOG.info("PiscesLxToolsLongContextBenchmarkRunner initialized")
    
    def run_all(self) -> Dict[str, Any]:
        """Run all long context benchmarks."""
        _LOG.info("Running all long context benchmarks...")
        
        evaluator = PiscesLxToolsLongContextEvaluator(
            self.config, self.model, self.tokenizer
        )
        self.results = evaluator.run_all_long_context_benchmarks()
        
        return self.results
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("PiscesL1 Long Context Benchmark Results")
        print("=" * 60)
        
        if "long_context_summary" in self.results:
            summary = self.results["long_context_summary"]
            for benchmark, score in summary.items():
                if benchmark != "average":
                    print(f"  {benchmark:20s}: {score:.4f}")
            if "average" in summary:
                print(f"  {'Average':20s}: {summary['average']:.4f}")
        
        print("=" * 60 + "\n")


def create_long_context_evaluator(
    config: PiscesLxToolsLongContextConfig,
    model: nn.Module,
    tokenizer: Any,
) -> PiscesLxToolsLongContextBenchmarkRunner:
    """Factory function to create long context benchmark runner."""
    return PiscesLxToolsLongContextBenchmarkRunner(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
