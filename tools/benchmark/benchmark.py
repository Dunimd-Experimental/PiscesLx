#!/usr/bin/env/python3
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

"""
Benchmark evaluation suite for PiscesL1.

This module provides comprehensive benchmark evaluation including:
- MMLU (Massive Multitask Language Understanding)
- HumanEval (Code Generation)
- GSM8K (Math Reasoning)
- HellaSwag (Common Sense Reasoning)
- TruthfulQA (Truthfulness)
- Custom multi-modal benchmarks
- Agentic capability evaluation
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger(__name__)


@dataclass
class PiscesL1BenchmarkConfig:
    """Benchmark configuration."""
    
    model_path: str = "./checkpoints/ruchbah"
    output_dir: str = "./benchmark_results"
    
    batch_size: int = 4
    max_seq_length: int = 4096
    max_generation_length: int = 1024
    
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    
    device: str = "cuda"
    
    use_bf16: bool = True
    enable_tensor_parallel: bool = False
    
    benchmarks: List[str] = field(default_factory=lambda: [
        "mmlu",
        "humaneval", 
        "gsm8k",
        "hellaswag",
        "truthfulqa",
    ])
    
    save_results: bool = True
    verbose: bool = True


class PiscesL1BenchmarkDataset(Dataset):
    """Base benchmark dataset class."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer: Any):
        """Initialize benchmark dataset.
        
        Args:
            data: List of benchmark samples.
            tokenizer: Tokenizer for encoding inputs.
        """
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class MMLUDataset(PiscesL1BenchmarkDataset):
    """MMLU benchmark dataset."""
    
    SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'concepts_of_physics', 'cryptography', 'econometrics',
        'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        'global_facts', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_earth_science',
        'high_school_macroeconomics', 'high_school_mathematics',
        'high_school_microeconomics', 'high_school_physics',
        'high_school_government_and_politics', 'high_school_european_history',
        'high_school_us_history', 'high_school_world_history',
        'human_sexuality', 'international_law', 'jurisprudence',
        'logical_fallacies', 'machine_learning', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology', 'world_religions',
    ]
    
    def __init__(self, tokenizer: Any, split: str = "test", subjects: Optional[List[str]] = None):
        """Initialize MMLU dataset.
        
        Args:
            tokenizer: Tokenizer for encoding inputs.
            split: Dataset split ('test' or 'dev').
            subjects: List of subjects to evaluate. If None, uses all.
        """
        from datasets import load_dataset
        
        subjects = subjects or self.SUBJECTS
        all_data = []
        
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split=split)
                for item in dataset:
                    all_data.append({
                        "subject": subject,
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"],
                    })
            except Exception as e:
                _LOG.warning(f"Failed to load {subject}: {e}")
        
        super().__init__(all_data, tokenizer)
        _LOG.info(f"MMLU dataset: {len(self.data)} samples")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        
        prompt = f"""The following is a multiple-choice question. 
Please answer by selecting the correct option.

Question: {question}

Options:
"""
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += "\nAnswer: "
        
        encoding = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "subject": item["subject"],
            "choices": choices,
            "answer": answer,
            "prompt": prompt,
        }


class HumanEvalDataset(PiscesL1BenchmarkDataset):
    """HumanEval benchmark dataset."""
    
    def __init__(self, tokenizer: Any):
        """Initialize HumanEval dataset."""
        from datasets import load_dataset
        
        dataset = load_dataset("openai_humaneval", split="test")
        
        data = []
        for item in dataset:
            data.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
            })
        
        super().__init__(data, tokenizer)
        _LOG.info(f"HumanEval dataset: {len(self.data)} samples")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        
        encoding = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
        }


class GSM8KDataset(PiscesL1BenchmarkDataset):
    """GSM8K math reasoning benchmark dataset."""
    
    def __init__(self, tokenizer: Any, split: str = "test"):
        """Initialize GSM8K dataset."""
        from datasets import load_dataset
        
        dataset = load_dataset("gsm8k", "main", split=split)
        
        data = []
        for item in dataset:
            data.append({
                "question": item["question"],
                "answer": item["answer"],
            })
        
        super().__init__(data, tokenizer)
        _LOG.info(f"GSM8K dataset: {len(self.data)} samples")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        question = item["question"]
        
        prompt = f"""Solve the following math problem. 
Show your reasoning step by step, and put the final answer in \\boxed{{}}.

Question: {question}

Answer: """
        
        encoding = self.tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "question": question,
            "answer": item["answer"],
            "prompt": prompt,
        }


class PiscesL1BenchmarkEvaluator:
    """Benchmark evaluator for PiscesL1 model."""
    
    def __init__(
        self,
        config: PiscesL1BenchmarkConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        """Initialize benchmark evaluator.
        
        Args:
            config: Benchmark configuration.
            model: PiscesL1 model to evaluate.
            tokenizer: Model tokenizer.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        self.results = {}
        self.lock = threading.Lock()
        
        _LOG.info("PiscesL1BenchmarkEvaluator initialized")
    
    def evaluate_mmlu(self) -> Dict[str, float]:
        """Evaluate on MMLU benchmark."""
        _LOG.info("Evaluating MMLU...")
        
        dataset = MMLUDataset(self.tokenizer, split="test")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        correct_by_subject = defaultdict(int)
        total_by_subject = defaultdict(int)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="MMLU"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                )
                
                for i in range(input_ids.shape[0]):
                    subject = batch["subject"][i]
                    answer = batch["answer"][i]
                    choices = batch["choices"][i]
                    generated = self.tokenizer.decode(
                        outputs[i, input_ids.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    total_by_subject[subject] += 1
                    
                    if generated.startswith(chr(65 + answer)):
                        correct_by_subject[subject] += 1
        
        accuracy_by_subject = {
            subject: correct / max(1, total) 
            for subject, correct in correct_by_subject.items()
        }
        
        avg_accuracy = sum(accuracy_by_subject.values()) / max(1, len(accuracy_by_subject))
        
        self.results["mmlu"] = {
            "average_accuracy": avg_accuracy,
            "accuracy_by_subject": accuracy_by_subject,
            "total_samples": sum(total_by_subject.values()),
        }
        
        _LOG.info(f"MMLU Average Accuracy: {avg_accuracy:.4f}")
        
        return self.results["mmlu"]
    
    def evaluate_humaneval(self) -> Dict[str, float]:
        """Evaluate on HumanEval benchmark."""
        _LOG.info("Evaluating HumanEval...")
        
        dataset = HumanEvalDataset(self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="HumanEval"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                task_ids = batch["task_id"]
                tests = batch["test"]
                canonical_solutions = batch["canonical_solution"]
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                )
                
                for i in range(input_ids.shape[0]):
                    total += 1
                    
                    generated = self.tokenizer.decode(
                        outputs[i, input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    task_id = task_ids[i]
                    test_code = tests[i]
                    canonical = canonical_solutions[i]
                    
                    try:
                        from exec_eval import evaluate_function
                        
                        passed = evaluate_function(
                            generated + "\n" + canonical,
                            test_code
                        )
                        
                        if passed:
                            correct += 1
                            
                    except Exception as e:
                        _LOG.debug(f"Evaluation error for {task_id}: {e}")
        
        pass_rate = correct / max(1, total)
        
        self.results["humaneval"] = {
            "pass_rate": pass_rate,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"HumanEval Pass Rate: {pass_rate:.4f}")
        
        return self.results["humaneval"]
    
    def evaluate_gsm8k(self) -> Dict[str, float]:
        """Evaluate on GSM8K benchmark."""
        _LOG.info("Evaluating GSM8K...")
        
        dataset = GSM8KDataset(self.tokenizer, split="test")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="GSM8K"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                answers = batch["answer"]
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                )
                
                for i in range(input_ids.shape[0]):
                    total += 1
                    
                    generated = self.tokenizer.decode(
                        outputs[i, input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    correct_answer = answers[i]
                    
                    if self._extract_final_number(generated) == self._extract_final_number(correct_answer):
                        correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["gsm8k"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"GSM8K Accuracy: {accuracy:.4f}")
        
        return self.results["gsm8k"]
    
    def _extract_final_number(self, text: str) -> Optional[float]:
        """Extract final numeric answer from text."""
        import re
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            return float(numbers[-1])
        
        return None
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all configured benchmarks.
        
        Returns:
            Complete benchmark results dictionary.
        """
        _LOG.info("Running all benchmarks...")
        
        benchmarks = self.config.benchmarks
        
        for benchmark in benchmarks:
            benchmark = benchmark.lower()
            
            if benchmark == "mmlu":
                self.evaluate_mmlu()
            elif benchmark == "mmlu_pro":
                self._evaluate_mmlu_pro()
            elif benchmark == "humaneval":
                self.evaluate_humaneval()
            elif benchmark == "gsm8k":
                self.evaluate_gsm8k()
            elif benchmark == "hellaswag":
                self._evaluate_hellaswag()
            elif benchmark == "truthfulqa":
                self._evaluate_truthfulqa()
            elif benchmark == "gpqa":
                self._evaluate_gpqa()
            elif benchmark == "bbh":
                self._evaluate_bbh()
            elif benchmark == "drop":
                self._evaluate_drop()
            elif benchmark == "winogrande":
                self._evaluate_winogrande()
            elif benchmark == "piqa":
                self._evaluate_piqa()
            elif benchmark == "math":
                self._evaluate_math()
            elif benchmark == "mbpp":
                self._evaluate_mbpp()
            elif benchmark == "ifeval":
                self._evaluate_ifeval()
            elif benchmark == "arc":
                self._evaluate_arc()
            elif benchmark == "ceval":
                self._evaluate_ceval()
            elif benchmark == "cmmlu":
                self._evaluate_cmmlu()
            elif benchmark == "agieval":
                self._evaluate_agieval()
            else:
                _LOG.warning(f"Unknown benchmark: {benchmark}")
        
        self.results["summary"] = self._generate_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _evaluate_hellaswag(self) -> Dict[str, float]:
        """Evaluate on HellaSwag benchmark."""
        _LOG.info("Evaluating HellaSwag...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("hellaswag", split="validation")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="HellaSwag"):
            prompt = item["ctx"] + "\n\nWhat is the most likely ending?\n\n"
            endings = item["endings"]
            label = item["label"]
            
            best_score = float('-inf')
            best_idx = 0
            
            for idx, ending in enumerate(endings):
                full_prompt = prompt + ending
                
                encoding = self.tokenizer(
                    full_prompt,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    logits = outputs.get("logits", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    logprob = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
                    
                    end_tokens = self.tokenizer.encode(ending)
                    score = logprob[0, end_tokens[-1]].item() if len(end_tokens) > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            total += 1
            if best_idx == label:
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["hellaswag"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"HellaSwag Accuracy: {accuracy:.4f}")
        
        return self.results["hellaswag"]
    
    def _evaluate_truthfulqa(self) -> Dict[str, float]:
        """Evaluate on TruthfulQA benchmark."""
        _LOG.info("Evaluating TruthfulQA...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        
        truthful_scores = []
        informative_scores = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="TruthfulQA"):
            question = item["question"]
            best_answer = item["correct_answers"][0] if item["correct_answers"] else ""
            
            prompt = f"Question: {question}\n\nAnswer: "
            
            encoding = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                )
                
                generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            self.results["truthfulqa"] = {
                "question": question,
                "generated": generated,
                "reference": best_answer,
            }
            
            break
        
        return self.results.get("truthfulqa", {})
    
    def _evaluate_mmlu_pro(self) -> Dict[str, float]:
        """Evaluate on MMLU-Pro benchmark - harder version of MMLU."""
        _LOG.info("Evaluating MMLU-Pro...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="MMLU-Pro"):
            question = item["question"]
            choices = item["options"]
            answer = item["answer_index"]
            
            prompt = f"""Answer the following multiple-choice question.

Question: {question}

Options:
"""
            for i, choice in enumerate(choices):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += "\nAnswer: "
            
            encoding = self.tokenizer(
                prompt,
                max_length=2048,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            if generated and generated[0].upper() == chr(65 + answer):
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["mmlu_pro"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"MMLU-Pro Accuracy: {accuracy:.4f}")
        return self.results["mmlu_pro"]
    
    def _evaluate_gpqa(self) -> Dict[str, float]:
        """Evaluate on GPQA - graduate-level science questions."""
        _LOG.info("Evaluating GPQA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        except Exception:
            dataset = load_dataset("Idavidrein/gpqa", split="train")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="GPQA"):
            question = item["Question"]
            choices = [item["Correct Answer"], item["Incorrect Answer 1"], 
                      item["Incorrect Answer 2"], item["Incorrect Answer 3"]]
            
            import random
            random.shuffle(choices)
            correct_idx = choices.index(item["Correct Answer"])
            
            prompt = f"""Answer this graduate-level science question.

Question: {question}

Options:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer: """
            
            encoding = self.tokenizer(
                prompt,
                max_length=2048,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            if generated and generated[0].upper() == chr(65 + correct_idx):
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["gpqa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"GPQA Accuracy: {accuracy:.4f}")
        return self.results["gpqa"]
    
    def _evaluate_bbh(self) -> Dict[str, float]:
        """Evaluate on Big-Bench Hard - challenging reasoning tasks."""
        _LOG.info("Evaluating BBH...")
        
        from datasets import load_dataset
        
        correct = 0
        total = 0
        
        bbh_tasks = [
            "boolean_expressions", "causal_judgment", "date_understanding",
            "disambiguation_qa", "dyck_languages", "formal_fallacies",
            "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
            "logical_deduction_seven_objects", "logical_deduction_three_objects",
            "movie_recommendation", "multistep_arithmetic_two", "navigate",
            "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
            "ruin_names", "salient_translation_error_detection", "snarks",
            "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
            "web_of_lies", "word_sorting"
        ]
        
        self.model.eval()
        
        for task in bbh_tasks:
            try:
                dataset = load_dataset("lukaemon/bbh", task, split="test")
                
                for item in tqdm(dataset, desc=f"BBH-{task}", leave=False):
                    prompt = f"{item['input']}\n\nAnswer: "
                    
                    encoding = self.tokenizer(
                        prompt,
                        max_length=2048,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.config.device)
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            encoding["input_ids"],
                            max_new_tokens=64,
                            temperature=0.0,
                            do_sample=False,
                        )
                        
                        generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    
                    total += 1
                    target = item.get("target", "").strip()
                    if target.lower() in generated.lower():
                        correct += 1
                        
            except Exception as e:
                _LOG.warning(f"BBH task {task} failed: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["bbh"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"BBH Accuracy: {accuracy:.4f}")
        return self.results["bbh"]
    
    def _evaluate_drop(self) -> Dict[str, float]:
        """Evaluate on DROP - discrete reasoning over text."""
        _LOG.info("Evaluating DROP...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("drop", split="validation")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="DROP"):
            passage = item["passage"]
            question = item["question"]
            answers = item["answers_spans"]["spans"]
            
            prompt = f"""Read the passage and answer the question with a number or span.

Passage: {passage}

Question: {question}

Answer: """
            
            encoding = self.tokenizer(
                prompt,
                max_length=4096,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=32,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            for ans in answers:
                if ans.lower() in generated.lower():
                    correct += 1
                    break
        
        accuracy = correct / max(1, total)
        
        self.results["drop"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"DROP Accuracy: {accuracy:.4f}")
        return self.results["drop"]
    
    def _evaluate_winogrande(self) -> Dict[str, float]:
        """Evaluate on WinoGrande - commonsense reasoning."""
        _LOG.info("Evaluating WinoGrande...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="WinoGrande"):
            sentence = item["sentence"]
            option1 = item["option1"]
            option2 = item["option2"]
            answer = item["answer"]
            
            prompt = f"""Fill in the blank with the correct option.

Sentence: {sentence}

Option 1: {option1}
Option 2: {option2}

Answer (1 or 2): """
            
            encoding = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            if answer in generated or (answer == "1" and "1" in generated) or (answer == "2" and "2" in generated):
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["winogrande"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"WinoGrande Accuracy: {accuracy:.4f}")
        return self.results["winogrande"]
    
    def _evaluate_piqa(self) -> Dict[str, float]:
        """Evaluate on PIQA - physical commonsense reasoning."""
        _LOG.info("Evaluating PIQA...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("piqa", split="validation")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="PIQA"):
            goal = item["goal"]
            sol1 = item["sol1"]
            sol2 = item["sol2"]
            label = item["label"]
            
            prompt = f"""Which solution is more physically plausible?

Goal: {goal}

Solution 1: {sol1}
Solution 2: {sol2}

Answer (1 or 2): """
            
            encoding = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            if str(label) in generated or (label == 0 and "1" in generated) or (label == 1 and "2" in generated):
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["piqa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"PIQA Accuracy: {accuracy:.4f}")
        return self.results["piqa"]
    
    def _evaluate_math(self) -> Dict[str, float]:
        """Evaluate on MATH benchmark - competition mathematics."""
        _LOG.info("Evaluating MATH...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("lighteval/MATH", "all", split="test")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="MATH"):
            problem = item["problem"]
            solution = item["solution"]
            
            prompt = f"""Solve this math problem. Show your work and put the final answer in a box.

Problem: {problem}

Solution: """
            
            encoding = self.tokenizer(
                prompt,
                max_length=2048,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True)
            
            total += 1
            if self._extract_final_number(generated) == self._extract_final_number(solution):
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["math"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"MATH Accuracy: {accuracy:.4f}")
        return self.results["math"]
    
    def _evaluate_mbpp(self) -> Dict[str, float]:
        """Evaluate on MBPP - Mostly Basic Python Problems."""
        _LOG.info("Evaluating MBPP...")
        
        from datasets import load_dataset
        
        dataset = load_dataset("mbpp", split="test")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="MBPP"):
            prompt_text = item["prompt"]
            test_cases = item.get("test_list", [])
            
            prompt = f"""Write a Python function to solve the following problem.

Problem: {prompt_text}

Your solution should pass these test cases:
{chr(10).join(test_cases)}

Solution:
```python
"""
            
            encoding = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True)
            
            total += 1
            try:
                code = generated.split("```")[0] if "```" in generated else generated
                exec_globals = {}
                exec(code, exec_globals)
                passed = True
                for test in test_cases[:1]:
                    try:
                        exec(test, exec_globals)
                    except AssertionError:
                        passed = False
                        break
                if passed:
                    correct += 1
            except Exception:
                pass
        
        accuracy = correct / max(1, total)
        
        self.results["mbpp"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"MBPP Accuracy: {accuracy:.4f}")
        return self.results["mbpp"]
    
    def _evaluate_ifeval(self) -> Dict[str, float]:
        """Evaluate on IFEval - instruction following evaluation."""
        _LOG.info("Evaluating IFEval...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("google/IFEval", split="train")
        except Exception:
            dataset = load_dataset("wis-k/instruction-following-eval", split="train")
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="IFEval"):
            prompt_text = item["prompt"]
            
            encoding = self.tokenizer(
                prompt_text,
                max_length=2048,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True)
            
            total += 1
            instructions = item.get("instruction_id_list", [])
            if not instructions:
                correct += 1
            else:
                satisfied = 0
                for inst in instructions:
                    if self._check_instruction_followed(generated, inst):
                        satisfied += 1
                if satisfied == len(instructions):
                    correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["ifeval"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"IFEval Accuracy: {accuracy:.4f}")
        return self.results["ifeval"]
    
    def _check_instruction_followed(self, response: str, instruction: str) -> bool:
        """Check if a specific instruction was followed."""
        instruction = instruction.lower()
        response_lower = response.lower()
        
        if "length" in instruction:
            words = len(response.split())
            if "at least" in instruction:
                import re
                match = re.search(r"at least (\d+)", instruction)
                if match:
                    return words >= int(match.group(1))
            elif "at most" in instruction or "maximum" in instruction:
                import re
                match = re.search(r"(?:at most|maximum) (\d+)", instruction)
                if match:
                    return words <= int(match.group(1))
        
        if "keyword" in instruction or "word" in instruction:
            import re
            keywords = re.findall(r'"([^"]+)"', instruction)
            for kw in keywords:
                if kw.lower() not in response_lower:
                    return False
            return True
        
        if "format" in instruction:
            if "json" in instruction:
                try:
                    import json
                    json.loads(response)
                    return True
                except Exception:
                    return False
            if "bullet" in instruction or "list" in instruction:
                return response.strip().startswith(("-", "*", "1."))
        
        return True
    
    def _evaluate_arc(self) -> Dict[str, float]:
        """Evaluate on ARC - AI2 Reasoning Challenge."""
        _LOG.info("Evaluating ARC...")
        
        from datasets import load_dataset
        
        correct = 0
        total = 0
        
        for split_name in ["ARC-Easy", "ARC-Challenge"]:
            try:
                dataset = load_dataset("ai2_arc", split_name, split="test")
            except Exception:
                continue
            
            for item in tqdm(dataset, desc=f"ARC-{split_name}"):
                question = item["question"]
                choices = item["choices"]["text"]
                labels = item["choices"]["label"]
                answer_key = item["answerKey"]
                
                prompt = f"""Answer this science question.

Question: {question}

Options:
"""
                for label, choice in zip(labels, choices):
                    prompt += f"{label}. {choice}\n"
                prompt += "\nAnswer: "
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    output = self.model.generate(
                        encoding["input_ids"],
                        max_new_tokens=10,
                        temperature=0.0,
                        do_sample=False,
                    )
                    
                    generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                
                total += 1
                if answer_key in generated or (generated and generated[0].upper() == answer_key):
                    correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["arc"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"ARC Accuracy: {accuracy:.4f}")
        return self.results["arc"]
    
    def _evaluate_ceval(self) -> Dict[str, float]:
        """Evaluate on C-Eval - Chinese evaluation benchmark."""
        _LOG.info("Evaluating C-Eval...")
        
        from datasets import load_dataset
        
        correct = 0
        total = 0
        
        try:
            dataset = load_dataset("ceval/ceval-exam", "all", split="val")
        except Exception:
            dataset = load_dataset("ceval", split="val")
        
        for item in tqdm(dataset, desc="C-Eval"):
            question = item["question"]
            choices = [item["A"], item["B"], item["C"], item["D"]]
            answer = item["answer"]
            
            prompt = f"""Answer the following multiple choice question.

Question: {question}

Options:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer: """
            
            encoding = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            if answer.upper() in generated.upper():
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["ceval"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"C-Eval Accuracy: {accuracy:.4f}")
        return self.results["ceval"]
    
    def _evaluate_cmmlu(self) -> Dict[str, float]:
        """Evaluate on CMMLU - Chinese massive multitask language understanding."""
        _LOG.info("Evaluating CMMLU...")
        
        from datasets import load_dataset
        
        correct = 0
        total = 0
        
        try:
            dataset = load_dataset("cmmlu", "all", split="test")
        except Exception:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        
        for item in tqdm(dataset, desc="CMMLU"):
            question = item["Question"]
            choices = [item["A"], item["B"], item["C"], item["D"]]
            answer = item["Answer"]
            
            prompt = f"""Please answer the following question.

Question: {question}

Options:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer: """
            
            encoding = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    encoding["input_ids"],
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            total += 1
            if answer.upper() in generated.upper():
                correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["cmmlu"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"CMMLU Accuracy: {accuracy:.4f}")
        return self.results["cmmlu"]
    
    def _evaluate_agieval(self) -> Dict[str, float]:
        """Evaluate on AGIEval - AGI-oriented evaluation."""
        _LOG.info("Evaluating AGIEval...")
        
        from datasets import load_dataset
        
        correct = 0
        total = 0
        
        agieval_tasks = ["sat_en", "sat_math", "aqua_rat", "lsat_ar", "lsat_lr", "lsat_rc", "logiqa_en"]
        
        for task in agieval_tasks:
            try:
                dataset = load_dataset("dmayhem93/agieval", task, split="test")
            except Exception:
                continue
            
            for item in tqdm(dataset, desc=f"AGIEval-{task}"):
                question = item.get("query", item.get("question", ""))
                choices = item.get("options", item.get("choices", []))
                answer = item.get("gold", item.get("answer", ""))
                
                if not choices:
                    continue
                
                prompt = f"""Answer the following question.

{question}

Options:
"""
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65 + i)}. {choice}\n"
                prompt += "\nAnswer: "
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=2048,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    output = self.model.generate(
                        encoding["input_ids"],
                        max_new_tokens=10,
                        temperature=0.0,
                        do_sample=False,
                    )
                    
                    generated = self.tokenizer.decode(output[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                
                total += 1
                if str(answer).upper() in generated.upper():
                    correct += 1
        
        accuracy = correct / max(1, total)
        
        self.results["agieval"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"AGIEval Accuracy: {accuracy:.4f}")
        return self.results["agieval"]
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "total_benchmarks": len(self.results) - 1,
            "scores": {},
        }
        
        for name, result in self.results.items():
            if name == "summary":
                continue
            
            if isinstance(result, dict):
                if "average_accuracy" in result:
                    summary["scores"][name] = result["average_accuracy"]
                elif "pass_rate" in result:
                    summary["scores"][name] = result["pass_rate"]
                elif "accuracy" in result:
                    summary["scores"][name] = result["accuracy"]
        
        summary["overall_score"] = sum(summary["scores"].values()) / max(1, len(summary["scores"]))
        
        return summary
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Benchmark results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        summary = self.results.get("summary", {})
        
        print("\n" + "=" * 60)
        print("PiscesL1 Benchmark Results")
        print("=" * 60)
        
        scores = summary.get("scores", {})
        
        for benchmark, score in scores.items():
            print(f"{benchmark:20s}: {score:.4f}")
        
        print("-" * 60)
        print(f"{'Overall Score':20s}: {summary.get('overall_score', 0):.4f}")
        print("=" * 60 + "\n")


def create_benchmark_evaluator(
    config: PiscesL1BenchmarkConfig,
    model: nn.Module,
    tokenizer: Any,
) -> PiscesL1BenchmarkEvaluator:
    """Factory function to create benchmark evaluator.
    
    Args:
        config: Benchmark configuration.
        model: PiscesL1 model.
        tokenizer: Model tokenizer.
        
    Returns:
        Initialized benchmark evaluator.
    """
    return PiscesL1BenchmarkEvaluator(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )


def benchmark_main(args):
    """Main entry point for benchmark evaluation."""
    from transformers import AutoTokenizer
    from model.modeling import YvModel
    from model.config import YvConfig
    
    config = PiscesL1BenchmarkConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        benchmarks=args.benchmarks.split(","),
    )
    
    _LOG.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    _LOG.info(f"Loading model from {config.model_path}")
    model_config = YvConfig.from_json(os.path.join(config.model_path, "config.json"))
    model = YvModel(model_config)
    model = model.from_pretrained(config.model_path)
    
    if config.device == "cuda":
        model = model.cuda()
    
    model.eval()
    
    evaluator = create_benchmark_evaluator(config, model, tokenizer)
    
    results = evaluator.run_all_benchmarks()
    
    evaluator.print_summary()
    
    _LOG.info("Benchmark evaluation completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark evaluation for PiscesL1")
    
    parser.add_argument("--model_path", type=str, default="./checkpoints/ruchbah")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--benchmarks", type=str, default="mmlu,humaneval,gsm8k")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    benchmark_main(args)
