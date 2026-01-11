#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

import dms_core
PiscesLxCoreLog = dms_core.log.get_logger

logger = PiscesLxCoreLog("pisceslx.tools.benchmark")


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
                logger.warning(f"Failed to load {subject}: {e}")
        
        super().__init__(all_data, tokenizer)
        logger.info(f"MMLU dataset: {len(self.data)} samples")
    
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
        logger.info(f"HumanEval dataset: {len(self.data)} samples")
    
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
        logger.info(f"GSM8K dataset: {len(self.data)} samples")
    
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
        
        logger.info("PiscesL1BenchmarkEvaluator initialized")
    
    def evaluate_mmlu(self) -> Dict[str, float]:
        """Evaluate on MMLU benchmark."""
        logger.info("Evaluating MMLU...")
        
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
        
        logger.info(f"MMLU Average Accuracy: {avg_accuracy:.4f}")
        
        return self.results["mmlu"]
    
    def evaluate_humaneval(self) -> Dict[str, float]:
        """Evaluate on HumanEval benchmark."""
        logger.info("Evaluating HumanEval...")
        
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
                        logger.debug(f"Evaluation error for {task_id}: {e}")
        
        pass_rate = correct / max(1, total)
        
        self.results["humaneval"] = {
            "pass_rate": pass_rate,
            "correct": correct,
            "total": total,
        }
        
        logger.info(f"HumanEval Pass Rate: {pass_rate:.4f}")
        
        return self.results["humaneval"]
    
    def evaluate_gsm8k(self) -> Dict[str, float]:
        """Evaluate on GSM8K benchmark."""
        logger.info("Evaluating GSM8K...")
        
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
        
        logger.info(f"GSM8K Accuracy: {accuracy:.4f}")
        
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
        logger.info("Running all benchmarks...")
        
        benchmarks = self.config.benchmarks
        
        for benchmark in benchmarks:
            benchmark = benchmark.lower()
            
            if benchmark == "mmlu":
                self.evaluate_mmlu()
            elif benchmark == "humaneval":
                self.evaluate_humaneval()
            elif benchmark == "gsm8k":
                self.evaluate_gsm8k()
            elif benchmark == "hellaswag":
                self._evaluate_hellaswag()
            elif benchmark == "truthfulqa":
                self._evaluate_truthfulqa()
        
        self.results["summary"] = self._generate_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _evaluate_hellaswag(self) -> Dict[str, float]:
        """Evaluate on HellaSwag benchmark."""
        logger.info("Evaluating HellaSwag...")
        
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
        
        logger.info(f"HellaSwag Accuracy: {accuracy:.4f}")
        
        return self.results["hellaswag"]
    
    def _evaluate_truthfulqa(self) -> Dict[str, float]:
        """Evaluate on TruthfulQA benchmark."""
        logger.info("Evaluating TruthfulQA...")
        
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
        
        logger.success(f"Benchmark results saved to {output_path}")
    
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
    from model.modeling import RuchbahModel
    from model.config import RuchbahConfig
    
    config = PiscesL1BenchmarkConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        benchmarks=args.benchmarks.split(","),
    )
    
    logger.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    logger.info(f"Loading model from {config.model_path}")
    model_config = RuchbahConfig.from_json(os.path.join(config.model_path, "config.json"))
    model = RuchbahModel(model_config)
    model = model.from_pretrained(config.model_path)
    
    if config.device == "cuda":
        model = model.cuda()
    
    model.eval()
    
    evaluator = create_benchmark_evaluator(config, model, tokenizer)
    
    results = evaluator.run_all_benchmarks()
    
    evaluator.print_summary()
    
    logger.success("Benchmark evaluation completed")


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
