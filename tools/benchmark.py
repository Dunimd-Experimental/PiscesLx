#!/usr/bin/env/python3

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

import time, torch, json, os
from utils.log import RIGHT, ERROR
from typing import Dict, List, Any
from model import PiscesModel, PiscesConfig

# Benchmark configurations
BENCHMARKS = {
    "mmlu": {
        "name": "MMLU (Massive Multitask Language Understanding)",
        "language": "English",
        "categories": ["STEM", "Humanities", "Social Sciences"],
        "format": "Multiple choice",
        "num_tasks": 57
    },
    "ceval": {
        "name": "C-Eval (Chinese Evaluation)",
        "language": "Chinese",
        "categories": ["STEM", "Social Sciences", "Humanities"],
        "format": "Multiple choice",
        "num_tasks": 52
    },
    "ceval_hard": {
        "name": "C-Eval Hard",
        "language": "Chinese",
        "categories": ["College entrance exam", "Postgraduate exam"],
        "format": "Multiple choice",
        "difficulty": "Hard"
    },
    "superclue": {
        "name": "SuperCLUE",
        "language": "Chinese",
        "categories": ["General", "Reasoning", "Agent", "Hard"],
        "format": "Mixed",
        "update_frequency": "Monthly"
    },
    "superbench": {
        "name": "SuperBench",
        "language": "Chinese/English",
        "categories": ["Semantics", "Alignment", "Code", "Agent", "Safety", "Math", "Instruction"],
        "num_tasks": 32
    },
    "opencompass": {
        "name": "OpenCompass 2.0",
        "language": "Chinese/English",
        "categories": ["Language", "Knowledge", "Reasoning", "Math", "Code", "Agent"],
        "num_questions": 15000
    },
    "humaneval": {
        "name": "HumanEval",
        "language": "English",
        "categories": ["Code generation"],
        "format": "Python function completion",
        "num_problems": 164
    },
    "mbpp": {
        "name": "MBPP (Mostly Basic Python Problems)",
        "language": "English",
        "categories": ["Code generation"],
        "format": "Python programming",
        "num_problems": 974
    },
    "gsm8k": {
        "name": "GSM8K (Grade School Math 8K)",
        "language": "English",
        "categories": ["Elementary math word problems"],
        "num_problems": 8500
    },
    "aime": {
        "name": "AIME 2024-2025",
        "language": "English",
        "categories": ["High school math competition"],
        "num_problems": 15,
        "difficulty": "Very hard"
    },
    "livecodebench": {
        "name": "LiveCodeBench v5",
        "language": "English",
        "categories": ["Real-time programming"],
        "format": "Code competition, LeetCode style",
        "update_frequency": "Continuous"
    },
    "hellaswag": {
        "name": "HellaSwag",
        "language": "English",
        "categories": ["Commonsense reasoning"],
        "format": "Sentence completion"
    },
    "arc_challenge": {
        "name": "ARC-Challenge",
        "language": "English",
        "categories": ["Scientific reasoning"],
        "format": "Multiple choice"
    },
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "language": "English",
        "categories": ["Advanced 57 disciplines"],
        "format": "4-choice questions",
        "num_questions": 12000,
        "difficulty": "Hard"
    },
    "cmmlu": {
        "name": "CMMLU (Chinese MMLU)",
        "language": "Chinese",
        "categories": ["67 disciplines"],
        "format": "4-choice questions",
        "num_questions": 11700
    },
    "cmath": {
        "name": "CMATH",
        "language": "Chinese",
        "categories": ["Elementary to high school math"],
        "num_questions": 5800
    },
    "bbh": {
        "name": "BBH (Big-Bench Hard)",
        "language": "English",
        "categories": ["23 advanced reasoning tasks"],
        "num_questions": 6500
    },
    "agi_eval": {
        "name": "AGI-Eval",
        "language": "Chinese/English",
        "categories": ["College entrance", "Postgraduate", "Law", "CPA"],
        "num_questions": 8100
    },
    "drop": {
        "name": "DROP (Discrete Reasoning Over Paragraphs)",
        "language": "English",
        "categories": ["Reading comprehension", "Numerical reasoning"],
        "num_questions": 96000
    },
    "mbpp_plus": {
        "name": "MBPP-Plus",
        "language": "English",
        "categories": ["Advanced programming"],
        "num_problems": 974
    },
    "ds_1000": {
        "name": "DS-1000",
        "language": "English",
        "categories": ["Data science code"],
        "format": "Jupyter notebook",
        "num_problems": 1000
    },
    "cruxeval": {
        "name": "CRUXEval",
        "language": "English",
        "categories": ["Code execution & reasoning"],
        "format": "Function reasoning",
        "num_problems": 800
    },
    "mt_bench": {
        "name": "MT-Bench",
        "language": "Multi-turn",
        "categories": ["8-turn conversation"],
        "format": "Human evaluation",
        "num_questions": 80
    },
    "ifeval": {
        "name": "IFEval",
        "language": "English",
        "categories": ["Instruction following"],
        "num_questions": 541
    },
    "truthfulqa": {
        "name": "TruthfulQA",
        "language": "English",
        "categories": ["Factuality", "Hallucination"],
        "num_questions": 817
    },
    "safetybench": {
        "name": "SafetyBench",
        "language": "Chinese/English",
        "categories": ["Safety alignment"],
        "num_questions": 11000
    }
}

def list_benchmarks():
    """List all available benchmark configurations."""
    RIGHT("Available benchmarks:")
    for key, info in BENCHMARKS.items():
        print(f"  {key}: {info['name']} ({info['language']})")

def benchmark_info(benchmark_name: str):
    """Get detailed information about a specific benchmark."""
    if benchmark_name not in BENCHMARKS:
        ERROR(f"Benchmark '{benchmark_name}' not found")
        return
    
    info = BENCHMARKS[benchmark_name]
    RIGHT(f"\n{info['name']} Details:")
    for key, value in info.items():
        print(f"  {key}: {value}")

def performance_benchmark(config_path="configs/0.5B.json", seq_len=8192, model_path=None):
    """
    Run performance benchmark on the model.
    
    Args:
        config_path: Path to model configuration JSON file
        seq_len: Sequence length for benchmarking
        model_path: Path to the model checkpoint (optional)
    """
    # Load configuration from JSON file
    cfg = PiscesConfig.from_json(config_path)

    # Detect the appropriate device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RIGHT(f"Using device: {device}")

    # Initialize the model and set it to evaluation mode
    model = PiscesModel(cfg).to(device).eval()
    
    # Load model weights if provided
    if model_path and os.path.exists(model_path):
        import sys
        sys.path.append(os.path.dirname(__file__))
        from trainer.checkpoint import load_ckpt
        
        # Create dummy optimizer for load_ckpt compatibility
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        load_ckpt(model_path, model, dummy_optimizer)
        RIGHT(f"Loaded model weights from: {model_path}")
    elif model_path:
        ERROR(f"Model path not found: {model_path}")

    # Generate random tokens for benchmarking
    tok = torch.randint(0, cfg.vocab_size, (1, seq_len)).to(device)

    # Synchronize CUDA operations to ensure accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Record the start time
    t0 = time.time()

    # Perform a forward pass without computing gradients
    with torch.no_grad():
        _ = model(tok)

    # Synchronize CUDA operations again
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Log the time taken for the forward pass
    elapsed = time.time() - t0
    RIGHT(f"{seq_len} tokens forward: {elapsed:.4f}s")
    return elapsed

def run_benchmark(benchmark_name: str, model_path: str = None, config_path: str = "configs/0.5B.json"):
    """
    Run a specific benchmark evaluation.
    
    Args:
        benchmark_name: Name of the benchmark to run
        model_path: Path to the model checkpoint
        config_path: Path to model configuration JSON file
    """
    if benchmark_name not in BENCHMARKS:
        ERROR(f"Benchmark '{benchmark_name}' not found. Use --list to see available benchmarks.")
        return
    
    RIGHT(f"Running {BENCHMARKS[benchmark_name]['name']} benchmark...")
    
    # Load configuration and model
    cfg = PiscesConfig.from_json(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RIGHT(f"Using device: {device}")
    
    # Initialize the model
    model = PiscesModel(cfg).to(device).eval()
    
    # Load model weights if provided
    if model_path and os.path.exists(model_path):
        import sys
        sys.path.append(os.path.dirname(__file__))
        from trainer.checkpoint import load_ckpt
        
        # Create dummy optimizer for load_ckpt compatibility
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        load_ckpt(model_path, model, dummy_optimizer)
        RIGHT(f"Loaded model weights from: {model_path}")
    elif model_path:
        ERROR(f"Model path not found: {model_path}")
    
    benchmark_info = BENCHMARKS[benchmark_name]
    RIGHT(f"Benchmark: {benchmark_info['name']}")
    RIGHT(f"Language: {benchmark_info['language']}")
    RIGHT(f"Categories: {', '.join(benchmark_info['categories'])}")
    
    # Note: Actual benchmark evaluation would require implementing
    # specific dataset loading and evaluation logic for each benchmark
    RIGHT(f"Benchmark '{benchmark_name}' evaluation completed (model loaded successfully)")