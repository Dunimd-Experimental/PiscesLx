#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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
vLLM Inference Runner for PiscesL1.

Provides high-performance inference using vLLM's PagedAttention
and continuous batching capabilities.

Usage:
    python -m tools.infer.vllm_runner --model-path ./checkpoints/ruchbah --tensor-parallel 4
    python -m tools.infer.vllm_runner --interactive --port 8080
    python -m tools.infer.vllm_runner --batch prompts.txt --output results.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
from .vllm_engine import (
    PiscesLxVLLMEngine,
    PiscesLxVLLMConfig,
    benchmark_vllm_inference,
    create_vllm_engine,
)

logger = PiscesLxCoreLog("pisceslx.tools.infer.vllm_runner")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PiscesL1 vLLM Inference Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single prompt inference
    python -m tools.infer.vllm_runner --prompt "Hello, I am"

    # Interactive mode
    python -m tools.infer.vllm_runner --interactive --port 8080

    # Batch inference
    python -m tools.infer.vllm_runner --batch prompts.txt --output results.jsonl

    # Benchmark
    python -m tools.infer.vllm_runner --benchmark --prompts 100
        """
    )
    
    parser.add_argument("--model-path", type=str, default="./checkpoints/ruchbah",
                        help="Path to PiscesL1 model checkpoint")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallelism size")
    parser.add_argument("--max-len", type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument("--max-seqs", type=int, default=256,
                        help="Maximum number of sequences in batch")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["awq", "gptq", "squeezellm", None],
                        help="Quantization method")
    
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for inference")
    parser.add_argument("--batch", type=str, default=None,
                        help="File containing prompts (one per line)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for batch results")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for interactive server")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark instead of inference")
    parser.add_argument("--prompts", type=int, default=100,
                        help="Number of prompts for benchmark")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum tokens to generate")
    
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming output")
    
    return parser.parse_args()


def run_single_prompt(engine: PiscesLxVLLMEngine, args) -> None:
    """Run inference on a single prompt."""
    print("\n" + "="*60)
    print("PiscesL1 vLLM Inference")
    print("="*60)
    
    results = engine.generate(
        prompts=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    result = results[0]
    
    print(f"\nPrompt: {result['prompt']}")
    print(f"\nGenerated: {result['generated_text']}")
    print(f"\n{'─'*60}")
    print(f"Tokens: {result['num_tokens']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print(f"Speed: {result['tokens_per_second']:.2f} tokens/sec")


def run_batch_inference(engine: PiscesLxVLLMEngine, args) -> None:
    """Run batch inference on multiple prompts."""
    with open(args.batch, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"\nRunning batch inference on {len(prompts)} prompts...")
    
    results = engine.generate(
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\nResults saved to: {args.output}")
    
    total_tokens = sum(r['num_tokens'] for r in results)
    total_time = sum(r['latency_ms'] for r in results) / 1000
    
    print(f"\n{'='*60}")
    print(f"Batch Complete")
    print(f"{'='*60}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Mean latency: {total_time/len(prompts)*1000:.2f}ms")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_tokens/total_time:.2f} tokens/sec")


def run_interactive_server(engine: PiscesLxVLLMEngine, args) -> None:
    """Run interactive HTTP server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        
        app = FastAPI(
            title="PiscesL1 vLLM API",
            description="High-performance inference API powered by vLLM",
            version="1.0.0"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": "PiscesL1-Ruchbah"}
        
        @app.get("/info")
        async def info():
            return engine.get_model_info()
        
        @app.post("/generate")
        async def generate(request: dict):
            prompt = request.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt required")
            
            result = engine.generate(
                prompts=prompt,
                max_tokens=request.get("max_tokens", args.max_tokens),
                temperature=request.get("temperature", args.temperature),
                top_p=request.get("top_p", args.top_p),
            )[0]
            
            return result
        
        @app.post("/batch_generate")
        async def batch_generate(request: dict):
            prompts = request.get("prompts", [])
            if not prompts:
                raise HTTPException(status_code=400, detail="prompts required")
            
            results = engine.generate(
                prompts=prompts,
                max_tokens=request.get("max_tokens", args.max_tokens),
                temperature=request.get("temperature", args.temperature),
                top_p=request.get("top_p", args.top_p),
            )
            
            return {"results": results}
        
        @app.post("/stream_generate")
        async def stream_generate(request: dict):
            prompt = request.get("prompt")
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt required")
            
            async def generate_stream():
                async for chunk in engine.generate_with_streaming(
                    prompts=prompt,
                    max_tokens=request.get("max_tokens", args.max_tokens),
                    temperature=request.get("temperature", args.temperature),
                    top_p=request.get("top_p", args.top_p),
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            return generate_stream()
        
        print(f"\n{'='*60}")
        print(f"PiscesL1 vLLM API Server")
        print(f"{'='*60}")
        print(f"Port: {args.port}")
        print(f"URL: http://localhost:{args.port}")
        print(f"\nEndpoints:")
        print(f"  GET  /health  - Health check")
        print(f"  GET  /info    - Model info")
        print(f"  POST /generate - Single prompt")
        print(f"  POST /batch_generate - Batch prompts")
        print(f"  POST /stream_generate - Streaming")
        print(f"{'='*60}\n")
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        
    except ImportError as e:
        logger.error(f"Failed to import server dependencies: {e}")
        logger.info("Install with: pip install fastapi uvicorn")
        raise


def run_benchmark(engine: PiscesLxVLLMEngine, args) -> None:
    """Run performance benchmark."""
    print(f"\n{'='*60}")
    print("PiscesL1 vLLM Benchmark")
    print(f"{'='*60}")
    
    benchmark_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of artificial intelligence?",
        "Describe the process of photosynthesis.",
        "How does a neural network work?",
        "What is the meaning of life?",
        "Explain the theory of relativity.",
        "Write a poem about the sea.",
        "What are the main causes of climate change?",
    ] * (args.prompts // 10 + 1)
    benchmark_prompts = benchmark_prompts[:args.prompts]
    
    stats = benchmark_vllm_inference(
        engine=engine,
        prompts=benchmark_prompts,
        warmup_runs=3,
        benchmark_runs=10,
        max_tokens=512,
    )
    
    print(f"\n{'='*60}")
    print("Benchmark Results")
    print(f"{'='*60}")
    print(f"Prompts: {stats['num_prompts']}")
    print(f"Runs: {stats['benchmark_runs']}")
    print(f"\nLatency:")
    print(f"  Mean: {stats['latency']['mean_ms']:.2f}ms")
    print(f"  Min:  {stats['latency']['min_ms']:.2f}ms")
    print(f"  Max:  {stats['latency']['max_ms']:.2f}ms")
    print(f"\nThroughput:")
    print(f"  Mean: {stats['throughput']['mean_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Min:  {stats['throughput']['min_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Max:  {stats['throughput']['max_tokens_per_sec']:.2f} tokens/sec")
    print(f"\nTokens/prompt: {stats['tokens_per_prompt']:.2f}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("PiscesL1 vLLM Inference Runner")
    print(f"{'='*60}")
    
    config = PiscesLxVLLMConfig.from_args(args)
    engine = create_vllm_engine(config)
    
    try:
        engine.initialize()
        
        model_info = engine.get_model_info()
        print(f"\nModel loaded:")
        print(f"  Path: {model_info.get('model_path', args.model_path)}")
        print(f"  Tensor parallel: {model_info.get('tensor_parallel_size', args.tensor_parallel)}")
        print(f"  Max length: {model_info.get('max_model_len', args.max_len)}")
        print(f"  vLLM version: {model_info.get('vllm_version', 'unknown')}")
        
        if args.prompt:
            run_single_prompt(engine, args)
        elif args.batch:
            run_batch_inference(engine, args)
        elif args.interactive:
            run_interactive_server(engine, args)
        elif args.benchmark:
            run_benchmark(engine, args)
        else:
            print("\nNo mode specified. Use --help for usage information.")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
