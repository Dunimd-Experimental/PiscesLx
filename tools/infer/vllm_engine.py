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
vLLM Integration for PiscesL1 Ruchbah Model.

This module provides high-performance inference acceleration using vLLM's
PagedAttention, continuous batching, and optimized CUDA kernels.

Key Features:
- PagedAttention for efficient KV cache management
- Continuous batching for maximum throughput
- Tensor parallelism support
- Speculative decoding integration
"""

import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.tools.infer.vllm_engine")


@dataclass
class PiscesLxVLLMConfig:
    """vLLM inference configuration for PiscesL1."""
    
    model_path: str = "./checkpoints/ruchbah"
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    
    max_model_len: int = 32768
    max_num_seqs: int = 256
    
    gpu_memory_utilization: float = 0.9
    
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    
    disable_log_requests: bool = False
    max_log_len: int = None
    
    enable_lora: bool = False
    lora_path: str = None
    lora_modules: List[str] = field(default_factory=lambda: ["decoder", "lm_head"])
    
    quantization: Optional[str] = None
    
    def __post_init__(self):
        if self.dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = "bfloat16"
            else:
                self.dtype = "float16"
    
    @classmethod
    def from_args(cls, args: Any) -> "PiscesLxVLLMConfig":
        config_dict = {}
        
        if getattr(args, "model_path", None):
            config_dict["model_path"] = args.model_path
        
        if getattr(args, "tensor_parallel", None):
            config_dict["tensor_parallel_size"] = args.tensor_parallel
        
        if getattr(args, "max_len", None):
            config_dict["max_model_len"] = args.max_len
        
        if getattr(args, "gpu_memory", None):
            config_dict["gpu_memory_utilization"] = args.gpu_memory
        
        if getattr(args, "quantize", None):
            config_dict["quantization"] = args.quantize
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "quantization": self.quantization,
        }


class PiscesLxVLLMEngine:
    """vLLM-powered inference engine for PiscesL1.
    
    Provides high-performance inference with:
    - PagedAttention for efficient KV cache management
    - Continuous batching for maximum throughput
    - Speculative decoding support
    - Multi-GPU tensor parallelism
    """
    
    def __init__(self, config: Optional[PiscesLxVLLMConfig] = None):
        """Initialize vLLM engine.
        
        Args:
            config: vLLM configuration. If None, uses defaults.
        """
        self.config = config or PiscesLxVLLMConfig()
        self._engine = None
        self._tokenizer = None
        self._initialized = False
        
        self._check_vllm_available()
    
    def _check_vllm_available(self):
        """Check if vLLM is available and install if needed."""
        try:
            import vllm
            self.vllm_version = vllm.__version__
            logger.info(f"vLLM version: {self.vllm_version}")
        except ImportError:
            logger.warning("vLLM not found. Installing...")
            self._install_vllm()
    
    def _install_vllm(self):
        """Install vLLM."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "vllm>=0.6.0"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Failed to install vLLM: {result.stderr}")
            raise ImportError("vLLM installation failed")
        
        import vllm
        self.vllm_version = vllm.__version__
        logger.success(f"vLLM installed successfully: {self.vllm_version}")
    
    def initialize(self) -> None:
        """Initialize the vLLM engine and load model."""
        if self._initialized:
            logger.warning("vLLM engine already initialized")
            return
        
        logger.info(f"Initializing vLLM engine from: {self.config.model_path}")
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
            )
            
            engine_config = self.config.to_dict()
            
            self._engine = LLM(
                model=self.config.model_path,
                **engine_config
            )
            
            self._sampling_params = sampling_params
            
            init_time = time.time() - start_time
            logger.success(f"vLLM engine initialized in {init_time:.2f}s")
            logger.info(f"  - Tensor parallelism: {self.config.tensor_parallel_size}")
            logger.info(f"  - Max model length: {self.config.max_model_len}")
            logger.info(f"  - Data type: {self.config.dtype}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate completions for given prompts.
        
        Args:
            prompts: Input prompts (single or batch)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop_tokens: Stop generation on these tokens
            
        Returns:
            List of generation results with generated text and metadata
        """
        if not self._initialized:
            self.initialize()
        
        from vllm import SamplingParams as VLLMSamplingParams
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        sampling_params = VLLMSamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
        )
        
        start_time = time.time()
        outputs = self._engine.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        results = []
        for i, output in enumerate(outputs):
            result = {
                "prompt": prompts[i],
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "num_tokens": len(output.outputs[0].token_ids),
                "latency_ms": total_time * 1000,
                "tokens_per_second": len(output.outputs[0].token_ids) / total_time if total_time > 0 else 0,
            }
            results.append(result)
        
        logger.debug(f"Generated {len(results)} responses in {total_time:.2f}s")
        
        return results
    
    def generate_with_streaming(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        """Generate completions with streaming output.
        
        Args:
            prompts: Input prompts (single or batch)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Yields:
            Streaming output chunks
        """
        if not self._initialized:
            self.initialize()
        
        from vllm import SamplingParams as VLLMSamplingParams
        
        sampling_params = VLLMSamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        for output in self._engine.generate(prompts, sampling_params, use_tqdm=True):
            yield {
                "text": output.outputs[0].text,
                "token_id": output.outputs[0].token_ids[-1] if output.outputs[0].token_ids else None,
                "finished": output.outputs[0].finish_reason is not None,
            }
    
    def batch_generate(
        self,
        requests: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> List[Dict[str, Any]]:
        """Batch generate for multiple requests with different parameters.
        
        Args:
            requests: List of request dicts with keys:
                - prompt: str
                - temperature: float (optional)
                - top_p: float (optional)
                - max_tokens: int (optional)
            max_tokens: Default max tokens
            
        Returns:
            List of generation results
        """
        if not self._initialized:
            self.initialize()
        
        from vllm import SamplingParams as VLLMSamplingParams
        
        prompts = []
        sampling_params_list = []
        
        for req in requests:
            prompts.append(req["prompt"])
            sampling_params_list.append(
                VLLMSamplingParams(
                    temperature=req.get("temperature", 0.7),
                    top_p=req.get("top_p", 0.95),
                    max_tokens=req.get("max_tokens", max_tokens),
                )
            )
        
        outputs = self._engine.generate(prompt_token_ids=None, sampling_params=sampling_params_list)
        
        results = []
        for i, output in enumerate(outputs):
            results.append({
                "prompt": requests[i]["prompt"],
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "num_tokens": len(output.outputs[0].token_ids),
            })
        
        return results
    
    def get_tokenizer(self):
        """Get the tokenizer used by the engine."""
        if not self._initialized:
            self.initialize()
        return self._engine.get_tokenizer()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not self._initialized:
            self.initialize()
        
        tokenizer = self.get_tokenizer()
        return len(tokenizer.encode(text))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if not self._initialized:
            return {}
        
        return {
            "vllm_version": self.vllm_version,
            "model_path": self.config.model_path,
            "dtype": self.config.dtype,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
        }
    
    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        if self._engine is not None:
            del self._engine
            self._engine = None
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        self._initialized = False
        logger.info("vLLM engine shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class PiscesLxVLLMWrapper:
    """Wrapper for integrating vLLM with PiscesL1 native model.
    
    This allows using vLLM acceleration with the RuchbahModel
    when the model is saved in vLLM-compatible format.
    """
    
    def __init__(self, engine: PiscesLxVLLMEngine):
        """Initialize wrapper.
        
        Args:
            engine: vLLM engine instance
        """
        self.engine = engine
        self._model = None
    
    def load_native_model(self, model_path: str) -> nn.Module:
        """Load native PiscesL1 model.
        
        Args:
            model_path: Path to PiscesL1 checkpoint
            
        Returns:
            Loaded RuchbahModel
        """
        from model.config import RuchbahConfig
        from model.modeling import RuchbahModel
        
        logger.info(f"Loading native PiscesL1 model from: {model_path}")
        
        config = RuchbahConfig.from_pretrained(model_path)
        self._model = RuchbahModel.from_pretrained(model_path)
        self._model.eval()
        
        logger.success("Native model loaded")
        
        return self._model
    
    def export_for_vllm(self, output_path: str, save_tensor_parallel: bool = False) -> str:
        """Export native model to vLLM-compatible format.
        
        Args:
            output_path: Output directory path
            save_tensor_parallel: Save with tensor parallelism shards
            
        Returns:
            Path to exported model
        """
        if self._model is None:
            raise ValueError("No native model loaded. Call load_native_model first.")
        
        import os
        os.makedirs(output_path, exist_ok=True)
        
        self._model.save_pretrained(
            output_path,
            save_tensor_parallel=save_tensor_parallel,
        )
        
        logger.success(f"Model exported to: {output_path}")
        
        return output_path


def create_vllm_engine(config: Optional[PiscesLxVLLMConfig] = None) -> PiscesLxVLLMEngine:
    """Factory function to create vLLM engine.
    
    Args:
        config: vLLM configuration
        
    Returns:
        Initialized vLLM engine
    """
    engine = PiscesLxVLLMEngine(config)
    return engine


def benchmark_vllm_inference(
    engine: PiscesLxVLLMEngine,
    prompts: List[str],
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Benchmark vLLM inference performance.
    
    Args:
        engine: vLLM engine instance
        prompts: List of test prompts
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        max_tokens: Max tokens to generate
        
    Returns:
        Benchmark results dictionary
    """
    logger.info(f"Starting benchmark with {len(prompts)} prompts")
    
    if not engine._initialized:
        engine.initialize()
    
    for _ in range(warmup_runs):
        engine.generate(prompts[0], max_tokens=max_tokens)
    
    latencies = []
    throughputs = []
    
    for _ in range(benchmark_runs):
        start = time.time()
        results = engine.generate(prompts, max_tokens=max_tokens)
        elapsed = time.time() - start
        
        total_tokens = sum(r["num_tokens"] for r in results)
        latencies.append(elapsed)
        throughputs.append(total_tokens / elapsed)
    
    stats = {
        "num_prompts": len(prompts),
        "benchmark_runs": benchmark_runs,
        "warmup_runs": warmup_runs,
        "latency": {
            "mean_ms": sum(latencies) / len(latencies) * 1000,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
        },
        "throughput": {
            "mean_tokens_per_sec": sum(throughputs) / len(throughputs),
            "min_tokens_per_sec": min(throughputs),
            "max_tokens_per_sec": max(throughputs),
        },
        "tokens_per_prompt": sum(r["num_tokens"] for r in results) / len(results),
    }
    
    logger.success("Benchmark complete:")
    logger.success(f"  - Mean latency: {stats['latency']['mean_ms']:.2f}ms")
    logger.success(f"  - Mean throughput: {stats['throughput']['mean_tokens_per_sec']:.2f} tokens/sec")
    
    return stats
