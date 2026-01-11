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
Multi-modal vLLM integration for PiscesL1.

This module provides multi-modal inference acceleration using vLLM's
PagedAttention with support for:
- Text input generation
- Image-text understanding
- Video-text processing
- Audio-text processing
- Cross-modal attention fusion
"""

import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from utils import PiscesLxCoreConfigManager

logger = PiscesLxCoreLog("pisceslx.tools.infer.multimodal_vllm")


@dataclass
class PiscesLxMultiModalVLLMConfig:
    """Multi-modal vLLM inference configuration."""
    
    model_path: str = "./checkpoints/ruchbah"
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    
    max_model_len: int = 32768
    max_num_seqs: int = 256
    max_input_len: int = 16384
    
    gpu_memory_utilization: float = 0.9
    
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    
    disable_log_requests: bool = False
    
    quantization: Optional[str] = None
    
    enable_image: bool = True
    enable_video: bool = True
    enable_audio: bool = True
    
    def __post_init__(self):
        if self.dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = "bfloat16"
            else:
                self.dtype = "float16"


class PiscesLxMultiModalVLLMEngine:
    """Multi-modal vLLM-powered inference engine for PiscesL1.
    
    Provides high-performance multi-modal inference with:
    - PagedAttention for efficient KV cache management
    - Continuous batching for maximum throughput
    - Multi-modal input processing (image, video, audio)
    - Tensor parallelism support
    """
    
    def __init__(self, config: Optional[PiscesLxMultiModalVLLMConfig] = None):
        """Initialize multi-modal vLLM engine.
        
        Args:
            config: Multi-modal vLLM configuration. If None, uses defaults.
        """
        self.config = config or PiscesLxMultiModalVLLMConfig()
        self._engine = None
        self._tokenizer = None
        self._initialized = False
        
        self._check_vllm_available()
        
        self.vision_processor = None
        self.audio_processor = None
    
    def _check_vllm_available(self):
        """Check if vLLM is available."""
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
    
    def _load_processors(self):
        """Load vision and audio processors."""
        try:
            from transformers import AutoProcessor
            
            logger.info("Loading vision and audio processors...")
            
            try:
                self.vision_processor = AutoProcessor.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True,
                )
                logger.info("Vision processor loaded")
            except Exception as e:
                logger.warning(f"Failed to load vision processor: {e}")
            
            try:
                self.audio_processor = AutoProcessor.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True,
                )
                logger.info("Audio processor loaded")
            except Exception as e:
                logger.warning(f"Failed to load audio processor: {e}")
                
        except ImportError:
            logger.warning("Transformers processors not available")
    
    def initialize(self) -> None:
        """Initialize the vLLM engine and load model."""
        if self._initialized:
            logger.warning("Multi-modal vLLM engine already initialized")
            return
        
        logger.info(f"Initializing multi-modal vLLM engine from: {self.config.model_path}")
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            from vllm.multimodal import MultiModalConfig
            from vllm.multimodal.image import ImageFeatureExtractor
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=4096,
            )
            
            mm_config = MultiModalConfig()
            
            if self.config.enable_image:
                mm_config.image_feature_extractor = ImageFeatureExtractor(
                    proc_type=ImageFeatureExtractor.FeatureExtractorType.TRANSFORMERS,
                )
            
            self._engine = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                max_model_len=self.config.max_model_len,
                max_num_seqs=self.config.max_num_seqs,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                enforce_eager=self.config.enforce_eager,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                quantization=self.config.quantization,
                multimodal_config=mm_config,
            )
            
            self._sampling_params = sampling_params
            
            self._load_processors()
            
            init_time = time.time() - start_time
            logger.success(f"Multi-modal vLLM engine initialized in {init_time:.2f}s")
            logger.info(f"  - Tensor parallelism: {self.config.tensor_parallel_size}")
            logger.info(f"  - Max model length: {self.config.max_model_len}")
            logger.info(f"  - Data type: {self.config.dtype}")
            logger.info(f"  - Image support: {self.config.enable_image}")
            logger.info(f"  - Video support: {self.config.enable_video}")
            logger.info(f"  - Audio support: {self.config.enable_audio}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-modal vLLM engine: {e}")
            raise
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        images: Optional[Union[str, List[str], torch.Tensor]] = None,
        videos: Optional[Union[str, List[str]]] = None,
        audio: Optional[Union[str, List[str], torch.Tensor]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """Generate completions for given prompts with multi-modal inputs.
        
        Args:
            prompts: Input prompts (single or batch)
            images: Input images (paths, URLs, or tensors)
            videos: Input video paths
            audio: Input audio (paths or tensors)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            List of generation results with generated text and metadata
        """
        if not self._initialized:
            self.initialize()
        
        from vllm import SamplingParams as VLLMSamplingParams
        
        if isinstance(prompts, str):
            prompts = [prompts]
            images = [images] if images is not None else None
            videos = [videos] if videos is not None else None
            audio = [audio] if audio is not None else None
        
        mm_inputs = []
        
        for i, prompt in enumerate(prompts):
            mm_data = {"prompt": prompt}
            
            if images is not None and i < len(images):
                image_data = images[i]
                if image_data is not None:
                    if isinstance(image_data, str):
                        if image_data.startswith("http"):
                            mm_data["image"] = image_data
                        else:
                            mm_data["image"] = self._load_image(image_data)
                    elif torch.is_tensor(image_data):
                        mm_data["image"] = image_data
            
            if videos is not None and i < len(videos):
                video_data = videos[i]
                if video_data is not None:
                    mm_data["video"] = video_data
            
            if audio is not None and i < len(audio):
                audio_data = audio[i]
                if audio_data is not None:
                    mm_data["audio"] = audio_data
            
            mm_inputs.append(mm_data)
        
        sampling_params = VLLMSamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        start_time = time.time()
        outputs = self._engine.generate(mm_inputs, sampling_params)
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
            
            if images is not None:
                result["has_image"] = True
            if videos is not None:
                result["has_video"] = True
            if audio is not None:
                result["has_audio"] = True
            
            results.append(result)
        
        logger.debug(f"Generated {len(results)} multi-modal responses in {total_time:.2f}s")
        
        return results
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for model input.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Preprocessed image tensor.
        """
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert("RGB")
            
            if self.vision_processor is not None:
                image_inputs = self.vision_processor(
                    images=image,
                    return_tensors="pt",
                )
                return image_inputs.get("pixel_values", image)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def generate_with_streaming(
        self,
        prompts: Union[str, List[str]],
        images: Optional[Union[str, List[str]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        """Generate completions with streaming output.
        
        Args:
            prompts: Input prompts (single or batch)
            images: Input images
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
        
        mm_inputs = []
        for i, prompt in enumerate(prompts):
            mm_data = {"prompt": prompt}
            if images is not None and i < len(images):
                mm_data["image"] = images[i]
            mm_inputs.append(mm_data)
        
        for output in self._engine.generate(mm_inputs, sampling_params, use_tqdm=True):
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
                - image: optional image path
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
        
        mm_inputs = []
        sampling_params_list = []
        
        for req in requests:
            mm_data = {"prompt": req["prompt"]}
            
            if "image" in req:
                mm_data["image"] = req["image"]
            
            mm_inputs.append(mm_data)
            sampling_params_list.append(
                VLLMSamplingParams(
                    temperature=req.get("temperature", 0.7),
                    top_p=req.get("top_p", 0.95),
                    max_tokens=req.get("max_tokens", max_tokens),
                )
            )
        
        outputs = self._engine.generate(mm_inputs, sampling_params_list)
        
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._initialized:
            return {}
        
        return {
            "vllm_version": self.vllm_version,
            "model_path": self.config.model_path,
            "dtype": self.config.dtype,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "image_support": self.config.enable_image,
            "video_support": self.config.enable_video,
            "audio_support": self.config.enable_audio,
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
        logger.info("Multi-modal vLLM engine shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


def create_multimodal_vllm_engine(
    config: Optional[PiscesLxMultiModalVLLMConfig] = None,
) -> PiscesLxMultiModalVLLMEngine:
    """Factory function to create multi-modal vLLM engine.
    
    Args:
        config: Multi-modal vLLM configuration
        
    Returns:
        Initialized multi-modal vLLM engine
    """
    engine = PiscesLxMultiModalVLLMEngine(config)
    return engine


def benchmark_multimodal_inference(
    engine: PiscesLxMultiModalVLLMEngine,
    prompts: List[str],
    images: Optional[List[str]] = None,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """Benchmark multi-modal inference performance.
    
    Args:
        engine: Multi-modal vLLM engine instance
        prompts: List of test prompts
        images: Optional list of test image paths
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
        engine.generate(prompts[0], images=images[0] if images else None, max_tokens=max_tokens)
    
    latencies = []
    throughputs = []
    
    for _ in range(benchmark_runs):
        start = time.time()
        results = engine.generate(prompts, images=images, max_tokens=max_tokens)
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
        "image_support": engine.config.enable_image,
        "video_support": engine.config.enable_video,
        "audio_support": engine.config.enable_audio,
    }
    
    logger.success("Benchmark complete:")
    logger.success(f"  - Mean latency: {stats['latency']['mean_ms']:.2f}ms")
    logger.success(f"  - Mean throughput: {stats['throughput']['mean_tokens_per_sec']:.2f} tokens/sec")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-modal vLLM inference for PiscesL1")
    
    parser.add_argument("--model_path", type=str, default="./checkpoints/ruchbah")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=32768)
    parser.add_argument("--gpu_memory", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--image", type=str, default=None)
    
    args = parser.parse_args()
    
    config = PiscesLxMultiModalVLLMConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_memory,
    )
    
    engine = create_multimodal_vllm_engine(config)
    engine.initialize()
    
    if args.image:
        result = engine.generate(args.prompt, images=args.image)
        print(f"Generated: {result[0]['generated_text']}")
    else:
        result = engine.generate(args.prompt)
        print(f"Generated: {result[0]['generated_text']}")
    
    engine.shutdown()
