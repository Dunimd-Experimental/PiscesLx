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
Inference Configuration Management System

This module provides comprehensive configuration management for the inference
pipeline. It uses Python dataclasses to define type-safe, hierarchical
configuration structures that can be easily serialized and deserialized.

Configuration Hierarchy:
    InferenceConfig (root)
    ├── ModelConfig: Model loading and tokenizer settings
    ├── GenerationConfig: Text generation parameters
    ├── AccelerationConfig: Inference acceleration options
    ├── QuantizationConfig: Model quantization settings
    └── SamplingConfig: Advanced sampling strategies

Features:
    - Type-safe configuration with automatic validation
    - JSON/YAML serialization support
    - Default value management
    - Configuration inheritance and composition

Usage Examples:
    Creating configuration programmatically:
    >>> config = InferenceConfig(
    ...     model=ModelConfig(model_path="meta-llama/Llama-2-7b"),
    ...     generation=GenerationConfig(temperature=0.8, max_new_tokens=512),
    ...     acceleration=AccelerationConfig(use_vllm=True, tensor_parallel_size=2)
    ... )
    
    Loading from JSON:
    >>> config = InferenceConfig.load_from_json("config.json")
    
    Converting to dictionary:
    >>> config_dict = config.to_dict()

See Also:
    - tools.infer.core: Core inference operator using these configs
    - tools.infer.orchestrator: High-level orchestration with config management
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import yaml


@dataclass
class ModelConfig:
    """
    Model Configuration
    
    Defines parameters for model loading, tokenizer initialization,
    and model-specific settings.
    
    Attributes:
        model_path: Path to model directory or HuggingFace model ID.
        model_type: Model architecture type ('auto', 'llama', 'gpt2', etc.).
        tokenizer_path: Path to tokenizer (defaults to model_path if None).
        trust_remote_code: Whether to trust and execute remote code from model.
    
    Example:
        >>> model_config = ModelConfig(
        ...     model_path="meta-llama/Llama-2-7b-hf",
        ...     model_type="llama",
        ...     trust_remote_code=False
        ... )
    """
    model_path: str = ""
    model_type: str = "auto"
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class GenerationConfig:
    """
    Text Generation Configuration
    
    Controls the behavior of text generation including sampling strategies,
    length constraints, and repetition handling.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).
        top_p: Nucleus sampling parameter (0.0-1.0).
        top_k: Top-k sampling parameter (limits vocabulary to k most likely tokens).
        repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition).
        do_sample: Whether to use sampling (False = greedy decoding).
        num_beams: Number of beams for beam search (1 = no beam search).
        early_stopping: Whether to stop when all beams reach EOS.
    
    Example:
        >>> gen_config = GenerationConfig(
        ...     max_new_tokens=512,
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1
        ... )
    """
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False


@dataclass
class AccelerationConfig:
    """
    Inference Acceleration Configuration
    
    Configures acceleration techniques for high-performance inference.
    
    Attributes:
        use_vllm: Whether to use VLLM inference engine.
        use_speculative_decoding: Whether to enable speculative decoding.
        spec_gamma: Number of tokens to speculate in speculative decoding.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of pipeline stages.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
        enforce_eager: Whether to disable CUDA graph optimization.
        max_num_seqs: Maximum number of concurrent sequences.
        min_new_tokens: Minimum number of new tokens to generate.
        use_ring_attention: Whether to enable Ring Attention for ultra-long context.
        ring_block_size: Block size for Ring Attention computation.
        max_sequence_length: Maximum sequence length for Ring Attention.
        use_flash_attention_3: Whether to enable FlashAttention-3 for H100.
        use_fp8: Whether to use FP8 quantization in FlashAttention-3.
        use_mamba: Whether to enable Mamba/SSM for linear complexity.
    
    Example:
        >>> accel_config = AccelerationConfig(
        ...     use_vllm=True,
        ...     tensor_parallel_size=2,
        ...     gpu_memory_utilization=0.9,
        ...     use_ring_attention=True,
        ...     max_sequence_length=1048576
        ... )
    """
    use_vllm: bool = False
    use_speculative_decoding: bool = False
    spec_gamma: int = 4
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    max_num_seqs: int = 256
    min_new_tokens: int = 16
    use_ring_attention: bool = False
    ring_block_size: int = 4096
    max_sequence_length: int = 1048576
    use_flash_attention_3: bool = True
    use_fp8: bool = True
    use_mamba: bool = False


@dataclass
class QuantizationConfig:
    """
    Model Quantization Configuration
    
    Defines parameters for model quantization to reduce memory usage
    and improve inference speed.
    
    Attributes:
        enable_quantization: Whether to apply quantization.
        quant_method: Quantization method ('awq', 'gptq', 'squeezellm', 'fp8').
        bits: Number of bits for quantization (4 or 8).
        group_size: Group size for quantization (affects accuracy/speed tradeoff).
        symmetric: Whether to use symmetric quantization.
        zero_point: Whether to use zero-point quantization.
    
    Example:
        >>> quant_config = QuantizationConfig(
        ...     enable_quantization=True,
        ...     quant_method="awq",
        ...     bits=4,
        ...     group_size=128
        ... )
    """
    enable_quantization: bool = False
    quant_method: str = "awq"  # awq, gptq, squeezellm, fp8
    bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    zero_point: bool = True


@dataclass
class SamplingConfig:
    """
    Advanced Sampling Configuration
    
    Configures advanced sampling strategies beyond basic temperature/top-p.
    
    Attributes:
        use_advanced_sampling: Whether to enable advanced sampling.
        nucleus_sampling: Whether to use nucleus (top-p) sampling.
        contrastive_search: Whether to use contrastive search decoding.
        beam_search: Whether to use beam search.
        diverse_beam_search: Whether to use diverse beam search.
    
    Note:
        Some sampling methods are mutually exclusive. Priority order:
        contrastive_search > diverse_beam_search > beam_search > nucleus_sampling
    """
    use_advanced_sampling: bool = True
    nucleus_sampling: bool = True
    contrastive_search: bool = False
    beam_search: bool = False
    diverse_beam_search: bool = False


@dataclass
class MoEConfig:
    """
    MoE Runtime Configuration
    
    Configures Mixture of Experts (MoE) runtime behavior during inference,
    including routing parameters, caching, and load balancing.
    
    Attributes:
        enable_moe: Whether to enable MoE runtime optimizations.
        routing_temp: Temperature for softmax routing (higher = more random).
        top_k: Number of experts to route each token to.
        capacity_factor: Multiplier for expert capacity.
        min_capacity: Minimum tokens per expert.
        enable_expert_caching: Whether to cache expert computations.
        expert_cache_size: Maximum number of cached expert states.
        enable_load_balancing: Whether to compute load balancing auxiliary loss.
        load_balance_alpha: Weight for load balancing loss.
        enable_adaptive_temp: Whether to adaptively adjust routing temperature.
        adaptive_temp_step: Temperature adjustment step size.
        adaptive_temp_interval: Steps between temperature adjustments.
        adaptive_temp_cap: Maximum routing temperature.
        enable_prefix_caching: Whether to cache KV caches for prefix tokens.
        priority_routing: Enable priority-based expert routing.
        drop_tokens: Whether to drop tokens when exceeding capacity.
    
    Example:
        >>> moe_config = MoEConfig(
        ...     enable_moe=True,
        ...     routing_temp=1.12,
        ...     top_k=2,
        ...     enable_expert_caching=True
        ... )
    """
    enable_moe: bool = False
    routing_temp: float = 1.12
    top_k: int = 2
    capacity_factor: float = 1.0
    min_capacity: int = 4
    enable_expert_caching: bool = True
    expert_cache_size: int = 10000
    enable_load_balancing: bool = True
    load_balance_alpha: float = 0.01
    enable_adaptive_temp: bool = True
    adaptive_temp_step: float = 0.03
    adaptive_temp_interval: int = 16
    adaptive_temp_cap: float = 1.30
    enable_prefix_caching: bool = True
    priority_routing: bool = False
    drop_tokens: bool = True


@dataclass
class InferenceConfig:
    """
    Complete Inference Configuration
    
    Root configuration class that aggregates all inference-related settings.
    Provides methods for serialization and deserialization.
    
    Attributes:
        model: Model configuration.
        generation: Generation configuration.
        device: Device to use ('cuda', 'cpu', 'auto').
        dtype: Data type ('auto', 'float16', 'bfloat16', 'float32').
        memory_limit: GPU memory limit in MB (None = no limit).
        acceleration: Acceleration configuration.
        quantization: Quantization configuration.
        sampling: Sampling configuration.
        moe: MoE runtime configuration.
        batch_size: Default batch size for inference.
        max_batch_size: Maximum batch size.
        enable_prefix_caching: Whether to enable prefix caching.
        enable_chunked_prefill: Whether to enable chunked prefill.
        enable_monitoring: Whether to enable monitoring.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        metrics_export_interval: Interval for metrics export.
    
    Example:
        >>> config = InferenceConfig(
        ...     model=ModelConfig(model_path="meta-llama/Llama-2-7b"),
        ...     generation=GenerationConfig(temperature=0.7),
        ...     acceleration=AccelerationConfig(use_vllm=True)
        ... )
        >>> config.save_to_json("inference_config.json")
    """
    # Base configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Hardware configuration
    device: str = "cuda"
    dtype: str = "auto"  # auto, float16, bfloat16, float32
    memory_limit: Optional[int] = None  # MB
    
    # Acceleration configuration
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    
    # Quantization configuration
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # Sampling configuration
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # MoE configuration
    moe: MoEConfig = field(default_factory=MoEConfig)
    
    # Batching configuration
    batch_size: int = 1
    max_batch_size: int = 32
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    
    # Monitoring configuration
    enable_monitoring: bool = True
    log_level: str = "INFO"
    metrics_export_interval: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "model": {
                "model_path": self.model.model_path,
                "model_type": self.model.model_type,
                "tokenizer_path": self.model.tokenizer_path,
                "trust_remote_code": self.model.trust_remote_code
            },
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "repetition_penalty": self.generation.repetition_penalty,
                "do_sample": self.generation.do_sample,
                "num_beams": self.generation.num_beams,
                "early_stopping": self.generation.early_stopping
            },
            "device": self.device,
            "dtype": self.dtype,
            "memory_limit": self.memory_limit,
            "acceleration": {
                "use_vllm": self.acceleration.use_vllm,
                "use_speculative_decoding": self.acceleration.use_speculative_decoding,
                "spec_gamma": self.acceleration.spec_gamma,
                "tensor_parallel_size": self.acceleration.tensor_parallel_size,
                "pipeline_parallel_size": self.acceleration.pipeline_parallel_size,
                "gpu_memory_utilization": self.acceleration.gpu_memory_utilization,
                "enforce_eager": self.acceleration.enforce_eager,
                "max_num_seqs": self.acceleration.max_num_seqs,
                "min_new_tokens": self.acceleration.min_new_tokens
            },
            "quantization": {
                "enable_quantization": self.quantization.enable_quantization,
                "quant_method": self.quantization.quant_method,
                "bits": self.quantization.bits,
                "group_size": self.quantization.group_size,
                "symmetric": self.quantization.symmetric,
                "zero_point": self.quantization.zero_point
            },
            "sampling": {
                "use_advanced_sampling": self.sampling.use_advanced_sampling,
                "nucleus_sampling": self.sampling.nucleus_sampling,
                "contrastive_search": self.sampling.contrastive_search,
                "beam_search": self.sampling.beam_search,
                "diverse_beam_search": self.sampling.diverse_beam_search
            },
            "moe": {
                "enable_moe": self.moe.enable_moe,
                "routing_temp": self.moe.routing_temp,
                "top_k": self.moe.top_k,
                "capacity_factor": self.moe.capacity_factor,
                "min_capacity": self.moe.min_capacity,
                "enable_expert_caching": self.moe.enable_expert_caching,
                "expert_cache_size": self.moe.expert_cache_size,
                "enable_load_balancing": self.moe.enable_load_balancing,
                "load_balance_alpha": self.moe.load_balance_alpha,
                "enable_adaptive_temp": self.moe.enable_adaptive_temp,
                "adaptive_temp_step": self.moe.adaptive_temp_step,
                "adaptive_temp_interval": self.moe.adaptive_temp_interval,
                "adaptive_temp_cap": self.moe.adaptive_temp_cap,
                "enable_prefix_caching": self.moe.enable_prefix_caching,
                "priority_routing": self.moe.priority_routing,
                "drop_tokens": self.moe.drop_tokens
            },
            "batching": {
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "enable_prefix_caching": self.enable_prefix_caching,
                "enable_chunked_prefill": self.enable_chunked_prefill
            },
            "monitoring": {
                "enable_monitoring": self.enable_monitoring,
                "log_level": self.log_level,
                "metrics_export_interval": self.metrics_export_interval
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
        
        Returns:
            InferenceConfig instance.
        """
        config = cls()
        
        # Model configuration
        if "model" in config_dict:
            model_dict = config_dict["model"]
            config.model.model_path = model_dict.get("model_path", config.model.model_path)
            config.model.model_type = model_dict.get("model_type", config.model.model_type)
            config.model.tokenizer_path = model_dict.get("tokenizer_path", config.model.tokenizer_path)
            config.model.trust_remote_code = model_dict.get("trust_remote_code", config.model.trust_remote_code)
        
        # Generation configuration
        if "generation" in config_dict:
            gen_dict = config_dict["generation"]
            config.generation.max_new_tokens = gen_dict.get("max_new_tokens", config.generation.max_new_tokens)
            config.generation.temperature = gen_dict.get("temperature", config.generation.temperature)
            config.generation.top_p = gen_dict.get("top_p", config.generation.top_p)
            config.generation.top_k = gen_dict.get("top_k", config.generation.top_k)
            config.generation.repetition_penalty = gen_dict.get("repetition_penalty", config.generation.repetition_penalty)
            config.generation.do_sample = gen_dict.get("do_sample", config.generation.do_sample)
            config.generation.num_beams = gen_dict.get("num_beams", config.generation.num_beams)
            config.generation.early_stopping = gen_dict.get("early_stopping", config.generation.early_stopping)
        
        # Base configuration
        config.device = config_dict.get("device", config.device)
        config.dtype = config_dict.get("dtype", config.dtype)
        config.memory_limit = config_dict.get("memory_limit", config.memory_limit)
        
        # Acceleration configuration
        if "acceleration" in config_dict:
            accel_dict = config_dict["acceleration"]
            config.acceleration.use_vllm = accel_dict.get("use_vllm", config.acceleration.use_vllm)
            config.acceleration.use_speculative_decoding = accel_dict.get("use_speculative_decoding", config.acceleration.use_speculative_decoding)
            config.acceleration.spec_gamma = accel_dict.get("spec_gamma", config.acceleration.spec_gamma)
            config.acceleration.tensor_parallel_size = accel_dict.get("tensor_parallel_size", config.acceleration.tensor_parallel_size)
            config.acceleration.pipeline_parallel_size = accel_dict.get("pipeline_parallel_size", config.acceleration.pipeline_parallel_size)
            config.acceleration.gpu_memory_utilization = accel_dict.get("gpu_memory_utilization", config.acceleration.gpu_memory_utilization)
            config.acceleration.enforce_eager = accel_dict.get("enforce_eager", config.acceleration.enforce_eager)
            config.acceleration.max_num_seqs = accel_dict.get("max_num_seqs", config.acceleration.max_num_seqs)
            config.acceleration.min_new_tokens = accel_dict.get("min_new_tokens", config.acceleration.min_new_tokens)
        
        # Quantization configuration
        if "quantization" in config_dict:
            quant_dict = config_dict["quantization"]
            config.quantization.enable_quantization = quant_dict.get("enable_quantization", config.quantization.enable_quantization)
            config.quantization.quant_method = quant_dict.get("quant_method", config.quantization.quant_method)
            config.quantization.bits = quant_dict.get("bits", config.quantization.bits)
            config.quantization.group_size = quant_dict.get("group_size", config.quantization.group_size)
            config.quantization.symmetric = quant_dict.get("symmetric", config.quantization.symmetric)
            config.quantization.zero_point = quant_dict.get("zero_point", config.quantization.zero_point)
        
        # Sampling configuration
        if "sampling" in config_dict:
            sample_dict = config_dict["sampling"]
            config.sampling.use_advanced_sampling = sample_dict.get("use_advanced_sampling", config.sampling.use_advanced_sampling)
            config.sampling.nucleus_sampling = sample_dict.get("nucleus_sampling", config.sampling.nucleus_sampling)
            config.sampling.contrastive_search = sample_dict.get("contrastive_search", config.sampling.contrastive_search)
            config.sampling.beam_search = sample_dict.get("beam_search", config.sampling.beam_search)
            config.sampling.diverse_beam_search = sample_dict.get("diverse_beam_search", config.sampling.diverse_beam_search)
        
        # MoE configuration
        if "moe" in config_dict:
            moe_dict = config_dict["moe"]
            config.moe.enable_moe = moe_dict.get("enable_moe", config.moe.enable_moe)
            config.moe.routing_temp = moe_dict.get("routing_temp", config.moe.routing_temp)
            config.moe.top_k = moe_dict.get("top_k", config.moe.top_k)
            config.moe.capacity_factor = moe_dict.get("capacity_factor", config.moe.capacity_factor)
            config.moe.min_capacity = moe_dict.get("min_capacity", config.moe.min_capacity)
            config.moe.enable_expert_caching = moe_dict.get("enable_expert_caching", config.moe.enable_expert_caching)
            config.moe.expert_cache_size = moe_dict.get("expert_cache_size", config.moe.expert_cache_size)
            config.moe.enable_load_balancing = moe_dict.get("enable_load_balancing", config.moe.enable_load_balancing)
            config.moe.load_balance_alpha = moe_dict.get("load_balance_alpha", config.moe.load_balance_alpha)
            config.moe.enable_adaptive_temp = moe_dict.get("enable_adaptive_temp", config.moe.enable_adaptive_temp)
            config.moe.adaptive_temp_step = moe_dict.get("adaptive_temp_step", config.moe.adaptive_temp_step)
            config.moe.adaptive_temp_interval = moe_dict.get("adaptive_temp_interval", config.moe.adaptive_temp_interval)
            config.moe.adaptive_temp_cap = moe_dict.get("adaptive_temp_cap", config.moe.adaptive_temp_cap)
            config.moe.enable_prefix_caching = moe_dict.get("enable_prefix_caching", config.moe.enable_prefix_caching)
            config.moe.priority_routing = moe_dict.get("priority_routing", config.moe.priority_routing)
            config.moe.drop_tokens = moe_dict.get("drop_tokens", config.moe.drop_tokens)
        
        # Batching configuration
        if "batching" in config_dict:
            batch_dict = config_dict["batching"]
            config.batch_size = batch_dict.get("batch_size", config.batch_size)
            config.max_batch_size = batch_dict.get("max_batch_size", config.max_batch_size)
            config.enable_prefix_caching = batch_dict.get("enable_prefix_caching", config.enable_prefix_caching)
            config.enable_chunked_prefill = batch_dict.get("enable_chunked_prefill", config.enable_chunked_prefill)
        
        # Monitoring configuration
        if "monitoring" in config_dict:
            monitor_dict = config_dict["monitoring"]
            config.enable_monitoring = monitor_dict.get("enable_monitoring", config.enable_monitoring)
            config.log_level = monitor_dict.get("log_level", config.log_level)
            config.metrics_export_interval = monitor_dict.get("metrics_export_interval", config.metrics_export_interval)
        
        return config
    
    def save_to_json(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to output JSON file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'InferenceConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to input JSON file.
        
        Returns:
            InferenceConfig instance.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'InferenceConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        if not isinstance(config_dict, dict):
            config_dict = {}
        return cls.from_dict(config_dict)

    def save_to_yaml(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def apply_cli_overrides(self, args: Dict[str, Any]) -> None:
        infer_mode = args.get("infer_mode")
        if infer_mode == "vllm":
            self.acceleration.use_vllm = True
        elif infer_mode == "standard":
            self.acceleration.use_vllm = False

        if args.get("speculative"):
            self.acceleration.use_speculative_decoding = True
        
        spec_gamma = args.get("spec_gamma")
        if spec_gamma is not None:
            try:
                self.acceleration.spec_gamma = int(spec_gamma)
            except Exception:
                pass
        
        min_new_tokens = args.get("min_new_tokens")
        if min_new_tokens is not None:
            try:
                self.acceleration.min_new_tokens = int(min_new_tokens)
            except Exception:
                pass

        if args.get("no_quant"):
            self.quantization.enable_quantization = False
        if args.get("quant") or args.get("force_quant"):
            self.quantization.enable_quantization = True
        quant_bits = args.get("quant_bits")
        if quant_bits is not None:
            try:
                self.quantization.bits = int(quant_bits)
            except Exception:
                pass


@dataclass
class ModelSpec:
    """
    Model Specification for Different Sizes
    
    Defines the architecture parameters for each model size variant,
    including hidden dimensions, layer counts, and MoE configurations.
    
    Attributes:
        name: Model size identifier (e.g., "0.5B", "7B", "671B")
        hidden_size: Hidden layer dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        context_length: Maximum context length in tokens
        moe_experts: Number of MoE experts (0 for dense models)
        moe_active: Number of active experts per token
        vocab_size: Vocabulary size
        intermediate_size: FFN intermediate dimension
        num_kv_heads: Number of KV heads for GQA
        rope_theta: RoPE base frequency
        max_position_embeddings: Maximum position embeddings
    """
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    context_length: int
    moe_experts: int = 0
    moe_active: int = 0
    vocab_size: int = 151936
    intermediate_size: int = 0
    num_kv_heads: int = 0
    rope_theta: float = 10000.0
    max_position_embeddings: int = 0
    
    def __post_init__(self):
        if self.intermediate_size == 0:
            self.intermediate_size = self.hidden_size * 4 // 3 * 2
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads
        if self.max_position_embeddings == 0:
            self.max_position_embeddings = self.context_length


MODEL_SPECS: Dict[str, ModelSpec] = {
    "0.5B": ModelSpec(
        name="0.5B",
        hidden_size=640,
        num_layers=16,
        num_heads=10,
        context_length=262144,
        vocab_size=151646,
        moe_experts=6,
        moe_active=2,
        num_kv_heads=5,
    ),
    "1.5B": ModelSpec(
        name="1.5B",
        hidden_size=896,
        num_layers=16,
        num_heads=14,
        context_length=262144,
        vocab_size=151646,
        moe_experts=6,
        moe_active=2,
        num_kv_heads=7,
    ),
    "7B": ModelSpec(
        name="7B",
        hidden_size=3584,
        num_layers=28,
        num_heads=32,
        context_length=1048576,
        vocab_size=151646,
        moe_experts=8,
        moe_active=2,
        num_kv_heads=8,
    ),
    "32B": ModelSpec(
        name="32B",
        hidden_size=5120,
        num_layers=64,
        num_heads=40,
        context_length=1048576,
        vocab_size=151646,
        moe_experts=8,
        moe_active=2,
        num_kv_heads=8,
    ),
    "64B": ModelSpec(
        name="64B",
        hidden_size=6656,
        num_layers=80,
        num_heads=52,
        context_length=262144,
        vocab_size=151646,
        moe_experts=8,
        moe_active=2,
        num_kv_heads=8,
    ),
    "70B": ModelSpec(
        name="70B",
        hidden_size=8192,
        num_layers=80,
        num_heads=64,
        context_length=10485760,
        vocab_size=151646,
        moe_experts=8,
        moe_active=2,
        num_kv_heads=8,
    ),
    "128B": ModelSpec(
        name="128B",
        hidden_size=10240,
        num_layers=120,
        num_heads=80,
        context_length=10485760,
        vocab_size=151646,
        moe_experts=8,
        moe_active=2,
        num_kv_heads=8,
    ),
    "314B": ModelSpec(
        name="314B",
        hidden_size=12288,
        num_layers=160,
        num_heads=96,
        context_length=10485760,
        vocab_size=151646,
        moe_experts=16,
        moe_active=4,
        num_kv_heads=12,
    ),
    "671B": ModelSpec(
        name="671B",
        hidden_size=16384,
        num_layers=200,
        num_heads=128,
        context_length=10485760,
        vocab_size=151646,
        moe_experts=32,
        moe_active=6,
        num_kv_heads=16,
    ),
    "1T": ModelSpec(
        name="1T",
        hidden_size=20480,
        num_layers=240,
        num_heads=160,
        context_length=10485760,
        vocab_size=151646,
        moe_experts=64,
        moe_active=8,
        num_kv_heads=20,
    ),
}


@dataclass
class OPSSConfig:
    """
    OPSS Integration Configuration
    
    Configures the integration with OPSS (Operator-based Production Service System)
    components including MCP Plaza, Swarm Coordinator, Orchestrator, and MCP Bridge.
    
    Attributes:
        enable_mcp_plaza: Enable MCP tool plaza
        enable_swarm_coordinator: Enable multi-agent swarm coordination
        enable_orchestrator: Enable dynamic task orchestration
        enable_mcp_bridge: Enable Agent-MCP bridge
        
        mcp_max_workers: Maximum concurrent MCP workers
        mcp_session_timeout: MCP session timeout in seconds
        swarm_max_agents: Maximum agents in swarm
        swarm_topology: Swarm topology (hierarchical, flat, mesh, star, ring)
        orchestrator_max_parallel: Maximum parallel agents in orchestration
        bridge_cache_ttl: Tool cache TTL in seconds
        bridge_retry_count: Number of retries for tool calls
        heartbeat_interval: Heartbeat interval in seconds
        heartbeat_timeout: Heartbeat timeout in seconds
    """
    enable_mcp_plaza: bool = True
    enable_swarm_coordinator: bool = True
    enable_orchestrator: bool = True
    enable_mcp_bridge: bool = True
    
    mcp_max_workers: int = 10
    mcp_session_timeout: int = 3600
    swarm_max_agents: int = 20
    swarm_topology: str = "hierarchical"
    orchestrator_max_parallel: int = 5
    bridge_cache_ttl: int = 300
    bridge_retry_count: int = 3
    heartbeat_interval: float = 10.0
    heartbeat_timeout: float = 60.0


@dataclass
class RunConfig:
    """
    Runtime Configuration for Service Management
    
    Configures the runtime behavior of the inference service,
    including resource monitoring, concurrency limits, and control settings.
    
    Attributes:
        run_id: Unique run identifier
        run_dir: Directory for run artifacts
        run_name: Human-readable run name
        control_interval_s: Control loop polling interval
        max_concurrency: Maximum concurrent requests
        request_timeout_s: Request timeout in seconds
        resource_interval_s: Resource monitoring interval
        enable_resource_monitor: Enable resource monitoring
        enable_heartbeat: Enable heartbeat monitoring
        enable_gpu_monitor: Enable GPU memory monitoring
    """
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    run_name: Optional[str] = None
    control_interval_s: float = 0.5
    max_concurrency: int = 8
    request_timeout_s: float = 120.0
    resource_interval_s: float = 5.0
    enable_resource_monitor: bool = True
    enable_heartbeat: bool = True
    enable_gpu_monitor: bool = True


@dataclass
class ServiceConfig:
    """
    Complete Service Configuration
    
    Aggregates all configuration for the inference service including
    model specification, inference settings, OPSS integration, and runtime.
    
    Attributes:
        model_size: Model size identifier
        model_spec: Model specification
        inference: Inference configuration
        opss: OPSS integration configuration
        run: Runtime configuration
        host: Service host address
        port: Service port
        api_key: Optional API key for authentication
        cors_origins: Allowed CORS origins
    """
    model_size: str = "7B"
    model_spec: Optional[ModelSpec] = None
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    opss: OPSSConfig = field(default_factory=OPSSConfig)
    run: RunConfig = field(default_factory=RunConfig)
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    def __post_init__(self):
        if self.model_spec is None:
            self.model_spec = MODEL_SPECS.get(self.model_size, MODEL_SPECS["7B"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_size": self.model_size,
            "model_spec": {
                "name": self.model_spec.name if self.model_spec else None,
                "hidden_size": self.model_spec.hidden_size if self.model_spec else None,
                "num_layers": self.model_spec.num_layers if self.model_spec else None,
                "num_heads": self.model_spec.num_heads if self.model_spec else None,
                "context_length": self.model_spec.context_length if self.model_spec else None,
                "moe_experts": self.model_spec.moe_experts if self.model_spec else 0,
                "moe_active": self.model_spec.moe_active if self.model_spec else 0,
            } if self.model_spec else None,
            "inference": self.inference.to_dict() if self.inference else None,
            "opss": {
                "enable_mcp_plaza": self.opss.enable_mcp_plaza,
                "enable_swarm_coordinator": self.opss.enable_swarm_coordinator,
                "enable_orchestrator": self.opss.enable_orchestrator,
                "enable_mcp_bridge": self.opss.enable_mcp_bridge,
                "mcp_max_workers": self.opss.mcp_max_workers,
                "swarm_max_agents": self.opss.swarm_max_agents,
                "swarm_topology": self.opss.swarm_topology,
            },
            "run": {
                "run_id": self.run.run_id,
                "max_concurrency": self.run.max_concurrency,
                "request_timeout_s": self.run.request_timeout_s,
            },
            "host": self.host,
            "port": self.port,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ServiceConfig':
        """Create from dictionary representation."""
        model_size = config_dict.get("model_size", "7B")
        
        config = cls(model_size=model_size)
        
        if "inference" in config_dict:
            config.inference = InferenceConfig.from_dict(config_dict["inference"])
        
        if "opss" in config_dict:
            opss_dict = config_dict["opss"]
            config.opss = OPSSConfig(
                enable_mcp_plaza=opss_dict.get("enable_mcp_plaza", True),
                enable_swarm_coordinator=opss_dict.get("enable_swarm_coordinator", True),
                enable_orchestrator=opss_dict.get("enable_orchestrator", True),
                enable_mcp_bridge=opss_dict.get("enable_mcp_bridge", True),
                mcp_max_workers=opss_dict.get("mcp_max_workers", 10),
                swarm_max_agents=opss_dict.get("swarm_max_agents", 20),
                swarm_topology=opss_dict.get("swarm_topology", "hierarchical"),
            )
        
        if "run" in config_dict:
            run_dict = config_dict["run"]
            config.run = RunConfig(
                run_id=run_dict.get("run_id"),
                max_concurrency=run_dict.get("max_concurrency", 8),
                request_timeout_s=run_dict.get("request_timeout_s", 120.0),
            )
        
        config.host = config_dict.get("host", "127.0.0.1")
        config.port = config_dict.get("port", 8000)
        config.api_key = config_dict.get("api_key")
        
        return config


def get_model_spec(model_size: str) -> ModelSpec:
    """
    Get model specification by size identifier.
    
    Args:
        model_size: Model size string (e.g., "7B", "671B")
    
    Returns:
        ModelSpec for the requested size
    
    Raises:
        ValueError: If model size is not found
    """
    if model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_SPECS.keys())}")
    return MODEL_SPECS[model_size]


def list_available_models() -> List[str]:
    """List all available model sizes."""
    return list(MODEL_SPECS.keys())
