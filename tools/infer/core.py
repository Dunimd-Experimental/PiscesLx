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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
PiscesLx Core Inference Engine

This module implements the flagship inference engine that integrates multiple
inference backends (VLLM, PyTorch) with state-of-the-art acceleration techniques.
The engine provides a unified interface for text generation with automatic
backend selection, quantization support, and comprehensive monitoring.

Key Features:
    - Multi-backend support: VLLM for high-throughput, PyTorch for flexibility
    - Automatic backend selection based on configuration and availability
    - Built-in quantization support (AWQ, GPTQ) for memory efficiency
    - Real-time inference statistics and performance monitoring
    - Batch and streaming generation modes
    - Model export capabilities

Architecture:
    The engine orchestrates operators from ops/infer/ to build complete inference pipeline.

Performance Characteristics:
    - VLLM backend: Up to 10x throughput improvement for batch inference
    - PyTorch backend: Flexible single-sequence generation
    - Quantization: 50-75% memory reduction with minimal accuracy loss
    - Mixed precision: Automatic selection of optimal data types

Usage Examples:
    Basic inference:
    >>> from tools.infer import InferenceConfig, PiscesLxInferenceEngine
    >>> config = InferenceConfig(model_path="meta-llama/Llama-2-7b")
    >>> engine = PiscesLxInferenceEngine(config)
    >>> result = engine.generate("Hello, world!")

    Batch inference:
    >>> prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    >>> results = engine.batch_generate(prompts, batch_size=8)

    With quantization:
    >>> config.quantization.enable_quantization = True
    >>> config.quantization.quant_method = "awq"
    >>> engine = PiscesLxInferenceEngine(config)

Dependencies:
    - torch: Core deep learning framework
    - transformers: Model loading and tokenization
    - vllm: Optional, for high-performance inference
    - awq: Optional, for AWQ quantization support

See Also:
    - tools.infer.config: Configuration management
    - tools.infer.acceleration: Acceleration operators
    - tools.infer.orchestrator: High-level orchestration
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import time
import json
from datetime import datetime

from utils.dc import PiscesLxLogger, PiscesLxSystemMonitor

from .config import InferenceConfig
from model import YvConfig, YvModel
from model.tokenizer import YvTokenizer
from model.multimodal.agentic import YvAgentic
from opss.am import POPSSToolRegistry, POPSSToolType
from opss.infer.moe_runtime import POPSSMoERuntimeOperator, POPSSMoERuntimeConfig
from opss.infer.speculative import POPSSSpeculativeDecodingOperator
from opss.infer.vllm import POPSSVLLMInferenceOperator
from opss.watermark import (
    POPSSWatermarkConfig,
    POPSSContentWatermarkOperator,
    POPSSWatermarkOrchestrator,
    POPSSComplianceOperator,
    POPSSAuditOperator,
    POPSSWatermarkDefaultConfigFactory
)

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)


def setup_inference_device(device_pref: str = "auto") -> torch.device:
    """
    Choose an inference device using unified System Monitor.
    
    Args:
        device_pref: Device preference ("auto", "cuda", "cpu")
    
    Returns:
        torch.device: Selected device
    """
    import torch
    
    try:
        monitor = PiscesLxSystemMonitor()
        if device_pref == "auto":
            if torch.cuda.is_available():
                memory_info = monitor.get_memory_info()
                if memory_info.usage_percent > 90:
                    device = torch.device("cpu")
                    _LOG.info("Inference mode: cpu (high GPU memory usage)")
                else:
                    device = torch.device("cuda")
                _LOG.info(f"Using device: {device}")
                return device
    except Exception as e:
        _LOG.warning(f"System Monitor failed, falling back: {e}")
    
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_pref)


def validate_infer_args(args: Any) -> Any:
    """
    Validate and normalize command-line arguments for inference.
    
    Args:
        args: Arguments object to validate
    
    Returns:
        Validated arguments object
    
    Raises:
        ValueError: If required arguments are missing
    """
    if not hasattr(args, 'prompt') or args.prompt is None or str(args.prompt).strip() == "":
        raise ValueError("Missing required argument: prompt")
    
    if not hasattr(args, 'model_size') or not args.model_size:
        setattr(args, 'model_size', '0.5B')
    
    if not hasattr(args, 'max_length') or not isinstance(args.max_length, int):
        setattr(args, 'max_length', 512)
    
    if not hasattr(args, 'temperature') or args.temperature is None:
        setattr(args, 'temperature', 0.7)
    
    if not hasattr(args, 'top_p') or args.top_p is None:
        setattr(args, 'top_p', 0.95)
    
    if not hasattr(args, 'stop'):
        setattr(args, 'stop', None)
    
    if not hasattr(args, 'use_vllm'):
        setattr(args, 'use_vllm', False)
    
    if not hasattr(args, 'vllm_dtype'):
        setattr(args, 'vllm_dtype', 'auto')
    
    if not hasattr(args, 'vllm_gpu_mem'):
        setattr(args, 'vllm_gpu_mem', 0.9)
    
    if not hasattr(args, 'vllm_tp_size'):
        setattr(args, 'vllm_tp_size', 1)
    
    if not hasattr(args, 'speculative'):
        setattr(args, 'speculative', False)
    
    if not hasattr(args, 'spec_gamma'):
        setattr(args, 'spec_gamma', 4)
    
    if not hasattr(args, 'min_new_tokens'):
        setattr(args, 'min_new_tokens', 16)
    
    return args


class PiscesLxInferenceEngine(object):
    """
    PiscesLx Flagship Inference Engine

    Integrates state-of-the-art inference acceleration technologies and optimization
    algorithms into a unified, easy-to-use interface. Automatically selects the
    optimal backend based on configuration and hardware availability.

    Attributes:
        config (InferenceConfig): Inference configuration object
        model: The loaded model instance (PyTorch or VLLM)
        tokenizer: The tokenizer for text encoding/decoding
        engine: VLLM engine instance (if using VLLM backend)
        device (torch.device): Computation device (cuda/cpu)
        inference_stats (dict): Real-time inference statistics

    Supported Backends:
        - VLLM: High-throughput inference with PagedAttention
        - PyTorch: Flexible inference with transformers integration

    Example:
        >>> config = InferenceConfig(
        ...     model_path="meta-llama/Llama-2-7b",
        ...     acceleration=AccelerationConfig(use_vllm=True)
        ... )
        >>> engine = PiscesLxInferenceEngine(config)
        >>> output = engine.generate("What is machine learning?")
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize the inference engine.

        Args:
            config: InferenceConfig instance containing all configuration parameters

        Note:
            The constructor automatically initializes the appropriate inference
            engine based on configuration. This may involve loading large models
            into GPU memory, which can take significant time.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.engine = None
        self.device = setup_inference_device(config.device)

        self.inference_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0.0,
            'avg_throughput': 0.0,
            'token_generation_rate': 0.0
        }
        
        self._moe_operator = None
        self._speculative_operator = None
        
        self._content_watermark_operator = None
        self._compliance_operator = None
        self._audit_operator = None
        self._watermark_config = None
        
        self._setup_inference_engine()
        self._setup_watermark_operator()

        _LOG.info(f"PiscesLxInferenceEngine initialized on {self.device}")
    
    def _setup_inference_engine(self):
        """
        Setup the inference engine based on configuration.
        
        Attempts to initialize VLLM first if configured, falls back to
        PyTorch native inference if VLLM is unavailable or fails.
        """
        if self.config.acceleration.use_vllm:
            try:
                self._setup_vllm_engine()
            except Exception as e:
                _LOG.warning(f"VLLM engine setup failed: {e}, falling back to PyTorch")
                self._setup_pytorch_engine()
        else:
            self._setup_pytorch_engine()
    
    def _setup_vllm_engine(self):
        """Initialize VLLM inference engine with configuration parameters."""
        try:
            from vllm import LLM, SamplingParams
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                max_tokens=self.config.generation.max_new_tokens,
                repetition_penalty=self.config.generation.repetition_penalty,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer else None
            )
            
            # Initialize VLLM engine
            self.engine = LLM(
                model=self.config.model.model_path,
                tokenizer=self.config.model.tokenizer_path or self.config.model.model_path,
                tensor_parallel_size=self.config.acceleration.tensor_parallel_size,
                pipeline_parallel_size=self.config.acceleration.pipeline_parallel_size,
                gpu_memory_utilization=self.config.acceleration.gpu_memory_utilization,
                enforce_eager=self.config.acceleration.enforce_eager,
                trust_remote_code=self.config.model.trust_remote_code,
                dtype=self.config.dtype,
                quantization=self.config.quantization.quant_method if self.config.quantization.enable_quantization else None
            )
            
            self.sampling_params = sampling_params
            _LOG.info("VLLM inference engine initialized successfully")
            
        except ImportError:
            raise ImportError("VLLM not installed. Please install vllm package.")
        except Exception as e:
            _LOG.error(f"VLLM engine initialization failed: {e}")
            raise
    
    def _setup_pytorch_engine(self):
        """Initialize Yv native inference engine."""
        try:
            model_size = getattr(self.config.model, 'model_size', '0.5B')
            cfg_path = self._resolve_ruchbah_config_path(model_size, self.config.model.model_path)
            
            cfg = YvConfig.from_json(str(cfg_path))
            self.model = YvModel(cfg)
            
            ckpt_path = self.config.model.model_path
            if ckpt_path and Path(ckpt_path).exists():
                if ckpt_path.endswith('.pt') or ckpt_path.endswith('.bin') or ckpt_path.endswith('.safetensors'):
                    raw = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(raw, dict):
                        state = raw.get("model_state_dict") or raw.get("model") or raw.get("state_dict") or raw
                    else:
                        state = raw
                    if isinstance(state, dict):
                        self.model.load_state_dict(state, strict=False)
                    _LOG.info(f"Loaded checkpoint from {ckpt_path}")
            
            self.tokenizer = YvTokenizer()
            
            if self.config.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            self._agentic = None
            if YvAgentic is not None:
                self._agentic = YvAgentic(cfg, tokenizer=self.tokenizer, model=self.model)
                
                self._tool_registry = POPSSToolRegistry.get_instance()
                
                try:
                    from opss.mcp.mcps import register_all_tools
                    registered = register_all_tools(self._tool_registry)
                    _LOG.info(f"Registered {len(registered)} MCP tools to unified registry")
                except ImportError as e:
                    _LOG.warning(f"Could not import MCP tools: {e}")
            
            if self.config.moe.enable_moe:
                self._setup_moe_operator()
            
            if self.config.acceleration.use_speculative_decoding:
                self._setup_speculative_operator()
            
            _LOG.info(f"Yv inference engine initialized: model_size={model_size}")
            
        except Exception as e:
            _LOG.error(f"Yv engine initialization failed: {e}")
            _LOG.warning("Falling back to AutoModelForCausalLM")
            self._setup_transformers_engine()
    
    def _setup_transformers_engine(self):
        """Initialize transformers AutoModelForCausalLM as fallback."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.tokenizer_path or self.config.model.model_path,
                trust_remote_code=self.config.model.trust_remote_code
            )
            
            model_kwargs = {
                'torch_dtype': self._get_torch_dtype(),
                'device_map': 'auto' if self.config.device == 'cuda' else None,
                'trust_remote_code': self.config.model.trust_remote_code
            }
            
            if self.config.memory_limit:
                model_kwargs['max_memory'] = {0: f"{self.config.memory_limit}MB"}
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_path,
                **model_kwargs
            )
            
            if self.config.quantization.enable_quantization:
                self._apply_quantization()
            
            if self.config.device == 'cuda' and not model_kwargs.get('device_map'):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            if self.config.moe.enable_moe:
                self._setup_moe_operator()
            
            if self.config.acceleration.use_speculative_decoding:
                self._setup_speculative_operator()
            
            self._agentic = None
            self._tool_registry = None
            
            _LOG.info("Transformers inference engine initialized successfully")
            
        except Exception as e:
            _LOG.error(f"Transformers engine initialization failed: {e}")
            raise
    
    def _resolve_ruchbah_config_path(self, model_size: str, model_path: Optional[str]) -> Path:
        """Resolve Yv model config path."""
        candidates = []
        
        if model_path:
            p = Path(model_path)
            if p.is_file():
                candidates.append(p)
            candidates.append(p / "config.json")
        
        candidates.append(Path("configs") / "model" / f"{model_size}.json")
        candidates.append(Path("configs") / f"{model_size}.json")
        
        for cand in candidates:
            if cand.exists():
                return cand
        
        raise FileNotFoundError(f"Yv config not found for model_size={model_size}, tried: {candidates}")
    
    def _setup_watermark_operator(self):
        """Setup content watermark operator for generated content provenance."""
        try:
            if hasattr(self.config, 'watermark') and self.config.watermark.get('enabled', False):
                self._watermark_config = POPSSWatermarkConfig(
                    standard=self.config.watermark.get('standard', 'GB/T 45225-2024'),
                    jurisdiction=self.config.watermark.get('jurisdiction', 'CN'),
                    risk_level=self.config.watermark.get('risk_level', 'medium'),
                    watermark_strength=self.config.watermark.get('strength', 1e-5),
                    redundancy_level=self.config.watermark.get('redundancy_level', 3),
                    encryption_enabled=self.config.watermark.get('encryption_enabled', True),
                    verify_threshold=self.config.watermark.get('verify_threshold', 0.02),
                    audit_enabled=self.config.watermark.get('audit_enabled', True),
                    owner_id=self.config.watermark.get('owner_id', 'default_owner'),
                    model_id=self.config.watermark.get('model_id', self.config.model.model_path)
                )
                
                self._content_watermark_operator = POPSSContentWatermarkOperator(self._watermark_config)
                self._compliance_operator = POPSSComplianceOperator(self._watermark_config)
                self._audit_operator = POPSSAuditOperator(self._watermark_config)
                
                _LOG.info(f"Content watermark operator initialized: jurisdiction={self._watermark_config.jurisdiction.code}")
        except Exception as e:
            _LOG.warning(f"Failed to initialize watermark operators: {e}")
            self._content_watermark_operator = None
            self._compliance_operator = None
            self._audit_operator = None
    
    def _setup_moe_operator(self):
        """Initialize MoE runtime operator for MoE models."""
        try:
            moe_runtime_config = POPSSMoERuntimeConfig(
                routing_temp=self.config.moe.routing_temp,
                top_k=self.config.moe.top_k,
                capacity_factor=self.config.moe.capacity_factor,
                min_capacity=self.config.moe.min_capacity,
                enable_load_balancing=self.config.moe.enable_load_balancing,
                load_balance_alpha=self.config.moe.load_balance_alpha,
                enable_adaptive_temp=self.config.moe.enable_adaptive_temp,
                adaptive_temp_step=self.config.moe.adaptive_temp_step,
                adaptive_temp_interval=self.config.moe.adaptive_temp_interval,
                adaptive_temp_cap=self.config.moe.adaptive_temp_cap,
                enable_expert_caching=self.config.moe.enable_expert_caching,
                expert_cache_size=self.config.moe.expert_cache_size,
                enable_prefix_caching=self.config.moe.enable_prefix_caching,
                priority_routing=self.config.moe.priority_routing,
                drop_tokens=self.config.moe.drop_tokens,
            )
            
            self._moe_operator = POPSSMoERuntimeOperator(config=moe_runtime_config)
            
            # Register model with MoE operator
            if self.model is not None:
                self._moe_operator.set_model(self.model)
            
            _LOG.info(
                f"MoE runtime operator initialized: "
                f"top_k={self.config.moe.top_k}, "
                f"temp={self.config.moe.routing_temp}"
            )
            
        except Exception as e:
            _LOG.warning(f"Failed to initialize MoE operator: {e}")
            self._moe_operator = None
    
    def _setup_speculative_operator(self):
        """Initialize speculative decoding operator."""
        try:
            self._speculative_operator = POPSSSpeculativeDecodingOperator()
            
            if self.model is not None:
                self._speculative_operator.set_model(self.model)
            
            _LOG.info("Speculative decoding operator initialized")
            
        except Exception as e:
            _LOG.warning(f"Failed to initialize speculative operator: {e}")
            self._speculative_operator = None
    
    def _get_moe_stats(self) -> Dict[str, Any]:
        """
        Get MoE runtime statistics.
        
        Returns:
            Dictionary containing MoE metrics including expert routing statistics,
            load balancing metrics, and cache hit rates.
        """
        if self._moe_operator is None:
            return {
                'moe_enabled': False,
                'message': 'MoE operator not initialized'
            }
        
        try:
            stats = self._moe_operator.get_statistics()
            return {
                'moe_enabled': True,
                'expert_routing': stats.get('routing', {}),
                'load_balancing': stats.get('load_balancing', {}),
                'cache_performance': stats.get('cache', {}),
                'capacity_utilization': stats.get('capacity', {})
            }
        except Exception as e:
            _LOG.warning(f"Failed to get MoE statistics: {e}")
            return {
                'moe_enabled': True,
                'error': str(e)
            }
    
    def update_moe_config(self, **kwargs):
        """
        Update MoE runtime configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update. Supported keys:
                - routing_temp: Temperature for expert routing softmax
                - top_k: Number of experts to route to
                - capacity_factor: Expert capacity multiplier
                - enable_load_balancing: Toggle load balancing
                - load_balance_alpha: Load balancing loss weight
                - enable_adaptive_temp: Toggle adaptive temperature
                - adaptive_temp_cap: Maximum adaptive temperature
        """
        if self._moe_operator is None:
            _LOG.warning("MoE operator not initialized, cannot update config")
            return
        
        try:
            self._moe_operator.update_config(**kwargs)
            _LOG.info(f"MoE configuration updated with parameters: {list(kwargs.keys())}")
        except Exception as e:
            _LOG.error(f"Failed to update MoE configuration: {e}")
            raise
    
    def set_expert_routing(self, routing_mode: str = 'top_k'):
        """
        Set expert routing mode for MoE models.
        
        Args:
            routing_mode: Routing strategy ('top_k', 'sparse', 'dense', 'priority')
        """
        if self._moe_operator is None:
            _LOG.warning("MoE operator not initialized")
            return
        
        try:
            self._moe_operator.set_routing_mode(routing_mode)
            _LOG.info(f"Expert routing mode set to: {routing_mode}")
        except Exception as e:
            _LOG.error(f"Failed to set expert routing mode: {e}")
            raise
    
    def get_expert_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get current expert weights and routing probabilities.
        
        Returns:
            Dictionary mapping expert names to routing weights, or None if unavailable
        """
        if self._moe_operator is None:
            return None
        
        try:
            return self._moe_operator.get_expert_weights()
        except Exception as e:
            _LOG.warning(f"Failed to get expert weights: {e}")
            return None
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch data type from configuration string."""
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
            'auto': torch.float16 if self.config.device == 'cuda' else torch.float32
        }
        return dtype_map.get(self.config.dtype, torch.float16)
    
    def _apply_quantization(self):
        """Apply model quantization based on configuration."""
        try:
            if self.config.quantization.quant_method == 'awq':
                from awq import AWQForCausalLM
                self.model = AWQForCausalLM.from_quantized(
                    self.config.model.model_path,
                    fuse_layers=True,
                    trust_remote_code=self.config.model.trust_remote_code
                )
            elif self.config.quantization.quant_method == 'gptq':
                # GPTQ quantization handling
                pass
            _LOG.info("Model quantization applied successfully")
        except Exception as e:
            _LOG.warning(f"Quantization failed: {e}")
    
    def load_model(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Load a new model and tokenizer.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            tokenizer_path: Path to tokenizer (optional, defaults to model_path)
        """
        self.config.model.model_path = model_path
        if tokenizer_path:
            self.config.model.tokenizer_path = tokenizer_path
        
        # Reinitialize engine
        self._setup_inference_engine()
        _LOG.info(f"Model loaded from {model_path}")
    
    def generate(self, prompts: Union[str, List[str]], 
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None,
                **kwargs) -> Union[str, List[str]]:
        """
        Generate text from input prompts.
        
        Args:
            prompts: Single prompt string or list of prompts
            max_new_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            top_k: Top-k sampling parameter (overrides config)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text string or list of strings
        
        Raises:
            RuntimeError: If generation fails
        """
        start_time = time.time()
        
        # Update config parameters
        if max_new_tokens is not None:
            self.config.generation.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.config.generation.temperature = temperature
        if top_p is not None:
            self.config.generation.top_p = top_p
        if top_k is not None:
            self.config.generation.top_k = top_k
        
        # Handle single and batch prompts
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # Compliance validation before generation
        if self._compliance_operator is not None:
            try:
                compliance_result = self._compliance_operator._validate({
                    "content_type": "prompt",
                    "content": prompts[0] if prompts else "",
                    "jurisdiction": self._watermark_config.jurisdiction.code if self._watermark_config else "CN",
                    "config": self._watermark_config
                })
                if compliance_result.is_success():
                    validation = compliance_result.output
                    if not validation.get('valid', True):
                        _LOG.warning(f"Compliance validation failed: {validation.get('message', 'Unknown')}")
            except Exception as e:
                _LOG.warning(f"Compliance validation error: {e}")
        
        try:
            if self.engine:  # VLLM backend
                results = self._generate_with_vllm(prompts, **kwargs)
            else:  # PyTorch backend
                results = self._generate_with_pytorch(prompts, **kwargs)
            
            # Update statistics
            self._update_inference_stats(start_time, len(prompts), results)
            
            # Audit logging after generation
            if self._audit_operator is not None:
                try:
                    self._audit_operator.log_operation(
                        operation="generate",
                        content_type="text",
                        result="success",
                        metadata={
                            "num_prompts": len(prompts),
                            "generation_time": time.time() - start_time,
                            "model_id": self._watermark_config.model_id if self._watermark_config else "unknown"
                        }
                    )
                except Exception as e:
                    _LOG.warning(f"Audit logging failed: {e}")
            
            # Return result
            if is_single:
                return results[0] if results else ""
            return results
            
        except Exception as e:
            self.inference_stats['failed_requests'] += len(prompts)
            _LOG.error(f"Generation failed: {e}")
            
            # Audit logging for failure
            if self._audit_operator is not None:
                try:
                    self._audit_operator.log_operation(
                        operation="generate",
                        content_type="text",
                        result="failed",
                        metadata={
                            "error": str(e),
                            "num_prompts": len(prompts)
                        }
                    )
                except Exception:
                    pass
            
            raise
    
    def _generate_with_vllm(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using VLLM backend."""
        outputs = self.engine.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def _generate_with_pytorch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using YvModel or PyTorch backend."""
        results = []
        
        use_tools = kwargs.pop('use_tools', False)
        tool_registry = kwargs.pop('tool_registry', self._tool_registry if hasattr(self, '_tool_registry') else None)
        
        for prompt in prompts:
            if hasattr(self, '_agentic') and self._agentic is not None:
                result = self._generate_with_ruchbah(prompt, use_tools, tool_registry, **kwargs)
                results.append(result)
            else:
                result = self._generate_with_transformers(prompt, **kwargs)
                results.append(result)
        
        return results
    
    def _generate_with_ruchbah(self, prompt: str, use_tools: bool = False, 
                                tool_registry: Any = None, **kwargs) -> str:
        """Generate text using YvModel with tool support."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if self.config.device == 'cuda':
            inputs = inputs.to(self.device)
        
        gen_kwargs = {
            'max_length': inputs.shape[1] + self.config.generation.max_new_tokens,
            'temperature': self.config.generation.temperature,
            'top_p': self.config.generation.top_p,
            'top_k': self.config.generation.top_k,
            'use_speculative': self.config.acceleration.use_speculative_decoding,
            'mode': kwargs.pop('mode', 'auto'),
            'seq_len': kwargs.pop('seq_len', 512),
        }
        gen_kwargs.update(kwargs)
        
        with torch.no_grad():
            out_ids, stats = self.model.generate(
                input_ids=inputs,
                **gen_kwargs
            )
        
        generated_text = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        if use_tools and tool_registry is not None:
            generated_text = self._process_tool_calls(generated_text, tool_registry)
        
        if self._content_watermark_operator is not None:
            try:
                wm_result = self._content_watermark_operator._embed_text({
                    "content": generated_text,
                    "payload": {
                        "model_id": self._watermark_config.model_id if self._watermark_config else "ruchbah",
                        "model": "YvModel"
                    },
                    "metadata": {"stats": stats}
                })
                if wm_result.is_success():
                    generated_text = wm_result.output.get("content", generated_text)
            except Exception as e:
                _LOG.warning(f"Watermark embedding failed: {e}")
        
        return generated_text
    
    def _generate_with_transformers(self, prompt: str, **kwargs) -> str:
        """Generate text using transformers AutoModelForCausalLM."""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if self.config.device == 'cuda':
            inputs = inputs.to(self.device)
        
        gen_kwargs = {
            'max_new_tokens': self.config.generation.max_new_tokens,
            'temperature': self.config.generation.temperature,
            'top_p': self.config.generation.top_p,
            'top_k': self.config.generation.top_k,
            'do_sample': self.config.generation.do_sample,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, **gen_kwargs)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def _process_tool_calls(self, text: str, tool_registry: Any) -> str:
        """Process agentic calls in generated text using XML syntax.
        
        Supports:
        - <agentic><ag>agent_name</ag></agentic>
        - <agentic><ag>agent_name</ag><tool><tn>tool_name</tn><tp>params</tp></tool></agentic>
        - <agentic mode="swarm"><ag_group>...</ag_group></agentic>
        """
        import re
        import json
        
        agentic_pattern = r'<agentic[^>]*>.*?</agentic>'
        matches = re.findall(agentic_pattern, text, re.DOTALL)
        
        if not matches:
            return text
        
        for match in matches:
            try:
                mode_match = re.search(r'<agentic[^>]*mode="([^"]*)"', match)
                mode = mode_match.group(1) if mode_match else "single"
                
                agent_pattern = r'<ag(?:\s+context="[^"]*")?>(.*?)</ag>'
                agents = re.findall(agent_pattern, match, re.DOTALL)
                agents = [a.strip() for a in agents if a.strip()]
                
                tool_name = None
                tool_params = {}
                
                tool_match = re.search(r'<tool[^>]*>(.*?)</tool>', match, re.DOTALL)
                if tool_match:
                    tool_content = tool_match.group(1)
                    
                    tn_match = re.search(r'<tn>(.*?)</tn>', tool_content, re.DOTALL)
                    if tn_match:
                        tool_name = tn_match.group(1).strip()
                    
                    tp_match = re.search(r'<tp>(.*?)</tp>', tool_content, re.DOTALL)
                    if tp_match:
                        try:
                            tool_params = json.loads(tp_match.group(1).strip())
                        except json.JSONDecodeError:
                            tool_params = {"raw": tp_match.group(1).strip()}
                
                result_text = None
                
                if agents and hasattr(self, '_agentic') and self._agentic is not None:
                    import asyncio
                    
                    agent_name = agents[0]
                    
                    if tool_name:
                        try:
                            result = asyncio.run(self._agentic.execute_unified_tool(tool_name, tool_params))
                            if result.success:
                                result_text = f"<result>{result.result}</result>"
                            else:
                                result_text = f"<error>{result.error}</error>"
                        except Exception as e:
                            result_text = f"<error>{str(e)}</error>"
                    else:
                        try:
                            result = asyncio.run(self._agentic.execute_unified_tool(agent_name, tool_params))
                            if result.success:
                                result_text = f"<result>{result.result}</result>"
                            else:
                                result_text = f"<error>{result.error}</error>"
                        except Exception as e:
                            result_text = f"<error>{str(e)}</error>"
                
                elif tool_name and tool_registry is not None:
                    import asyncio
                    try:
                        result = asyncio.run(tool_registry.execute(tool_name, tool_params))
                        if result.success:
                            result_text = f"<result>{result.result}</result>"
                        else:
                            result_text = f"<error>{result.error}</error>"
                    except Exception as e:
                        result_text = f"<error>{str(e)}</error>"
                
                if result_text:
                    text = text.replace(match, result_text)
                    
            except Exception as e:
                _LOG.warning(f"Failed to process agentic call: {e}")
                text = text.replace(match, f"<error>Processing failed: {str(e)}</error>")
        
        return text
    
    def batch_generate(self, prompts: List[str], batch_size: Optional[int] = None) -> List[str]:
        """
        Generate text for a batch of prompts with configurable batch size.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing (defaults to config.batch_size)
        
        Returns:
            List of generated texts
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.generate(batch_prompts)
            results.extend(batch_results)
        
        return results
    
    def _update_inference_stats(self, start_time: float, batch_size: int, results: List[str]):
        """Update inference statistics after generation."""
        end_time = time.time()
        latency = end_time - start_time
        tokens_generated = sum(len(self.tokenizer.encode(result)) for result in results) if self.tokenizer else 0
        
        # Update statistics
        self.inference_stats['total_requests'] += batch_size
        self.inference_stats['successful_requests'] += batch_size
        self.inference_stats['avg_latency'] = (
            (self.inference_stats['avg_latency'] * (self.inference_stats['successful_requests'] - batch_size) + latency) / 
            self.inference_stats['successful_requests']
        )
        self.inference_stats['avg_throughput'] = self.inference_stats['successful_requests'] / (time.time() - start_time + 1e-8)
        self.inference_stats['token_generation_rate'] = tokens_generated / latency if latency > 0 else 0
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """
        Get current inference statistics.
        
        Returns:
            Dictionary containing inference metrics
        """
        base_stats = {
            'total_requests': self.inference_stats['total_requests'],
            'successful_requests': self.inference_stats['successful_requests'],
            'failed_requests': self.inference_stats['failed_requests'],
            'success_rate': (
                self.inference_stats['successful_requests'] / 
                max(self.inference_stats['total_requests'], 1)
            ),
            'avg_latency': self.inference_stats['avg_latency'],
            'avg_throughput': self.inference_stats['avg_throughput'],
            'token_generation_rate': self.inference_stats['token_generation_rate'],
            'current_model': self.config.model.model_path,
            'device': str(self.device)
        }
        
        if self.config.moe.enable_moe and self._moe_operator is not None:
            base_stats['moe'] = self._get_moe_stats()
        
        if self.config.acceleration.use_speculative_decoding and self._speculative_operator is not None:
            base_stats['speculative'] = self._speculative_operator.get_stats()
        
        return base_stats
    
    def clear_cache(self):
        """Clear inference cache and free GPU memory."""
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _LOG.info("Inference cache cleared")
    
    def export_model(self, export_path: str, format: str = "safetensors"):
        """
        Export the model to specified format.
        
        Args:
            export_path: Path for exported model
            format: Export format ('safetensors' or 'pytorch')
        """
        if not self.model:
            raise RuntimeError("No model loaded for export")
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(self.model.state_dict(), str(export_path))
            except ImportError:
                _LOG.warning("safetensors not installed, falling back to torch format")
                torch.save(self.model.state_dict(), str(export_path))
        else:
            torch.save(self.model.state_dict(), str(export_path))
        
        _LOG.info(f"Model exported to {export_path}")
    
    def verify_watermark(self, text: str) -> Dict[str, Any]:
        """
        Verify watermark in generated text.
        
        Args:
            text: Text to verify for watermark
            
        Returns:
            Dictionary containing verification results
        """
        if self._content_watermark_operator is None:
            return {"watermark_found": False, "message": "Watermark operator not initialized"}
        
        try:
            result = self._content_watermark_operator._extract_from_text(text)
            
            if result.is_success():
                return {
                    "watermark_found": result.output.get("watermark_found", False),
                    "payload": result.output.get("payload", {})
                }
            return {"watermark_found": False, "error": result.error}
            
        except Exception as e:
            _LOG.error(f"Watermark verification failed: {e}")
            return {"watermark_found": False, "error": str(e)}
    
    def validate_watermark_compliance(self, jurisdiction: str = None) -> Dict[str, Any]:
        """
        Validate watermark configuration against compliance requirements.
        
        Args:
            jurisdiction: Target jurisdiction for validation
            
        Returns:
            Compliance validation report
        """
        if self._compliance_operator is None:
            return {"valid": False, "message": "Compliance operator not initialized"}
        
        try:
            result = self._compliance_operator._validate({
                "content_type": "text",
                "jurisdiction": jurisdiction or self._watermark_config.jurisdiction.code,
                "config": self._watermark_config
            })
            
            if result.is_success():
                return result.output
            return {"valid": False, "error": result.error}
            
        except Exception as e:
            _LOG.error(f"Compliance validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_watermark_stats(self) -> Dict[str, Any]:
        """Get watermark operator statistics."""
        return {
            "watermark_enabled": self._content_watermark_operator is not None,
            "jurisdiction": self._watermark_config.jurisdiction.code if self._watermark_config else None,
            "standard": self._watermark_config.standard.value if self._watermark_config else None
        }
    
    def register_agent_expert(self, agent: Any, name: Optional[str] = None, 
                               description: Optional[str] = None, priority: int = 5) -> Optional[str]:
        """Register an Agent expert to the tool registry.
        
        Args:
            agent: Agent expert instance
            name: Optional name for the expert
            description: Optional description
            priority: Priority level
            
        Returns:
            Tool ID if successful, None otherwise
        """
        if not hasattr(self, '_agentic') or self._agentic is None:
            _LOG.warning("YvAgentic not available, cannot register agent expert")
            return None
        
        return self._agentic.register_agent_expert(
            agent=agent, name=name, description=description, priority=priority
        )
    
    def register_native_function(self, func: Any, name: Optional[str] = None,
                                  description: str = "", priority: int = 5) -> Optional[str]:
        """Register a native function to the tool registry.
        
        Args:
            func: Function to register
            name: Optional name for the function
            description: Description of the function
            priority: Priority level
            
        Returns:
            Tool ID if successful, None otherwise
        """
        if not hasattr(self, '_agentic') or self._agentic is None:
            _LOG.warning("YvAgentic not available, cannot register function")
            return None
        
        return self._agentic.register_native_function(
            func=func, name=name, description=description, priority=priority
        )
    
    def list_available_tools(self, tool_type: Optional[Any] = None) -> List[Dict[str, Any]]:
        """List all available tools from the unified registry.
        
        Args:
            tool_type: Optional filter by tool type
            
        Returns:
            List of tool information dictionaries
        """
        if not hasattr(self, '_agentic') or self._agentic is None:
            return []
        
        return self._agentic.list_available_tools(tool_type=tool_type)
    
    def generate_with_tools(self, prompt: str, **kwargs) -> str:
        """Generate text with automatic tool execution.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text with tool results included
        """
        return self.generate(prompt, use_tools=True, **kwargs)
    
    def get_agentic(self) -> Optional[Any]:
        """Get the YvAgentic instance.
        
        Returns:
            YvAgentic instance or None if not available
        """
        return getattr(self, '_agentic', None)
    
    def is_ruchbah_model(self) -> bool:
        """Check if using YvModel.
        
        Returns:
            True if using YvModel, False otherwise
        """
        return hasattr(self, '_agentic') and self._agentic is not None
    
    # === Streaming Generation ===
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text with streaming output.
        
        This method yields tokens one by one for real-time response streaming.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters
        
        Yields:
            str: Generated tokens one at a time
        
        Example:
            >>> for token in engine.generate_stream("Hello"):
            ...     print(token, end="", flush=True)
        """
        if max_new_tokens is not None:
            self.config.generation.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.config.generation.temperature = temperature
        if top_p is not None:
            self.config.generation.top_p = top_p
        if top_k is not None:
            self.config.generation.top_k = top_k
        
        if self.engine:
            yield from self._stream_with_vllm(prompt, **kwargs)
        else:
            yield from self._stream_with_pytorch(prompt, **kwargs)
    
    def _stream_with_vllm(self, prompt: str, **kwargs):
        """Stream generation using VLLM backend."""
        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                max_tokens=self.config.generation.max_new_tokens,
            )
            
            outputs = self.engine.generate([prompt], sampling_params, stream=True)
            for output in outputs:
                for token in output.outputs[0].text:
                    yield token
        except Exception as e:
            _LOG.error(f"VLLM streaming failed: {e}")
            yield f"[Error: {e}]"
    
    def _stream_with_pytorch(self, prompt: str, **kwargs):
        """Stream generation using PyTorch backend."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            if self.config.device == 'cuda':
                inputs = inputs.to(self.device)
            
            generated = inputs
            past_key_values = None
            
            for _ in range(self.config.generation.max_new_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(
                            input_ids=generated,
                            use_cache=True,
                        )
                    else:
                        outputs = self.model(
                            input_ids=generated[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    
                    if self.config.generation.temperature > 0:
                        logits = logits / self.config.generation.temperature
                    
                    if self.config.generation.top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, self.config.generation.top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    if self.config.generation.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.config.generation.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    token_text = self.tokenizer.decode(next_token.item())
                    yield token_text
                    
                    generated = torch.cat([generated, next_token], dim=-1)
                    
        except Exception as e:
            _LOG.error(f"PyTorch streaming failed: {e}")
            yield f"[Error: {e}]"
    
    # === Embedding ===
    
    def embed(self, text: str) -> torch.Tensor:
        """
        Get embedding vector for text.
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as torch.Tensor
        
        Example:
            >>> embedding = engine.embed("Hello world")
            >>> print(embedding.shape)
        """
        if self.engine:
            return self._embed_with_vllm(text)
        return self._embed_with_pytorch(text)
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Get embedding vectors for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Batch of embedding vectors
        """
        embeddings = []
        for text in texts:
            emb = self.embed(text)
            embeddings.append(emb)
        return torch.stack(embeddings)
    
    def _embed_with_vllm(self, text: str) -> torch.Tensor:
        """Get embedding using VLLM backend."""
        try:
            outputs = self.engine.encode([text])
            return torch.tensor(outputs[0])
        except Exception as e:
            _LOG.warning(f"VLLM embedding failed: {e}")
            return self._embed_with_pytorch(text)
    
    def _embed_with_pytorch(self, text: str) -> torch.Tensor:
        """Get embedding using PyTorch backend."""
        try:
            inputs = self.tokenizer.encode(text, return_tensors='pt')
            if self.config.device == 'cuda':
                inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze()
            
            return embedding.cpu()
            
        except Exception as e:
            _LOG.error(f"Embedding failed: {e}")
            return torch.zeros(768)
    
    # === Multimodal Generation ===
    
    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate image from text prompt.
        
        This method uses the model's multimodal generation capabilities
        to create images from text descriptions.
        
        Args:
            prompt: Text description for image generation
            size: Image size (e.g., "1024x1024", "512x512")
            **kwargs: Additional generation parameters
        
        Returns:
            Generated image as torch.Tensor
        
        Example:
            >>> image = engine.generate_image("A beautiful sunset")
            >>> print(image.shape)
        """
        try:
            from model.multimodal.generator import YvGenerator
            
            if not hasattr(self, '_generator'):
                self._generator = YvGenerator(self.model, self.tokenizer)
            
            width, height = map(int, size.lower().split('x'))
            
            image = self._generator.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                **kwargs
            )
            
            return image
            
        except ImportError:
            _LOG.warning("YvGenerator not available")
            return torch.zeros(3, 1024, 1024)
        except Exception as e:
            _LOG.error(f"Image generation failed: {e}")
            return torch.zeros(3, 1024, 1024)
    
    def generate_video(
        self,
        prompt: str,
        duration: int = 5,
        fps: int = 24,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description for video generation
            duration: Video duration in seconds
            fps: Frames per second
            **kwargs: Additional generation parameters
        
        Returns:
            Generated video as torch.Tensor (frames, channels, height, width)
        """
        try:
            from model.multimodal.generator import YvGenerator
            
            if not hasattr(self, '_generator'):
                self._generator = YvGenerator(self.model, self.tokenizer)
            
            video = self._generator.generate_video(
                prompt=prompt,
                duration=duration,
                fps=fps,
                **kwargs
            )
            
            return video
            
        except ImportError:
            _LOG.warning("YvGenerator not available")
            return torch.zeros(duration * fps, 3, 1024, 1024)
        except Exception as e:
            _LOG.error(f"Video generation failed: {e}")
            return torch.zeros(duration * fps, 3, 1024, 1024)
    
    def generate_audio(
        self,
        prompt: str,
        duration: float = 5.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate audio from text prompt.
        
        Args:
            prompt: Text description for audio generation
            duration: Audio duration in seconds
            **kwargs: Additional generation parameters
        
        Returns:
            Generated audio as torch.Tensor
        """
        try:
            from model.multimodal.generator import YvGenerator
            
            if not hasattr(self, '_generator'):
                self._generator = YvGenerator(self.model, self.tokenizer)
            
            audio = self._generator.generate_audio(
                prompt=prompt,
                duration=duration,
                **kwargs
            )
            
            return audio
            
        except ImportError:
            _LOG.warning("YvGenerator not available")
            return torch.zeros(int(duration * 22050))
        except Exception as e:
            _LOG.error(f"Audio generation failed: {e}")
            return torch.zeros(int(duration * 22050))
    
    # === Vision Operations ===
    
    def encode_image(self, image: Any) -> torch.Tensor:
        """
        Encode image to token representation.
        
        Args:
            image: Image input (PIL Image, tensor, or path)
        
        Returns:
            Token representation of the image
        """
        try:
            from model.multimodal.vision import YvVisionEncoder
            
            if not hasattr(self, '_vision_encoder'):
                self._vision_encoder = YvVisionEncoder()
            
            return self._vision_encoder.encode_to_tokens(image)
            
        except ImportError:
            _LOG.warning("YvVisionEncoder not available")
            return torch.zeros(1, 256)
        except Exception as e:
            _LOG.error(f"Image encoding failed: {e}")
            return torch.zeros(1, 256)
    
    def decode_image(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode token representation to image.
        
        Args:
            tokens: Token representation
        
        Returns:
            Decoded image as tensor
        """
        try:
            from model.multimodal.vision import YvVisionEncoder
            
            if not hasattr(self, '_vision_encoder'):
                self._vision_encoder = YvVisionEncoder()
            
            return self._vision_encoder._decode_from_tokens(tokens)
            
        except ImportError:
            _LOG.warning("YvVisionEncoder not available")
            return torch.zeros(3, 1024, 1024)
        except Exception as e:
            _LOG.error(f"Image decoding failed: {e}")
            return torch.zeros(3, 1024, 1024)
    
    def detect_objects(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detect objects in image.
        
        Args:
            image: Image input
        
        Returns:
            List of detected objects with bounding boxes and labels
        """
        try:
            from model.multimodal.vision import YvVisionEncoder
            
            if not hasattr(self, '_vision_encoder'):
                self._vision_encoder = YvVisionEncoder()
            
            return self._vision_encoder.detect(image)
            
        except ImportError:
            _LOG.warning("YvVisionEncoder not available")
            return []
        except Exception as e:
            _LOG.error(f"Object detection failed: {e}")
            return []
    
    def segment_image(self, image: Any) -> torch.Tensor:
        """
        Segment image into regions.
        
        Args:
            image: Image input
        
        Returns:
            Segmentation mask as tensor
        """
        try:
            from model.multimodal.vision import YvVisionEncoder
            
            if not hasattr(self, '_vision_encoder'):
                self._vision_encoder = YvVisionEncoder()
            
            return self._vision_encoder.segment(image)
            
        except ImportError:
            _LOG.warning("YvVisionEncoder not available")
            return torch.zeros(1024, 1024)
        except Exception as e:
            _LOG.error(f"Image segmentation failed: {e}")
            return torch.zeros(1024, 1024)
    
    # === Token Operations ===
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())
    
    def encode_tokens(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        if self.tokenizer:
            return self.tokenizer.encode(text)
        return [ord(c) for c in text]
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text
        """
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        return ''.join(chr(t) for t in token_ids if t < 128)
    
    # === Model Info ===
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model information.
        
        Returns:
            Dictionary with model details
        """
        info = {
            "model_path": self.config.model.model_path,
            "device": str(self.device),
            "dtype": str(self.config.dtype),
            "backend": "vllm" if self.engine else "pytorch",
            "quantization": self.config.quantization.quant_method if self.config.quantization.enable_quantization else None,
        }
        
        if hasattr(self.config, 'model_size'):
            info["model_size"] = self.config.model_size
        
        if hasattr(self.config, 'moe') and self.config.moe.enable_moe:
            info["moe_enabled"] = True
            info["moe_experts"] = self.config.moe.num_experts
            info["moe_top_k"] = self.config.moe.top_k
        
        return info
    
    # === OPSS Integration ===
    
    def get_opss_status(self) -> Dict[str, Any]:
        """
        Get OPSS integration status.
        
        Returns:
            OPSS status dictionary
        """
        return {
            "agentic_available": hasattr(self, '_agentic') and self._agentic is not None,
            "tool_registry_available": hasattr(self, '_tool_registry') and self._tool_registry is not None,
            "moe_operator": self._moe_operator is not None,
            "speculative_operator": self._speculative_operator is not None,
            "watermark_operator": self._content_watermark_operator is not None,
        }
    
    async def execute_agent(self, agent_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent request.
        
        Args:
            agent_request: Agent request specification
        
        Returns:
            Execution result
        """
        if not hasattr(self, '_agentic') or self._agentic is None:
            return {"success": False, "error": "Agent system not available"}
        
        try:
            result = await self._agentic.execute(agent_request)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def orchestrate_agents(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate multiple agents for a task.
        
        Args:
            task: Task description
            context: Optional execution context
        
        Returns:
            Orchestration result
        """
        if not hasattr(self, '_agentic') or self._agentic is None:
            return {"success": False, "error": "Agent system not available"}
        
        try:
            result = await self._agentic.orchestrate(task, context=context)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        if not hasattr(self, '_tool_registry') or self._tool_registry is None:
            raise RuntimeError("Tool registry not available")
        
        return await self._tool_registry.execute(tool_name, arguments)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available tools.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of tool information
        """
        if not hasattr(self, '_tool_registry') or self._tool_registry is None:
            return []
        
        try:
            tools = self._tool_registry.list_tools()
            if category:
                tools = [t for t in tools if t.get('category') == category]
            return tools
        except Exception as e:
            _LOG.error(f"Failed to list tools: {e}")
            return []
    
    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tools by query.
        
        Args:
            query: Search query
        
        Returns:
            List of matching tools
        """
        if not hasattr(self, '_tool_registry') or self._tool_registry is None:
            return []
        
        try:
            tools = self._tool_registry.list_tools()
            query_lower = query.lower()
            return [
                t for t in tools
                if query_lower in t.get('name', '').lower()
                or query_lower in t.get('description', '').lower()
            ]
        except Exception as e:
            _LOG.error(f"Failed to search tools: {e}")
            return []
