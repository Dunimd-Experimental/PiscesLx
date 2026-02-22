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

"""
Inference Orchestrator
Unified management and coordination of all inference components.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
import time
from datetime import datetime
import json
import yaml
import os

# OPSC operator system integration
from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

from .config import InferenceConfig
from .core import PiscesLxInferenceEngine
from .pipeline import InferencePipelineOperator, PromptEngineeringOperator


from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)


@PiscesLxOperatorRegistrar()
class PiscesLxInferOrchestrator(PiscesLxBaseOperator):
    """
    Inference Orchestrator.
    Unified management of all components in the inference pipeline.
    """
    
    def __init__(self, config: Optional[Union[PiscesLxOperatorConfig, InferenceConfig, str, Dict[str, Any]]] = None):
        op_config = config if isinstance(config, PiscesLxOperatorConfig) else None
        super().__init__(op_config)

        self.infer_config = self._normalize_infer_config(config)
        
        # Initialize core components
        self.inferencer = None
        self.pipeline = None
        self.accelerators = {}
        self.quantizers = {}
        
        # Inference state
        self.is_initialized = False
        self.current_session = "default"
        self.inference_sessions = {}
        
        _LOG.info("PiscesLxInferOrchestrator initialized")

    def run(self, args) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        try:
            if hasattr(args, "__dict__"):
                params = dict(vars(args))
        except Exception:
            params = {}

        infer_cfg_path = params.get("infer_config")
        if isinstance(infer_cfg_path, str) and infer_cfg_path.strip():
            self.infer_config = self._load_config_from_file(infer_cfg_path.strip())
        try:
            self.infer_config.apply_cli_overrides(params)
        except Exception:
            pass

        if params.get("dry_run"):
            out = {"status": "dry_run", "infer_config": self.infer_config.to_dict(), "infer_mode": params.get("infer_mode", "standard")}
            try:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            except Exception:
                pass
            return out

        if params.get("serve"):
            from .server import PiscesLxInferService
            svc = PiscesLxInferService(
                host=str(params.get("host") or "127.0.0.1"),
                port=int(params.get("port") or 8000),
                ckpt=str(params.get("ckpt") or "").strip() or None,
                model_path=str(params.get("model_path") or "").strip() or None,
                model_size=str(params.get("model_size") or "0.5B").strip(),
                run_id=str(params.get("run_id") or "").strip() or None,
                run_dir=params.get("run_dir"),
                run_name=params.get("run_name"),
                control_interval_s=float(params.get("control_interval") or 0.5),
                max_concurrency=int(params.get("max_concurrency") or 2),
                request_timeout_s=float(params.get("request_timeout") or 120.0),
            )
            svc.serve()
            return {"status": "stopped"}

        ckpt = str(params.get("ckpt") or "").strip()
        prompt = params.get("prompt")
        prompt = str(prompt) if prompt is not None else "Hello, please introduce yourself"
        model_path = params.get("model_path")
        model_path = str(model_path).strip() if model_path is not None else ""

        run_id = str(params.get("run_id") or "").strip()
        run_ctl = None
        if run_id:
            try:
                from opss.run import POPSSRunController, POPSSRunStore
                st = POPSSRunStore(run_id, run_dir=params.get("run_dir"))
                run_ctl = POPSSRunController(st)
                run_ctl.init_run(
                    {
                        "run_id": run_id,
                        "run_name": str(params.get("run_name") or "").strip(),
                        "type": "infer",
                        "infer_mode": str(params.get("infer_mode") or "standard"),
                        "model_size": str(params.get("model_size") or "0.5B").strip(),
                        "ckpt": ckpt,
                        "model_path": model_path,
                    },
                    state={"status": "running", "phase": "infer", "pid": int(os.getpid())},
                )
            except Exception:
                run_ctl = None

        if not ckpt and not model_path:
            _LOG.info("No --ckpt provided; inference not started")
            return {"status": "ready", "reason": "no_ckpt"}

        infer_mode = str(params.get("infer_mode") or "standard").strip() or "standard"

        use_spec = bool(params.get("speculative") or False)

        if infer_mode == "vllm":
            candidate = model_path or ckpt
            if os.path.isfile(candidate):
                _LOG.warning("vllm_requires_model_dir_or_hf_id_falling_back", ckpt=candidate)
            else:
                self.infer_config.acceleration.use_vllm = True
                if model_path:
                    self.infer_config.model.model_path = model_path
                else:
                    self.infer_config.model.model_path = ckpt
                engine = PiscesLxInferenceEngine(self.infer_config)
                engine.load_model(self.infer_config.model.model_path, self.infer_config.model.tokenizer_path)
                out = engine.generate(prompt)
                try:
                    print(out)
                except Exception:
                    pass
                if run_ctl is not None:
                    try:
                        run_ctl.append_event("infer_ok", payload={"backend": "vllm", "output_preview": str(out)[:1000]})
                        run_ctl.update_state({"status": "completed", "phase": "completed"})
                    except Exception:
                        pass
                return {"status": "ok", "output": out, "backend": "vllm"}

        if os.path.isfile(ckpt):
            from opss.infer.native_ruchbah import POPSSNativeInferenceOperator
            op = POPSSNativeInferenceOperator()
            res = op.execute(
                {
                    "ckpt": ckpt,
                    "prompt": prompt,
                    "model_size": str(params.get("model_size") or "0.5B").strip(),
                    "seq_len": int(params.get("seq_len") or 512),
                    "generation": {
                        "max_new_tokens": int(self.infer_config.generation.max_new_tokens),
                        "temperature": float(self.infer_config.generation.temperature),
                        "top_p": float(self.infer_config.generation.top_p),
                        "top_k": int(self.infer_config.generation.top_k),
                        "use_speculative": bool(use_spec),
                        "mode": "thinking" if use_spec else "auto",
                    },
                }
            )
            if not res.is_success():
                if run_ctl is not None:
                    try:
                        run_ctl.append_event("infer_failed", level="error", payload={"backend": "native_ruchbah", "error": str(res.error)})
                        run_ctl.update_state({"status": "failed", "phase": "failed", "error": str(res.error)})
                    except Exception:
                        pass
                return {"status": "error", "reason": res.error, "backend": "native_ruchbah"}
            out_obj = res.output or {}
            out = out_obj.get("text", "")
            try:
                print(out)
            except Exception:
                pass
            if run_ctl is not None:
                try:
                    run_ctl.append_event("infer_ok", payload={"backend": out_obj.get("backend", "native_ruchbah"), "output_preview": str(out)[:1000]})
                    run_ctl.update_state({"status": "completed", "phase": "completed"})
                except Exception:
                    pass
            return {"status": "ok", "output": out, "backend": out_obj.get("backend", "native_ruchbah"), "stats": out_obj.get("stats", {})}

        self.infer_config.acceleration.use_vllm = False
        self.infer_config.model.model_path = model_path or ckpt
        engine = PiscesLxInferenceEngine(self.infer_config)
        engine.load_model(self.infer_config.model.model_path, self.infer_config.model.tokenizer_path)
        out = engine.generate(prompt)
        try:
            print(out)
        except Exception:
            pass
        if run_ctl is not None:
            try:
                run_ctl.append_event("infer_ok", payload={"backend": "transformers", "output_preview": str(out)[:1000]})
                run_ctl.update_state({"status": "completed", "phase": "completed"})
            except Exception:
                pass
        return {"status": "ok", "output": out, "backend": "transformers"}

    def _normalize_infer_config(self, config: Optional[Union[PiscesLxOperatorConfig, InferenceConfig, str, Dict[str, Any]]]) -> InferenceConfig:
        if isinstance(config, InferenceConfig):
            return config
        if isinstance(config, str):
            return self._load_config_from_file(config)
        if isinstance(config, dict):
            return InferenceConfig.from_dict(config)
        if isinstance(config, PiscesLxOperatorConfig):
            params = getattr(config, "parameters", {}) or {}
            cfg = params.get("inference_config", None)
            if isinstance(cfg, InferenceConfig):
                return cfg
            if isinstance(cfg, str):
                return self._load_config_from_file(cfg)
            if isinstance(cfg, dict):
                return InferenceConfig.from_dict(cfg)
            cfg_path = params.get("config_path")
            if isinstance(cfg_path, str) and cfg_path:
                return self._load_config_from_file(cfg_path)
        return InferenceConfig()
    
    def _load_config_from_file(self, config_path: str) -> InferenceConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        if config_path.suffix.lower() == '.json':
            return InferenceConfig.load_from_json(str(config_path))
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            return InferenceConfig.load_from_yaml(str(config_path))
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def initialize_inference(self, model_path: str,
                           tokenizer_path: Optional[str] = None,
                           **model_kwargs) -> 'PiscesLxInferOrchestrator':
        """
        Initialize complete inference environment.
        
        Args:
            model_path: Model path.
            tokenizer_path: Tokenizer path.
            **model_kwargs: Model initialization parameters.
            
        Returns:
            Initialized orchestrator instance.
        """
        _LOG.info("Initializing inference environment...")
        
        # Initialize core inference engine
        self.inferencer = PiscesLxInferenceEngine(self.infer_config)
        
        # 2. Load model
        self.inferencer.load_model(model_path, tokenizer_path)
        
        # 3. Initialize acceleration components
        self._setup_acceleration()
        
        # 4. Initialize quantization components (if enabled)
        if self.infer_config.quantization.enable_quantization:
            self._setup_quantization()
        
        # 5. Initialize watermark components (if enabled)
        if getattr(self.infer_config, 'enable_watermark', False):
            self._setup_watermark()
        
        # 6. Initialize inference pipeline
        self.pipeline = InferencePipelineOperator(self.infer_config)
        
        # 7. Initialize prompt engineering
        self.prompt_engineer = PromptEngineeringOperator()
        
        # 8. Setup watermark pipeline (if watermark enabled)
        if getattr(self.infer_config, 'enable_watermark', False):
            self._setup_watermark_pipeline()
        
        self.is_initialized = True
        _LOG.info("Inference environment initialization completed")
        return self
    
    def _setup_acceleration(self):
        """Setup acceleration components."""
        # VLLM acceleration
        if self.infer_config.acceleration.use_vllm:
            self.accelerators['vllm'] = VLLMAccelerationOperator(
                tensor_parallel_size=self.infer_config.acceleration.tensor_parallel_size,
                pipeline_parallel_size=self.infer_config.acceleration.pipeline_parallel_size,
                gpu_memory_utilization=self.infer_config.acceleration.gpu_memory_utilization,
                enforce_eager=self.infer_config.acceleration.enforce_eager
            )
        
        # Speculative decoding
        if self.infer_config.acceleration.use_speculative_decoding:
            self.accelerators['speculative'] = SpeculativeDecodingOperator()
        
        # Attention optimization
        self.accelerators['attention'] = AttentionOptimizationOperator()
        
        # KV cache
        self.accelerators['kv_cache'] = KVCacheOperator()
        
        _LOG.info(f"Acceleration components setup: {list(self.accelerators.keys())}")
    
    def _setup_quantization(self):
        """Setup quantization components."""
        # Quantized inference
        self.quantizers['main'] = QuantizationInferenceOperator(
            quant_method=self.infer_config.quantization.quant_method,
            bits=self.infer_config.quantization.bits,
            group_size=self.infer_config.quantization.group_size,
            symmetric=self.infer_config.quantization.symmetric
        )
        
        # Mixed precision
        self.quantizers['mixed_precision'] = MixedPrecisionInferenceOperator(
            precision=self.infer_config.dtype
        )
        
        _LOG.info(f"Quantization components setup: {list(self.quantizers.keys())}")
    
    def _setup_watermark(self):
        """Setup watermark components."""
        self.watermark_ops = {}
        
        # Inference watermark integration
        self.watermark_ops['integration'] = InferenceWatermarkIntegrationOperator(
            enable_content_verification=getattr(self.infer_config, 'enable_content_verification', True),
            enable_model_verification=getattr(self.infer_config, 'enable_model_verification', True),
            strict_mode=getattr(self.infer_config, 'watermark_strict_mode', False)
        )
        
        _LOG.info(f"Watermark components setup: {list(self.watermark_ops.keys())}")
    
    def _setup_watermark_pipeline(self):
        """Setup watermark pipeline."""
        if 'integration' in self.watermark_ops:
            self.watermark_pipeline = InferencePipelineWatermarkOperator(
                self.watermark_ops['integration']
            )
            _LOG.info("Watermark pipeline setup completed")
    
    def add_inference_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Add inference callback function.
        
        Args:
            callback: Callback function that receives (stage, kwargs) parameters.
        """
        if self.pipeline:
            self.pipeline.add_callback(callback)
            _LOG.info("Inference callback added")

    def _execute_impl(self, inputs: Dict[str, Any], **kwargs):
        action = inputs.get("action") or inputs.get("mode") or "generate"
        if action in ("generate", "text", "completion"):
            prompt = inputs.get("prompt", None)
            prompts = inputs.get("prompts", None)
            session_id = inputs.get("session_id", None)
            gen_kwargs = inputs.get("generation", None) or {}
            gen_kwargs.update(kwargs)
            if prompt is None and prompts is None:
                raise ValueError("Missing 'prompt' or 'prompts' for generate action")
            if prompts is not None:
                return {"result": self.batch_inference(prompts, **gen_kwargs)}
            return {"result": self.generate_text(prompt, session_id=session_id, **gen_kwargs)}
        if action in ("stats", "metrics"):
            return {"stats": self.get_inference_stats()}
        if action in ("clear_cache", "reset_cache"):
            self.clear_inference_cache()
            return {"cleared": True}
        raise ValueError(f"Unsupported action: {action}")
    
    def generate_text(self, prompt: Union[str, List[str]], 
                     session_id: Optional[str] = None,
                     **kwargs) -> Union[str, List[str]]:
        """
        Generate text.
        
        Args:
            prompt: Input prompt.
            session_id: Session ID.
            **kwargs: Generation parameters.
            
        Returns:
            Generated text.
        """
        if not self.is_initialized:
            raise RuntimeError("Inference environment not initialized. Call initialize_inference() first.")
        
        # Set session
        if session_id:
            self.current_session = session_id
        
        # Preprocess inputs
        processed_inputs = self.pipeline.preprocess_inputs(prompt)
        
        # Execute inference
        if len(processed_inputs) == 1:
            result = self.inferencer.generate(processed_inputs[0], **kwargs)
        else:
            result = self.pipeline.batch_inference(processed_inputs, **kwargs)
        
        # Postprocess outputs
        final_result = self.pipeline.postprocess_outputs(result, processed_inputs)
        
        return final_result
    
    def stream_generate(self, prompt: str, **kwargs):
        """
        Stream generate text.
        
        Args:
            prompt: Input prompt.
            **kwargs: Generation parameters.
            
        Yields:
            Generated text chunks.
        """
        if not self.is_initialized:
            raise RuntimeError("Inference environment not initialized")
        
        yield from self.pipeline.generate_stream(prompt, **kwargs)
    
    def batch_inference(self, prompts: List[str], 
                       batch_size: Optional[int] = None,
                       **kwargs) -> List[str]:
        """
        Batch inference.
        
        Args:
            prompts: List of input prompts.
            batch_size: Batch processing size.
            **kwargs: Inference parameters.
            
        Returns:
            List of inference results.
        """
        if not self.is_initialized:
            raise RuntimeError("Inference environment not initialized")
        
        return self.pipeline.batch_inference(prompts, batch_size, **kwargs)
    
    def async_inference(self, prompts: List[str], 
                       callback: Callable,
                       **kwargs):
        """
        Asynchronous inference.
        
        Args:
            prompts: List of input prompts.
            callback: Completion callback function.
            **kwargs: Inference parameters.
        """
        if not self.is_initialized:
            raise RuntimeError("Inference environment not initialized")
        
        return self.pipeline.async_inference(prompts, callback, **kwargs)
    
    def render_prompt_template(self, template_name: str, **kwargs) -> str:
        """
        Render prompt template.
        
        Args:
            template_name: Template name.
            **kwargs: Template variables.
            
        Returns:
            Rendered prompt text.
        """
        return self.prompt_engineer.render_template(template_name, **kwargs)
    
    def add_prompt_template(self, name: str, template: str,
                           variables: Optional[List[str]] = None):
        """
        Add prompt template.
        
        Args:
            name: Template name.
            template: Template string.
            variables: Variable list.
        """
        self.prompt_engineer.add_template(name, template, variables)
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """
        Get inference statistics.
        
        Returns:
            Inference statistics dictionary.
        """
        stats = {
            'basic_stats': self.inferencer.get_inference_stats() if self.inferencer else {},
            'session_info': {
                'current_session': self.current_session,
                'total_sessions': len(self.inference_sessions)
            },
            'components': {
                'inferencer': self.inferencer is not None,
                'pipeline': self.pipeline is not None,
                'accelerators': list(self.accelerators.keys()),
                'quantizers': list(self.quantizers.keys())
            }
        }
        
        # Add accelerator statistics
        if 'kv_cache' in self.accelerators:
            stats['kv_cache_stats'] = self.accelerators['kv_cache'].get_cache_stats()
        
        return stats
    
    def clear_inference_cache(self):
        """Clear inference cache."""
        if self.inferencer:
            self.inferencer.clear_cache()
        if 'kv_cache' in self.accelerators:
            self.accelerators['kv_cache'].clear_cache()
        _LOG.info("All inference caches cleared")
    
    def export_inference_model(self, export_path: str, format: str = "safetensors"):
        """
        Export inference model.
        
        Args:
            export_path: Export path.
            format: Export format.
        """
        if not self.inferencer:
            raise RuntimeError("No inference model available")
        
        self.inferencer.export_model(export_path, format)
        _LOG.info(f"Inference model exported to {export_path}")
    
    def save_inference_state(self, filepath: str):
        """
        Save inference state.
        
        Args:
            filepath: State file path.
        """
        state = {
            'config': self.infer_config.to_dict(),
            'current_session': self.current_session,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Inference state saved to {filepath}")
    
    def load_inference_state(self, filepath: str):
        """
        Load inference state.
        
        Args:
            filepath: State file path.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.infer_config = InferenceConfig.from_dict(state['config'])
        self.current_session = state.get('current_session', 'default')

        _LOG.info(f"Inference state loaded from {filepath}")
