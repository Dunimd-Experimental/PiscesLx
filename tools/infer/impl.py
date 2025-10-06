#!/usr/bin/env python3

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

import os
import sys
import time
import contextlib
from utils import PiscesLxCoreLog as LOG
RIGHT = LOG.info; ERROR = LOG.error; DEBUG = LOG.debug
from typing import Any, Optional, List

# Runtime context
_HOOKS = None
_PROFILER = None
_CFG = None

def set_context(*, hooks: Any = None, profiler: Any = None, cfg: Any = None) -> None:
    """
    Set runtime context for hooks, profiler, and configuration.
    
    This function is designed to maintain parity with train.impl. It updates the global 
    variables with the provided values for hooks, profiler, and configuration.
    
    Args:
        hooks (Any, optional): Hooks object to be set in the runtime context. Defaults to None.
        profiler (Any, optional): Profiler object to be set in the runtime context. Defaults to None.
        cfg (Any, optional): Configuration object to be set in the runtime context. Defaults to None.
    """
    global _HOOKS, _PROFILER, _CFG
    _HOOKS = hooks
    _PROFILER = profiler
    _CFG = cfg

def _emit(event: str, **kwargs: Any) -> None:
    """
    Emit an event through the hooks if available.
    
    This function attempts to emit an event using the global _HOOKS object. 
    If an exception occurs during the emission, it silently passes.
    
    Args:
        event (str): The name of the event to emit.
        **kwargs: Additional keyword arguments to pass to the emit method.
    """
    try:
        if _HOOKS is not None:
            _HOOKS.emit(event, **kwargs)
    except Exception:
        pass

def setup_inference_device(device_pref: str):
    """
    Choose an inference device based on the preference with GPUManager fallback.
    
    This function selects an appropriate device for inference according to the provided 
    preference. It supports "auto", "cpu", and "cuda[:id]" as device preferences. 
    When "auto" is selected, it tries to use GPUManager to determine the best strategy.
    
    Args:
        device_pref (str): Device preference. Options: "auto", "cpu", "cuda[:id]".
    Returns:
        torch.device: The selected device for inference.
    """
    import torch

    if device_pref == "auto":
        # Check if CUDA is available
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            RIGHT("CUDA not available, falling back to CPU inference mode")
            RIGHT("Inference mode: cpu")
        else:
            try:
                from utils.device import setup_devices
                device_config = setup_devices()
                # Apply device selection from orchestrator
                if device_config.get('device_type') == 'cpu':
                    device = torch.device("cpu")
                    RIGHT("CPU inference mode (via device orchestrator)")
                else:
                    gpu_ids = device_config.get('gpu_ids', [])
                    if gpu_ids:
                        device = torch.device(f"cuda:{gpu_ids[0]}")
                        torch.cuda.set_device(gpu_ids[0])
                    else:
                        device = torch.device("cuda")
                # Apply silently without printing suggestions
            except Exception as e:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    RIGHT(f"DeviceOrchestrator unavailable, using default CUDA device: {e}")
                else:
                    device = torch.device("cpu")
                    RIGHT(f"DeviceOrchestrator unavailable and CUDA not available, using CPU: {e}")
    else:
        device = torch.device(device_pref)

    RIGHT(f"Using device: {device}")
    return device

class VLLMEngine:
    """
    VLLM-based high-performance inference engine for Pisces L1 models.

    This class provides a wrapper around the VLLM library to perform inference on Pisces L1 models.
    If VLLM is not available, it falls back to native inference.
    """
    def __init__(self, model_path: str, dtype: str = "auto", gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1) -> None:
        """
        Initialize the VLLMEngine.

        Attempts to import the VLLM library. If successful, initializes the LLM with the provided parameters.
        If the import fails, sets the engine to unavailable and logs an error.

        Args:
            model_path (str): Path to the model checkpoint.
            dtype (str, optional): Data type for model inference. Defaults to "auto".
            gpu_memory_utilization (float, optional): Proportion of GPU memory to use. Defaults to 0.9.
            tensor_parallel_size (int, optional): Number of GPUs to use for tensor parallelism. Defaults to 1.
        """
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
            self.LLM = LLM
            self.SamplingParams = SamplingParams
        except ImportError:
            self.vllm_available = False
            ERROR("VLLM not available, falling back to native inference")
            return

        self.model_path = model_path
        # Determine dtype and tensor_parallel from device orchestrator if dtype is auto
        try:
            if dtype == "auto":
                from utils.device import setup_devices as _setup_devices_for_vllm
                _dev_cfg = _setup_devices_for_vllm()
                _dtype = str(_dev_cfg.get('dtype', 'fp16')).lower()
                # Map to vLLM dtype names
                dtype = 'bfloat16' if _dtype == 'bf16' else ('float16' if _dtype == 'fp16' else 'float32')
                # If not explicitly set, derive TP from gpu_ids length
                if not tensor_parallel_size or tensor_parallel_size == 1:
                    _gpu_ids = _dev_cfg.get('gpu_ids', [])
                    tensor_parallel_size = max(1, len(_gpu_ids) or 1)
        except Exception:
            pass
        # Initialize the VLLM LLM with the provided parameters
        self.llm = self.LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size
        )

    def infer(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512, top_p: float = 0.95, stop: Optional[List[str]] = None) -> Optional[str]:
        """
        Perform inference using the VLLM engine.

        If the VLLM engine is available, generates a response to the given prompt with specified sampling parameters,
        then adds a watermark to the generated text before returning it.

        Args:
            prompt (str): Input prompt for the model.
            temperature (float, optional): Temperature for sampling. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
            top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
            stop (Optional[List[str]], optional): List of stop sequences. Defaults to None.

        Returns:
            Optional[str]: Watermarked generated text if VLLM is available, otherwise None.
        """
        if not self.vllm_available:
            return None
        # Configure sampling parameters for text generation
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        # Prepare metadata for watermarking
        watermark_metadata = {
            "prompt": prompt,
            "params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "use_vllm": True
            },
            "user_id": "vllm_user",
            "timestamp": str(int(time.time()))
        }
        from tools import watermark_text
        return watermark_text(generated_text, prompt, watermark_metadata)

def _get_attr(obj: Any, name: str, default: Any) -> Any:
    """
    Safely get an attribute from an object with a default value.
    
    This function is kept for maintaining parity with other implementations.
    It returns the value of the named attribute of the object if it exists;
    otherwise, it returns the provided default value.

    Args:
        obj (Any): The object from which to get the attribute.
        name (str): The name of the attribute to retrieve.
        default (Any): The value to return if the attribute does not exist.

    Returns:
        Any: The value of the attribute or the default value.
    """
    return getattr(obj, name, default)

def validate_infer_args(args: Any) -> Any:
    """
    Validate and normalize command-line arguments for inference.
    
    This function checks if the necessary arguments are present and valid.
    If an argument is missing, it sets a default value. If an argument is invalid,
    it raises a ValueError.

    Args:
        args (Any): An object containing the command-line arguments.

    Returns:
        Any: The validated and normalized arguments object.

    Raises:
        ValueError: If any required argument is missing or any argument value is invalid.
    """
    # Check if the prompt argument is provided and not empty
    if not hasattr(args, 'prompt') or args.prompt is None or str(args.prompt).strip() == "":
        raise ValueError("Missing required argument: prompt")
    
    # Set default model size if not provided
    if not hasattr(args, 'model_size') or not args.model_size:
        setattr(args, 'model_size', '0.5B')
    
    # Set default max length if not provided or not an integer
    if not hasattr(args, 'max_length') or not isinstance(args.max_length, int):
        setattr(args, 'max_length', 512)
    
    # Set default temperature if not provided
    if not hasattr(args, 'temperature') or args.temperature is None:
        setattr(args, 'temperature', 0.7)
    
    # Set default top_p if not provided
    if not hasattr(args, 'top_p') or args.top_p is None:
        setattr(args, 'top_p', 0.95)
    
    # Set default stop sequence if not provided
    if not hasattr(args, 'stop'):
        setattr(args, 'stop', None)
    
    # Set default use_vllm flag if not provided
    if not hasattr(args, 'use_vllm'):
        setattr(args, 'use_vllm', False)
    
    # Set default vllm_dtype if not provided
    if not hasattr(args, 'vllm_dtype'):
        setattr(args, 'vllm_dtype', 'auto')
    
    # Set default vllm_gpu_mem if not provided
    if not hasattr(args, 'vllm_gpu_mem'):
        setattr(args, 'vllm_gpu_mem', 0.9)
    
    # Set default vllm_tp_size if not provided
    if not hasattr(args, 'vllm_tp_size'):
        setattr(args, 'vllm_tp_size', 1)
    
    # Set default speculative flag if not provided
    if not hasattr(args, 'speculative'):
        setattr(args, 'speculative', False)
    
    # Set default spec_gamma if not provided
    if not hasattr(args, 'spec_gamma'):
        setattr(args, 'spec_gamma', 4)
    
    # Set default force_lora flag if not provided
    if not hasattr(args, 'force_lora'):
        setattr(args, 'force_lora', False)
    
    # Set default lora_r if not provided
    if not hasattr(args, 'lora_r'):
        setattr(args, 'lora_r', 8)
    
    # Set default lora_alpha if not provided
    if not hasattr(args, 'lora_alpha'):
        setattr(args, 'lora_alpha', 32)
    
    # Set default lora_dropout if not provided
    if not hasattr(args, 'lora_dropout'):
        setattr(args, 'lora_dropout', 0.05)
    
    # Set default lora_bias if not provided
    if not hasattr(args, 'lora_bias'):
        setattr(args, 'lora_bias', "none")

    # Validate temperature value
    try:
        temp = float(args.temperature)
        if not (0.0 <= temp <= 2.0):
            raise ValueError
    except Exception:
        raise ValueError("temperature must be a float in [0.0, 2.0]")

    # Validate top_p value
    try:
        topp = float(args.top_p)
        if not (0.0 < topp <= 1.0):
            raise ValueError
    except Exception:
        raise ValueError("top_p must be a float in (0.0, 1.0]")

    # Validate max_length value
    if not isinstance(args.max_length, int) or args.max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    import os as _os
    # Validate image path if provided
    if hasattr(args, 'image') and args.image:
        if not _os.path.exists(args.image):
            raise ValueError(f"image path does not exist: {args.image}")
    
    # Validate checkpoint path if provided
    if hasattr(args, 'ckpt') and args.ckpt:
        if not _os.path.exists(args.ckpt):
            raise ValueError(f"ckpt path does not exist: {args.ckpt}")
    
    # Validate checkpoint path when using VLLM
    if args.use_vllm and (not hasattr(args, 'ckpt') or not args.ckpt or not _os.path.exists(args.ckpt)):
        raise ValueError("use_vllm requires a valid --ckpt path to a model checkpoint")
    
    # Validate stop sequence type
    if args.stop is not None and not isinstance(args.stop, (list, tuple)):
        raise ValueError("stop must be a list of strings or None")
    
    # Validate LoRA parameters if force_lora is enabled
    if args.force_lora:
        try:
            lora_r = int(args.lora_r)
            if lora_r <= 0:
                raise ValueError("lora_r must be a positive integer")
        except Exception:
            raise ValueError("lora_r must be a positive integer")
        
        try:
            lora_alpha = int(args.lora_alpha)
            if lora_alpha <= 0:
                raise ValueError("lora_alpha must be a positive integer")
        except Exception:
            raise ValueError("lora_alpha must be a positive integer")
        
        try:
            lora_dropout = float(args.lora_dropout)
            if not (0.0 <= lora_dropout < 1.0):
                raise ValueError("lora_dropout must be a float in [0.0, 1.0)")
        except Exception:
            raise ValueError("lora_dropout must be a float in [0.0, 1.0)")
        
        if args.lora_bias not in ["none", "all", "lora_only"]:
            raise ValueError("lora_bias must be one of: 'none', 'all', 'lora_only'")
    
    return args

    

class PiscesLxToolsInferImpl:
    """Class-based facade for inference implementation in a unified style.

    This class provides a unified interface for inference operations by exposing 
    module-level functions to maintain backward compatibility.
    """

    def __init__(self) -> None:
        # Initialize the runtime context attributes
        self._hooks = None
        self._profiler = None
        self._cfg = None

    def set_context(self, *, hooks: Any = None, profiler: Any = None, cfg: Any = None) -> None:
        """Set the runtime context for the inference operations.

        This method updates both the instance-level context attributes and the 
        module-level global variables to ensure compatibility with code paths 
        using module-level helpers.

        Args:
            hooks (Any, optional): Hooks object for the runtime context. Defaults to None.
            profiler (Any, optional): Profiler object for the runtime context. Defaults to None.
            cfg (Any, optional): Configuration object for the runtime context. Defaults to None.
        """
        global _HOOKS, _PROFILER, _CFG
        self._hooks = hooks
        self._profiler = profiler
        self._cfg = cfg
        # Set module context for code paths using module-level helpers
        set_context(hooks=hooks, profiler=profiler, cfg=cfg)

    def infer(self, args: Any) -> None:
        """Perform inference using the existing module-level implementation.

        This method delegates the inference task to the `_infer_impl` function 
        to avoid code duplication.

        Args:
            args (Any): Arguments for the inference process.

        Returns:
            None: The return value from the `_infer_impl` function.
        """
        return _infer_impl(args)

    def validate_args(self, args: Any) -> Any:
        """Validate and normalize the inference arguments.

        This method provides a convenient way to validate inference arguments, 
        maintaining parity with the training facade.

        Args:
            args (Any): An object containing the command-line arguments.

        Returns:
            Any: The validated and normalized arguments object.
        """
        return validate_infer_args(args)

    def setup_inference_device(self, device_pref: str):
        """Choose an inference device based on the preference.

        This method delegates the device selection task to the `setup_inference_device` 
        function to preserve behavior while supporting class-only runners.

        Args:
            device_pref (str): Device preference. Options: "auto", "cpu", "cuda[:id]".

        Returns:
            torch.device: The selected device for inference.
        """
        return setup_inference_device(device_pref)

    def get_attr(self, obj: Any, name: str, default: Any) -> Any:
        """Safely get an attribute from an object with a default value.

        This method delegates the attribute retrieval task to the `_get_attr` function 
        to preserve behavior while supporting class-only runners.

        Args:
            obj (Any): The object from which to get the attribute.
            name (str): The name of the attribute to retrieve.
            default (Any): The value to return if the attribute does not exist.

        Returns:
            Any: The value of the attribute or the default value.
        """
        return _get_attr(obj, name, default)

def _infer_impl(args: Any) -> None:
    """
    Perform the core inference implementation extracted from the `infer()` function.
    
    This function handles the complete inference pipeline, including argument validation,
    model loading, device setup, and text generation. It supports both VLLM and native 
    inference, as well as speculative decoding.
    Args:
        args (Any): An object containing the command-line arguments for inference.
    """
    import torch
    from PIL import Image
    from model.tokenizer import get_tokenizer
    from model import PiscesModel, PiscesConfig
    from transformers import BitsAndBytesConfig
    from torchvision.transforms import functional as TF
    import torch.nn.functional as F
    # Start the profiler if available
    if _PROFILER is not None and hasattr(_PROFILER, 'start'):
        try:
            _PROFILER.start('infer', args=args)
        except Exception:
            pass
    # Validate and normalize inference arguments
    try:
        args = validate_infer_args(args)
    except Exception as e:
        ERROR(f"Invalid inference arguments: {e}")
        raise
    RIGHT("Starting Pisces L1 Inference with MCP Integration...")
    _emit('on_infer_start', args=args)
    # Load model configuration and inference settings
    model_size = getattr(args, "model_size", "0.5B").upper()
    config_path = f"configs/{model_size}.json"
    cfg = PiscesConfig.from_json(config_path)
    import json as _json
    with open(config_path, 'r', encoding='utf-8') as _f:
        _full_cfg = _json.load(_f)
    inference_cfg = _full_cfg.get('inference_config', {})
    # Automatically select the inference device (GPU if available, otherwise CPU)
    device = setup_inference_device('auto')
    # Prepare mixed precision settings from device orchestrator
    try:
        from utils.device import setup_devices as _setup_devices_for_infer
        _dev_cfg = _setup_devices_for_infer()
        _mp_enabled = bool(_dev_cfg.get('mixed_precision', torch.cuda.is_available())) and torch.cuda.is_available()
        _dtype_str = str(_dev_cfg.get('dtype', 'fp16')).lower()
        _amp_dtype = torch.float16 if _dtype_str == 'fp16' else (torch.bfloat16 if _dtype_str == 'bf16' else torch.float32)
    except Exception:
        _mp_enabled = torch.cuda.is_available()
        _amp_dtype = torch.float16
    # Ensure model is on the selected device and uses mixed precision if available
    try:
        model = model.to(device, dtype=_amp_dtype if _mp_enabled else None)
    except Exception:
        pass
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        RIGHT(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel inference")
        model = torch.nn.DataParallel(model)
    lora_used = False
{{ ... }}
        RIGHT(f"Loading model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # Adjust vocabulary size if there's a mismatch between checkpoint and model
        ckpt_vocab_size = state_dict['embed.weight'].shape[0] if 'embed.weight' in state_dict else None
        model_vocab_size = model.module.embed.weight.shape[0] if hasattr(model, 'module') else model.embed.weight.shape[0]
        if ckpt_vocab_size and ckpt_vocab_size != model_vocab_size:
            DEBUG(f"Vocab size mismatch: checkpoint={ckpt_vocab_size}, model={model_vocab_size}. Auto resizing...")
            if hasattr(model, 'module'):
                model.module.resize_token_embeddings(ckpt_vocab_size)
            else:
                model.resize_token_embeddings(ckpt_vocab_size)
        # Wrap the model with LoRA configuration if LoRA weights are detected
        lora_weights_detected = False
        if checkpoint is not None:
            # Check for explicit LoRA flag in checkpoint
            if checkpoint.get('lora', False):
                lora_weights_detected = True
            # Check for LoRA-specific keys in state dict
            elif any('lora_' in key for key in state_dict.keys()):
                lora_weights_detected = True
            # Check for PEFT model config
            elif 'peft_config' in checkpoint:
                lora_weights_detected = True
        
        if lora_weights_detected:
            try:
                from peft import PeftModel
                # Load LoRA weights
                model = PeftModel.from_pretrained(model, args.ckpt)
                lora_used = True
                RIGHT(f"LoRA weights loaded from {args.ckpt}")
            except ImportError:
                ERROR("PEFT library not available for LoRA loading")
            except Exception as e:
                ERROR(f"Failed to load LoRA weights: {e}")
        else:
            # Load standard model weights
            model.load_state_dict(state_dict, strict=False)
            
        # Apply LoRA configuration from inference config if specified
        if not lora_used and (inference_cfg.get('force_lora', False) or args.force_lora):
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                # Get LoRA config from inference_cfg or use defaults (aligned with training config)
                lora_cfg = inference_cfg.get('lora_config', {})
                lora_config = LoraConfig(
                    r=lora_cfg.get('r', args.lora_r),
                    lora_alpha=lora_cfg.get('lora_alpha', args.lora_alpha),
                    target_modules=lora_cfg.get('target_modules', ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                    lora_dropout=lora_cfg.get('lora_dropout', args.lora_dropout),
                    bias=lora_cfg.get('bias', args.lora_bias),
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True  # Enable inference mode for better performance
                )
                model = get_peft_model(model, lora_config)
                lora_used = True
                RIGHT(f"LoRA configuration applied: r={lora_config.r}, alpha={lora_config.lora_alpha}, target_modules={lora_config.target_modules}")
                try:
                    model.print_trainable_parameters()
                except Exception:
                    pass
            except ImportError:
                ERROR("PEFT library not available for LoRA configuration")
            except Exception as e:
                ERROR(f"Failed to apply LoRA configuration: {e}")
        elif not lora_used:
            # Load standard model weights if no LoRA is used
            model.load_state_dict(state_dict, strict=False)