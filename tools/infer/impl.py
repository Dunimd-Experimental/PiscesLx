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

import os
import sys
import time
import contextlib
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.data.download")
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
    Choose an inference device using unified Device Facade.
    """
    import torch
    from utils.device.facade import PiscesLxCoreDeviceFacade
    facade = PiscesLxCoreDeviceFacade(args=None)
    if device_pref == "auto":
        dev_cfg = facade.setup_devices(mode="auto")
        if dev_cfg.get("device_type") == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
            logger.success("Inference mode: cpu (via Device Facade)")
        else:
            gpu_ids = dev_cfg.get("gpu_ids", [])
            if gpu_ids:
                device = torch.device(f"cuda:{gpu_ids[0]}")
                try:
                    torch.cuda.set_device(gpu_ids[0])
                except Exception:
                    pass
            else:
                device = torch.device("cuda")
    else:
        device = torch.device(device_pref)
    logger.success(f"Using device: {device}")
    return device

class VLLMEngine:
    """
    VLLM-based high-performance inference engine for PiscesL1 models.

    This class provides a wrapper around the VLLM library to perform inference on PiscesL1 models.
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
            logger.error("VLLM not available, falling back to native inference")
            return

        self.model_path = model_path
        # Determine dtype and tensor_parallel from device orchestrator if dtype is auto
        try:
            if dtype == "auto":
                from utils.device.facade import PiscesLxCoreDeviceFacade
                f = PiscesLxCoreDeviceFacade(args=None)
                dev_cfg = f.setup_devices(mode="auto")
                torch_dtype = f.amp_dtype(dev_cfg.get("dtype", "auto"))
                dtype = f.map_vllm_dtype(torch_dtype)
                if not tensor_parallel_size or tensor_parallel_size == 1:
                    gpu_ids = dev_cfg.get("gpu_ids", [])
                    tensor_parallel_size = max(1, len(gpu_ids) or 1)
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
        from tools.infer.watermark import watermark_text
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

    # Ensure a minimum number of new tokens are generated unless EOS encountered
    if not hasattr(args, 'min_new_tokens'):
        setattr(args, 'min_new_tokens', 16)
    
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

    # Inference-time MoE balancing knobs (borrowed from training ideas)
    if not hasattr(args, 'routing_temp'):
        setattr(args, 'routing_temp', None)  # float or None; when set, overrides gate.temperature
    if not hasattr(args, 'moe_top_k_override'):
        setattr(args, 'moe_top_k_override', None)  # int or None; when set, overrides gate.top_k

    # Adaptive MoE runtime adjuster (optional)
    if not hasattr(args, 'adaptive_moe'):
        setattr(args, 'adaptive_moe', False)
    if not hasattr(args, 'adaptive_moe_temp_step'):
        setattr(args, 'adaptive_moe_temp_step', 0.03)
    if not hasattr(args, 'adaptive_moe_interval'):
        setattr(args, 'adaptive_moe_interval', 16)
    if not hasattr(args, 'adaptive_moe_temp_cap'):
        setattr(args, 'adaptive_moe_temp_cap', 1.30)

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
    from model import ArcticModel, ArcticConfig
    from transformers import BitsAndBytesConfig
    import torch.nn.functional as F
    import asyncio
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
        logger.error(f"Invalid inference arguments: {e}")
        raise
    logger.success("Starting PiscesL1 Inference with MCP Integration...")
    _emit('on_infer_start', args=args)
    
    # Initialize MCP integration (optional, only if module exists and enabled)
    try:
        import importlib.util as _importlib_util
        mcp_enabled = bool(getattr(args, 'mcp_enable', True))
        if mcp_enabled and _importlib_util.find_spec("tools.infer.agentic.integration") is not None:
            from tools.infer.agentic.integration import initialize_mcp_for_inference  # type: ignore
            mcp_initialized = asyncio.run(initialize_mcp_for_inference())
            if mcp_initialized:
                logger.success("MCP集成初始化成功")
            else:
                logger.warning("MCP集成初始化失败，继续推理流程")
        else:
            # Skip MCP init silently if module not found or disabled
            pass
    except Exception as e:
        logger.debug("MCP init skipped", error=str(e))
    # Load model configuration and inference settings
    model_size = getattr(args, "model_size", "0.5B").upper()
    # Resolve config file path (support both new and legacy locations)
    _candidates = [
        f"configs/model/{model_size}.json",
        f"configs/{model_size}.json",
    ]
    config_path = next((p for p in _candidates if os.path.exists(p)), None)
    if not config_path:
        logger.error(f"Config file not found for model_size={model_size}. Tried: {_candidates}")
        raise FileNotFoundError(f"Config file not found. Tried: {_candidates}")
    cfg = ArcticConfig.from_json(config_path)
    import json as _json
    with open(config_path, 'r', encoding='utf-8') as _f:
        _full_cfg = _json.load(_f)
    inference_cfg = _full_cfg.get('inference_config', {})
    # Automatically select the inference device (GPU if available, otherwise CPU)
    device = setup_inference_device('auto')
    # Prepare mixed precision settings from unified Device Facade
    from utils.device.facade import PiscesLxCoreDeviceFacade
    f = PiscesLxCoreDeviceFacade(args)
    _dev_cfg = f.setup_devices(mode="auto")
    _amp_dtype = f.amp_dtype(_dev_cfg.get("dtype", "auto"))
    _mp_enabled = bool(_dev_cfg.get("mixed_precision", torch.cuda.is_available())) and torch.cuda.is_available()
    # Initialize model
    try:
        model = ArcticModel(cfg)
        logger.success(f"Model initialized with config: {model_size}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    
    # Ensure model is on the selected device and uses mixed precision if available
    try:
        model = model.to(device, dtype=_amp_dtype if _mp_enabled else None)
    except Exception:
        pass
    # DataParallel is removed; rely on engine tensor-parallel or single-device execution via unified device facade
    lora_used = False
    
    # Load model checkpoint if provided
    if hasattr(args, 'ckpt') and args.ckpt:
        logger.success(f"Loading model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # Adjust vocabulary size if there's a mismatch between checkpoint and model
        ckpt_vocab_size = state_dict['embed.weight'].shape[0] if 'embed.weight' in state_dict else None
        model_vocab_size = model.module.embed.weight.shape[0] if hasattr(model, 'module') else model.embed.weight.shape[0]
        if ckpt_vocab_size and ckpt_vocab_size != model_vocab_size:
            logger.debug(f"Vocab size mismatch: checkpoint={ckpt_vocab_size}, model={model_vocab_size}. Auto resizing...")
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
                logger.success(f"LoRA weights loaded from {args.ckpt}")
            except ImportError:
                logger.error("PEFT library not available for LoRA loading")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
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
                logger.success(f"LoRA configuration applied: r={lora_config.r}, alpha={lora_config.lora_alpha}, target_modules={lora_config.target_modules}")
                try:
                    model.print_trainable_parameters()
                except Exception:
                    pass
            except ImportError:
                logger.error("PEFT library not available for LoRA configuration")
            except Exception as e:
                logger.error(f"Failed to apply LoRA configuration: {e}")
        elif not lora_used:
            # Load standard model weights if no LoRA is used
            model.load_state_dict(state_dict, strict=False)

    # Auto-enable MoE balancing (Option B) when not explicitly provided via CLI
    try:
        if getattr(args, 'routing_temp', None) is None:
            args.routing_temp = 1.12  # sensible default start
        if not getattr(args, 'adaptive_moe', False):
            args.adaptive_moe = True
            if not hasattr(args, 'adaptive_moe_temp_step'):
                args.adaptive_moe_temp_step = 0.03
            if not hasattr(args, 'adaptive_moe_interval'):
                args.adaptive_moe_interval = 16
            if not hasattr(args, 'adaptive_moe_temp_cap'):
                args.adaptive_moe_temp_cap = 1.30
    except Exception:
        pass

    # Apply inference-time MoE runtime overrides before switching to eval
    try:
        def _apply_moe_overrides(_model, routing_temp=None, top_k=None):
            for m in _model.modules():
                # Override routing temperature if gate exposes a tensor buffer 'temperature'
                if routing_temp is not None and hasattr(m, 'temperature'):
                    try:
                        # Handle tensor buffer or float attribute
                        t = float(routing_temp)
                        if isinstance(m.temperature, torch.Tensor):
                            m.temperature.fill_(t)
                        else:
                            setattr(m, 'temperature', t)
                    except Exception:
                        pass
                # Optionally override top-k for gates exposing 'top_k'
                if top_k is not None and hasattr(m, 'top_k'):
                    try:
                        setattr(m, 'top_k', int(top_k))
                    except Exception:
                        pass
        _apply_moe_overrides(model, routing_temp=args.routing_temp, top_k=args.moe_top_k_override)
    except Exception as _e:
        logger.debug(f"MoE override skipped: {_e}")

    # Set model to evaluation mode
    model.eval()
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Process input prompt
    prompt = str(args.prompt).strip()
    if not prompt:
        raise ValueError("Empty prompt provided")
    
    # Handle image input if provided
    if hasattr(args, 'image') and args.image:
        try:
            image = Image.open(args.image).convert('RGB')
            try:
                # Try torchvision if available
                from torchvision.transforms import functional as TF  # type: ignore
                image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
            except Exception:
                # Fallback: pure PIL + torch without torchvision
                import numpy as _np  # type: ignore
                _arr = _np.asarray(image, dtype=_np.float32) / 255.0  # HWC, [0,1]
                import torch as _t
                image_tensor = _t.from_numpy(_arr).permute(2, 0, 1).unsqueeze(0).to(device)
            # Process image through vision encoder if available
            if hasattr(model, 'vision_encoder'):
                _ = model.vision_encoder(image_tensor)
                prompt = f"<image>{prompt}"
            else:
                logger.warning("Model does not have vision encoder, treating as text-only input")
        except Exception as e:
            logger.warning(f"Failed to process image: {e}, treating as text-only input")
    
    # Tokenize input
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    except Exception as e:
        logger.error(f"Failed to tokenize input: {e}")
        raise
    
    # Generate text based on inference method
    if args.use_vllm:
        # Use VLLM for inference
        try:
            vllm_engine = VLLMEngine(
                model_path=args.ckpt if hasattr(args, 'ckpt') and args.ckpt else "piscesl1-model",
                dtype=args.vllm_dtype,
                gpu_memory_utilization=args.vllm_gpu_mem,
                tensor_parallel_size=args.vllm_tp_size
            )
            
            result = vllm_engine.infer(
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_length,
                top_p=args.top_p,
                stop=args.stop
            )
            
            if result:
                logger.success("VLLM inference completed successfully")
                print(f"Generated text: {result}")
                _emit('on_infer_end', result=result, args=args)
                return
            else:
                logger.warning("VLLM inference failed, falling back to native inference")
        except Exception as e:
            logger.error(f"VLLM inference error: {e}, falling back to native inference")
    
    # Native PyTorch inference
    try:
        logger.info("Starting native PyTorch inference...")
        
        # Prepare generation parameters
        generation_config = {
            'max_length': min(args.max_length, 2048),  # Safety limit
            'temperature': args.temperature,
            'top_p': args.top_p,
            'do_sample': True,
            'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'min_new_tokens': max(0, int(getattr(args, 'min_new_tokens', 16))),
        }
        # Optional: enable adaptive MoE runtime adjustment during generation
        if getattr(args, 'adaptive_moe', False):
            generation_config['adaptive_moe'] = {
                'enabled': True,
                'temp_step': float(getattr(args, 'adaptive_moe_temp_step', 0.03)),
                'interval': int(getattr(args, 'adaptive_moe_interval', 16)),
                'temp_cap': float(getattr(args, 'adaptive_moe_temp_cap', 1.30)),
            }
        
        # Add stop sequences if provided
        if args.stop:
            stop_sequences = args.stop if isinstance(args.stop, (list, tuple)) else [args.stop]
            generation_config['stop_sequences'] = stop_sequences
        
        # Generate text
        with torch.no_grad():
            if args.speculative and hasattr(model, 'speculative_generate'):
                # Use speculative decoding if available
                logger.info("Using speculative decoding...")
                output_ids = model.speculative_generate(
                    input_ids,
                    gamma=args.spec_gamma,
                    **generation_config
                )
            else:
                # Standard generation
                outputs = model.generate(input_ids, **generation_config)
                if isinstance(outputs, tuple):
                    output_ids = outputs[0]
                else:
                    output_ids = outputs[0] if outputs.dim() > 1 else outputs
            
            # Decode only newly generated tokens beyond the prompt length
            try:
                in_len = input_ids.shape[1] if hasattr(input_ids, 'shape') and len(input_ids.shape) == 2 else input_ids.shape[-1]
                if hasattr(output_ids, 'dim') and output_ids.dim() > 1:
                    new_ids = output_ids[:, in_len:]
                else:
                    # 1D tensor path
                    new_ids = output_ids[in_len:]
                if new_ids.numel() > 0:
                    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                else:
                    # Fallback: decode full output (may include prompt)
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            except Exception:
                # Conservative fallback
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Apply watermark
            try:
                from tools.infer.watermark import watermark_text
                watermark_metadata = {
                    "prompt": prompt,
                    "params": {
                        "temperature": args.temperature,
                        "max_length": args.max_length,
                        "top_p": args.top_p,
                        "use_vllm": False,
                        "speculative": args.speculative
                    },
                    "user_id": "piscesl1_user",
                    "timestamp": str(int(time.time()))
                }
                watermarked_text = watermark_text(generated_text, prompt, watermark_metadata)
                result = watermarked_text
            except Exception as e:
                logger.warning(f"Watermarking failed: {e}, using raw generated text")
                result = generated_text
            
            logger.success("Native inference completed successfully")
            print(f"Generated text: {result}")
            
            # Emit completion event
            _emit('on_infer_end', result=result, args=args)
            
            # Stop profiler if available
            if _PROFILER is not None and hasattr(_PROFILER, 'stop'):
                try:
                    _PROFILER.stop('infer')
                except Exception:
                    pass
            
            return result
            
    except Exception as e:
        logger.error(f"Native inference failed: {e}")
        _emit('on_infer_error', error=str(e), args=args)
        raise

# Module-level infer function for backward compatibility
def infer(args):
    """
    Perform inference using the module-level implementation.
    
    This function provides backward compatibility by delegating to the 
    PiscesLxToolsInferImpl class.
    
    Args:
        args: Arguments for the inference process.
        
    Returns:
        The result of the inference operation.
    """
    impl = PiscesLxToolsInferImpl()
    return impl.infer(args)
