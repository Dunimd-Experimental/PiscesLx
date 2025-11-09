#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
import time
import torch
from utils.log.core import PiscesLxCoreLog
from typing import Any, Dict, Optional
from .config import PiscesLxCoreDeviceConfig
from .manager import PiscesLxCoreDeviceManager
from utils.config.loader import PiscesLxCoreConfigLoader
from utils.hooks.bus import PiscesLxCoreHookBus, get_global_hook_bus
from utils.observability.decorators import PiscesLxCoreDecorators as ObsDec
from utils.error import PiscesLxCoreDeviceError, PiscesLxCoreNoGPUError, PiscesLxCoreGPUInsufficientError

logger = PiscesLxCoreLog("PiscesLx.Core.Device.Facade")

class PiscesLxCoreDeviceFacade:
    """
    Unified device management facade for PiscesL1.
    
    Consolidates device detection, configuration, and orchestration into a single
    entry point, eliminating redundant abstraction layers.
    """
    
    def __init__(self, args: Any = None) -> None:
        """
        Initialize the device facade with integrated orchestration.
        
        Args:
            args: Command line arguments or configuration object.
        """
        self.args = args or {}
        self._config_cache = None

        self.hooks = get_global_hook_bus()
        
        # Detect and store project root for local file lookups
        try:
            self.project_root = PiscesLxCoreConfigLoader()._project_root
        except Exception:
            self.project_root = None
        
        # Initialize configuration
        self.cfg = PiscesLxCoreDeviceConfig.from_args(args) if args else PiscesLxCoreDeviceConfig({})
        
        # Defer GPU manager construction to avoid heavy detection during __init__
        self.gpu_manager = None
                
        self._mode = "auto"
    
    @ObsDec.log_span("device.setup")
    def _auto_setup(self) -> Dict[str, Any]:
        """
        Automatically set up devices based on inference strategy.
        
        Returns:
            Dict[str, Any]: Device configuration dictionary with recommended settings.
        """
        # Temporary fast-return to unblock training if detection is fragile/slow
        try:
            if os.environ.get("PISCESLX_DEVICE_FAST_RETURN", "0").strip() in ("1", "true", "True"): 
                try:
                    import torch as _t
                    cuda_ok = bool(getattr(_t, "cuda", None) and _t.cuda.is_available())
                except Exception:
                    cuda_ok = False
                cfg = {
                    "device_type": "cuda" if cuda_ok else "cpu",
                    "strategy": "single_gpu" if cuda_ok else "cpu",
                    "gpu_ids": [0] if cuda_ok else [],
                    "batch_size": 1,
                    "mixed_precision": bool(cuda_ok),
                    "dtype": "bf16" if cuda_ok else "fp32",
                    "reason": "FAST_RETURN",
                    "memory_efficient": True,
                }
                try:
                    print(f"[device] FAST_RETURN active device_type={cfg['device_type']} strategy={cfg['strategy']}", flush=True)
                except Exception:
                    pass
                return cfg
        except Exception:
            pass
        # Simple path: skip advanced strategy computation if enabled
        try:
            if os.environ.get("PISCESLX_DEVICE_SIMPLE", "0").strip() in ("1", "true", "True"):
                try:
                    import torch as _t
                    cuda_ok = bool(getattr(_t, "cuda", None) and _t.cuda.is_available())
                except Exception:
                    cuda_ok = False
                return {
                    "device_type": "cuda" if cuda_ok else "cpu",
                    "strategy": "single_gpu" if cuda_ok else "cpu",
                    "gpu_ids": [0] if cuda_ok else [],
                    "batch_size": 1,
                    "mixed_precision": bool(cuda_ok),
                    "dtype": "bf16" if cuda_ok else "fp32",
                    "reason": "SIMPLE",
                    "memory_efficient": True,
                }
        except Exception:
            pass
        logger.info("_auto_setup enter")
        self._ensure_gpu_manager()
        try:
            self.gpu_manager.print_summary()
        except Exception:
            pass
        model_size = self.cfg.get("model.size", None)
        seq_len = int(self.cfg.get("inference.sequence_length", 1024))
        strategy = self.gpu_manager.get_inference_strategy(model_size=model_size, sequence_length=seq_len)
        try:
            print(f"[device] strategy computed mode={strategy.get('mode')} gpus={len(self.gpu_manager.gpu_info)}", flush=True)
        except Exception:
            pass
        
        if self.gpu_manager.gpu_info:
            first_type = self.gpu_manager.gpu_info[0].get('type', 'nvidia')
            device_type = 'cuda' if first_type == 'nvidia' else ('rocm' if first_type == 'amd' else 'cuda')
        else:
            device_type = 'cpu'
        
        gpu_ids = strategy.get('gpu_ids', [])
        try:
            print(f"[device] device_type={device_type} gpu_ids={gpu_ids}", flush=True)
        except Exception:
            pass
        
        def _recommend_dtype(device_type: str, gpu_ids: list) -> str:
            """
            Recommend data type conservatively without querying device capability to avoid hangs.
            CPU -> fp32; CUDA -> fp16 (safe on Turing+); BF16 decision left to higher-level configs.
            """
            try:
                if device_type == 'cpu':
                    return 'fp32'
                # Avoid get_device_capability; default to fp16 for CUDA
                if torch.cuda.is_available():
                    return 'fp16'
                return 'fp32'
            except Exception:
                return 'fp16' if device_type != 'cpu' else 'fp32'
        
        config: Dict[str, Any] = {
            "device_type": device_type,
            "strategy": strategy.get('mode', 'cpu' if device_type == 'cpu' else 'single_gpu'),
            "gpu_ids": gpu_ids,
            "batch_size": strategy.get('batch_size', 1),
            "mixed_precision": strategy.get('mixed_precision', device_type != 'cpu'),
            "dtype": _recommend_dtype(device_type, gpu_ids),
            "reason": strategy.get('reason', ''),
            "memory_efficient": True,
        }
        
        # Add distributed/cluster parameters if present
        for k in ("world_size", "rank", "local_rank", "master_addr", "master_port"):
            if k in strategy:
                config[k] = strategy[k]
        
        # Add memory and warning info
        for k in ("estimated_memory", "memory_margin_mb", "warning", "fallbacks"):
            if k in strategy:
                config[k] = strategy[k]
        
        if strategy.get('mode') == 'cpu':
            config["device_type"] = 'cpu'
        
        # Log only key fields to avoid heavy/unstable serialization
        logger.info(
            "Auto device config summary",
            device_type=config.get("device_type"),
            strategy=config.get("strategy"),
            gpu_ids_count=len(config.get("gpu_ids", [])),
            batch_size=config.get("batch_size"),
            mixed_precision=bool(config.get("mixed_precision")),
            dtype=config.get("dtype"),
        )
        logger.info("_auto_setup exit", device_type=config.get("device_type"), strategy=config.get("strategy"))
        try:
            print(f"[device] _auto_setup exit device_type={config.get('device_type')} strategy={config.get('strategy')}", flush=True)
        except Exception:
            pass
        return config
    
    def _manual_setup(self) -> dict:
        """
        Perform manual device setup with strict validation.
        
        Returns:
            dict: Manual device configuration.
            
        Raises:
            RuntimeError: If GPU is requested but not available.
            ValueError: If an unknown device type is specified.
        """
        self._ensure_gpu_manager()
        device_type = getattr(self.args, 'device', 'auto') if self.args else 'auto'
        batch_size = getattr(self.args, 'batch_size', None) if self.args else None
        
        if device_type == "auto":
            return self._auto_setup()
        
        gpu_info = self.gpu_manager.get_gpu_info()
        gpu_ids = None
        
        if device_type == "cuda":
            gpu_ids = getattr(self.args, 'gpu_ids', None)
            if gpu_ids is None:
                gpu_ids = [gpu["index"] for gpu in gpu_info]
            elif isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]
            elif isinstance(gpu_ids, str):
                gpu_ids = [int(x) for x in gpu_ids.split(",")]
            
            available_gpu_ids = [gpu["index"] for gpu in gpu_info]
            for gpu_id in gpu_ids:
                if gpu_id not in available_gpu_ids:
                    logger.error("GPU not found", requested_gpu=gpu_id, available_gpus=available_gpu_ids)
                    raise RuntimeError(f"Requested GPU {gpu_id} not available. Available: {available_gpu_ids}")
            
            if not gpu_info:
                logger.error("No GPU detected but CUDA device requested", requested_device="cuda", available_devices="cpu")
                raise RuntimeError("Configuration requires CUDA but no GPUs detected")
        
        elif device_type == "cpu":
            gpu_ids = []
        else:
            logger.error("Unknown device type", device_type=device_type)
            raise ValueError(f"Unknown device type: {device_type}")
        
        # Large model batch size validation
        if batch_size is not None and batch_size > 1:
            model_params = self._get_model_params_from_config()
            if model_params > 100:
                logger.error("Large model batch size validation failed", 
                                model_params=model_params, batch_size=batch_size)
                raise RuntimeError(f"Large model ({model_params:.1f}B) requires batch_size=1, got {batch_size}")
        
        config = {
            "device_type": device_type,
            "strategy": "manual",
            "gpu_ids": gpu_ids,
            "batch_size_recommendation": batch_size or 1,
            "memory_efficient": True,
            "reason": "Manual configuration (validated)"
        }
        
        logger.info("Manual device config", **config)
        return config
    
    def _distributed_setup(self) -> dict:
        """
        Set up devices for distributed training.
        
        Returns:
            dict: Distributed training configuration.
        """
        self._ensure_gpu_manager()
        recommendation = self.gpu_manager.get_recommendation()
        strategy = recommendation.get("strategy")
        gpu_info = self.gpu_manager.get_gpu_info()
        
        if strategy not in ["distributed", "ddp", "distributed_cluster"] and len(gpu_info) > 1:
            strategy = "ddp"
        
        config = {
            "device_type": "cuda",
            "strategy": strategy or "ddp",
            "batch_size_recommendation": recommendation.get("batch_size", 1),
            "memory_efficient": True,
            "world_size": recommendation.get("world_size", 1),
            "rank": recommendation.get("rank", 0),
        }
        
        # Add distributed parameters
        for k in ("local_rank", "master_addr", "master_port", "gpu_ids"):
            if k in recommendation:
                config[k] = recommendation[k]
        
        logger.info("Distributed device config", **config)
        return config
    
    def _cluster_setup(self) -> dict:
        """
        Set up devices for multi-node cluster training.
        
        Returns:
            dict: Cluster configuration dictionary.
        """
        recommendation = self.gpu_manager.get_recommendation()
        
        # Ensure cluster parameters are set
        cluster_params = {
            "world_size": getattr(self.args, 'world_size', 1) if self.args else 1,
            "rank": getattr(self.args, 'rank', 0) if self.args else 0,
            "local_rank": getattr(self.args, 'local_rank', 0) if self.args else 0,
            "master_addr": getattr(self.args, 'master_addr', "localhost") if self.args else "localhost",
            "master_port": getattr(self.args, 'master_port', "29500") if self.args else "29500"
        }
        
        for k, v in cluster_params.items():
            if k not in recommendation:
                recommendation[k] = v
        
        return self._auto_setup()  # Reuse auto setup logic
    
    def _get_model_params_from_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model parameters from configuration file.
        
        Args:
            model_name (Optional[str]): Specific model name to retrieve parameters for.
            
        Returns:
            Dict[str, Any]: Model parameters dictionary.
        """
        try:
            if not self.project_root:
                return {}
            config_path = self.project_root / "config" / "models.json"
            from utils.config.loader import load_config_from_file
            config = load_config_from_file(config_path)
            if model_name:
                return config.get(model_name, {})
            return config
        except Exception:
            return {}

    def _emit_hook_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Emit hook events safely via global hook bus (no-op on failure).
        
        Args:
            event_name (str): Name of the event to emit.
            payload (Dict[str, Any]): Event payload data.
        """
        try:
            if hasattr(self, "hooks") and self.hooks:
                # Assuming bus API has emit/publish interface; try emit then publish
                if hasattr(self.hooks, "emit"):
                    if isinstance(payload, dict):
                        self.hooks.emit(event_name, **payload)
                    else:
                        self.hooks.emit(event_name, payload=payload)
                elif hasattr(self.hooks, "publish"):
                    if isinstance(payload, dict):
                        self.hooks.publish(event_name, **payload)
                    else:
                        self.hooks.publish(event_name, payload=payload)
        except Exception as e:
            # Keep failures non-fatal and only debug-log
            logger.debug("hook emit failed", event=event_name, error=str(e))
    
    @ObsDec.log_span("device.setup")
    def setup_devices(self, mode: str = "auto") -> dict:
        """
        Set up devices based on mode.
        
        Args:
            mode (str): Device setup mode ("auto", "manual", "distributed", "cluster")
            
        Returns:
            dict: Device configuration dictionary
            
        Raises:
            RuntimeError: If device setup fails
            ValueError: If an unknown setup mode is provided
        """
        self._mode = mode
        logger.info("setup_devices enter", mode=mode)
        
        try:
            if mode == "auto":
                config = self._auto_setup()
            elif mode == "manual":
                config = self._manual_setup()
            elif mode == "distributed":
                config = self._distributed_setup()
            elif mode == "cluster":
                config = self._cluster_setup()
            else:
                logger.error("Unknown setup mode", mode=mode)
                raise ValueError(f"Unknown setup mode: {mode}")
            
            # Prepare to emit hook event (non-blocking)
            logger.info("setup_devices prepared config", **config)
            # Cache current config for downstream consumers (dtype/ids decisions)
            self._config_cache = config
            # Log and print exit BEFORE emitting hooks to avoid blocking training startup
            logger.info("setup_devices exit", device_type=config.get("device_type"), strategy=config.get("strategy"))
            try:
                print(f"[device] setup_devices exit device_type={config.get('device_type')} strategy={config.get('strategy')}")
            except Exception:
                pass
            # Fire hook asynchronously so it cannot block the caller
            try:
                import threading as _threading
                def _emit_async():
                    try:
                        self._emit_hook_event("device.setup_completed", {
                            "mode": mode,
                            "config": config,
                            "timestamp": time.time()
                        })
                    except Exception:
                        pass
                _t = _threading.Thread(target=_emit_async, daemon=True)
                _t.start()
            except Exception:
                pass
            return config
            
        except Exception as e:
            logger.error("Device setup failed", error=str(e), mode=mode)
            
            # Fallback to CPU if GPU setup fails
            if mode != "manual" and "gpu" in str(e).lower():
                logger.warning("GPU setup failed, falling back to CPU")
                return self.setup_devices("manual")
            
            raise RuntimeError(f"Device setup failed: {str(e)}") from e

    def setup_cluster(self, world_size: int = None, rank: int = None, local_rank: int = None, 
                       master_addr: str = None, master_port: int = None) -> dict:
        """
        Set up devices for cluster training.
        
        Args:
            world_size (int): Total number of processes
            rank (int): Global rank of this process
            local_rank (int): Local rank within this node
            master_addr (str): Master node address
            master_port (int): Master node port
            
        Returns:
            dict: Cluster configuration dictionary
            
        Raises:
            RuntimeError: If cluster setup fails or GPUs are required but not available
        """
        try:
            # Create cluster configuration
            cluster_config = {
                "world_size": world_size or 1,
                "rank": rank or 0,
                "local_rank": local_rank or 0,
                "master_addr": master_addr or "localhost",
                "master_port": master_port or 29500,
                "device_type": "cuda",
                "strategy": "distributed_cluster"
            }
            
            # Get GPU recommendation
            recommendation = self.gpu_manager.get_recommendation()
            cluster_config.update(recommendation)
            
            # Validate cluster setup
            if cluster_config["world_size"] > 1 and not self.gpu_manager.gpu_info:
                logger.error("Cluster setup requires GPUs but none detected")
                raise RuntimeError("Cluster training requires GPU devices")
            
            logger.info("Cluster setup completed", **cluster_config)
            return cluster_config
            
        except Exception as e:
            logger.error("Cluster setup failed", error=str(e))
            raise RuntimeError(f"Cluster setup failed: {str(e)}") from e
    
    def get_gpu_info(self) -> list:
        """
        Get detailed GPU information.
        
        Returns:
            list: List of GPU information dictionaries
        """
        self._ensure_gpu_manager()
        return self.gpu_manager.gpu_info

    def get_cluster_status(self) -> dict:
        """
        Get cluster status and configuration.

        Returns:
            dict: Cluster status information including distributed settings and GPU info.
        """
        cluster_info = {
            "enabled": getattr(self.args, 'distributed', {}).get('enabled', False),
            "world_size": getattr(self.args, 'distributed', {}).get('world_size', 1),
            "rank": getattr(self.args, 'distributed', {}).get('rank', 0),
            "local_rank": getattr(self.args, 'distributed', {}).get('local_rank', 0),
            "master_addr": getattr(self.args, 'distributed', {}).get('master_addr', 'localhost'),
            "master_port": getattr(self.args, 'distributed', {}).get('master_port', '29500'),
            "backend": getattr(self.args, 'distributed', {}).get('backend', 'nccl'),
            "gpu_info": self.get_gpu_info(),
        }
        
        if self._config_cache:
            cluster_info["current_strategy"] = self._config_cache.get("device_type", "unknown")
            cluster_info["batch_size"] = self._config_cache.get("batch_size", 1)
        
        return cluster_info
    
    def recommend_batch_size(self, model_size: str = None, sequence_length: int = 1024, 
                              precision: str = "auto") -> int:
        """
        Recommend optimal batch size based on model and hardware.
        
        Args:
            model_size (str): Model size specification (e.g., "7B", "13B", "70B")
            sequence_length (int): Input sequence length
            precision (str): Precision mode ("fp32", "fp16", "bf16", "auto")
            
        Returns:
            int: Recommended batch size
        """
        self._ensure_gpu_manager()
        return self.gpu_manager.recommend_batch_size(model_size, sequence_length, precision)
    
    def get_current_config(self) -> Optional[dict]:
        """
        Get the current device configuration if available.

        Returns:
            Optional[dict]: Current device configuration, or None if not set.
        """
        return self._config_cache
    
    def _ensure_gpu_manager(self) -> None:
        """Lazily construct GPU manager only when first needed."""
        if getattr(self, "gpu_manager", None) is not None:
            return
        try:
            self.gpu_manager = PiscesLxCoreDeviceManager(self.cfg)
        except PiscesLxCoreNoGPUError:
            logger.error("GPU detection failed. PiscesL1 requires CUDA-capable GPU for optimal performance.")
            gm = PiscesLxCoreDeviceManager.__new__(PiscesLxCoreDeviceManager)
            gm.cfg = self.cfg
            gm.gpu_info = []
            gm.strategy = {"mode": "cpu_fallback", "reason": "No GPU detected"}
            self.gpu_manager = gm
        except RuntimeError:
            raise
    
    def print_device_summary(self) -> None:
        """
        Print a summary of device information.
        """
        self.gpu_manager.print_summary()
    
    def validate_device_requirements(self, required_memory_gb: float, 
                                     required_gpu_count: int = 1) -> bool:
        """
        Validate that devices meet requirements.
        
        Args:
            required_memory_gb (float): Required memory in GB
            required_gpu_count (int): Required number of GPUs
            
        Returns:
            bool: True if requirements are met, False otherwise
        """
        gpu_info = self.gpu_manager.get_gpu_info()
        
        # Check GPU count
        available_gpus = len(gpu_info)
        if available_gpus < required_gpu_count:
            logger.warning("Insufficient GPUs", {
                "available": available_gpus,
                "required": required_gpu_count
            })
            return False
        
        # Check memory requirements
        if required_memory_gb > 0:
            max_available = max(gpu.get("memory_total", 0) for gpu in gpu_info) if gpu_info else 0
            max_available_gb = max_available / 1024**3
            
            if max_available_gb < required_memory_gb:
                logger.warning("Insufficient memory", {
                    "available_gb": max_available_gb,
                    "required_gb": required_memory_gb
                })
                return False
        
        logger.info(
            "Device requirements validated",
            required_memory_gb=required_memory_gb,
            required_gpu_count=required_gpu_count,
            available_gpus=available_gpus,
            max_available_gb=max_available_gb if required_memory_gb > 0 else "N/A",
        )
        return True

    def init_distributed(self, backend: str = "nccl") -> dict:
        """
        Initialize torch.distributed process group and set current device.
        Returns a dictionary with distributed metadata that train/infer can consume.
        """
        import torch
        import torch.distributed as dist
        try:
            world_size = int(os.environ.get('WORLD_SIZE', self.cfg.get("distributed.world_size", 1)))
            rank = int(os.environ.get('RANK', self.cfg.get("distributed.rank", 0)))
            local_rank = int(os.environ.get('LOCAL_RANK', self.cfg.get("distributed.local_rank", 0)))
            master_addr = os.environ.get('MASTER_ADDR', self.cfg.get("distributed.master_addr", "localhost"))
            master_port = os.environ.get('MASTER_PORT', str(self.cfg.get("distributed.master_port", "29500")))

            distributed = world_size > 1 and dist.is_available()
            if distributed and not dist.is_initialized():
                os.environ.setdefault('MASTER_ADDR', master_addr)
                os.environ.setdefault('MASTER_PORT', master_port)
                dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method="env://")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.set_device(local_rank)
                    except Exception as e:
                        logger.debug("set_device failed", error=str(e))

            cfg = {
                "distributed": distributed,
                "backend": backend if distributed else None,
                "world_size": world_size,
                "rank": rank,
                "local_rank": local_rank,
                "master_addr": master_addr,
                "master_port": master_port,
            }
            self._emit_hook_event("device.distributed_initialized", cfg)
            return cfg
        except Exception as e:
            logger.error("Distributed init failed", error=str(e))
            return {
                "distributed": False,
                "backend": None,
                "world_size": 1,
                "rank": 0,
                "local_rank": 0,
            }

    def amp_dtype(self, dtype_str: str = "auto"):
        """
        Map precision preference to torch.dtype, with 'auto' selecting BF16 on Ampere+ else FP16,
        and FP32 on CPU.
        """
        import torch
        s = (dtype_str or "auto").lower()
        if s == "fp16":
            return torch.float16
        if s == "bf16":
            return torch.bfloat16
        if s == "fp32":
            return torch.float32
        # auto decision
        if not torch.cuda.is_available():
            return torch.float32
        try:
            gpu_ids = []
            if self._config_cache and self._config_cache.get("gpu_ids"):
                gpu_ids = self._config_cache["gpu_ids"]
            elif getattr(self, "gpu_manager", None) and self.gpu_manager.gpu_info:
                gpu_ids = [self.gpu_manager.gpu_info[0].get("index", 0)]
            idx = gpu_ids[0] if gpu_ids else 0
            major, _ = torch.cuda.get_device_capability(idx)
            return torch.bfloat16 if major >= 8 else torch.float16
        except Exception:
            return torch.float16

    def map_vllm_dtype(self, torch_dtype) -> str:
        """
        Convert torch.dtype to vLLM dtype string.
        """
        import torch
        if torch_dtype == torch.bfloat16:
            return "bfloat16"
        if torch_dtype == torch.float16:
            return "float16"
        return "float32"

