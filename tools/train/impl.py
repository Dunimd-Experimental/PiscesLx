#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import json
import torch
import warnings
from utils.cache import get_cache_manager
from utils.device.facade import PiscesLxCoreDeviceFacade
from utils.device.manager import PiscesLxCoreDeviceManager
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
from utils import PiscesLxCoreConfigManager, PiscesLxCoreCheckpointManager

_HOOKS = None
_PROFILER = None
_CFG = None
COLLATE_MAX_SEQ_LEN = 96

# Local compatibility wrapper to avoid importing utils functions
def get_cache_manager():
    try:
        from utils.cache import get_cache_manager as get_cache_manager_facade
        return get_cache_manager_facade()
    except Exception:
        from utils.cache import PiscesLxCoreCacheManagerFacade
        return PiscesLxCoreCacheManagerFacade.get_instance()

def set_context(*, hooks=None, profiler=None, cfg=None):
    """
    Set the runtime context for hooks, profiler, and configuration.

    Args:
        hooks: Hooks object, used to trigger events during the training process. Defaults to None.
        profiler: Profiler object, used for performance profiling. Defaults to None.
        cfg: Configuration object, used to store training configurations. Defaults to None.
    """
    global _HOOKS, _PROFILER, _CFG
    _HOOKS = hooks
    _PROFILER = profiler
    _CFG = cfg

def _emit(event: str, **kwargs):
    """
    Emit a specified event through the hooks object.

    Args:
        event (str): The name of the event to emit.
        **kwargs: Additional keyword arguments to pass to the event handler.
    """
    try:
        if _HOOKS is not None:
            _HOOKS.emit(event, **kwargs)
    except Exception:
        # Never break training on hook emission
        pass

def setup_distributed_training():
    """
    Unified distributed training setup via utils.device.PiscesLxCoreDeviceFacade.
    Initializes process group, selects device, and recommends simple 3D parallel config.
    """
    import torch
    from utils.device.facade import PiscesLxCoreDeviceFacade

    logger = PiscesLxCoreLog("pisceslx.tools.train.impl.setup_distributed_training")

    # Use facade for device + distributed init
    facade = PiscesLxCoreDeviceFacade()
    dev_cfg = facade.setup_devices(mode="distributed")
    dist_cfg = facade.init_distributed(backend="nccl")

    is_distributed = bool(dist_cfg.get("distributed", dev_cfg.get("world_size", 1) > 1))
    local_rank = int(dist_cfg.get("local_rank", dev_cfg.get("local_rank", 0)))
    world_size = int(dist_cfg.get("world_size", dev_cfg.get("world_size", 1)))
    rank = int(dist_cfg.get("rank", dev_cfg.get("rank", 0)))

    # Select device based on facade config
    if dev_cfg.get("device_type") == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("Training mode: CPU (Device Facade)")
    else:
        device = torch.device(f"cuda:{local_rank}")
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        logger.info(f"Training mode: CUDA device {local_rank} (Device Facade)")

    # Simple 3D parallel suggestion (engine-agnostic)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dp_size = max(1, world_size) if is_distributed else 1
    pp_size = 1
    mp_size = 1
    if num_gpus >= 8 and is_distributed:
        dp_size = max(1, world_size // 2)
        pp_size = 2
        mp_size = 2
    elif num_gpus >= 4 and is_distributed:
        dp_size = max(1, world_size // 2)
        pp_size = 2

    _emit("train.distributed.setup",
          world_size=world_size, rank=rank, local_rank=local_rank,
          dp_size=dp_size, pp_size=pp_size, mp_size=mp_size)

    parallel_config = {
        "dp_size": dp_size,
        "pp_size": pp_size,
        "mp_size": mp_size,
        "total_gpus": dp_size * pp_size * mp_size,
        "device_type": device.type,
        "device_index": device.index if hasattr(device, "index") else 0
    }

    return device, is_distributed, local_rank, world_size, parallel_config

def create_ddp_model(model, device, is_distributed, local_rank):
    """
    Create a distributed data parallel model.
    Enhanced with utils device management for optimal model placement.

    Args:
        model (torch.nn.Module): The model to be wrapped.
        device (torch.device): The device to use for training.
        is_distributed (bool): Whether the training is distributed.
        local_rank (int): The local rank of the current process.

    Returns:
        torch.nn.Module: The wrapped model.
    """
    # from utils.device import PiscesLxCoreModelParallelizer  # Commented out as it's not available
    
    # Initialize logger for this function
    logger = PiscesLxCoreLog("pisceslx.tools.train.impl.create_ddp_model")
    
    if is_distributed:
        # Move the model to the specified device with utils-enhanced placement
        model = model.to(device)
        
        # Emit model distribution event for observability
        _emit("train.model.distribute", 
              device=str(device), local_rank=local_rank, 
              model_params=sum(p.numel() for p in model.parameters()))
        
        # Wrap the model with DistributedDataParallel using utils recommendations
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    else:
        # Single-GPU mode with utils device management
        device_facade = PiscesLxCoreDeviceFacade()
        device_config = device_facade.setup_devices(mode="auto")
        
        # Move the model to the specified device
        model = model.to(device)
        
        # If multiple GPUs are available but not using distributed training, use DataParallel
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            logger.info(f"Detected {gpu_count} GPUs, using DataParallel")
            
            # Emit multi-GPU setup event for observability
            _emit("train.model.dataparallel", gpu_count=gpu_count)
            
            model = torch.nn.DataParallel(model)
        
        # Emit model placement event for observability
        _emit("train.model.place", device=str(device), gpu_count=gpu_count)
            
    return model

def create_dataloader(dataset, batch_size, is_distributed, world_size=1, rank=0, local_rank=0):
    """
    Create an optimized data loader with async prefetching and multi-threaded augmentation.

    Args:
        dataset: The dataset to load data from.
        batch_size (int): The number of samples per batch to load.
        is_distributed (bool): Whether the training is distributed.
        world_size (int, optional): The total number of processes in the distributed training. Defaults to 1.
        rank (int, optional): The global rank of the current process. Defaults to 0.
        local_rank (int, optional): The local rank of the current process. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: An optimized data loader.
    """
    import torch
    from torch.utils.data import DistributedSampler
    import multiprocessing as mp
    
    # Initialize logger for this function
    logger = PiscesLxCoreLog("pisceslx.tools.train.impl.create_dataloader")
    
    # Handle empty dataset to avoid deadlock
    if len(dataset) == 0:
        from torch.utils.data import Dataset
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            def __getitem__(self, idx):
                return {}
        return torch.utils.data.DataLoader(EmptyDataset(), batch_size=1)
    
    if is_distributed:
        # Create a distributed sampler for distributed training
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Optimized configuration for async prefetching and multi-threading
    num_workers = min(mp.cpu_count(), 8)  # Use up to 8 CPU cores
    prefetch_factor = 4 if num_workers > 0 else 2  # Increase prefetch for better GPU utilization
    persistent_workers = num_workers > 0  # Keep workers alive between epochs
    pin_memory = torch.cuda.is_available()  # Enable pin memory for GPU training
    
    # Async data augmentation for multimodal inputs
    def async_collate_fn(batch):
        """
        Enhanced collate function with async data augmentation.

        Args:
            batch: A batch of data samples.

        Returns:
            The augmented and collated batch data.
        """
        # Pre-fetch next batch while processing current batch
        augmented_batch = []
        for item in batch:
            # Apply lightweight data augmentation for multimodal inputs
            if 'pixel_values' in item and item['pixel_values'] is not None:
                # Random horizontal flip for images (10% probability)
                if torch.rand(1).item() < 0.1:
                    item['pixel_values'] = torch.flip(item['pixel_values'], dims=[-1])
            
            if 'audio_input' in item and item['audio_input'] is not None:
                # Random noise injection for audio (5% probability, 0.01 amplitude)
                if torch.rand(1).item() < 0.05:
                    if isinstance(item['audio_input'], dict) and 'input_values' in item['audio_input']:
                        noise = torch.randn_like(item['audio_input']['input_values']) * 0.01
                        item['audio_input']['input_values'] = item['audio_input']['input_values'] + noise
            
            augmented_batch.append(item)
        
        # Use original collate function for final processing
        return collate_fn(augmented_batch)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=async_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,  # Drop incomplete batches for better GPU utilization
        multiprocessing_context='spawn' if num_workers > 0 else None  # Use spawn for stability
    )

def collate_fn(batch):
    """
    Memory-optimized collate function for Arctic architecture.

    Args:
        batch: A batch of data samples.

    Returns:
        dict: A dictionary containing collated and processed data, including input_ids, labels, pixel_values, and audio_input.
    """
    import torch
    # Sequence length now comes from model training_config (set in _train_impl)
    global COLLATE_MAX_SEQ_LEN
    MAX_SEQ_LEN = COLLATE_MAX_SEQ_LEN
    
    # Extract input_ids from the batch and pad them
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, :MAX_SEQ_LEN]
    
    # Handle pixel_values for vision modality to limit memory usage
    pixel_values = None
    if any(item.get("pixel_values") is not None for item in batch):
        pixel_values_list = [item["pixel_values"] for item in batch if item.get("pixel_values") is not None]
        if pixel_values_list:
            pixel_values = torch.stack(pixel_values_list)
    
    # Handle audio_input for audio modality
    audio_input = None
    if any(item.get("audio_input") is not None for item in batch):
        audio_input_list = [item["audio_input"]['input_values'] for item in batch if item.get("audio_input") is not None and item["audio_input"].get('input_values') is not None]
        if audio_input_list:
            audio_input = {'input_values': torch.nn.utils.rnn.pad_sequence(audio_input_list, batch_first=True, padding_value=0)}
    
    # Create labels based on input_ids
    labels = input_ids.clone()
    if labels.shape[1] > MAX_SEQ_LEN:
        labels = labels[:, :MAX_SEQ_LEN]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "audio_input": audio_input
    }

def _train_impl(args):
    """
    Implement the actual training logic of the Pisces model.

    Args:
        args: Command line arguments or configuration object containing training parameters.
    """
    import torch
    from tools.data.dataset.core import PiscesDataset
    from torch.utils.data import DataLoader
    from model.tokenizer import get_tokenizer
    from model import ArcticModel, ArcticConfig
    from utils.checkpoint import save_ckpt, load_ckpt
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from transformers.optimization import get_linear_schedule_with_warmup
    
    # Initialize logger for this function
    logger = PiscesLxCoreLog("pisceslx.tools.train.impl._train_impl")
    
    # Get the model size and construct the configuration file path
    model_size = getattr(args, 'model_size', '0.5B').upper()
    config_path = f"configs/{model_size}.json"
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found. Please provide a valid --model_size.")
        sys.exit(1)

    # Load the full configuration file
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    if 'training_config' not in full_config:
        logger.error(f"training_config not found in {config_path}")
        sys.exit(1)
    
    # Extract training configuration parameters (config-first, with legacy defaults)
    training_config = full_config['training_config']
    batch_size = training_config.get('batch_size', 1)
    accum = training_config.get('accum', 1)
    max_accum = training_config.get('max_accum', accum)
    seq_len = training_config.get('seq_len', 96)
    force_quant = training_config.get('force_quant', False)
    force_lora = training_config.get('force_lora', False)
    lr = training_config.get('lr', 1e-4)
    # Newly configurable (was hardcoded)
    epochs = int(training_config.get('epochs', 1))
    min_plateau_epoch = int(training_config.get('min_plateau_epoch', 5))
    loss_stop_threshold = float(training_config.get('loss_stop_threshold', 2.8))
    max_epochs_stop = int(training_config.get('max_epochs_stop', 6))
    auto_export_safetensors = bool(training_config.get('auto_export_safetensors', False))

    # De-hardcode collate max seq len: take from training_config if provided
    # Default to training seq_len to avoid duplication
    global COLLATE_MAX_SEQ_LEN
    try:
        COLLATE_MAX_SEQ_LEN = int(training_config.get('collate_seq_len', seq_len))
    except Exception:
        COLLATE_MAX_SEQ_LEN = seq_len
    
    # Quantization settings - only from command line or auto-detection, not from config
    quant_bits = 4  # Default quantization bits
    if hasattr(args, 'quant_bits') and args.quant_bits is not None:
        try:
            quant_bits = int(args.quant_bits)
        except Exception:
            logger.error("quant_bits must be integer; falling back to default (4)")
        
    if hasattr(args, 'force_quant') and args.force_quant:
        force_quant = True
        logger.info(f"Command line override: force_quant = true ({quant_bits}-bit)")
    if hasattr(args, 'force_lora') and args.force_lora:
        force_lora = True
        logger.info("Command line override: force_lora = true")
    if hasattr(args, 'quant') and args.quant:
        force_quant = True
        logger.info(f"Command line override: quant = true ({quant_bits}-bit)")
    if hasattr(args, 'no_quant') and args.no_quant:
        force_quant = False
        logger.info("Command line override: force_quant = false")
    
    # Auto-detect quantization needs based on hardware resources (only if not explicitly disabled)
    if not hasattr(args, 'no_quant') and not hasattr(args, 'force_quant') and not hasattr(args, 'quant'):
        try:
            from utils.device.manager import PiscesLxCoreDeviceManager
            device_manager = PiscesLxCoreDeviceManager()
            # Get inference strategy to check quantization recommendations
            strategy = device_manager.get_inference_strategy(model_size=model_size)
            if strategy.get('quantization_needed', False):
                recommended_bits = strategy.get('recommended_quant_bits', 4)
                if recommended_bits in [2, 4, 8]:
                    quant_bits = recommended_bits
                    force_quant = True
                    logger.info(f"Auto-detected quantization need: {quant_bits}-bit quantization recommended based on hardware resources")
        except Exception as e:
            logger.debug(f"Failed to auto-detect quantization needs: {e}")
    
    # Honor epochs from training_config. epochs=0 means no epoch cap (auto convergence)
    cache_manager = get_cache_manager()
    save_dir = str(cache_manager.get_cache_dir("ckpt"))
    os.makedirs(save_dir, exist_ok=True)
    
    class DynamicGradientAccumulator:
        """
        Safe dynamic gradient accumulation based on gradient norm monitoring.

        Args:
            base_accum (int): The base gradient accumulation steps.
            max_accum (int): The maximum gradient accumulation steps.
            target_grad_norm (float, optional): The target gradient norm. Defaults to 1.0.
            safety_factor (float, optional): The safety factor. Defaults to 0.8.
        """
        def __init__(self, base_accum, max_accum, target_grad_norm=1.0, safety_factor=0.8):
            self.base_accum = base_accum
            self.max_accum = max_accum
            self.target_grad_norm = target_grad_norm
            self.safety_factor = safety_factor
            self.current_accum = base_accum
            self.grad_norm_history = []
            self.stable_steps = 0
            
        def should_step(self, grad_norm):
            """
            Determine whether to perform a gradient update based on the gradient norm.

            Args:
                grad_norm (float): The current gradient norm.

            Returns:
                bool: Whether to perform a gradient update.
            """
            self.grad_norm_history.append(grad_norm)
            if len(self.grad_norm_history) > 10:
                self.grad_norm_history.pop(0)
            if len(self.grad_norm_history) >= 3:
                recent_avg = sum(self.grad_norm_history[-3:]) / 3
                if recent_avg < self.target_grad_norm * 0.5 and self.current_accum < self.max_accum:
                    self.current_accum = min(self.current_accum + 1, self.max_accum)
                    self.stable_steps = 0
                elif recent_avg > self.target_grad_norm * 2.0 and self.current_accum > self.base_accum:
                    self.current_accum = max(self.current_accum - 1, self.base_accum)
                    self.stable_steps = 0
                else:
                    self.stable_steps += 1
            return len(self.grad_norm_history) >= self.current_accum
            
        def get_current_accum(self):
            """
            Get the current gradient accumulation steps.

            Returns:
                int: The current gradient accumulation steps.
            """
            return max(1, int(self.current_accum * self.safety_factor))
            
        def reset(self):
            """
            Reset the gradient accumulation state.
            """
            self.current_accum = self.base_accum
            self.grad_norm_history.clear()
            self.stable_steps = 0
    
    # Initialize the dynamic gradient accumulator
    grad_accumulator = DynamicGradientAccumulator(accum, max_accum)
    
    min_plateau_epoch = 5
    scheduler = None
    
    # Handle dataset selection: command line takes priority over model.txt
    dataset_list = []
    if hasattr(args, 'dataset') and args.dataset:
        dataset_list = [args.dataset]
        logger.info(f"Using command line dataset: {args.dataset}")
    else:
        cache_manager = get_cache_manager()
        data_cache_dir = str(cache_manager.get_cache_dir("data_cache"))
        os.makedirs(data_cache_dir, exist_ok=True)
        model_txt = os.path.join(data_cache_dir, "model.txt")
        if not os.path.exists(model_txt):
            logger.error(f"{model_txt} not found! Please create it with one dataset name per line, or use --dataset argument.")
            sys.exit(1)
        with open(model_txt, "r", encoding="utf-8") as f:
            dataset_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        if not dataset_list:
            logger.error(f"No dataset names found in {model_txt}! Please use --dataset argument instead.")
            sys.exit(1)
    
    # Set up distributed training
    device, is_distributed, local_rank, world_size, _ = setup_distributed_training()
    
    # Adjust batch size according to hardware and distributed environment
    if is_distributed:
        effective_batch_size = batch_size // world_size
        if effective_batch_size < 1:
            effective_batch_size = 1
            batch_size = world_size
        logger.info(f"Distributed training: global batch_size={batch_size}, per-GPU batch_size={effective_batch_size}")
    else:
        # Use unified device facade's batch_size unless CLI explicitly overrides
        try:
            from utils.device.facade import PiscesLxCoreDeviceFacade
            f = PiscesLxCoreDeviceFacade(args)
            dev_cfg_b = f.setup_devices(mode="auto")
            dev_bs = int(dev_cfg_b.get("batch_size", batch_size))
            cli_batch = getattr(args, "batch_size", None)
            batch_size = int(cli_batch) if cli_batch is not None else dev_bs
        except Exception:
            pass
        effective_batch_size = batch_size
    
    # Set up mixed precision training via unified device facade
    from utils.device.facade import PiscesLxCoreDeviceFacade
    f = PiscesLxCoreDeviceFacade(args)
    _dev_cfg = f.setup_devices(mode="distributed" if is_distributed else "auto")
    _amp_dtype = f.amp_dtype(_dev_cfg.get("dtype", "auto"))
    _mp_enabled = bool(_dev_cfg.get("mixed_precision", torch.cuda.is_available())) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if _mp_enabled else None
    # Apply silently without printing suggestions
        
    logger.info(f"Device setup completed: {device}")
    logger.info("Loading ArcticConfig...")
    config = f"configs/{model_size}.json"
    if not os.path.exists(config):
        logger.error(f"Config file {config} not found. Please provide a valid --model_size")
        sys.exit(1)
    logger.info(f"Loading config file: {config}")
    cfg = ArcticConfig.from_json(config)
    logger.info("ArcticConfig loaded.")

    logger.info("Initializing ArcticModel with Reasoner...")
    _emit('on_train_start', args=args)
    
    # Always - on Reasoner: Tokenizer setup
    tokenizer = get_tokenizer()
    special_tokens = training_config.get('special_tokens', ["<think>", "</think>"])
    tokenizer.add_tokens(special_tokens)

    # Handle quantization override from command line
    force_quant_override = None
    if hasattr(args, 'quant') and args.quant:
        force_quant_override = True
    elif hasattr(args, 'no_quant') and args.no_quant:
        force_quant_override = False
    
    # Use override if provided, otherwise use default config
    use_quant = force_quant_override if force_quant_override is not None else force_quant
    
    model = None
    if use_quant or force_lora:
        from transformers.utils.quantization_config import BitsAndBytesConfig
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            # If peft is not available, create dummy classes
            class LoraConfig:
                def __init__(self, **kwargs):
                    pass
            
            def get_peft_model(model, config):
                return model
            
            class TaskType:
                CAUSAL_LM = "CAUSAL_LM"
        quant_config = None
        if use_quant:
            if quant_bits == 2:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                logger.info(f"Using 2 - bit quantization (experimental, based on 4 - bit+compression)")
            elif quant_bits == 4:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info(f"Using 4 - bit quantization (NF4)")
            elif quant_bits == 8:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                logger.info(f"Using 8 - bit quantization (INT8)")
            else:
                logger.error(f"Unsupported quantization bits: {quant_bits}. Supported: 2, 4, 8")
                sys.exit(1)
        model = ArcticModel(cfg, quantization_config=quant_config) if quant_config else ArcticModel(cfg)
        if force_lora:
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
            )
            lora_model = get_peft_model(model, lora_config)
            for attr in ["cfg", "quantization_config", "lora_config", "forward", "prepare_inputs_for_generation"]:
                if hasattr(model, attr):
                    setattr(lora_model, attr, getattr(model, attr))
            model = lora_model
            try:
                if hasattr(model, 'print_trainable_parameters'):
                    model.print_trainable_parameters()
            except Exception:
                # If print_trainable_parameters is not available, calculate manually
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    else:
        model = ArcticModel(cfg)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        start_id = tokenizer.encoder.get("<think>")
        end_id = tokenizer.encoder.get("</think>")
        if start_id is None or end_id is None:
            raise ValueError("Special reasoning tokens could not be added to the tokenizer.")
        if hasattr(model, 'reasoner') and hasattr(model.reasoner, 'start_thinking_id'):
            model.reasoner.start_thinking_id = start_id
        if hasattr(model, 'reasoner') and hasattr(model.reasoner, 'end_thinking_id'):
            model.reasoner.end_thinking_id = end_id
        logger.info(f"Reasoner is integral and configured with token IDs: start={start_id}, end={end_id}")
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info(f"Gradient Checkpointing enabled.")
    else:
        logger.info(f"Gradient Checkpointing not available for this model.")
        
    model = model.to(device)
    logger.info("ArcticModel initialized.")
    # DataParallel removed; rely on DDP when is_distributed=True
    logger.info("Initializing optimizer and scheduler...")
    
    def get_parameter_groups(model, base_lr):
        """
        Group model parameters by modality and set different learning rates for each group.

        Args:
            model (torch.nn.Module): The model to group parameters from.
            base_lr (float): The base learning rate.

        Returns:
            list: A list of parameter groups, each containing parameters and their corresponding learning rate.
        """
        param_groups = []
        text_params = []
        vision_params = []
        audio_params = []
        fusion_params = []
        other_params = []
        
        model_params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'vision' in name.lower() or 'visual' in name.lower():
                vision_params.append(param)
            elif 'audio' in name.lower() or 'speech' in name.lower():
                audio_params.append(param)
            elif 'fusion' in name.lower() or 'multimodal' in name.lower():
                fusion_params.append(param)
            elif 'text' in name.lower() or 'language' in name.lower():
                text_params.append(param)
            else:
                other_params.append(param)
        
        if text_params:
            param_groups.append({'params': text_params, 'lr': base_lr, 'name': 'text'})
        if vision_params:
            param_groups.append({'params': vision_params, 'lr': base_lr * 0.5, 'name': 'vision'})
        if audio_params:
            param_groups.append({'params': audio_params, 'lr': base_lr * 0.3, 'name': 'audio'})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': base_lr * 1.5, 'name': 'fusion'})
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr, 'name': 'other'})
        
        return param_groups
    
    param_groups = get_parameter_groups(model, lr)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False
    )
    
    def create_modality_scheduler(optimizer, modality_name, base_scheduler_config):
        """
        Create a learning rate scheduler for a specific modality.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to.
            modality_name (str): The name of the modality, e.g., 'text', 'vision', 'audio', 'fusion', 'other'.
            base_scheduler_config (dict): The base configuration for the scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: A learning rate scheduler object.
        """
        if modality_name == 'vision':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
        elif modality_name == 'audio':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=5e-8
            )
        elif modality_name == 'fusion':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=1, eta_min=2e-7
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=8, T_mult=1, eta_min=1e-7
            )
    
    # Modality schedulers use config overrides if provided
    modality_schedulers = {}
    sched_cfg = training_config.get('schedulers', {})
    for modality in ['text', 'vision', 'audio', 'fusion', 'other']:
        scheduler_obj = create_modality_scheduler(optimizer, modality, sched_cfg.get(modality, {}))
        if scheduler_obj is not None:
            modality_schedulers[modality] = scheduler_obj
    
    logger.info("Optimizer and modality-aware schedulers ready.")
    
    clip_cfg = training_config.get('grad_clip', {})
    grad_clipper = AdaptiveGradientClipper(
        initial_max_norm=float(clip_cfg.get('initial_max_norm', 1.0)),
        history_length=int(clip_cfg.get('history_length', 100)),
        percentile=int(clip_cfg.get('percentile', 95)),
        min_clip=float(clip_cfg.get('min_clip', 0.1)),
        max_clip=float(clip_cfg.get('max_clip', 10.0)),
        warmup_steps=int(clip_cfg.get('warmup_steps', 10)),
        use_kfac=bool(clip_cfg.get('use_kfac', True)),
        kfac_update_freq=int(clip_cfg.get('kfac_update_freq', 100)),
        kfac_damping=float(clip_cfg.get('kfac_damping', 0.001)),
    )
    resume_ckpt = getattr(args, 'resume_ckpt', None)
    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        start_epoch = load_ckpt(resume_ckpt, model, optimizer)
        logger.info(f"Resumed at epoch {start_epoch}")

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            logger.info(f"Learning rate auto - reset to {lr}")
        min_lr_threshold = lr * 0.5
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr_threshold:
                param_group['lr'] = lr
                logger.info(f"Learning rate auto - reset to {lr}")
    for dataset in dataset_list:
        logger.debug(f"\n==============================")
        logger.info(f"Training dataset: {dataset}")
        logger.info(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
        cache_manager = get_cache_manager()
        data_cache_dir = str(cache_manager.get_cache_dir("data_cache"))
        os.makedirs(data_cache_dir, exist_ok=True)
        cache_path = os.path.join(data_cache_dir, dataset)
        if not os.path.exists(cache_path):
            logger.error(f"Local dataset not found: {cache_path}")
            continue
        # Convert ArcticConfig to dict if needed
        if hasattr(cfg, 'to_dict'):
            config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else {}
        elif hasattr(cfg, '__dict__'):
            config_dict = cfg.__dict__
        else:
            config_dict = {}
        train_ds = PiscesDataset(subset=dataset, split="train", config=config_dict)
        if len(train_ds) == 0:
            logger.debug(f"Warning: Dataset '{dataset}' is empty after filtering. Skipping.")
            continue
        logger.info(f"Dataset loaded successfully, size: {len(train_ds)}")
        logger.info("Creating DataLoader...")
        rank = int(os.environ.get('RANK', 0))
        train_loader = create_dataloader(
            train_ds,
            batch_size=effective_batch_size,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank if is_distributed else 0
        )
        logger.info("DataLoader created successfully")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Starting training loop...")
        model.train()
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stop_training = False
        epoch = start_epoch
        
        if is_distributed:
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
        
        try:
            while not stop_training:
                if is_distributed:
                    train_loader.sampler.set_epoch(epoch)
                
                logger.debug(f"Starting epoch {epoch+1}")
                _emit('on_epoch_start', epoch=epoch+1)
                total_loss = 0
                step_loss = 0
                optimizer.zero_grad()
                grad_accumulator.reset()
                grad_clipper.reset()
                
                for step, batch in enumerate(train_loader):
                    model_keys = ["input_ids", "labels", "pixel_values", "audio_input"]
                    device_batch = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items() if k in model_keys and v is not None
                    }
                    
                    pixel_values = device_batch.get("pixel_values")
                    audio_input = device_batch.get("audio_input")
                    if audio_input is not None and isinstance(audio_input, dict):
                        audio_input = {k: v.to(device) for k, v in audio_input.items()}
                    
                    loss = None
                    with torch.amp.autocast('cuda', enabled=_mp_enabled, dtype=_amp_dtype):
                        outputs = model(
                            input_ids=device_batch["input_ids"],
                            labels=device_batch["labels"],
                            images=pixel_values,
                            audio=audio_input
                        )
                        loss = outputs.get("loss")
                        
                        if hasattr(outputs, 'task_losses') and outputs.task_losses is not None:
                            task_losses = outputs.task_losses
                            if not hasattr(model, 'log_vars'):
                                num_tasks = len(task_losses)
                                model.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))
                                model.log_vars.to(device)
                            precision = torch.exp(-model.log_vars)
                            weighted_losses = precision * task_losses + 0.5 * model.log_vars
                            loss = weighted_losses.sum()
                            if step % 100 == 0:
                                uncertainties = torch.exp(model.log_vars)
                                for i, uncertainty in enumerate(uncertainties):
                                    logger.debug(f"Task {i} uncertainty: {uncertainty.item():.4f}")
                        
                        if loss is None and outputs.get("loss") is not None:
                            loss = outputs.get("loss")

                    if loss is not None and loss.requires_grad:
                        if is_distributed or torch.cuda.device_count() > 1:
                            loss = loss.mean()
                        
                        current_accum = grad_accumulator.get_current_accum()
                        
                        if scaler is not None:
                            scaler.scale(loss / current_accum).backward()
                        else:
                            (loss / current_accum).backward()
                        
                        step_loss += loss.item()
                        
                        grad_norm = 0.0
                        params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
                        for param in params:
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        
                        if grad_accumulator.should_step(grad_norm):
                            params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
                            total_norm, clip_coef, was_clipped = grad_clipper.update_and_clip(params, optimizer, scaler)
                            torch.cuda.empty_cache()
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            step_loss = 0
                            if not is_distributed or local_rank == 0:
                                if step % 50 == 0:
                                    current_threshold = grad_clipper.get_current_threshold()
                                    clip_ratio = (1.0 - clip_coef) * 100 if was_clipped else 0.0
                                    logger.info(f"GradClip: norm={total_norm:.4f}, threshold={current_threshold:.4f}, clipped={was_clipped}, clip_ratio={clip_ratio:.1f}%")
                    else:
                        logger.debug(f"Warning: Skipping step {step} due to invalid loss (None or no grad).")
                        continue

                    if loss is not None:
                        current_accum = grad_accumulator.get_current_accum()
                        total_loss += loss.item() * current_accum
                        _emit('on_step_end', epoch=epoch+1, step=step, loss=float(loss.item()))
                        for modality, modality_scheduler in modality_schedulers.items():
                            if modality_scheduler is not None:
                                modality_scheduler.step()
                        if epoch+1 > min_plateau_epoch and scheduler is not None:
                            scheduler.step(loss.item())
                        if not is_distributed or local_rank == 0:
                            if step % 10 == 0:
                                effective_batch = batch_size * current_accum
                                avg_loss = total_loss / (step + 1)
                                logger.info(f"Epoch {epoch + 1} | Step {step} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Accum: {current_accum} | Batch: {effective_batch}")
                
                if not train_loader:
                    logger.debug("Skipping epoch end logic for empty loader.")
                    continue

                if step_loss > 0:
                    params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
                    total_norm, clip_coef, was_clipped = grad_clipper.update_and_clip(params, optimizer, scaler)
                    torch.cuda.empty_cache()
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if not is_distributed or local_rank == 0:
                        current_threshold = grad_clipper.get_current_threshold()
                        clip_ratio = (1.0 - clip_coef) * 100 if was_clipped else 0.0
                        logger.info(f"Final GradClip: norm={total_norm:.4f}, threshold={current_threshold:.4f}, clipped={was_clipped}, clip_ratio={clip_ratio:.1f}%")

                avg_loss = total_loss / (step + 1)
                
                if not is_distributed or local_rank == 0:
                    checkpoint_path = f"{save_dir}/pisces_{dataset}_epoch{epoch + 1}.pt"
                    save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                    _emit('on_checkpoint_saved', path=checkpoint_path, epoch=epoch+1)
                    logger.info(f"Epoch {epoch + 1} complete | Final accum: {grad_accumulator.get_current_accum()} | Avg grad norm: {sum(grad_accumulator.grad_norm_history[-5:])/min(5, len(grad_accumulator.grad_norm_history)):.4f}")
                    
                    if epoch+1 == min_plateau_epoch:
                        from torch.optim.lr_scheduler import ReduceLROnPlateau
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-8)
                        logger.info(f"ReduceLROnPlateau scheduler enabled after {min_plateau_epoch} epochs.")
                    
                    # Stopping criteria: loss-driven with optional epoch caps.
                    # If epochs <= 0, epoch cap is disabled. If max_epochs_stop <= 0, that cap is disabled.
                    reached_loss = (avg_loss < loss_stop_threshold)
                    reached_max_epochs = (epochs > 0 and (epoch + 1) >= epochs)
                    reached_max_epochs_stop = (max_epochs_stop > 0 and (epoch + 1) >= max_epochs_stop)
                    if reached_loss or reached_max_epochs or reached_max_epochs_stop:
                        logger.info(f"Dataset {dataset} training complete (loss={avg_loss:.4f}, epochs={epoch+1}).")
                        _emit('on_epoch_end', epoch=epoch+1, avg_loss=float(avg_loss))
                        stop_training = True
                    else:
                        _emit('on_epoch_end', epoch=epoch+1, avg_loss=float(avg_loss))
                        epoch += 1
        except KeyboardInterrupt:
            logger.error("Training interrupted by user (Ctrl - C). Saving checkpoint...")
            interrupt_ckpt = f"{save_dir}/latest_interrupt.pt"
            save_ckpt(model, optimizer, epoch + 1, interrupt_ckpt)
            logger.info(f"Checkpoint saved: {interrupt_ckpt}")
            logger.info(f"You can resume training with: python manage.py train --model_size {model_size} --resume_ckpt {interrupt_ckpt}")
            _emit('on_train_interrupted', epoch=epoch+1, ckpt=interrupt_ckpt)
            sys.exit(0)
        logger.info(f"Dataset {dataset} training completed!")

    logger.info("All datasets finished. Saving final model weights...")

    final_weight_path = os.path.join(save_dir, f"pisces-l1-{model_size.lower()}-final.pt")
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), final_weight_path)
    else:
        torch.save(model.state_dict(), final_weight_path)
    logger.info(f"All datasets finished. Final model weights saved to: {final_weight_path}")
    _emit('on_train_end', final_weight_path=final_weight_path)

    # Optional: auto export to safetensors if enabled in config
    if auto_export_safetensors:
        try:
            from tools.train.quant_export import PiscesLxToolsQuantExporter
            exporter = PiscesLxToolsQuantExporter(cfg=_CFG, hooks=_HOOKS, profiler=_PROFILER)
            exporter.export(final_weight_path, ["safetensors"])
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Auto safetensors export failed: {e}")

class AdaptiveGradientClipper:
    """
    Adaptive gradient clipper with K - FAC Hessian approximation for second - order optimization.

    Args:
        initial_max_norm (float, optional): The initial maximum gradient norm. Defaults to 1.0.
        history_length (int, optional): The length of the gradient norm history. Defaults to 100.
        percentile (int, optional): The percentile used to calculate the target gradient norm. Defaults to 95.
        min_clip (float, optional): The minimum clipping value. Defaults to 0.1.
        max_clip (float, optional): The maximum clipping value. Defaults to 10.0.
        warmup_steps (int, optional): The number of warmup steps. Defaults to 10.
        use_kfac (bool, optional): Whether to use K - FAC preconditioning. Defaults to True.
        kfac_update_freq (int, optional): The frequency of updating K - FAC matrices. Defaults to 100.
        kfac_damping (float, optional): The damping factor for K - FAC preconditioning. Defaults to 0.001.
    """
    
    def __init__(self, initial_max_norm=1.0, history_length=100, percentile=95, 
                 min_clip=0.1, max_clip=10.0, warmup_steps=10, use_kfac=True, 
                 kfac_update_freq=100, kfac_damping=0.001):
        self.initial_max_norm = initial_max_norm
        self.current_max_norm = initial_max_norm
        self.history_length = history_length
        self.percentile = percentile
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.warmup_steps = warmup_steps
        self.use_kfac = use_kfac
        self.kfac_update_freq = kfac_update_freq
        self.kfac_damping = kfac_damping
        
        self.grad_norm_history = []
        self.step_count = 0
        self.kfac_state = {}
        self.fisher_matrices = {}
        self.grad_momentum = {}
        self.kfac_step = 0
    
    def update_and_clip(self, parameters, optimizer=None, scaler=None):
        """
        Update the gradient clipping threshold and clip the gradients.

        Args:
            parameters (iterable): An iterable of parameters to clip gradients for.
            optimizer (torch.optim.Optimizer, optional): The optimizer used for training. Defaults to None.
            scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler for mixed precision training. Defaults to None.

        Returns:
            tuple: A tuple containing the total gradient norm, the clipping coefficient, and a boolean indicating whether clipping occurred.
        """
        self.step_count += 1
        self.kfac_step += 1
        parameters = list(parameters)
        if self.use_kfac and self.step_count > self.warmup_steps:
            total_norm = self._calculate_natural_gradient_norm(parameters)
        else:
            total_norm = 0.0
            for param in parameters:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
        if self.use_kfac and self.kfac_step % self.kfac_update_freq == 0:
            self._update_kfac_matrices(parameters)
        if self.step_count <= self.warmup_steps:
            max_norm = self.initial_max_norm
        else:
            self.grad_norm_history.append(total_norm)
            if len(self.grad_norm_history) > self.history_length:
                self.grad_norm_history.pop(0)
            if len(self.grad_norm_history) >= 10:
                try:
                    import numpy as np
                    percentile_value = np.percentile(self.grad_norm_history, self.percentile)
                    curvature_factor = self._get_curvature_factor()
                    adjusted_percentile = percentile_value * curvature_factor
                    target_norm = max(self.min_clip, min(self.max_clip, adjusted_percentile * 1.2))
                    alpha = 0.1
                    self.current_max_norm = (1 - alpha) * self.current_max_norm + alpha * target_norm
                except ImportError:
                    avg_norm = sum(self.grad_norm_history) / len(self.grad_norm_history)
                    curvature_factor = self._get_curvature_factor()
                    adjusted_avg = avg_norm * curvature_factor
                    self.current_max_norm = max(self.min_clip, min(self.max_clip, adjusted_avg * 1.5))
            else:
                max_norm = self.initial_max_norm
        max_norm = self.current_max_norm
        if total_norm > 0:
            if self.use_kfac and self.step_count > self.warmup_steps:
                self._apply_kfac_preconditioning(parameters)
            clip_coef = max_norm / (total_norm + 1e-6)
            was_clipped = clip_coef < 1.0
            if scaler is not None and optimizer is not None:
                scaler.unscale_(optimizer)
            clip_coef_clamped = min(clip_coef, 1.0)
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef_clamped)
            return total_norm, clip_coef_clamped, was_clipped
        return total_norm, 1.0, False
    
    def _calculate_natural_gradient_norm(self, parameters):
        """
        Calculate the natural gradient norm using K - FAC preconditioning.

        Args:
            parameters (iterable): An iterable of parameters to calculate the natural gradient norm for.

        Returns:
            float: The natural gradient norm.
        """
        natural_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                grad = param.grad.data
                param_id = id(param)
                if param_id in self.fisher_matrices:
                    fisher = self.fisher_matrices[param_id]
                    if len(grad.shape) == 2:
                        fisher_diag = fisher.expand_as(grad)
                        natural_grad = grad / (fisher_diag + self.kfac_damping)
                    elif len(grad.shape) == 1:
                        natural_grad = grad / (fisher + self.kfac_damping)
                    else:
                        natural_grad = grad
                    natural_norm += natural_grad.norm(2).item() ** 2
                else:
                    natural_norm += grad.norm(2).item() ** 2
        return natural_norm ** 0.5
    
    def _update_kfac_matrices(self, parameters):
        """
        Update the K - FAC Fisher matrices based on the current gradients.

        Args:
            parameters (iterable): An iterable of parameters to update the K - FAC matrices for.
        """
        for param in parameters:
            if param.grad is not None:
                param_id = id(param)
                grad = param.grad.data
                if len(grad.shape) == 2:
                    fisher_approx = (grad * grad).mean(dim=1, keepdim=True)
                elif len(grad.shape) == 1:
                    fisher_approx = grad * grad
                else:
                    continue
                if param_id in self.fisher_matrices:
                    alpha = 0.01
                    self.fisher_matrices[param_id] = (
                        (1 - alpha) * self.fisher_matrices[param_id] + 
                        alpha * fisher_approx.detach()
                    )
                else:
                    self.fisher_matrices[param_id] = fisher_approx.detach()
    
    def _get_curvature_factor(self):
        """
        Calculate the curvature factor based on the K - FAC Fisher matrices.

        Returns:
            float: The curvature factor.
        """
        if not self.fisher_matrices:
            return 1.0
        total_curvature = 0.0
        count = 0
        for fisher in self.fisher_matrices.values():
            if isinstance(fisher, torch.Tensor):
                avg_curvature = torch.abs(fisher).mean().item()
                total_curvature += avg_curvature
                count += 1
        if count == 0:
            return 1.0
        avg_curvature = total_curvature / count
        base_curvature = 1.0
        curvature_factor = base_curvature / (1.0 + avg_curvature * 0.1)
        return max(0.5, min(2.0, curvature_factor))
    
    def _apply_kfac_preconditioning(self, parameters):
        """
        Apply K - FAC preconditioning to the gradients.

        Args:
            parameters (iterable): An iterable of parameters to apply K - FAC preconditioning to.
        """
        for param in parameters:
            if param.grad is not None and param.requires_grad:
                param_id = id(param)
                grad = param.grad.data
                if param_id in self.fisher_matrices:
                    fisher = self.fisher_matrices[param_id]
                    if len(grad.shape) == 2:
                        fisher_diag = fisher.expand_as(grad)
                        preconditioned_grad = grad / (fisher_diag + self.kfac_damping)
                    elif len(grad.shape) == 1:
                        preconditioned_grad = grad / (fisher + self.kfac_damping)
                    else:
                        continue
                    param.grad.data = preconditioned_grad
    
    def get_current_threshold(self):
        """
        Get the current gradient clipping threshold.

        Returns:
            float: The current gradient clipping threshold.
        """
        return self.current_max_norm
        
    def reset(self):
        """
        Reset the gradient clipping state.
        """
        self.grad_norm_history.clear()
        self.step_count = 0
        self.current_max_norm = self.initial_max_norm

def validate_train_args(args):
    """
    Validate and normalize arguments for tools.train.train().

    Args:
        args: Command line arguments or configuration object to validate.

    Returns:
        The validated and normalized arguments.
    """
    # Set default model size if not provided
    if not hasattr(args, 'model_size') or not args.model_size:
        setattr(args, 'model_size', '0.5B')

    # Check dataset presence
    if not getattr(args, 'dataset', None):
        cache_manager = get_cache_manager()
        data_cache_dir = str(cache_manager.get_cache_dir("data_cache"))
        os.makedirs(data_cache_dir, exist_ok=True)
        model_txt = os.path.join(data_cache_dir, "model.txt")
        if not os.path.exists(model_txt):
            raise ValueError("dataset not provided and data_cache/model.txt not found")

    # Default flags
    for flag in ("force_quant", "force_lora", "quant", "no_quant"):
        if not hasattr(args, flag):
            setattr(args, flag, False)

    # Quant bits
    if hasattr(args, 'quant_bits') and args.quant_bits is not None:
        try:
            qb = int(args.quant_bits)
        except Exception:
            raise ValueError("quant_bits must be integer")
        if qb not in (2, 4, 8):
            raise ValueError("quant_bits must be one of {2,4,8}")

    # Resume path
    if getattr(args, 'resume_ckpt', None):
        if not os.path.exists(args.resume_ckpt):
            raise ValueError(f"resume_ckpt not found: {args.resume_ckpt}")

    # Default save dir
    if not getattr(args, 'save_dir', None):
        setattr(args, 'save_dir', 'ckpt')

    return args

# Module-level train function for backward compatibility
def train(args):
    """Run training by delegating to the existing implementation."""
    return _train_impl(args)

class PiscesLxToolsTrainImpl:
    """Class-based facade for training implementation (unified style).

    This facade mirrors the infer side (`PiscesLxToolsInferImpl`) and allows
    callers to use a class with `set_context()` and `train()` methods while
    preserving the existing module-level API and behavior.
    """

    def __init__(self) -> None:
        self._hooks = None
        self._profiler = None
        self._cfg = None

    def set_context(self, *, hooks=None, profiler=None, cfg=None) -> None:
        """Set runtime context and synchronize module-level context."""
        self._hooks = hooks
        self._profiler = profiler
        self._cfg = cfg
        # keep module-level helpers working
        try:
            set_context(hooks=hooks, profiler=profiler, cfg=cfg)
        except Exception:
            pass

    def train(self, args) -> None:
        """Run training by delegating to the existing implementation."""
        return _train_impl(args)

    # Optional convenience: expose validator
    def validate_args(self, args):
        return validate_train_args(args)

    # Helper delegations to preserve behavior while supporting class-only runners
    def setup_distributed_training(self):
        return setup_distributed_training()

    def create_dataloader(self, *a, **kw):
        return create_dataloader(*a, **kw)

    def collate_fn(self, batch):
        return collate_fn(batch)
