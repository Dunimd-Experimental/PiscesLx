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
import json
import torch
import warnings
from utils.gpu_manager import GPUManager
from utils.log import RIGHT, DEBUG, ERROR

def setup_distributed_training():
    """
    Set up the distributed training environment with 3D parallel support.

    Returns:
        torch.device: The device to use for training.
        bool: Whether the training is distributed.
        int: The local rank of the current process.
        int: The total number of processes in the distributed training.
        dict: 3D parallel configuration.
    """
    import torch
    import torch.distributed as dist
    
    # Automatically detect the distributed environment
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    is_distributed = local_rank >= 0 and world_size > 1
    
    # 3D parallel configuration
    dp_size = max(1, world_size)  # Data parallel size
    pp_size = 1  # Pipeline parallel size
    mp_size = 1  # Model parallel size
    
    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # Dynamically calculate 3D parallel configuration
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 8:
            dp_size = max(1, num_gpus // 4)
            pp_size = 2
            mp_size = 2
        elif num_gpus >= 4:
            dp_size = max(1, num_gpus // 2)
            pp_size = 2
            
        RIGHT(f"3D Parallel: DP={dp_size}, PP={pp_size}, MP={mp_size}, rank {rank}/{world_size}")
    else:
        # Single-GPU or multi-GPU mode
        gpu_manager = GPUManager()
        strategy = gpu_manager.strategy
        
        if strategy['mode'] == 'single_gpu':
            device = torch.device(f"cuda:{strategy['gpu_ids'][0]}")
            torch.cuda.set_device(strategy['gpu_ids'][0])
        else:
            device = torch.device('cuda')
            
        RIGHT(f"Training mode: {strategy['mode']}")
        
    parallel_config = {
        'dp_size': dp_size,
        'pp_size': pp_size,
        'mp_size': mp_size,
        'total_gpus': dp_size * pp_size * mp_size
    }
    
    return device, is_distributed, local_rank, world_size, parallel_config

def create_ddp_model(model, device, is_distributed, local_rank):
    """
    Create a distributed data parallel model.

    Args:
        model (torch.nn.Module): The model to be wrapped.
        device (torch.device): The device to use for training.
        is_distributed (bool): Whether the training is distributed.
        local_rank (int): The local rank of the current process.

    Returns:
        torch.nn.Module: The wrapped model.
    """
    if is_distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    else:
        # Single-GPU mode
        model = model.to(device)
        
        # Check if multiple GPUs are available but distributed training is not used
        if torch.cuda.device_count() > 1:
            RIGHT(f"Detected {torch.cuda.device_count()} GPUs, using DataParallel")
            model = torch.nn.DataParallel(model)
            
    return model

def create_dataloader(dataset, batch_size, is_distributed, world_size=1, rank=0, local_rank=0):
    """
    Create a data loader that supports distributed training.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be loaded.
        batch_size (int): The batch size.
        is_distributed (bool): Whether the training is distributed.
        world_size (int, optional): The total number of processes in the distributed training. Defaults to 1.
        rank (int, optional): The rank of the current process. Defaults to 0.
        local_rank (int, optional): The local rank of the current process. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: The created data loader.
    """
    import torch
    from torch.utils.data import DistributedSampler
    
    # Handle empty dataset to avoid deadlock
    if len(dataset) == 0:
        return torch.utils.data.DataLoader([], batch_size=1)  # Return empty loader to avoid deadlock
    
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
        
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

def collate_fn(batch):
    """
    Memory-optimized collate function for Arctic architecture.

    Args:
        batch (list): A list of data items, each is a dictionary containing model inputs.

    Returns:
        dict: A dictionary containing collated model inputs, including input_ids, labels, 
              pixel_values, and audio_input.
    """
    import torch
    # Ultra-conservative sequence length for 14GB GPU with 1.5B model
    MAX_SEQ_LEN = 96  # Reduced from 128 to 96 for critical memory efficiency
    
    # Extract and pad input_ids
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, :MAX_SEQ_LEN]
    
    # Handle pixel_values for vision modality - limit size for memory
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
    
    labels = input_ids.clone()
    if labels.shape[1] > MAX_SEQ_LEN:
        labels = labels[:, :MAX_SEQ_LEN]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "audio_input": audio_input
    }

def train(args):
    """
    Train the Pisces model based on the given arguments.

    Args:
        args (argparse.Namespace): Command line arguments containing training configuration.
    """
    try:
        _train_impl(args)
    except torch.cuda.OutOfMemoryError as e:
        # CUDA out of memory error handler with quantization suggestion
        model_size = getattr(args, 'model_size', '0.5B').upper()
        dataset = getattr(args, 'dataset', 'Chinese2')

        ERROR("CUDA OUT OF MEMORY ERROR DETECTED!")
        DEBUG(f"python manage.py train --config configs/{model_size}.json --dataset {dataset} --force_quant --force_lora")

        sys.exit(1)
    except Exception as e:
        ERROR(f"Training failed with error: {str(e)}")
        import traceback
        ERROR(traceback.format_exc())
        sys.exit(1)

def _train_impl(args):
    import torch
    from data.dataset import PiscesDataset
    from torch.utils.data import DataLoader
    from model.tokenizer import get_tokenizer
    from model import PiscesModel, PiscesConfig
    from trainer.checkpoint import save_ckpt, load_ckpt
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from transformers import get_linear_schedule_with_warmup
    
    model_size = getattr(args, 'model_size', '0.5B').upper()
    config_path = f"configs/{model_size}.json"
    if not os.path.exists(config_path):
        ERROR(f"Config file {config_path} not found. Please provide a valid --model_size.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    if 'training_config' not in full_config:
        ERROR(f"training_config not found in {config_path}")
        sys.exit(1)
    
    training_config = full_config['training_config']
    batch_size = training_config['batch_size']
    accum = training_config['accum']
    max_accum = training_config['max_accum']
    seq_len = training_config['seq_len']
    force_quant = training_config['force_quant']
    force_lora = training_config['force_lora']
    lr = training_config['lr']
    
    # Override quantization and LoRA settings from command line arguments
    quant_bits = 4  # Default quantization bits
    if hasattr(args, 'quant_bits'):
        quant_bits = args.quant_bits
        
    if hasattr(args, 'force_quant') and args.force_quant:
        force_quant = True
        RIGHT(f"Command line override: force_quant = true ({quant_bits}-bit)")
    if hasattr(args, 'force_lora') and args.force_lora:
        force_lora = True
        RIGHT("Command line override: force_lora = true")
    if hasattr(args, 'quant') and args.quant:
        force_quant = True
        RIGHT(f"Command line override: quant = true ({quant_bits}-bit)")
    if hasattr(args, 'no_quant') and args.no_quant:
        force_quant = False
        RIGHT("Command line override: force_quant = false")
    
    epochs = 1
    save_dir = "ckpt"
    
    class DynamicGradientAccumulator:
        """Safe dynamic gradient accumulation based on gradient norm monitoring"""
        def __init__(self, base_accum, max_accum, target_grad_norm=1.0, safety_factor=0.8):
            """
            Initialize the dynamic gradient accumulator.

            Args:
                base_accum (int): The base number of gradient accumulations.
                max_accum (int): The maximum number of gradient accumulations.
                target_grad_norm (float, optional): The target gradient norm. Defaults to 1.0.
                safety_factor (float, optional): The safety factor. Defaults to 0.8.
            """
            self.base_accum = base_accum
            self.max_accum = max_accum
            self.target_grad_norm = target_grad_norm
            self.safety_factor = safety_factor
            self.current_accum = base_accum
            self.grad_norm_history = []
            self.stable_steps = 0
            
        def should_step(self, grad_norm):
            """
            Determine if the optimizer should perform a step based on the gradient norm.

            Args:
                grad_norm (float): The current gradient norm.

            Returns:
                bool: Whether the optimizer should perform a step.
            """
            self.grad_norm_history.append(grad_norm)
            if len(self.grad_norm_history) > 10:
                self.grad_norm_history.pop(0)
            
            # Calculate recent average gradient norm
            if len(self.grad_norm_history) >= 3:
                recent_avg = sum(self.grad_norm_history[-3:]) / 3
                
                # Dynamic adjustment logic
                if recent_avg < self.target_grad_norm * 0.5 and self.current_accum < self.max_accum:
                    # Gradients are small, increase accumulation for better efficiency
                    self.current_accum = min(self.current_accum + 1, self.max_accum)
                    self.stable_steps = 0
                elif recent_avg > self.target_grad_norm * 2.0 and self.current_accum > self.base_accum:
                    # Gradients are large, decrease accumulation to prevent explosion
                    self.current_accum = max(self.current_accum - 1, self.base_accum)
                    self.stable_steps = 0
                else:
                    self.stable_steps += 1
            
            # Safety check: never exceed max_accum
            return len(self.grad_norm_history) >= self.current_accum
            
        def get_current_accum(self):
            """
            Get the current number of gradient accumulations.

            Returns:
                int: The current number of gradient accumulations.
            """
            return max(1, int(self.current_accum * self.safety_factor))
            
        def reset(self):
            """
            Reset the gradient accumulator to its initial state.
            """
            self.current_accum = self.base_accum
            self.grad_norm_history.clear()
            self.stable_steps = 0
    
    grad_accumulator = DynamicGradientAccumulator(accum, max_accum)
    
    min_plateau_epoch = 5
    scheduler = None
    
    # Handle dataset selection: command line takes priority over model.txt
    dataset_list = []
    if hasattr(args, 'dataset') and args.dataset:
        # Use command line dataset
        dataset_list = [args.dataset]
        RIGHT(f"Using command line dataset: {args.dataset}")
    else:
        # Fallback to model.txt file
        data_cache_dir = "data_cache"
        model_txt = os.path.join(data_cache_dir, "model.txt")
        if not os.path.exists(model_txt):
            ERROR(f"{model_txt} not found! Please create it with one dataset name per line, or use --dataset argument.")
            sys.exit(1)
        with open(model_txt, "r", encoding="utf-8") as f:
            dataset_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        if not dataset_list:
            ERROR(f"No dataset names found in {model_txt}! Please use --dataset argument instead.")
            sys.exit(1)
    # Set up distributed training
    device, is_distributed, local_rank, world_size, _ = setup_distributed_training()
    
    # Adjust batch size according to hardware and distributed environment
    import torch
    if is_distributed:
        effective_batch_size = batch_size // world_size
        if effective_batch_size < 1:
            effective_batch_size = 1
            batch_size = world_size
        RIGHT(f"Distributed training: global batch_size={batch_size}, per-GPU batch_size={effective_batch_size}")
    else:
        # Single-GPU mode, optimize using GPU manager
        gpu_manager = GPUManager()
        recommended_batch = gpu_manager.recommend_batch_size(model_size, seq_len)
        
        # Special handling for 1.9B Arctic architecture
        # The model has grown from 0.5B to 1.9B, need conservative batch size
        if "0.5B" in model_size and recommended_batch > 8:
            # For Arctic 1.9B architecture, limit batch size to avoid OOM
            conservative_batch = min(8, recommended_batch)  # Max 8 for 1.9B model
            RIGHT(f"Arctic 1.9B memory optimization: {recommended_batch} -> {conservative_batch}")
            recommended_batch = conservative_batch
            
        if batch_size != recommended_batch:
            RIGHT(f"Adjust batch_size based on hardware: {batch_size} -> {recommended_batch}")
            batch_size = recommended_batch
        effective_batch_size = batch_size
        
    # Set up mixed precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    RIGHT("Mixed precision training enabled")
        
    RIGHT(f"Device setup completed: {device}")
    RIGHT("Loading PiscesConfig...")
    config = f"configs/{model_size}.json"
    if not os.path.exists(config):
        ERROR(f"Config file {config} not found. Please provide a valid --model_size")
        sys.exit(1)
    RIGHT(f"Loading config file: {config}")
    cfg = PiscesConfig.from_json(config)
    RIGHT("PiscesConfig loaded.")

    RIGHT("Initializing PiscesModel with Reasoner...")
    
    # Always-on Reasoner: Tokenizer setup
    tokenizer = get_tokenizer()
    special_tokens = ["<think>", "</think>"]
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
        from transformers import BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        quant_config = None
        if use_quant:
            # Configure quantization based on bits
            if quant_bits == 2:
                # 2-bit quantization (experimental)
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Use 4bit as base, will be further compressed
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    # Additional compression for 2-bit simulation
                    llm_int8_enable_fp32_cpu_offload=True
                )
                RIGHT(f"Using 2-bit quantization (experimental, based on 4-bit+compression)")
            elif quant_bits == 4:
                # Standard 4-bit quantization
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                RIGHT(f"Using 4-bit quantization (NF4)")
            elif quant_bits == 8:
                # 8-bit quantization
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                RIGHT(f"Using 8-bit quantization (INT8)")
            else:
                ERROR(f"Unsupported quantization bits: {quant_bits}. Supported: 2, 4, 8")
                sys.exit(1)
        model = PiscesModel(cfg, quantization_config=quant_config) if quant_config else PiscesModel(cfg)
        if force_lora:
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
            )
            lora_model = get_peft_model(model, lora_config)
            # Assign custom properties/methods of PiscesModel back to LoRA model
            for attr in ["cfg", "quantization_config", "lora_config", "forward", "prepare_inputs_for_generation"]:
                if hasattr(model, attr):
                    setattr(lora_model, attr, getattr(model, attr))
            model = lora_model
            try:
                model.print_trainable_parameters()
            except Exception:
                pass
    else:
        model = PiscesModel(cfg)

    # Always-on Reasoner: Model and Reasoner setup
    model.resize_token_embeddings(len(tokenizer))
    start_id = tokenizer.encoder.get("<think>")
    end_id = tokenizer.encoder.get("</think>")
    if start_id is None or end_id is None:
        raise ValueError("Special reasoning tokens could not be added to the tokenizer.")
    model.reasoner.start_thinking_id = start_id
    model.reasoner.end_thinking_id = end_id
    RIGHT(f"Reasoner is integral and configured with token IDs: start={start_id}, end={end_id}")
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        RIGHT(f"Gradient Checkpointing enabled.")
        
    model = model.to(device)
    RIGHT("PiscesModel initialized.")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        RIGHT(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    RIGHT("Initializing optimizer and scheduler...")
    # Memory-optimized AdamW configuration for 14GB GPU
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters() if not hasattr(model, 'module') else model.module.parameters()),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,  # Disable amsgrad to save memory
        maximize=False,
        foreach=False,  # Disable foreach to save memory
        capturable=False,  # Disable capturable to save memory
        differentiable=False,  # Disable differentiable to save memory
        fused=False  # Disable fused implementation which can use more memory
    )
    RIGHT("Optimizer and scheduler ready.")
    # Initialize the adaptive gradient clipper with K-FAC support
    grad_clipper = AdaptiveGradientClipper(
        initial_max_norm=1.0,
        history_length=100,
        percentile=95,
        min_clip=0.1,
        max_clip=10.0,
        warmup_steps=10,
        use_kfac=True,
        kfac_update_freq=100,
        kfac_damping=0.001
    )
    resume_ckpt = getattr(args, 'resume_ckpt', None)
    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        RIGHT(f"Resuming from checkpoint: {resume_ckpt}")
        start_epoch = load_ckpt(resume_ckpt, model, optimizer)
        RIGHT(f"Resumed at epoch {start_epoch}")

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            RIGHT(f"Learning rate auto-reset to {lr}")
        min_lr_threshold = lr * 0.5
        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr_threshold:
                param_group['lr'] = lr
                RIGHT(f"Learning rate auto-reset to {lr}")
    for dataset in dataset_list:
        DEBUG(f"\n==============================")
        RIGHT(f"Training dataset: {dataset}")
        RIGHT(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
        data_cache_dir = "data_cache"  # Define data_cache_dir here
        cache_path = os.path.join(data_cache_dir, dataset)
        if not os.path.exists(cache_path):
            ERROR(f"Local dataset not found: {cache_path}")
            continue
        train_ds = PiscesDataset(subset=dataset, split="train", config=cfg)
        if len(train_ds) == 0:
            DEBUG(f"Warning: Dataset '{dataset}' is empty after filtering. Skipping.")
            continue
        RIGHT(f"Dataset loaded successfully, size: {len(train_ds)}")
        RIGHT("Creating DataLoader...")
        rank = int(os.environ.get('RANK', 0))
        train_loader = create_dataloader(
            train_ds,
            batch_size=effective_batch_size,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank if is_distributed else 0
        )
        RIGHT("DataLoader created successfully")
        os.makedirs(save_dir, exist_ok=True)
        RIGHT("Starting training loop...")
        model.train()
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stop_training = False
        epoch = start_epoch
        
        # Set epoch sampler for distributed training
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        try:
            while not stop_training:
                if is_distributed:
                    train_loader.sampler.set_epoch(epoch)
                
                DEBUG(f"Starting epoch {epoch+1}")
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
                    
                    # Handle multi-modal inputs
                    pixel_values = device_batch.get("pixel_values")
                    audio_input = device_batch.get("audio_input")
                    if audio_input is not None and isinstance(audio_input, dict):
                        audio_input = {k: v.to(device) for k, v in audio_input.items()}
                    
                    loss = None
                    with torch.amp.autocast('cuda', enabled=scaler is not None):
                        outputs = model(
                            input_ids=device_batch["input_ids"],
                            labels=device_batch["labels"],
                            images=pixel_values,
                            audio=audio_input
                        )
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
                        
                        # Calculate gradient norm
                        grad_norm = 0.0
                        params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
                        for param in params:
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        
                        if grad_accumulator.should_step(grad_norm):
                            params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
                            
                            # Use the adaptive gradient clipper
                            total_norm, clip_coef, was_clipped = grad_clipper.update_and_clip(params, optimizer, scaler)
                            
                            # Critical memory optimization for 14GB GPU
                            torch.cuda.empty_cache()  # Clear unused memory before optimizer step
                            
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            
                            # Additional memory cleanup after optimizer step
                            optimizer.zero_grad(set_to_none=True)  # More aggressive gradient cleanup
                            torch.cuda.empty_cache()  # Clear cache again after step
                            step_loss = 0
                            
                            # Record gradient clipping information
                            if not is_distributed or local_rank == 0:
                                if step % 50 == 0:  # Record clipping information every 50 steps
                                    current_threshold = grad_clipper.get_current_threshold()
                                    clip_ratio = (1.0 - clip_coef) * 100 if was_clipped else 0.0
                                    RIGHT(f"GradClip: norm={total_norm:.4f}, threshold={current_threshold:.4f}, clipped={was_clipped}, clip_ratio={clip_ratio:.1f}%")
                    else:
                        DEBUG(f"Warning: Skipping step {step} due to invalid loss (None or no grad).")
                        continue

                    if loss is not None:
                        current_accum = grad_accumulator.get_current_accum()
                        total_loss += loss.item() * current_accum
                        
                        if epoch+1 > min_plateau_epoch and scheduler is not None:
                            scheduler.step(loss.item())
                        
                        # Print logs only on the main process
                        if not is_distributed or local_rank == 0:
                            if step % 10 == 0:
                                effective_batch = batch_size * current_accum
                                avg_loss = total_loss / (step + 1)
                                RIGHT(f"Epoch {epoch + 1} | Step {step} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Accum: {current_accum} | Batch: {effective_batch}")
                
                if not train_loader:
                    DEBUG("Skipping epoch end logic for empty loader.")
                    continue

                # Handle remaining gradients
                if step_loss > 0:
                    params = model.parameters() if not hasattr(model, 'module') else model.module.parameters()
                    
                    # Use the adaptive gradient clipper to handle remaining gradients
                    total_norm, clip_coef, was_clipped = grad_clipper.update_and_clip(params, optimizer, scaler)
                    
                    # Critical memory optimization for 14GB GPU
                    torch.cuda.empty_cache()  # Clear unused memory before optimizer step
                    
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Additional memory cleanup after optimizer step
                    optimizer.zero_grad(set_to_none=True)  # More aggressive gradient cleanup
                    torch.cuda.empty_cache()  # Clear cache again after step
                    
                    # Record the final gradient clipping information
                    if not is_distributed or local_rank == 0:
                        current_threshold = grad_clipper.get_current_threshold()
                        clip_ratio = (1.0 - clip_coef) * 100 if was_clipped else 0.0
                        RIGHT(f"Final GradClip: norm={total_norm:.4f}, threshold={current_threshold:.4f}, clipped={was_clipped}, clip_ratio={clip_ratio:.1f}%")

                avg_loss = total_loss / (step + 1)
                
                # Save checkpoints only on the main process
                if not is_distributed or local_rank == 0:
                    checkpoint_path = f"{save_dir}/pisces_{dataset}_epoch{epoch + 1}.pt"
                    save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
                    RIGHT(f"Checkpoint saved: {checkpoint_path}")
                    RIGHT(f"Epoch {epoch + 1} complete | Final accum: {grad_accumulator.get_current_accum()} | Avg grad norm: {sum(grad_accumulator.grad_norm_history[-5:])/min(5, len(grad_accumulator.grad_norm_history)):.4f}")
                    
                    if epoch+1 == min_plateau_epoch:
                        from torch.optim.lr_scheduler import ReduceLROnPlateau
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-8)
                        RIGHT(f"ReduceLROnPlateau scheduler enabled after {min_plateau_epoch} epochs.")
                    
                    if avg_loss < 2.8 or epoch+1 >= 6:
                        RIGHT(f"Dataset {dataset} training complete (loss={avg_loss:.4f}, epochs={epoch+1}).")
                        stop_training = True
                    else:
                        epoch += 1
        except KeyboardInterrupt:
            ERROR("Training interrupted by user (Ctrl-C). Saving checkpoint...")
            interrupt_ckpt = f"{save_dir}/latest_interrupt.pt"
            save_ckpt(model, optimizer, epoch + 1, interrupt_ckpt)
            RIGHT(f"Checkpoint saved: {interrupt_ckpt}")
            RIGHT(f"You can resume training with: python manage.py train --model_size {model_size} --resume_ckpt {interrupt_ckpt}")
            sys.exit(0)
        RIGHT(f"Dataset {dataset} training completed!")

    RIGHT("All datasets finished. Saving final model weights...")

    final_weight_path = os.path.join(save_dir, f"pisces-l1-{model_size.lower()}-final.pt")
    if hasattr(model, "module"):  # DataParallel
        torch.save(model.module.state_dict(), final_weight_path)
    else:
        torch.save(model.state_dict(), final_weight_path)
    RIGHT(f"All datasets finished. Final model weights saved to: {final_weight_path}")

class AdaptiveGradientClipper:
    """Adaptive gradient clipper with K-FAC Hessian approximation for second-order optimization."""
    
    def __init__(self, initial_max_norm=1.0, history_length=100, percentile=95, 
                 min_clip=0.1, max_clip=10.0, warmup_steps=10, use_kfac=True, 
                 kfac_update_freq=100, kfac_damping=0.001):
        """
        Initialize the adaptive gradient clipper with K-FAC support.
        
        Args:
            initial_max_norm (float): Initial maximum gradient norm.
            history_length (int): Length of gradient history used to calculate statistics.
            percentile (int): Percentile used to calculate the adaptive threshold.
            min_clip (float): Minimum clipping threshold.
            max_clip (float): Maximum clipping threshold.
            warmup_steps (int): Warm-up steps during which a fixed threshold is used.
            use_kfac (bool): Whether to use K-FAC Hessian approximation.
            kfac_update_freq (int): Frequency of K-FAC Fisher matrix updates.
            kfac_damping (float): Damping parameter for K-FAC.
        """
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
        Update the gradient clipping threshold and perform gradient clipping with K-FAC support.
        
        Args:
            parameters (Iterable[torch.Tensor]): Model parameters.
            optimizer (torch.optim.Optimizer, optional): Optimizer object for scaler.unscale_.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler (for mixed precision training).
        
        Returns:
            tuple: (total_norm, clip_coef, was_clipped)
        """
        self.step_count += 1
        self.kfac_step += 1
        
        parameters = list(parameters)
        
        # Calculate natural gradient norm using K-FAC if enabled
        if self.use_kfac and self.step_count > self.warmup_steps:
            total_norm = self._calculate_natural_gradient_norm(parameters)
        else:
            # Standard gradient norm calculation
            total_norm = 0.0
            for param in parameters:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
        
        # Update K-FAC Fisher matrices periodically
        if self.use_kfac and self.kfac_step % self.kfac_update_freq == 0:
            self._update_kfac_matrices(parameters)
        
        # Use a fixed threshold during the warm-up period
        if self.step_count <= self.warmup_steps:
            max_norm = self.initial_max_norm
        else:
            # Update the gradient history
            self.grad_norm_history.append(total_norm)
            if len(self.grad_norm_history) > self.history_length:
                self.grad_norm_history.pop(0)
            
            # Calculate the adaptive threshold with K-FAC adjustment
            if len(self.grad_norm_history) >= 10:
                try:
                    import numpy as np
                    percentile_value = np.percentile(self.grad_norm_history, self.percentile)
                    
                    # Adjust target norm based on K-FAC curvature information
                    curvature_factor = self._get_curvature_factor()
                    adjusted_percentile = percentile_value * curvature_factor
                    
                    # Smoothly update the threshold
                    target_norm = max(self.min_clip, min(self.max_clip, adjusted_percentile * 1.2))
                    
                    # Use exponential moving average to smooth threshold changes
                    alpha = 0.1
                    self.current_max_norm = (1 - alpha) * self.current_max_norm + alpha * target_norm
                except ImportError:
                    # If numpy is not available, use a simple average
                    avg_norm = sum(self.grad_norm_history) / len(self.grad_norm_history)
                    curvature_factor = self._get_curvature_factor()
                    adjusted_avg = avg_norm * curvature_factor
                    self.current_max_norm = max(self.min_clip, min(self.max_clip, adjusted_avg * 1.5))
            else:
                max_norm = self.initial_max_norm
        
        max_norm = self.current_max_norm
        
        # Perform gradient clipping with K-FAC preconditioning
        if total_norm > 0:
            if self.use_kfac and self.step_count > self.warmup_steps:
                # Apply K-FAC preconditioning
                self._apply_kfac_preconditioning(parameters)
            
            clip_coef = max_norm / (total_norm + 1e-6)
            was_clipped = clip_coef < 1.0
            
            if scaler is not None and optimizer is not None:
                # Mixed precision training requires unscaling first
                scaler.unscale_(optimizer)
            
            clip_coef_clamped = min(clip_coef, 1.0)
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef_clamped)
                    
            return total_norm, clip_coef_clamped, was_clipped
        
        return total_norm, 1.0, False
    
    def _calculate_natural_gradient_norm(self, parameters):
        """Calculate natural gradient norm using diagonal K-FAC approximation."""
        natural_norm = 0.0
        
        for param in parameters:
            if param.grad is not None:
                grad = param.grad.data
                
                # Get Fisher matrix for this parameter
                param_id = id(param)
                if param_id in self.fisher_matrices:
                    fisher = self.fisher_matrices[param_id]
                    
                    # Apply diagonal Fisher matrix to gradient
                    if len(grad.shape) == 2:  # Weight matrix
                        # fisher is [out_features, 1], broadcast to match grad shape
                        fisher_diag = fisher.expand_as(grad)
                        natural_grad = grad / (fisher_diag + self.kfac_damping)
                    elif len(grad.shape) == 1:  # Bias vector
                        natural_grad = grad / (fisher + self.kfac_damping)
                    else:
                        natural_grad = grad
                    
                    natural_norm += natural_grad.norm(2).item() ** 2
                else:
                    natural_norm += grad.norm(2).item() ** 2
        
        return natural_norm ** 0.5
    
    def _update_kfac_matrices(self, parameters):
        """Update K-FAC Fisher matrix approximations with memory optimization."""
        for param in parameters:
            if param.grad is not None:
                param_id = id(param)
                grad = param.grad.data
                
                # Memory-efficient Fisher matrix approximation using diagonal elements only
                if len(grad.shape) == 2:  # Weight matrix
                    # Use element-wise squares instead of full outer product to save memory
                    fisher_approx = (grad * grad).mean(dim=1, keepdim=True)  # [out_features, 1]
                elif len(grad.shape) == 1:  # Bias vector
                    fisher_approx = grad * grad  # Element-wise squares
                else:
                    continue
                
                # Update Fisher matrix with exponential moving average
                if param_id in self.fisher_matrices:
                    alpha = 0.01  # Slow update rate for stability
                    self.fisher_matrices[param_id] = (
                        (1 - alpha) * self.fisher_matrices[param_id] + 
                        alpha * fisher_approx.detach()
                    )
                else:
                    self.fisher_matrices[param_id] = fisher_approx.detach()
    
    def _get_curvature_factor(self):
        """Get curvature factor based on K-FAC Fisher matrices."""
        if not self.fisher_matrices:
            return 1.0
        
        # Calculate average curvature across all layers
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
        
        # Adjust factor based on curvature (higher curvature -> more conservative clipping)
        base_curvature = 1.0
        curvature_factor = base_curvature / (1.0 + avg_curvature * 0.1)
        
        return max(0.5, min(2.0, curvature_factor))
    
    def _apply_kfac_preconditioning(self, parameters):
        """Apply diagonal K-FAC preconditioning to gradients."""
        for param in parameters:
            if param.grad is not None and param.requires_grad:
                param_id = id(param)
                grad = param.grad.data
                
                if param_id in self.fisher_matrices:
                    fisher = self.fisher_matrices[param_id]
                    
                    # Apply diagonal Fisher preconditioning
                    if len(grad.shape) == 2:  # Weight matrix
                        # fisher is [out_features, 1], broadcast to match grad shape
                        fisher_diag = fisher.expand_as(grad)
                        preconditioned_grad = grad / (fisher_diag + self.kfac_damping)
                    elif len(grad.shape) == 1:  # Bias vector
                        preconditioned_grad = grad / (fisher + self.kfac_damping)
                    else:
                        continue
                    
                    # Update gradient with preconditioning
                    param.grad.data = preconditioned_grad
        
    def get_current_threshold(self):
        """Get the current clipping threshold."""
        return self.current_max_norm
        
    def reset(self):
        """Reset the clipper state."""
        self.grad_norm_history.clear()
        self.step_count = 0
        self.current_max_norm = self.initial_max_norm