#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces L1.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from utils.log import RIGHT, DEBUG, ERROR

def setup_device(device_pref):
    """
    Set up the training device based on the device preference.

    Args:
        device_pref (str): Device preference, e.g., "auto", "cuda", "cpu".

    Returns:
        torch.device: The selected device for training.
    """
    import torch
    DEBUG("Pisces L1 Training Start!")
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)
    RIGHT(f"Using device: {device}")
    if torch.cuda.is_available():
        RIGHT(f"GPU: {torch.cuda.get_device_name(0)}")
        RIGHT(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        ERROR("No GPU available, using CPU")
    return device

def collate_fn(batch):
    """
    Collate a batch of data into a format suitable for training.

    Args:
        batch (list): A list of data items, each is a dictionary containing model inputs.

    Returns:
        dict: A dictionary containing collated model inputs, including input_ids, labels, 
              pixel_values, and audio_input.
    """
    import torch
    MAX_SEQ_LEN = 256
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, :MAX_SEQ_LEN]
    
    # Handle pixel_values for vision modality
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
    import torch
    from data.dataset import PiscesDataset
    from torch.utils.data import DataLoader
    from model.tokenizer import get_tokenizer
    from model import PiscesModel, PiscesConfig
    from trainer.checkpoint import save_ckpt, load_ckpt
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from transformers import get_linear_schedule_with_warmup
    
    AUTO_CONFIG = {
        "0.5B":  dict(batch_size=4,  accum=8,  seq_len=384,  force_quant=False, force_lora=False, lr=3e-5),
        "1.5B":  dict(batch_size=2,  accum=16, seq_len=512,  force_quant=False, force_lora=False, lr=2e-5),
        "7B":    dict(batch_size=1,  accum=32, seq_len=384,  force_quant=True, force_lora=True,  lr=2e-5),
        "32B":   dict(batch_size=1,  accum=64, seq_len=256,  force_quant=True, force_lora=True,  lr=1e-5),
        "64B":   dict(batch_size=1,  accum=64, seq_len=192,  force_quant=True, force_lora=True,  lr=1e-5),
        "70B":   dict(batch_size=1,  accum=64, seq_len=128,  force_quant=True, force_lora=True,  lr=8e-6),
    }
    model_size = getattr(args, 'model_size', '0.5B').upper()
    if model_size not in AUTO_CONFIG:
        ERROR(f"Unsupported model_size: {model_size}")
        sys.exit(1)
    cfg_dict = AUTO_CONFIG[model_size]
    batch_size = cfg_dict['batch_size']
    accum = cfg_dict['accum']
    seq_len = cfg_dict['seq_len']
    force_quant = cfg_dict['force_quant']
    force_lora = cfg_dict['force_lora']
    epochs = 1
    lr = cfg_dict['lr']
    save_dir = "ckpt"
    
    min_plateau_epoch = 5
    scheduler = None
    
    data_cache_dir = "data_cache"
    model_txt = os.path.join(data_cache_dir, "model.txt")
    if not os.path.exists(model_txt):
        ERROR(f"{model_txt} not found! Please create it with one dataset name per line.")
        sys.exit(1)
    with open(model_txt, "r", encoding="utf-8") as f:
        dataset_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    if not dataset_list:
        ERROR(f"No dataset names found in {model_txt}!")
        sys.exit(1)
    device = setup_device("auto")
    RIGHT(f"Device set: {device}")
    RIGHT("Loading PiscesConfig...")
    config = f"configs/{model_size}.json"
    if not os.path.exists(config):
        ERROR(f"Config file {config} not found. Please provide a valid --model_size.")
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
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
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
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    RIGHT("Optimizer and scheduler ready.")
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
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        RIGHT("DataLoader created successfully")
        os.makedirs(save_dir, exist_ok=True)
        RIGHT("Starting training loop...")
        model.train()
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stop_training = False
        epoch = start_epoch
        try:
            while not stop_training:
                DEBUG(f"Starting epoch {epoch+1}")
                total_loss = 0
                accum_counter = 0
                optimizer.zero_grad()
                for step, batch in enumerate(train_loader):
                    model_keys = ["input_ids", "labels"]
                    device_batch = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items() if k in model_keys and v is not None
                    }
                    loss = None
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(**device_batch)
                            loss = outputs.get("loss")

                        if loss is not None and loss.requires_grad:
                            if torch.cuda.device_count() > 1:
                                loss = loss.mean()
                            
                            scaler.scale(loss / accum).backward()
                            
                            accum_counter += 1
                            if accum_counter % accum == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                                accum_counter = 0
                        else:
                            DEBUG(f"Warning: Skipping step {step} due to invalid loss (None or no grad).")
                            
                    else: # Non-scaler path
                        outputs = model(**device_batch)
                        loss = outputs.get("loss")
                        
                        if loss is not None and loss.requires_grad:
                            if torch.cuda.device_count() > 1:
                                loss = loss.mean()
                            loss.backward()

                            accum_counter += 1
                            if accum_counter % accum == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                optimizer.step()
                                optimizer.zero_grad()
                                accum_counter = 0
                        else:
                            DEBUG(f"Warning: Skipping step {step} due to invalid loss (None or no grad).")

                    if loss is not None:
                        total_loss += loss.item() * accum
                        
                        if epoch+1 > min_plateau_epoch and scheduler is not None:
                            scheduler.step(loss.item())
                        if step % 10 == 0:
                            avg_loss = total_loss / (step + 1)
                            RIGHT(f"Epoch {epoch + 1} | Step {step} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                if not train_loader:
                    DEBUG("Skipping epoch end logic for empty loader.")
                    continue

                avg_loss = total_loss / (step + 1)
                checkpoint_path = f"{save_dir}/pisces_{dataset}_epoch{epoch + 1}.pt"
                save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
                RIGHT(f"Checkpoint saved: {checkpoint_path}")
                
                if epoch+1 == min_plateau_epoch:
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-8)
                    RIGHT(f"ReduceLROnPlateau scheduler enabled after {min_plateau_epoch} epochs.")
                if avg_loss < 1.0:
                    RIGHT(f"Loss < 1.0, stopping training for dataset {dataset}.")
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
        RIGHT("Training completed!")

    final_weight_path = os.path.join(save_dir, f"pisces-l1-{model_size.lower()}-final.pt")
    if hasattr(model, "module"):  # DataParallel
        torch.save(model.module.state_dict(), final_weight_path)
    else:
        torch.save(model.state_dict(), final_weight_path)
    RIGHT(f"All datasets finished. Final model weights saved to: {final_weight_path}")