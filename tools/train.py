#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import argparse
import subprocess

def setup_device(device_pref):
    import torch
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)
    print(f"✅\tUsing device: {device}")
    if torch.cuda.is_available():
        print(f"✅\tGPU: {torch.cuda.get_device_name(0)}")
        print(f"✅\tGPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌\tNo GPU available, using CPU")
    return device

def collate_fn(batch):
    import torch
    MAX_SEQ_LEN = 256
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, :MAX_SEQ_LEN]
    pixel_values = None
    audio_input = None
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
    import torch
    from data.dataset import PiscesDataset
    from torch.utils.data import DataLoader
    from model import PiscesModel, PiscesConfig
    from trainer.checkpoint import save_ckpt, load_ckpt
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from transformers import get_linear_schedule_with_warmup
    
    AUTO_CONFIG = {
        "0.5B":  dict(batch_size=32,  accum=2,  seq_len=1024, force_quant=True, force_lora=False),
        "1.5B":  dict(batch_size=4,  accum=16, seq_len=1024, force_quant=True, force_lora=True),
        "7B":    dict(batch_size=2,  accum=32, seq_len=1024, force_quant=True, force_lora=True),
        "32B":   dict(batch_size=1,  accum=32, seq_len=512,  force_quant=True, force_lora=True),
        "64B":   dict(batch_size=1,  accum=32, seq_len=384,  force_quant=True, force_lora=True),
        "70B":   dict(batch_size=1,  accum=32, seq_len=256,  force_quant=True, force_lora=True),
    }
    model_size = getattr(args, 'model_size', '0.5B').upper()
    if model_size not in AUTO_CONFIG:
        print(f"❌ Unsupported model_size: {model_size}")
        sys.exit(1)
    cfg_dict = AUTO_CONFIG[model_size]
    batch_size = cfg_dict['batch_size']
    accum = cfg_dict['accum']
    seq_len = cfg_dict['seq_len']
    force_quant = cfg_dict['force_quant']
    force_lora = cfg_dict['force_lora']
    epochs = 1
    lr = 5e-5
    save_dir = "ckpt"
    
    data_cache_dir = "data_cache"
    model_txt = os.path.join(data_cache_dir, "model.txt")
    if not os.path.exists(model_txt):
        print(f"❌\t{model_txt} not found! Please create it with one dataset name per line.")
        sys.exit(1)
    with open(model_txt, "r", encoding="utf-8") as f:
        dataset_list = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    if not dataset_list:
        print(f"❌\tNo dataset names found in {model_txt}!")
        sys.exit(1)
    device = setup_device("auto")
    print(f"✅\tDevice set: {device}")
    print("✅\tLoading PiscesConfig...")
    config = f"configs/{model_size}.json"
    if not os.path.exists(config):
        print(f"❌\tConfig file {config} not found. Please provide a valid --model_size.")
        sys.exit(1)
    print(f"✅\tLoading config file: {config}")
    cfg = PiscesConfig.from_json(config)
    print("✅\tPiscesConfig loaded.")
    print("✅\tInitializing PiscesModel...")
    
    model = None
    if force_quant or force_lora:
        from transformers import BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        quant_config = None
        if force_quant:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
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
    model = model.to(device)
    print("✅\tPiscesModel initialized.")
    
    if getattr(args, 'resume_ckpt', ''):
        print(f"✅\tResuming from checkpoint: {args.resume_ckpt}")
        load_ckpt(args.resume_ckpt, model, optimizer)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"✅\tUsing {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    print("✅\tInitializing optimizer and scheduler...")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6)
    print("✅\tOptimizer and scheduler ready.")
    for dataset in dataset_list:
        print(f"\n==============================")
        print(f"✅\tTraining dataset: {dataset}")
        print(f"✅\tBatch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
        cache_path = os.path.join(data_cache_dir, dataset)
        if not os.path.exists(cache_path):
            print(f"❌\tLocal dataset not found: {cache_path}")
            continue
        train_ds = PiscesDataset(subset=dataset, split="train", config=cfg)
        print(f"✅\tDataset loaded successfully, size: {len(train_ds)}")
        print("✅\tCreating DataLoader...")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        print("✅\tDataLoader created successfully")
        os.makedirs(save_dir, exist_ok=True)
        print("✅\tStarting training loop...")
        model.train()
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stop_training = False
        epoch = 0
        while not stop_training:
            print(f"✅\tStarting epoch {epoch+1}")
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
                        _, loss, _, _, _ = model(**device_batch)
                        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                            loss = loss.mean()
                        loss = loss / accum
                    try:
                        scaler.scale(loss).backward()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"❌\tOOM at step {step}, skipping batch...")
                            torch.cuda.empty_cache()
                            import gc; gc.collect()
                            optimizer.zero_grad()
                            continue
                        else:
                            raise
                    accum_counter += 1
                    if accum_counter % accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                        scheduler.step(loss.item())
                        scaler.update()
                        accum_counter = 0
                    del batch, device_batch
                    torch.cuda.empty_cache()
                else:
                    _, loss, _, _, _ = model(**device_batch)
                    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                        loss = loss.mean()
                    loss = loss / accum
                    loss.backward()
                    accum_counter += 1
                    if accum_counter % accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        accum_counter = 0
                    del batch, device_batch
                    torch.cuda.empty_cache()
                if loss is not None:
                    total_loss += loss.item() * accum
                    scheduler.step(loss.item())
                    if step % 50 == 0:
                        avg_loss = total_loss / (step + 1)
                        print(f"✅\tEpoch {epoch + 1} | Step {step} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            avg_loss = total_loss / (step + 1)
            checkpoint_path = f"{save_dir}/pisces_{dataset}_epoch{epoch + 1}.pt"
            save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
            print(f"✅\tCheckpoint saved: {checkpoint_path}")
            if avg_loss < 1.0:
                print(f"✅\tLoss < 1.0, stopping training for dataset {dataset}.")
                stop_training = True
            else:
                epoch += 1
        print("✅\tTraining completed!")

def add_train_args(parser):
    parser.add_argument('--model_size', default='0.5B', type=str, help='Model size, e.g. 0.5B, 1.5B, 7B, 70B')
    parser.add_argument('--resume_ckpt', default='', type=str, help='Path to checkpoint to resume training')
    return parser 