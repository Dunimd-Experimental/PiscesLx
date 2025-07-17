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
    if not torch.cuda.is_available():
        print("❌\tTraining requires a CUDA-capable GPU. Exiting.")
        exit(1)
    from torch.utils.data import DataLoader
    from data.dataset import PiscesDataset
    from model import PiscesModel, PiscesConfig
    from trainer.checkpoint import save_ckpt, load_ckpt
    from transformers import get_linear_schedule_with_warmup
    # Only require model_size
    if not hasattr(args, 'model_size') or not args.model_size:
        print("❌\tYou must provide --model_size (e.g. 0.5B, 1.5B, 7B, etc.)")
        sys.exit(1)
    model_size = args.model_size
    config = f"configs/{model_size}.json"
    if not os.path.exists(config):
        print(f"❌\tConfig file {config} not found. Please provide a valid --model_size.")
        sys.exit(1)
    print(f"✅\tLoading config file: {config}")
    # Hardcoded training parameters
    batch_size = 2
    epochs = 1
    lr = 5e-5
    save_dir = "ckpt"
    # Find all datasets in data_cache
    data_cache_dir = "data_cache"
    if not os.path.exists(data_cache_dir):
        print(f"❌\t{data_cache_dir} directory does not exist!")
        sys.exit(1)
    dataset_list = [d for d in os.listdir(data_cache_dir) if os.path.isdir(os.path.join(data_cache_dir, d))]
    if not dataset_list:
        print(f"❌\tNo datasets found in {data_cache_dir}!")
        sys.exit(1)
    device = setup_device("auto")
    print(f"✅\tDevice set: {device}")
    print("✅\tLoading PiscesConfig...")
    cfg = PiscesConfig.from_json(config)
    print("✅\tPiscesConfig loaded.")
    print("✅\tInitializing PiscesModel...")
    model = PiscesModel(cfg).to(device)
    print("✅\tPiscesModel initialized.")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"✅\tUsing {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    print("✅\tInitializing optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=100000
    )
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
            num_workers=0,
            pin_memory=False,
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
        for epoch in range(epochs):
            print(f"✅\tStarting epoch {epoch+1}/{epochs}")
            total_loss = 0
            for step, batch in enumerate(train_loader):
                print(f"✅\tEpoch {epoch+1} Step {step} - Batch keys: {list(batch.keys())}")
                model_keys = ["input_ids", "labels"]
                device_batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items() if k in model_keys and v is not None
                }
                print(f"✅\tBatch moved to device.")
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        _, loss, _, _, _ = model(**device_batch)
                        print(f"✅\tForward pass done. Loss: {loss.item() if hasattr(loss, 'item') else loss}")
                        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                            loss = loss.mean()
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scheduler.step()
                    scaler.update()
                    del loss, batch, device_batch
                    torch.cuda.empty_cache()
                else:
                    _, loss, _, _, _ = model(**device_batch)
                    print(f"✅\tForward pass done. Loss: {loss.item() if hasattr(loss, 'item') else loss}")
                    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                        loss = loss.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    del loss, batch, device_batch
                    torch.cuda.empty_cache()
                    total_loss += loss.item()
                    if step % 50 == 0:
                        avg_loss = total_loss / (step + 1)
                        print(f"✅\tEpoch {epoch + 1}/{epochs} | Step {step} | Loss: {avg_loss:.4f}")
                checkpoint_path = f"{save_dir}/pisces_{dataset}_epoch{epoch + 1}.pt"
                save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
                print(f"✅\tCheckpoint saved: {checkpoint_path}")
            print("✅\tTraining completed!") 