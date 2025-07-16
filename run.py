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
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"
import sys
import subprocess
import platform
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def setup_env():
    """Auto setup venv and install requirements if needed, then auto-enter venv shell"""
    print("✅\tPisces auto environment setup...")
    py_exec = sys.executable
    venv_dir = os.path.join(os.getcwd(), "pisces_env")
    is_windows = platform.system().lower().startswith("win")
    
    # Check if in venv
    if sys.prefix == sys.base_prefix:
        print("✅\tNot in virtual environment. Creating venv...")
        subprocess.check_call([py_exec, "-m", "venv", venv_dir])
        print(f"✅\tVirtual environment created at {venv_dir}")
        # Re-run in venv python
        python_bin = os.path.join(venv_dir, "Scripts" if is_windows else "bin", "python" + (".exe" if is_windows else ""))
        print("✅\tRe-running setup in venv...")
        os.execv(python_bin, [python_bin] + sys.argv)
        return
    else:
        print("✅\tAlready in virtual environment.")
    # Upgrade pip
    print("✅\tUpgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    # Install requirements
    print("✅\tInstalling requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅\tPisces environment setup complete!")
    # Auto enter venv shell
    if is_windows:
        shell = os.environ.get("COMSPEC", "cmd.exe")
        activate = os.path.join(venv_dir, "Scripts", "activate.bat")
        print("✅\tAuto-entering Pisces venv shell (Windows)...")
        os.execv(shell, [shell, "/K", activate])
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        activate = os.path.join(venv_dir, "bin", "activate")
        print("✅\tAuto-entering Pisces venv shell (Linux/Mac)...")
        os.execv(shell, [shell, "-i", "-c", f"source '{activate}'; exec {shell}"])
    sys.exit(0)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Pisces L1 - One-Click Training and Inference")
    parser.add_argument("mode", nargs='?', default="train", choices=["train", "infer", "setup"], help="Mode: train, infer, or setup (default: train)")
    # Simple parameters with defaults
    parser.add_argument("--ckpt", default="", help="Model file for inference")
    parser.add_argument("--prompt", default="Hello, please introduce yourself", help="Prompt text for inference")
    parser.add_argument("--image", default="", help="Image path for inference")
    return parser.parse_args()


def setup_device(device_pref):
    """Setup device with smart detection"""
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
    """Custom batch processing for variable length data"""
    import torch
    MAX_SEQ_LEN = 256  # Safer, memory-friendly length for more model structures
    # Text data
    input_ids = [item["input_ids"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    # Truncate length
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, :MAX_SEQ_LEN]
    # Force multimodal input to None (save memory during training)
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
    """Training function"""
    import torch
    if not torch.cuda.is_available():
        print("❌\tTraining requires a CUDA-capable GPU. Exiting.")
        exit(1)
    from torch.utils.data import DataLoader
    from data.dataset import PiscesDataset
    from model import PiscesModel, PiscesConfig
    from trainer.checkpoint import save_ckpt, load_ckpt
    from transformers import get_linear_schedule_with_warmup
    import subprocess
    # Default configuration
    config = "configs/0.5B.json"
    print("✅\tStarting Pisces L1 Training...")
    print("✅\tLoading config file:", config)
    dataset = "tiny_stories"
    batch_size = 2
    epochs = 1
    lr = 5e-5
    save_dir = "ckpt"
    
    # Setup device
    device = setup_device("auto")
    print(f"✅\tDevice set: {device}")
    
    # Load config and model
    print("✅\tLoading PiscesConfig...")
    cfg = PiscesConfig.from_json(config)
    print("✅\tPiscesConfig loaded.")
    print("✅\tInitializing PiscesModel...")
    model = PiscesModel(cfg).to(device)
    print("✅\tPiscesModel initialized.")

    # Multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"✅\tUsing {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Setup optimizer and scheduler
    print("✅\tInitializing optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=100000
    )
    print("✅\tOptimizer and scheduler ready.")

    # Load dataset (auto download if needed)
    print(f"✅\tLoading dataset: {dataset}")
    cache_path = os.path.join("data_cache", dataset)
    if not os.path.exists(cache_path):
        print(f"❌\tLocal dataset not found: {cache_path}")
        print("✅\tAttempting to download ModelScope dataset via data/download.py ...")
        subprocess.run([sys.executable, "data/download.py", "--max_samples", "50000"])
        if not os.path.exists(cache_path):
            raise RuntimeError(f"❌\tDataset download failed or not found at {cache_path}")
        print(f"✅\tDataset downloaded to {cache_path}")
    train_ds = PiscesDataset(subset=dataset, split="train", config=cfg)
    print(f"✅\tDataset loaded successfully, size: {len(train_ds)}")

    # Create data loader
    print("✅\tCreating DataLoader...")
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    print("✅\tDataLoader created successfully")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    print("✅\tStarting training loop...")
    model.train()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        print(f"✅\tStarting epoch {epoch+1}/{epochs}")
        total_loss = 0
        for step, batch in enumerate(train_loader):
            print(f"✅\tEpoch {epoch+1} Step {step} - Batch keys: {list(batch.keys())}")
            # Move to device
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
                torch.cuda.empty_cache()  # Clean up memory fragments
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
                torch.cuda.empty_cache()  # Clean up memory fragments
                total_loss += loss.item()
                if step % 50 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"✅\tEpoch {epoch + 1}/{epochs} | Step {step} | Loss: {avg_loss:.4f}")
            # Save checkpoint
            checkpoint_path = f"{save_dir}/pisces_{dataset}_epoch{epoch + 1}.pt"
            save_ckpt(model, optimizer, epoch + 1, checkpoint_path)
            print(f"✅\tCheckpoint saved: {checkpoint_path}")
        print("✅\tTraining completed!")


def infer(args):
    """Inference function"""
    import torch
    from model import PiscesModel, PiscesConfig
    from transformers import LlamaTokenizerFast
    from PIL import Image
    from torchvision.transforms import functional as TF
    print("✅\tStarting Pisces L1 Inference...")
    
    # Setup device
    device = setup_device("auto")
    
    # Load config and model
    cfg = PiscesConfig.from_json("configs/0.5B.json")
    model = PiscesModel(cfg).to(device).eval()
    
    # Load checkpoint
    if args.ckpt:
        print(f"✅\tLoading model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("✅\tModel loaded successfully")
    else:
        print("❌\tNo model file provided, using random weights")

    # Load tokenizer
    print("✅\tLoading tokenizer...")
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        print("✅\tTokenizer loaded successfully")
    except Exception as e:
        print(f"❌\tError loading tokenizer: {e}")
        print("❌\tCreating dummy tokenizer...")
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=None,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            vocab_size=1000
        )

    # Prepare input
    print(f"✅\tProcessing prompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    # Process image if provided
    pixel_values = None
    if args.image and os.path.exists(args.image):
        print(f"✅\tProcessing image: {args.image}")
        try:
            img = Image.open(args.image).convert("RGB").resize((224, 224))
            pixel_values = TF.to_tensor(img).unsqueeze(0).to(device)
            print("✅\tImage processed successfully")
        except Exception as e:
            print(f"❌\tError processing image: {e}")
            pixel_values = None

    # Generate
    print("✅\tGenerating response...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=input_ids.shape[1] + getattr(args, 'max_length', 100),
            temperature=getattr(args, 'temperature', 0.7),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("✅\tGenerated Response:")
    print("="*50)
    print(generated_text)
    print("="*50)


def main():
    """Main function"""
    args = parse_args()
    
    if args.mode == "setup":
        setup_env()
    elif args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    else:
        print("❌\tInvalid mode. Use 'train', 'infer' or 'setup'")


if __name__ == "__main__":
    main()