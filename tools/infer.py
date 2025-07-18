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

def infer(args):
    import torch
    from model import PiscesModel, PiscesConfig
    from model.tokenizer import get_tokenizer
    from transformers import BitsAndBytesConfig
    from PIL import Image
    from torchvision.transforms import functional as TF
    print("✅\tStarting Pisces L1 Inference ...")
    device = setup_device("auto")
    # 自动检测模型规模
    model_size = getattr(args, "model_size", "0.5B").upper()
    cfg = PiscesConfig.from_json(f"configs/{model_size}.json")
    # 自动4bit/LoRA/混合精度推理
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = PiscesModel(cfg, quantization_config=quant_config)
    lora_used = False
    if args.ckpt:
        print(f"✅\tLoading model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # 检查是否为LoRA权重
        lora_keys = [k for k in state_dict.keys() if k.startswith('base_model.model.') or '.lora_A.' in k or '.lora_B.' in k]
        if lora_keys:
            from peft import get_peft_model, LoraConfig, TaskType
            print("✅\tDetected LoRA/QLoRA checkpoint, wrapping PiscesModel with LoRA config...")
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
            )
            lora_model = get_peft_model(model, lora_config)
            # 保证PiscesModel自定义属性/方法不丢失
            for attr in ["cfg", "quantization_config", "lora_config", "forward", "prepare_inputs_for_generation"]:
                if hasattr(model, attr):
                    setattr(lora_model, attr, getattr(model, attr))
            model = lora_model
            lora_used = True
        model = model.to(device).eval()
        model.load_state_dict(state_dict, strict=False)
        print("✅\tModel loaded successfully")
    else:
        model = model.to(device).eval()
        print("❌\tNo model file provided, using random weights")
    print("✅\tLoading Pisces BPETokenizer...")
    tokenizer = get_tokenizer()
    print("✅\tPisces BPETokenizer loaded successfully")
    print(f"✅\tProcessing prompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
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
    print("✅\tGenerating response (自动分块/混合精度/4bit)...")
    max_gen_len = getattr(args, 'max_length', 100)
    chunk_size = min(getattr(cfg, 'max_position_embeddings', 2048), 512)
    generated_ids = []
    # 兼容不同PyTorch版本的autocast
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    with torch.no_grad(), autocast_ctx:
        cur_input = input_ids
        for _ in range(max_gen_len):
            # 自动分块推理，防止OOM
            logits_chunks = []
            for i in range(0, cur_input.shape[1], chunk_size):
                chunk = cur_input[:, i:i+chunk_size]
                logits, *_ = model(chunk, images=pixel_values)
                logits_chunks.append(logits)
            logits = torch.cat(logits_chunks, dim=1)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token.item())
            cur_input = torch.cat([cur_input, next_token], dim=1)
    output_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("\n" + "="*50)
    print("✅\tGenerated Response:")
    print("="*50)
    print(generated_text)
    print("="*50) 