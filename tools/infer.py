#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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
from utils.log import RIGHT, DEBUG, ERROR

def setup_device(device_pref):
    """
    Set up the computing device based on the given preference.

    Args:
        device_pref (str): Device preference, e.g., "auto", "cpu", "cuda".

    Returns:
        torch.device: The selected computing device.
    """
    import torch
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

def infer(args):
    """
    Perform inference using the Pisces model.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration parameters.

    Returns:
        None: The function prints the generated response directly.
    """
    import torch
    from PIL import Image
    from model.tokenizer import get_tokenizer
    from model import PiscesModel, PiscesConfig
    from transformers import BitsAndBytesConfig
    from torchvision.transforms import functional as TF
    import torch.nn.functional as F
    RIGHT("Starting Pisces L1 Inference ...")
    device = setup_device("auto")
    
    model_size = getattr(args, "model_size", "0.5B").upper()
    cfg = PiscesConfig.from_json(f"configs/{model_size}.json")
    # Automatic 4-bit/LoRA/mixed precision inference
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = PiscesModel(cfg, quantization_config=quant_config)
    lora_used = False
    if args.ckpt:
        RIGHT(f"Loading model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        ckpt_vocab_size = state_dict['embed.weight'].shape[0] if 'embed.weight' in state_dict else None
        model_vocab_size = model.embed.weight.shape[0]
        if ckpt_vocab_size and ckpt_vocab_size != model_vocab_size:
            DEBUG(f"Vocab size mismatch: checkpoint={ckpt_vocab_size}, model={model_vocab_size}. Auto resizing...")
            model.resize_token_embeddings(ckpt_vocab_size)

        lora_keys = [k for k in state_dict.keys() if k.startswith('base_model.model.') or '.lora_A.' in k or '.lora_B.' in k]
        if lora_keys:
            from peft import get_peft_model, LoraConfig, TaskType
            RIGHT("Detected LoRA/QLoRA checkpoint, wrapping PiscesModel with LoRA config...")
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
            )
            lora_model = get_peft_model(model, lora_config)
            for attr in ["cfg", "quantization_config", "lora_config", "forward", "prepare_inputs_for_generation"]:
                if hasattr(model, attr):
                    setattr(lora_model, attr, getattr(model, attr))
            model = lora_model
            lora_used = True
        model = model.to(device).eval()
        model.load_state_dict(state_dict, strict=False)
        RIGHT("Model loaded successfully")
    else:
        model = model.to(device).eval()
        ERROR("No model file provided, using random weights")
    RIGHT("Loading Pisces BPETokenizer...")
    tokenizer = get_tokenizer()
    RIGHT("Pisces BPETokenizer loaded successfully")
    RIGHT(f"Processing prompt: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    pixel_values = None
    if args.image and os.path.exists(args.image):
        RIGHT(f"Processing image: {args.image}")
        try:
            img = Image.open(args.image).convert("RGB").resize((224, 224))
            pixel_values = TF.to_tensor(img).unsqueeze(0).to(device)
            RIGHT("Image processed successfully")
        except Exception as e:
            ERROR(f"Error processing image: {e}")
            pixel_values = None
    RIGHT("Generating response (Automatic blocking/Mixed precision/4-bit)...")
    max_gen_len = getattr(args, 'max_length', 100)
    
    prompt_len = input_ids.shape[1]
    if prompt_len < 20:
        top_k = 40
        top_p = 0.9
    else:
        top_k = 20
        top_p = 0.8
    chunk_size = min(getattr(cfg, 'max_position_embeddings', 2048), 512)
    
    # Speculative decoding support
    draft_model = None
    if getattr(args, 'speculative', False) and getattr(args, 'draft_model', None):
        try:
            draft_cfg = PiscesConfig.from_json(f"configs/{args.draft_model.upper()}.json")
            draft_model = PiscesModel(draft_cfg, quantization_config=quant_config)
            if args.ckpt:
                draft_ckpt = args.ckpt.replace('.pth', '_draft.pth')
                if os.path.exists(draft_ckpt):
                    draft_checkpoint = torch.load(draft_ckpt, map_location=device)
                    draft_state_dict = draft_checkpoint['model'] if 'model' in draft_checkpoint else draft_checkpoint
                    draft_model.load_state_dict(draft_state_dict, strict=False)
                    RIGHT("Draft model loaded for speculative decoding")
                else:
                    RIGHT("Draft model checkpoint not found, using random weights")
            draft_model = draft_model.to(device).eval()
        except Exception as e:
            ERROR(f"Failed to load draft model: {e}")
            draft_model = None
    
    generated_ids = []
    
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    
    def greedy_generate(model, input_ids, max_new_tokens, images=None):
        """Greedy generation for speculative decoding."""
        cur_input = input_ids
        new_tokens = []
        
        with torch.no_grad(), autocast_ctx:
            for _ in range(max_new_tokens):
                logits_chunks = []
                for i in range(0, cur_input.shape[1], chunk_size):
                    chunk = cur_input[:, i:i+chunk_size]
                    outputs = model(chunk, images=images)
                    logits = outputs["logits"]
                    logits_chunks.append(logits)
                logits = torch.cat(logits_chunks, dim=1)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                new_tokens.append(next_token.item())
                cur_input = torch.cat([cur_input, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        return new_tokens
    
    with torch.no_grad(), autocast_ctx:
        cur_input = input_ids
        
        if draft_model is not None:
            # Speculative decoding
            RIGHT("Using speculative decoding with draft model...")
            gamma = getattr(args, 'spec_gamma', 4)
            
            while len(generated_ids) < max_gen_len:
                # Draft model generates gamma tokens
                draft_tokens = greedy_generate(draft_model, cur_input, gamma, pixel_values)
                
                # Verify with target model
                verify_input = torch.cat([cur_input, torch.tensor([draft_tokens], device=device)], dim=1)
                
                logits_chunks = []
                for i in range(0, verify_input.shape[1], chunk_size):
                    chunk = verify_input[:, i:i+chunk_size]
                    outputs = model(chunk, images=pixel_values)
                    logits = outputs["logits"]
                    logits_chunks.append(logits)
                target_logits = torch.cat(logits_chunks, dim=1)
                
                # Find first mismatch
                accepted = 0
                for i, draft_token in enumerate(draft_tokens):
                    target_token = torch.argmax(target_logits[0, cur_input.shape[1] + i, :])
                    if draft_token == target_token.item():
                        accepted += 1
                        generated_ids.append(draft_token)
                    else:
                        generated_ids.append(target_token.item())
                        break
                
                if accepted == len(draft_tokens):
                    # All tokens accepted, add last token from target
                    last_token = torch.argmax(target_logits[0, -1, :])
                    generated_ids.append(last_token.item())
                
                cur_input = torch.cat([input_ids, torch.tensor([generated_ids], device=device)], dim=1)
                
                if any(t == tokenizer.eos_token_id for t in generated_ids[-len(draft_tokens)-1:]):
                    break
        else:
            # Standard generation
            for _ in range(max_gen_len):
                logits_chunks = []
                for i in range(0, cur_input.shape[1], chunk_size):
                    chunk = cur_input[:, i:i+chunk_size]
                    outputs = model(chunk, images=pixel_values)
                    logits = outputs["logits"]
                    logits_chunks.append(logits)
                logits = torch.cat(logits_chunks, dim=1)
                next_token_logits = logits[:, -1, :]
                
                filtered_logits = next_token_logits.clone()
                if top_k > 0:
                    top_k = min(top_k, filtered_logits.size(-1))
                    values, _ = torch.topk(filtered_logits, top_k)
                    min_values = values[:, -1].unsqueeze(-1)
                    filtered_logits[filtered_logits < min_values] = -float('Inf')
                if 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    if sorted_indices_to_remove[..., 1:].size(-1) > 0:
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    filtered_logits[0, indices_to_remove] = -float('Inf')
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token.item())
                cur_input = torch.cat([cur_input, next_token], dim=1)
    output_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    DEBUG("\n" + "="*50)
    RIGHT("Generated Response:")
    DEBUG("="*50)
    DEBUG(generated_text)
    DEBUG("="*50)
