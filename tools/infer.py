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
    from transformers import LlamaTokenizerFast
    from PIL import Image
    from torchvision.transforms import functional as TF
    print("✅\tStarting Pisces L1 Inference...")
    device = setup_device("auto")
    cfg = PiscesConfig.from_json("configs/0.5B.json")
    model = PiscesModel(cfg).to(device).eval()
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
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("✅\tGenerated Response:")
    print("="*50)
    print(generated_text)
    print("="*50) 