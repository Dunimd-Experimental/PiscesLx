#!/usr/bin/env/python3

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
import asyncio
import time
import contextlib
from utils.gpu_manager import GPUManager
from utils.log import RIGHT, DEBUG, ERROR
from tools.watermark import watermark_manager, watermark_text

def setup_inference_device(device_pref):
    """
    Set up the inference device, supporting automatic selection of single-GPU/multi-GPU.

    Args:
        device_pref (str): Device preference, e.g., "auto", "cpu", "cuda".

    Returns:
        torch.device: The selected device.
    """
    import torch

    if device_pref == "auto":
        # Prefer a fast non-blocking decision path: if CUDA isn't available, go CPU directly
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            RIGHT("CUDA not available, falling back to CPU inference mode")
            RIGHT("Inference mode: cpu")
        else:
            # Try GPUManager with a safety net to avoid crashes/hangs
            try:
                gpu_manager = GPUManager()
                gpu_manager.print_summary()
                strategy = gpu_manager.get_inference_strategy()

                if strategy['mode'] == 'cpu':
                    device = torch.device("cpu")
                    RIGHT("CPU inference mode")
                elif strategy['mode'] == 'single_gpu':
                    device = torch.device(f"cuda:{strategy['gpu_ids'][0]}")
                    torch.cuda.set_device(strategy['gpu_ids'][0])
                else:
                    # Multi-GPU inference, use the first GPU or DataParallel
                    device = torch.device("cuda")

                RIGHT(f"Inference mode: {strategy['mode']}")
            except Exception as e:
                # Any issue with GPUManager -> safe fallback to single CUDA device or CPU
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    RIGHT(f"GPUManager unavailable, using default CUDA device: {e}")
                else:
                    device = torch.device("cpu")
                    RIGHT(f"GPUManager unavailable and CUDA not available, using CPU: {e}")
    else:
        device = torch.device(device_pref)

    RIGHT(f"Using device: {device}")
    return device

class VLLMEngine:
    """VLLM-based high-performance inference engine for Pisces L1 models."""
    
    def __init__(self, model_path, dtype="auto", gpu_memory_utilization=0.9, tensor_parallel_size=1):
        """
        Initialize the VLLMEngine instance.
        
        Args:
            model_path (str): Path to the model to be loaded.
            dtype (str, optional): Data type for the model. Defaults to "auto".
            gpu_memory_utilization (float, optional): Proportion of GPU memory to use. Defaults to 0.9.
            tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism. Defaults to 1.
        """
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
            self.LLM = LLM
            self.SamplingParams = SamplingParams
        except ImportError:
            self.vllm_available = False
            ERROR("VLLM not available, falling back to native inference")
            return
            
        self.model_path = model_path
        self.llm = self.LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size
        )

    def infer(self, prompt, temperature=0.7, max_tokens=512, top_p=0.95, stop=None):
        """
        Generate text using VLLM engine.
        
        Args:
            prompt (str): Input prompt for text generation.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 512.
            top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
            stop (list, optional): Stop sequences. Defaults to None.
            
        Returns:
            str: Generated text or None if VLLM unavailable.
        """
        if not self.vllm_available:
            return None
            
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Add hidden watermark for VLLM generated content
        watermark_metadata = {
            "prompt": prompt,
            "params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "use_vllm": True
            },
            "user_id": "vllm_user",
            "timestamp": str(int(time.time()))
        }
        
        return watermark_text(generated_text, prompt, watermark_metadata)

def infer(args):
    """
    Perform inference using the Pisces model with integrated MCP server.

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
    
    # Validate and normalize args first
    try:
        args = validate_infer_args(args)
    except Exception as e:
        ERROR(f"Invalid inference arguments: {e}")
        raise

    RIGHT("Starting Pisces L1 Inference with MCP Integration...")
    
    device = setup_inference_device("auto")
    
    # Get the model size from arguments, default to "0.5B"
    model_size = getattr(args, "model_size", "0.5B").upper()
    cfg = PiscesConfig.from_json(f"configs/{model_size}.json")
    # Enable automatic 4-bit/LoRA/mixed precision inference
    use_quantization = cfg.force_quant if hasattr(cfg, 'force_quant') else False
    
    if use_quantization:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = PiscesModel(cfg, quantization_config=quant_config)
    else:
        model = PiscesModel(cfg)
    
    # Support multi-GPU inference
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        RIGHT(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel inference")
        model = torch.nn.DataParallel(model)
    
    lora_used = False
    if args.ckpt:
        RIGHT(f"Loading model: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        ckpt_vocab_size = state_dict['embed.weight'].shape[0] if 'embed.weight' in state_dict else None
        model_vocab_size = model.module.embed.weight.shape[0] if hasattr(model, 'module') else model.embed.weight.shape[0]
        if ckpt_vocab_size and ckpt_vocab_size != model_vocab_size:
            DEBUG(f"Vocab size mismatch: checkpoint={ckpt_vocab_size}, model={model_vocab_size}. Auto resizing...")
            if hasattr(model, 'module'):
                model.module.resize_token_embeddings(ckpt_vocab_size)
            else:
                model.resize_token_embeddings(ckpt_vocab_size)

        lora_keys = [k for k in state_dict.keys() if k.startswith('base_model.model.') or '.lora_A.' in k or '.lora_B.' in k]
        if lora_keys:
            from peft import get_peft_model, LoraConfig, TaskType
            RIGHT("Detected LoRA/QLoRA checkpoint, wrapping PiscesModel with LoRA config...")
            lora_config = LoraConfig(
                r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
            )
            
            # Handle DataParallel wrapping
            if hasattr(model, 'module'):
                base_model = model.module
            else:
                base_model = model
                
            lora_model = get_peft_model(base_model, lora_config)
            for attr in ["cfg", "quantization_config", "lora_config", "forward", "prepare_inputs_for_generation"]:
                if hasattr(base_model, attr):
                    setattr(lora_model, attr, getattr(base_model, attr))
            
            if hasattr(model, 'module'):
                model.module = lora_model
            else:
                model = lora_model
            lora_used = True
            
        model = model.to(device).eval()
        
        # Handle DataParallel's state_dict
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
            
        RIGHT("Model loaded successfully")
    else:
        model = model.to(device).eval()
        ERROR("No model file provided, using random weights")
    # Check if VLLM mode is requested
    use_vllm = getattr(args, 'use_vllm', False) and args.ckpt and os.path.exists(args.ckpt)
    vllm_engine = None
    
    if use_vllm:
        try:
            RIGHT("Initializing VLLM engine...")
            vllm_engine = VLLMEngine(
                model_path=args.ckpt,
                dtype=getattr(args, 'vllm_dtype', 'auto'),
                gpu_memory_utilization=getattr(args, 'vllm_gpu_mem', 0.9),
                tensor_parallel_size=getattr(args, 'vllm_tp_size', 1)
            )
            if vllm_engine.vllm_available:
                RIGHT("VLLM engine initialized successfully")
            else:
                use_vllm = False
                RIGHT("VLLM not available, using native inference")
        except Exception as e:
            ERROR(f"Failed to initialize VLLM: {e}, using native inference")
            use_vllm = False
    
    if use_vllm:
        # VLLM inference path
        RIGHT("Using VLLM for high-performance inference...")
        generated_text = vllm_engine.infer(
            prompt=args.prompt,
            temperature=getattr(args, 'temperature', 0.7),
            max_tokens=getattr(args, 'max_length', 512),
            top_p=getattr(args, 'top_p', 0.95),
            stop=getattr(args, 'stop', None)
        )
        
        if generated_text:
            # Skip MCP processing to prevent hanging
                    RIGHT("\n" + "="*50)
                    RIGHT("Generated Response:")
                    RIGHT("="*50)
                    RIGHT(generated_text)
                    return
    
    # Native Pisces inference path
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
    
    # Support speculative decoding
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
    
    # Choose proper autocast context based on device to avoid CUDA init on CPU-only systems
    if device.type == 'cuda':
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()
    
    def greedy_generate(model, input_ids, max_new_tokens, images=None):
        """
        Perform greedy generation for speculative decoding.

        Args:
            model: The model used for generation.
            input_ids (torch.Tensor): Input token IDs.
            max_new_tokens (int): Maximum number of new tokens to generate.
            images (torch.Tensor, optional): Input images. Defaults to None.

        Returns:
            list: Generated token IDs.
        """
        cur_input = input_ids
        new_tokens = []
        
        with torch.no_grad(), autocast_ctx:
            for _ in range(max_new_tokens):
                logits_chunks = []
                for i in range(0, cur_input.shape[1], chunk_size):
                    chunk = cur_input[:, i:i+chunk_size]
                    actual_model = model.module if hasattr(model, 'module') else model
                    outputs = actual_model(chunk, images=images)
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
            # Perform speculative decoding
            RIGHT("Using speculative decoding with draft model...")
            gamma = getattr(args, 'spec_gamma', 4)
            
            while len(generated_ids) < max_gen_len:
                # The draft model generates gamma tokens
                draft_tokens = greedy_generate(draft_model, cur_input, gamma, pixel_values)
                
                # Verify with the target model
                verify_input = torch.cat([cur_input, torch.tensor([draft_tokens], device=device)], dim=1)
                
                logits_chunks = []
                for i in range(0, verify_input.shape[1], chunk_size):
                    chunk = verify_input[:, i:i+chunk_size]
                    actual_model = model.module if hasattr(model, 'module') else model
                    outputs = actual_model(chunk, images=pixel_values)
                    logits = outputs["logits"]
                    logits_chunks.append(logits)
                target_logits = torch.cat(logits_chunks, dim=1)
                
                # Find the first mismatch
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
                    # All tokens are accepted, add the last token from the target
                    last_token = torch.argmax(target_logits[0, -1, :])
                    generated_ids.append(last_token.item())
                
                cur_input = torch.cat([input_ids, torch.tensor([generated_ids], device=device)], dim=1)
                
                if any(t == tokenizer.eos_token_id for t in generated_ids[-len(draft_tokens)-1:]):
                    break
                
                # Clear cache after each iteration to prevent memory explosion
                if torch.cuda.is_available():
                    del draft_tokens
                    torch.cuda.empty_cache()
                    
        else:
            # Perform standard generation
            for _ in range(max_gen_len):
                logits_chunks = []
                for i in range(0, cur_input.shape[1], chunk_size):
                    chunk = cur_input[:, i:i+chunk_size]
                    actual_model = model.module if hasattr(model, 'module') else model
                    outputs = actual_model(chunk, images=pixel_values)
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
                
                # Clear cache after each iteration to prevent memory explosion
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    output_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # Add hidden watermark to generated content with 2025 mandatory standard
    watermark_metadata = {
        "prompt": args.prompt,
        "params": {
            "temperature": getattr(args, 'temperature', 0.7),
            "max_tokens": getattr(args, 'max_length', 512),
            "top_p": getattr(args, 'top_p', 0.95),
            "model_size": getattr(args, 'model_size', '1.5B'),
            "use_vllm": getattr(args, 'use_vllm', False),
            "use_speculative": getattr(args, 'use_speculative', False)
        },
        "user_id": getattr(args, 'user_id', 'anonymous'),
        "timestamp": str(int(time.time())),
        "generation_method": "text_inference",
        
    }
    
    from tools.watermark import watermark_manager
    watermarked_text = watermark_manager.add_watermark(generated_text, watermark_metadata)
    generated_text = watermarked_text
    
    # MCP processing: only import and run if output contains <agent> tag
    if "<agent>" in generated_text:
        try:
            from model.mcp.translator import MCPTranslationLayer
            translator = MCPTranslationLayer()
            if translator._wait_for_ready(timeout=0.05):
                RIGHT("Processing with MCP tools (config-driven)...")
                generated_text = translator.remove_agent_tags(generated_text)
            else:
                RIGHT("MCP tools not ready, returning direct response")
        except Exception as e:
            RIGHT(f"MCP unavailable, returning direct response: {e}")
    else:
        RIGHT("No agent tags detected, returning direct response")

    RIGHT("\n" + "="*50)
    RIGHT("Generated Response:")
    RIGHT("="*50)
    RIGHT(generated_text)
    RIGHT("="*50)

def _get_attr(obj, name, default):
    return getattr(obj, name, default)

def validate_infer_args(args):
    """Validate and normalize arguments for infer().
    Ensures required fields exist, fills defaults, and performs basic range/path checks.
    Returns the same args (possibly with attributes set to defaults).
    Raises ValueError on fatal validation errors.
    """
    # Required: prompt
    if not hasattr(args, 'prompt') or args.prompt is None or str(args.prompt).strip() == "":
        raise ValueError("Missing required argument: prompt")

    # Defaults
    if not hasattr(args, 'model_size') or not args.model_size:
        setattr(args, 'model_size', '0.5B')
    if not hasattr(args, 'max_length') or not isinstance(args.max_length, int):
        setattr(args, 'max_length', 512)
    if not hasattr(args, 'temperature') or args.temperature is None:
        setattr(args, 'temperature', 0.7)
    if not hasattr(args, 'top_p') or args.top_p is None:
        setattr(args, 'top_p', 0.95)
    if not hasattr(args, 'stop'):
        setattr(args, 'stop', None)
    if not hasattr(args, 'use_vllm'):
        setattr(args, 'use_vllm', False)
    if not hasattr(args, 'vllm_dtype'):
        setattr(args, 'vllm_dtype', 'auto')
    if not hasattr(args, 'vllm_gpu_mem'):
        setattr(args, 'vllm_gpu_mem', 0.9)
    if not hasattr(args, 'vllm_tp_size'):
        setattr(args, 'vllm_tp_size', 1)
    if not hasattr(args, 'speculative'):
        setattr(args, 'speculative', False)
    if not hasattr(args, 'spec_gamma'):
        setattr(args, 'spec_gamma', 4)

    # Ranges
    try:
        temp = float(args.temperature)
        if not (0.0 <= temp <= 2.0):
            raise ValueError
    except Exception:
        raise ValueError("temperature must be a float in [0.0, 2.0]")

    try:
        topp = float(args.top_p)
        if not (0.0 < topp <= 1.0):
            raise ValueError
    except Exception:
        raise ValueError("top_p must be a float in (0.0, 1.0]")

    if not isinstance(args.max_length, int) or args.max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    # Paths
    if hasattr(args, 'image') and args.image:
        if not os.path.exists(args.image):
            raise ValueError(f"image path does not exist: {args.image}")

    if hasattr(args, 'ckpt') and args.ckpt:
        # ckpt is optional but if provided must exist
        if not os.path.exists(args.ckpt):
            raise ValueError(f"ckpt path does not exist: {args.ckpt}")

    # VLLM constraints
    if args.use_vllm and (not hasattr(args, 'ckpt') or not args.ckpt or not os.path.exists(args.ckpt)):
        raise ValueError("use_vllm requires a valid --ckpt path to a model checkpoint")

    # Stop sequences
    if args.stop is not None and not isinstance(args.stop, (list, tuple)):
        raise ValueError("stop must be a list of strings or None")

    return args