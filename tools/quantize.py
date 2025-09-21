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

import torch, os
from utils import RIGHT
from model import PiscesModel, PiscesConfig
from transformers import BitsAndBytesConfig

def quantize(checkpoint, save_path, bits=8):
    """
    Quantize the model loaded from the given checkpoint and save the quantized model.

    Args:
        checkpoint (str): Path to the checkpoint file containing the model state.
        save_path (str): Path where the quantized model state will be saved.
        bits (int, optional): Quantization bits, either 4 or 8. Defaults to 8.
    """
    # Validate input arguments
    _validate_quantize_args(checkpoint, save_path, bits)

    # Load model configuration
    cfg = PiscesConfig.from_json("configs/0.5B.json")
    
    # Initialize model with the loaded configuration
    model = PiscesModel(cfg)
    
    # Load model state from the checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

    # Perform 8-bit quantization if bits is set to 8
    if bits == 8:
        import bitsandbytes as bnb
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight = bnb.nn.Params8bit(m.weight)

    # Create the directory for the save path if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the quantized model state
    torch.save(model.state_dict(), save_path)
    
    # Log the save path
    RIGHT("Quantized model saved to", save_path)


def _validate_quantize_args(checkpoint: str, save_path: str, bits: int):
    """
    Validate and normalize arguments for quantize().

    Args:
        checkpoint (str): Path to the checkpoint file containing the model state.
        save_path (str): Path where the quantized model state will be saved.
        bits (int): Quantization bits, either 4 or 8.

    Raises:
        ValueError: If checkpoint does not exist.
        ValueError: If save_path is empty.
        ValueError: If bits is not an integer.
        ValueError: If bits is not 4 or 8.
    """
    # Check if the checkpoint exists
    if not checkpoint or not os.path.exists(checkpoint):
        raise ValueError(f"checkpoint not found: {checkpoint}")
    
    # Check if save_path is provided
    if not save_path:
        raise ValueError("save_path is required")
    
    # Check if bits can be converted to an integer
    try:
        b = int(bits)
    except Exception:
        raise ValueError("bits must be integer")
    
    # Check if bits is either 4 or 8
    if b not in (4, 8):
        raise ValueError("bits must be one of {4, 8}")