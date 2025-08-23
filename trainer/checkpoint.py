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
import torch

def save_ckpt(model, optimizer, epoch, path):
    """
    Save the model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        epoch (int): The current training epoch.
        path (str): The path where the checkpoint will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'epoch': epoch
    }, path)


def load_ckpt(path, model, optimizer):
    """
    Load the model checkpoint from the specified path.

    Args:
        path (str): The path from which the checkpoint will be loaded.
        model (torch.nn.Module): The PyTorch model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The epoch number stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']