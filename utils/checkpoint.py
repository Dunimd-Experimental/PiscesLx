#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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

# Use dms_core logging instead of standard logging
import dms_core
logger = dms_core.log

class PiscesLxCoreCheckpointManager:
    """
    A high-level checkpoint manager for handling PyTorch model checkpoints.
    This class provides functionality to save, load, and load the latest checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir (str): Directory to store checkpoints. Defaults to "checkpoints".
        """
        self.checkpoint_dir = checkpoint_dir
        
    def save(self, model, optimizer, epoch: int, name: str = None) -> str:
        """
        Save a model checkpoint with automatic path generation.

        Args:
            model (torch.nn.Module): The PyTorch model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer to be saved.
            epoch (int): The current training epoch.
            name (str, optional): Custom checkpoint name. If None, uses epoch-based naming.

        Returns:
            str: Path to the saved checkpoint.
        """
        if name is None:
            name = f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, name)
        self._save_ckpt(model, optimizer, epoch, path)
        return path
        
    def load(self, path: str, model, optimizer) -> int:
        """
        Load a checkpoint from the specified path into the model and optimizer.

        Args:
            path (str): Path to the checkpoint file.
            model (torch.nn.Module): The PyTorch model to load the state into.
            optimizer (torch.optim.Optimizer): The optimizer to load the state into.

        Returns:
            int: The epoch number stored in the checkpoint.
        """
        return self._load_ckpt(path, model, optimizer)
        
    def load_latest(self, model, optimizer) -> tuple[int, str]:
        """
        Load the latest checkpoint from the checkpoint directory.

        Args:
            model (torch.nn.Module): The PyTorch model to load the state into.
            optimizer (torch.optim.Optimizer): The optimizer to load the state into.

        Returns:
            tuple[int, str]: A tuple containing the epoch number and the path of the loaded checkpoint.

        Raises:
            PiscesLxCoreIOError: If no checkpoints or valid checkpoints are found.
        """
        import glob
        # Generate a pattern to match all epoch-based checkpoint files
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            raise IOError(f"No checkpoints found in {self.checkpoint_dir}")
            
        # Extract epoch numbers from checkpoint filenames and pair them with their paths
        epochs = []
        for ckpt in checkpoints:
            try:
                # Extract the epoch number from the filename
                epoch_str = ckpt.split("_epoch_")[1].split(".pt")[0]
                epochs.append((int(epoch_str), ckpt))
            except (IndexError, ValueError):
                continue
                
        if not epochs:
            raise IOError(f"No valid checkpoints found in {self.checkpoint_dir}")
            
        # Find the checkpoint with the largest epoch number
        latest_epoch, latest_path = max(epochs, key=lambda x: x[0])
        epoch = self.load(latest_path, model, optimizer)
        return epoch, latest_path
    
    @staticmethod
    def _save_ckpt(model, optimizer, epoch, path):
        """
        Save the model, optimizer state, and epoch number to the specified path.

        Args:
            model (torch.nn.Module): The PyTorch model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer to be saved.
            epoch (int): The current training epoch.
            path (str): The path where the checkpoint will be saved.

        Raises:
            IOError: If the directory creation fails or the checkpoint save fails.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except OSError as e:
            logger._Ferror("PiscesLx.Core.Checkpoint", f"failed to create checkpoint directory: {e}")
            raise IOError(f"failed to create checkpoint directory: {e}") from e

        try:
            # Save the model state, optimizer state, and epoch number to the checkpoint file
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, path)
            logger._Fdebug("PiscesLx.Core.Checkpoint", f"checkpoint saved: {path}")
        except Exception as e:
            logger._Ferror("PiscesLx.Core.Checkpoint", f"failed to save checkpoint: {e}")
            raise IOError(f"failed to write checkpoint: {e}") from e
    
    @staticmethod
    def _load_ckpt(path, model, optimizer):
        """
        Load the model and optimizer states from the specified checkpoint file.

        Args:
            path (str): The path from which the checkpoint will be loaded.
            model (torch.nn.Module): The PyTorch model to load the state into.
            optimizer (torch.optim.Optimizer): The optimizer to load the state into.

        Returns:
            int: The epoch number stored in the checkpoint.

        Raises:
            IOError: If the checkpoint is not found, loading fails, 
                    or the checkpoint format is invalid.
        """
        try:
            # Load the checkpoint data to CPU
            ckpt = torch.load(path, map_location='cpu')
        except FileNotFoundError as e:
            logger._Ferror("PiscesLx.Core.Checkpoint", f"checkpoint not found: {e}")
            raise IOError(f"checkpoint not found: {path}") from e
        except Exception as e:
            logger._Ferror("PiscesLx.Core.Checkpoint", f"failed to load checkpoint: {e}")
            raise IOError(f"failed to load checkpoint: {e}") from e

        try:
            # Load the model and optimizer states from the checkpoint
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            epoch = ckpt['epoch']
            logger._Fdebug("PiscesLx.Core.Checkpoint", f"checkpoint loaded: {path}")
            return epoch
        except KeyError as e:
            logger._Ferror("PiscesLx.Core.Checkpoint", f"invalid checkpoint format: missing key {e}")
            raise IOError(f"invalid checkpoint format: missing key {e}") from e
        except Exception as e:
            logger._Ferror("PiscesLx.Core.Checkpoint", f"failed to restore states from checkpoint: {e}")
            raise IOError(f"failed to restore states from checkpoint: {e}") from e