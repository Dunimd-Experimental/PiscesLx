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

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup and restart mechanism (SGDR).
    
    This scheduler combines:
    - Warmup phase: Linearly increase learning rate from warmup_start_lr to base_lr
    - Cosine annealing: Decrease learning rate following cosine curve
    - Restarts: Periodically reset learning rate to base_lr (SGDR mechanism)
    
    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer
        warmup_steps (int): Number of warmup steps
        T_0 (int): Number of iterations for the first restart
        T_mult (int, optional): Factor to increase T_i after each restart. Defaults to 2.
        eta_min (float, optional): Minimum learning rate. Defaults to 0.
        warmup_start_lr (float, optional): Starting learning rate for warmup. Defaults to 0.
        last_epoch (int, optional): The index of last epoch. Defaults to -1.
    """
    
    def __init__(self, optimizer, warmup_steps, T_0, T_mult=2, eta_min=0, warmup_start_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.T_cur = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur < self.warmup_steps:
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.T_cur / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        
        T_cur = self.T_cur - self.warmup_steps
        T_i = self.T_0
        
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ModalityAwareCosineScheduler:
    """
    Modality-aware cosine annealing scheduler with warmup and restarts.
    
    This scheduler manages multiple optimizers for different modalities (text, vision, audio, fusion, other)
    with independent learning rate schedules but coordinated warmup and restart cycles.
    
    Args:
        optimizers (dict): Dictionary mapping modality names to optimizers
        base_lrs (dict): Dictionary mapping modality names to base learning rates
        warmup_steps (int): Number of warmup steps
        T_0 (int): Number of iterations for the first restart
        T_mult (int, optional): Factor to increase T_i after each restart. Defaults to 2.
        eta_min (dict, optional): Dictionary mapping modality names to minimum learning rates. Defaults to None.
        warmup_start_lr (dict, optional): Dictionary mapping modality names to warmup start learning rates. Defaults to None.
    """
    
    def __init__(self, optimizers, base_lrs, warmup_steps, T_0, T_mult=2, eta_min=None, warmup_start_lr=None):
        self.optimizers = optimizers
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min or {modality: 0 for modality in optimizers.keys()}
        self.warmup_start_lr = warmup_start_lr or {modality: 0 for modality in optimizers.keys()}
        
        self.schedulers = {}
        for modality, optimizer in optimizers.items():
            self.schedulers[modality] = CosineAnnealingWarmupRestarts(
                optimizer,
                warmup_steps=warmup_steps,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=self.eta_min[modality],
                warmup_start_lr=self.warmup_start_lr[modality]
            )
    
    def step(self):
        for scheduler in self.schedulers.values():
            scheduler.step()
    
    def get_lr(self, modality):
        if modality in self.schedulers:
            return self.schedulers[modality].get_last_lr()[0]
        return None
    
    def get_all_lrs(self):
        return {modality: self.get_lr(modality) for modality in self.schedulers.keys()}


def get_scheduler(optimizer, T_0=1000):
    """
    Create and return an instance of the CosineAnnealingWarmRestarts learning rate scheduler.

    This function initializes a CosineAnnealingWarmRestarts scheduler with the given optimizer 
    and the number of iterations for the first restart.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be wrapped by the scheduler.
        T_0 (int, optional): Number of iterations for the first restart. The scheduler will restart 
                            the cosine annealing process every T_0 iterations. Defaults to 1000.

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts: An instance of the 
                                                             CosineAnnealingWarmRestarts scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)
