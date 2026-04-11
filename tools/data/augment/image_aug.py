#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Image Augmentation for enhancing visual data diversity.

This module provides image augmentation capabilities using Albumentations
or fallback implementations. These augmentations help improve model
generalization for vision tasks.

Key Features:
    - Geometric transforms: Flip, rotate, crop, scale
    - Color transforms: Brightness, contrast, saturation
    - Noise and blur: Gaussian noise, motion blur
    - Configurable augmentation probability

Usage:
    >>> from tools.data.augment import PiscesLxDataImageAugmenter
    >>> augmenter = PiscesLxDataImageAugmenter(aug_prob=0.5)
    >>> augmented_image = augmenter.augment(image)
"""

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class PiscesLxDataImageAugmenter:
    """
    Image augmenter with multiple transformation strategies.
    
    This class provides various image augmentation techniques for
    improving model robustness and generalization on vision tasks.
    
    Attributes:
        aug_prob: Probability of applying augmentation.
        transforms: List of transformation names to use.
        seed: Random seed for reproducibility.
    
    Example:
        >>> augmenter = PiscesLxDataImageAugmenter(
        ...     transforms=['flip', 'rotate', 'brightness'],
        ...     aug_prob=0.5
        ... )
        >>> aug_image = augmenter.augment(image)
    """
    
    def __init__(
        self,
        transforms: Optional[List[str]] = None,
        aug_prob: float = 0.5,
        seed: int = 42
    ) -> None:
        """
        Initialize the image augmenter.
        
        Args:
            transforms: List of transforms to use. Options: 'flip_h', 'flip_v',
                'rotate', 'brightness', 'contrast', 'blur', 'noise', 'crop'.
                Defaults to all transforms.
            aug_prob: Probability of applying augmentation. Defaults to 0.5.
            seed: Random seed. Defaults to 42.
        """
        self.transforms = transforms or [
            'flip_h', 'flip_v', 'rotate', 'brightness', 
            'contrast', 'blur', 'noise'
        ]
        self.aug_prob = aug_prob
        self.seed = seed
        
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        
        self._use_albumentations = self._check_albumentations()
        
        if self._use_albumentations:
            self._albumentations_transform = self._build_albumentations()
    
    def _check_albumentations(self) -> bool:
        """
        Check if Albumentations is available.
        
        Returns:
            bool: True if available.
        """
        try:
            import albumentations
            return True
        except ImportError:
            return False
    
    def _build_albumentations(self) -> Any:
        """
        Build Albumentations transform pipeline.
        
        Returns:
            Any: Albumentations Compose object.
        """
        try:
            import albumentations as A
            
            transform_list = []
            
            if 'flip_h' in self.transforms:
                transform_list.append(A.HorizontalFlip(p=0.5))
            if 'flip_v' in self.transforms:
                transform_list.append(A.VerticalFlip(p=0.5))
            if 'rotate' in self.transforms:
                transform_list.append(A.Rotate(limit=30, p=0.5))
            if 'brightness' in self.transforms:
                transform_list.append(A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0, p=0.5
                ))
            if 'contrast' in self.transforms:
                transform_list.append(A.RandomBrightnessContrast(
                    brightness_limit=0, contrast_limit=0.2, p=0.5
                ))
            if 'blur' in self.transforms:
                transform_list.append(A.GaussianBlur(blur_limit=5, p=0.3))
            if 'noise' in self.transforms:
                transform_list.append(A.GaussNoise(var_limit=(10, 50), p=0.3))
            
            return A.Compose(transform_list) if transform_list else None
        except Exception:
            return None
    
    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return np.fliplr(image).copy()
    
    def _flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically."""
        return np.flipud(image).copy()
    
    def _rotate(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Rotate image by angle degrees."""
        if angle is None:
            angle = random.uniform(-30, 30)
        
        try:
            from scipy.ndimage import rotate
            return rotate(image, angle, reshape=False, mode='reflect')
        except ImportError:
            return image
    
    def _adjust_brightness(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Adjust image brightness."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        
        adjusted = image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    def _adjust_contrast(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Adjust image contrast."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        
        mean = image.mean()
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(image.dtype)
    
    def _add_gaussian_noise(self, image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise to image."""
        if sigma is None:
            sigma = random.uniform(5, 20)
        
        noise = self._rng.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(image.dtype)
    
    def _gaussian_blur(self, image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        """Apply Gaussian blur to image."""
        if kernel_size is None:
            kernel_size = random.choice([3, 5])
        
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image, sigma=kernel_size / 3).astype(image.dtype)
        except ImportError:
            return image
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Augment a single image.
        
        Args:
            image: Input image as numpy array (H, W, C).
            
        Returns:
            np.ndarray: Augmented image.
        """
        if random.random() > self.aug_prob:
            return image
        
        if self._use_albumentations and self._albumentations_transform is not None:
            try:
                result = self._albumentations_transform(image=image)
                return result['image']
            except Exception:
                pass
        
        augmented = image.copy()
        
        num_transforms = random.randint(1, min(3, len(self.transforms)))
        selected_transforms = random.sample(self.transforms, num_transforms)
        
        for transform in selected_transforms:
            if transform == 'flip_h':
                augmented = self._flip_horizontal(augmented)
            elif transform == 'flip_v':
                augmented = self._flip_vertical(augmented)
            elif transform == 'rotate':
                augmented = self._rotate(augmented)
            elif transform == 'brightness':
                augmented = self._adjust_brightness(augmented)
            elif transform == 'contrast':
                augmented = self._adjust_contrast(augmented)
            elif transform == 'blur':
                augmented = self._gaussian_blur(augmented)
            elif transform == 'noise':
                augmented = self._add_gaussian_noise(augmented)
        
        return augmented
    
    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment multiple images.
        
        Args:
            images: List of input images.
            
        Returns:
            List[np.ndarray]: List of augmented images.
        """
        return [self.augment(img) for img in images]
