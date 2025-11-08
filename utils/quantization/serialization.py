"""
Model serialization utilities for quantized models.

This module provides comprehensive save/load functionality for quantized models,
including quantization parameters, configuration, and model state.
"""

import json
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from ..core import PiscesLxCoreLog, PiscesLxCoreException
from .config import QuantizationConfig

logger = PiscesLxCoreLog("quantization.serialization")


class QuantizedModelSerializer:
    """Handles serialization of quantized models with full state preservation."""
    
    def __init__(self):
        from configs.version import PVERSION
        self.serialization_version = PVERSION
    
    def save_quantized_model(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        filepath: Union[str, Path],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a quantized model with full state and configuration.
        
        Args:
            model: The quantized model to save
            config: Quantization configuration used
            filepath: Path to save the model
            additional_metadata: Additional metadata to include
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model state
            model_state = {
                "model_state_dict": model.state_dict(),
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
            }
            
            # Extract quantization parameters from model
            quantization_params = self._extract_quantization_params(model)
            
            # Prepare complete save package
            save_package = {
                "serialization_version": self.serialization_version,
                "quantization_config": asdict(config),
                "model_state": model_state,
                "quantization_params": quantization_params,
                "metadata": {
                    "torch_version": torch.__version__,
                    "python_version": self._get_python_version(),
                    "save_timestamp": self._get_timestamp(),
                    **(additional_metadata or {})
                }
            }
            
            # Save to file
            if filepath.suffix == '.json':
                self._save_json(save_package, filepath)
            else:
                self._save_pickle(save_package, filepath)
            
            logger.info("quantized model saved successfully", 
                       filepath=str(filepath),
                       size_mb=filepath.stat().st_size / (1024 * 1024))
            
        except Exception as e:
            logger.error("failed to save quantized model", error=str(e))
            raise PiscesLxCoreException(f"Failed to save quantized model: {str(e)}")
    
    def load_quantized_model(
        self,
        filepath: Union[str, Path],
        model_class: Optional[type] = None,
        device: Optional[str] = None
    ) -> tuple[nn.Module, QuantizationConfig]:
        """
        Load a quantized model with full state and configuration.
        
        Args:
            filepath: Path to the saved model
            model_class: Model class to instantiate (if not in save file)
            device: Device to load model to
            
        Returns:
            Tuple of (loaded_model, quantization_config)
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load from file
            if filepath.suffix == '.json':
                save_package = self._load_json(filepath)
            else:
                save_package = self._load_pickle(filepath)
            
            # Validate serialization version
            self._validate_serialization_version(save_package)
            
            # Reconstruct quantization config
            config = QuantizationConfig(**save_package["quantization_config"])
            
            # Reconstruct model
            model = self._reconstruct_model(save_package, model_class, device)
            
            # Apply quantization parameters
            self._apply_quantization_params(model, save_package["quantization_params"])
            
            logger.info("quantized model loaded successfully",
                       filepath=str(filepath),
                       model_class=model.__class__.__name__)
            
            return model, config
            
        except Exception as e:
            logger.error("failed to load quantized model", error=str(e))
            raise PiscesLxCoreException(f"Failed to load quantized model: {str(e)}")
    
    def _extract_quantization_params(self, model: nn.Module) -> Dict[str, Any]:
        """Extract quantization parameters from model."""
        params = {
            "quantized_layers": [],
            "scale_factors": {},
            "zero_points": {},
            "bit_widths": {}
        }
        
        for name, module in model.named_modules():
            # Check for quantization-specific attributes
            if hasattr(module, 'weight_scale'):
                params["quantized_layers"].append(name)
                params["scale_factors"][name] = module.weight_scale.cpu()
            
            if hasattr(module, 'weight_zero_point'):
                params["zero_points"][name] = module.weight_zero_point.cpu()
            
            if hasattr(module, 'quantization_bits'):
                params["bit_widths"][name] = module.quantization_bits
        
        return params
    
    def _reconstruct_model(self, save_package: Dict[str, Any], 
                          model_class: Optional[type], device: Optional[str]) -> nn.Module:
        """Reconstruct model from saved state."""
        model_state = save_package["model_state"]
        
        # Determine model class
        if model_class is None:
            class_name = model_state["model_class"]
            module_name = model_state["model_module"]
            
            # Try to import and instantiate model class
            try:
                import importlib
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
            except (ImportError, AttributeError):
                raise PiscesLxCoreException(
                    f"Cannot reconstruct model class {class_name} from {module_name}. "
                    f"Please provide model_class parameter."
                )
        
        # Instantiate model
        model = model_class()
        
        # Load state dict
        model.load_state_dict(model_state["model_state_dict"])
        
        # Move to device if specified
        if device:
            model = model.to(device)
        
        return model
    
    def _apply_quantization_params(self, model: nn.Module, params: Dict[str, Any]) -> None:
        """Apply quantization parameters to model."""
        for layer_name in params["quantized_layers"]:
            # Find layer by name
            layer = dict(model.named_modules()).get(layer_name)
            if layer:
                # Restore quantization attributes
                if layer_name in params["scale_factors"]:
                    layer.weight_scale = params["scale_factors"][layer_name]
                
                if layer_name in params["zero_points"]:
                    layer.weight_zero_point = params["zero_points"][layer_name]
                
                if layer_name in params["bit_widths"]:
                    layer.quantization_bits = params["bit_widths"][layer_name]
    
    def _save_pickle(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data using pickle format."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load data from pickle format."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data using JSON format (for human readability)."""
        def convert_to_serializable(obj):
            """Convert PyTorch tensors and other objects to JSON-serializable format."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, (torch.dtype, torch.device)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_to_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load data from JSON format."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _validate_serialization_version(self, save_package: Dict[str, Any]) -> None:
        """Validate serialization version compatibility."""
        saved_version = save_package.get("serialization_version", "0.0.0")
        current_version = self.serialization_version
        
        # Simple version compatibility check
        if saved_version.split('.')[0] != current_version.split('.')[0]:
            logger.warning("serialization version mismatch",
                         saved_version=saved_version,
                         current_version=current_version)
    
    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class QuantizedModelManager:
    """High-level interface for managing quantized model lifecycle."""
    
    def __init__(self):
        self.serializer = QuantizedModelSerializer()
        self.model_cache = {}
    
    def save_model_with_metadata(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        base_path: Union[str, Path],
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[list] = None
    ) -> Dict[str, Path]:
        """
        Save model with comprehensive metadata and multiple formats.
        
        Args:
            model: Quantized model to save
            config: Quantization configuration
            base_path: Base directory for saving
            model_name: Name for the model
            description: Model description
            tags: Tags for categorization
            
        Returns:
            Dictionary mapping format names to file paths
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "model_name": model_name,
            "description": description,
            "tags": tags or [],
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        saved_files = {}
        
        # Save in multiple formats
        formats = {
            "pickle": base_path / f"{model_name}.pkl",
            "json": base_path / f"{model_name}.json",
            "torch": base_path / f"{model_name}.pt"
        }
        
        for format_name, filepath in formats.items():
            try:
                self.serializer.save_quantized_model(
                    model, config, filepath, metadata
                )
                saved_files[format_name] = filepath
                logger.info(f"saved model in {format_name} format", filepath=str(filepath))
            except Exception as e:
                logger.warning(f"failed to save {format_name} format", error=str(e))
        
        # Save metadata separately
        metadata_path = base_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files["metadata"] = metadata_path
        
        return saved_files
    
    def load_model_from_directory(
        self,
        directory: Union[str, Path],
        model_name: str,
        preferred_format: str = "pickle"
    ) -> tuple[nn.Module, QuantizationConfig]:
        """
        Load model from a saved directory.
        
        Args:
            directory: Directory containing saved models
            model_name: Name of the model to load
            preferred_format: Preferred format to load
            
        Returns:
            Tuple of (loaded_model, quantization_config)
        """
        directory = Path(directory)
        
        # Try to load from preferred format
        format_extensions = {
            "pickle": ".pkl",
            "json": ".json",
            "torch": ".pt"
        }
        
        if preferred_format in format_extensions:
            filepath = directory / f"{model_name}{format_extensions[preferred_format]}"
            if filepath.exists():
                return self.serializer.load_quantized_model(filepath)
        
        # Try other formats if preferred format not available
        for format_name, extension in format_extensions.items():
            if format_name == preferred_format:
                continue
            
            filepath = directory / f"{model_name}{extension}"
            if filepath.exists():
                logger.info(f"loading from {format_name} format", filepath=str(filepath))
                return self.serializer.load_quantized_model(filepath)
        
        raise FileNotFoundError(f"No saved model found for {model_name} in {directory}")


# Convenience functions for direct usage
def save_quantized_model(
    model: nn.Module,
    config: QuantizationConfig,
    filepath: Union[str, Path],
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to save a quantized model."""
    serializer = QuantizedModelSerializer()
    serializer.save_quantized_model(model, config, filepath, additional_metadata)


def load_quantized_model(
    filepath: Union[str, Path],
    model_class: Optional[type] = None,
    device: Optional[str] = None
) -> tuple[nn.Module, QuantizationConfig]:
    """Convenience function to load a quantized model."""
    serializer = QuantizedModelSerializer()
    return serializer.load_quantized_model(filepath, model_class, device)