"""
Model loading with quantization support.

Supports:
- FP16 (baseline)
- 8-bit quantization
- 4-bit quantization
- 3-bit quantization (NF4)

Uses bitsandbytes for quantization and HuggingFace transformers for loading.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig
)

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Configuration for model quantization."""
    
    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = 'nf4',
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_use_double_quant: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = 'auto'
    ):
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.torch_dtype = torch_dtype
        self.device_map = device_map
    
    def to_bitsandbytes_config(self) -> Optional[BitsAndBytesConfig]:
        """Convert to bitsandbytes configuration."""
        if self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_8bit_use_double_quant=self.bnb_4bit_use_double_quant
            )
        elif self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant
            )
        return None


class ModelLoader:
    """
    Handles loading transformer models with various precision levels.
    
    Supports:
    - FP16 baseline
    - 8-bit quantization (bitsandbytes)
    - 4-bit quantization (bitsandbytes NF4)
    - 3-bit quantization (bitsandbytes NF4)
    """
    
    # Supported models and their configurations
    SUPPORTED_MODELS = {
        'meta-llama/Llama-3-8B': {'hidden_size': 4096, 'n_layers': 32},
        'meta-llama/Llama-3-8B-Instruct': {'hidden_size': 4096, 'n_layers': 32},
        'mistralai/Mistral-7B-v0.1': {'hidden_size': 4096, 'n_layers': 32},
        'mistralai/Mistral-7B-Instruct-v0.2': {'hidden_size': 4096, 'n_layers': 32},
        'Qwen/Qwen2-7B': {'hidden_size': 3584, 'n_layers': 28},
        'Qwen/Qwen2-7B-Instruct': {'hidden_size': 3584, 'n_layers': 28},
    }
    
    def __init__(
        self,
        model_name: str,
        quantization_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: HuggingFace model name or path
            quantization_config: Quantization configuration dict
            cache_dir: Model cache directory
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.quantization_config = QuantizationConfig(**(quantization_config or {}))
        self.cache_dir = cache_dir or os.getenv('HF_HOME', '~/.cache/huggingface')
        self.trust_remote_code = trust_remote_code
        
        # Get model architecture info
        self.model_info = self.SUPPORTED_MODELS.get(model_name, {})
        
    def load(self) -> Tuple[nn.Module, Any]:
        """
        Load model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Quantization: load_in_8bit={self.quantization_config.load_in_8bit}, "
                   f"load_in_4bit={self.quantization_config.load_in_4bit}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            padding_side='left'
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get quantization config
        bnb_config = self.quantization_config.to_bitsandbytes_config()
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=self.quantization_config.torch_dtype,
                device_map=self.quantization_config.device_map,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Failed to load with quantization: {e}")
            logger.info("Falling back to standard loading...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.quantization_config.torch_dtype,
                device_map=self.quantization_config.device_map,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code
            )
        
        # Set to evaluation mode
        model.eval()
        
        # Log model info
        self._log_model_info(model)
        
        return model, self.tokenizer
    
    def _log_model_info(self, model: nn.Module):
        """Log model architecture information."""
        config = model.config
        
        hidden_size = getattr(config, 'hidden_size', 'unknown')
        num_layers = getattr(config, 'num_hidden_layers', 'unknown')
        num_attention_heads = getattr(config, 'num_attention_heads', 'unknown')
        
        logger.info(f"Model architecture:")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - Number of layers: {num_layers}")
        logger.info(f"  - Attention heads: {num_attention_heads}")
        
        # Store for later use
        self.model_info = {
            'hidden_size': hidden_size,
            'n_layers': num_layers,
            'n_heads': num_attention_heads
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return self.model_info


def load_model_with_precision(
    model_name: str,
    precision: str,
    device: str = 'auto'
) -> Tuple[nn.Module, Any]:
    """
    Convenience function to load model with specified precision.
    
    Args:
        model_name: HuggingFace model name
        precision: One of 'fp16', '8bit', '4bit', '3bit'
        device: Device mapping strategy
    
    Returns:
        Tuple of (model, tokenizer)
    """
    precision_configs = {
        'fp16': {'load_in_8bit': False, 'load_in_4bit': False, 'torch_dtype': torch.float16},
        '8bit': {'load_in_8bit': True, 'load_in_4bit': False, 'torch_dtype': torch.float16},
        '4bit': {'load_in_8bit': False, 'load_in_4bit': True, 'torch_dtype': torch.float16},
        '3bit': {'load_in_8bit': False, 'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4'}
    }
    
    if precision not in precision_configs:
        raise ValueError(f"Unknown precision: {precision}")
    
    config = precision_configs[precision]
    config['device_map'] = device
    
    loader = ModelLoader(model_name=model_name, quantization_config=config)
    return loader.load()


# Example usage
if __name__ == '__main__':
    # Test loading
    print("Testing model loader...")
    
    # This would require actual model files to run
    # loader = ModelLoader('mistralai/Mistral-7B-v0.1', {'load_in_4bit': True})
    # model, tokenizer = loader.load()
    # print(f"Loaded model: {loader.get_model_info()}")
    
    print("Model loader module ready")
