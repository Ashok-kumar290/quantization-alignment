"""
Model loading and quantization utilities.

This module provides:
- Model loading with various precision levels (FP16, 8-bit, 4-bit, 3-bit)
- Integration with bitsandbytes for quantization
- Hook-based activation extraction
"""

from .model_loader import ModelLoader, QuantizationConfig
from .activation_collector import ActivationCollector

__all__ = ['ModelLoader', 'QuantizationConfig', 'ActivationCollector']
