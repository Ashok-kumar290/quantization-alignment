"""
Activation collection using PyTorch hooks.

This module provides utilities to collect internal activations from
transformer models during inference for mechanistic interpretability analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import copy

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class ActivationHook:
    """Wrapper for PyTorch forward hooks to capture activations."""
    
    def __init__(self, name: str):
        self.name = name
        self.activations = []
    
    def __call__(self, module: nn.Module, input: Tuple, output: Any):
        """Hook callback to capture output."""
        # Handle different output types
        if isinstance(output, torch.Tensor):
            # Clone to avoid memory issues
            self.activations.append(output.detach().clone())
        elif isinstance(output, tuple):
            # Usually (hidden_states, attention_mask) or similar
            # Take first element which is typically the hidden states
            if len(output) > 0 and isinstance(output[0], torch.Tensor):
                self.activations.append(output[0].detach().clone())
        elif isinstance(output, dict):
            # Some models return dicts
            if 'hidden_states' in output:
                self.activations.append(output['hidden_states'].detach().clone())
    
    def get_activations(self) -> torch.Tensor:
        """Get all collected activations stacked."""
        if not self.activations:
            return torch.tensor([])
        return torch.cat(self.activations, dim=0)
    
    def clear(self):
        """Clear stored activations."""
        self.activations = []


class ActivationCollector:
    """
    Collects internal activations from transformer models using hooks.
    
    Supports collection from:
    - Residual stream (after each layer)
    - Attention outputs
    - MLP outputs
    - Embeddings
    
    Usage:
        collector = ActivationCollector(model, tokenizer)
        activations = collector.collect_from_dataset(dataset)
    """
    
    # Layer names to hook for different component types
    LAYER_PATTERNS = {
        'residual': ['mlp.output', 'block_outputs', 'layer_outputs'],
        'attention': ['attn.output', 'attention_output'],
        'mlp': ['mlp.hidden_states', 'ffn_output'],
        'embed': ['embed_tokens', 'embedding']
    }
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        layer_types: Optional[List[str]] = None,
        pooling: str = 'last_token'
    ):
        """
        Initialize activation collector.
        
        Args:
            model: The transformer model
            tokenizer: The tokenizer
            layer_types: Which layer types to collect (default: ['residual', 'mlp'])
            pooling: How to pool activations ('last_token', 'mean', 'all')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        
        # Default to collecting residual stream and MLP outputs
        self.layer_types = layer_types or ['residual', 'mlp']
        
        # Storage for hooks
        self.hooks: Dict[str, ActivationHook] = {}
        self.hook_handles: List[Any] = []
        
        # Collected activations
        self.collected_activations: Dict[str, List] = defaultdict(list)
        
        # Model info
        self._model_info = self._extract_model_info()
    
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract model architecture information."""
        config = self.model.config
        
        return {
            'hidden_size': getattr(config, 'hidden_size', 4096),
            'num_layers': getattr(config, 'num_hidden_layers', 32),
            'num_attention_heads': getattr(config, 'num_attention_heads', 32)
        }
    
    def _get_layer_names(self) -> List[str]:
        """Get list of layer names to hook based on model architecture."""
        layer_names = []
        num_layers = self._model_info['num_layers']
        
        for layer_idx in range(num_layers):
            # Common patterns in different model architectures
            if hasattr(self.model, 'model'):
                # Llama-style: model.layers
                layer_names.append(f'model.layers.{layer_idx}')
            elif hasattr(self.model, 'transformer'):
                # GPT-NeoX style: transformer.h
                layer_names.append(f'transformer.h.{layer_idx}')
            else:
                # Generic
                layer_names.append(f'layers.{layer_idx}')
        
        return layer_names
    
    def _register_hooks(self):
        """Register forward hooks on model layers."""
        # Clear existing hooks
        self._clear_hooks()
        
        # Get layer names
        layer_names = self._get_layer_names()
        
        for layer_name in layer_names:
            try:
                # Try to get the layer module
                layer = self.model.get_submodule(layer_name)
                
                # Hook into the forward pass
                # We'll hook the output of each layer
                hook = ActivationHook(layer_name)
                handle = layer.register_forward_hook(hook)
                
                self.hooks[layer_name] = hook
                self.hook_handles.append(handle)
                
            except AttributeError:
                logger.warning(f"Could not find layer: {layer_name}")
                continue
        
        logger.info(f"Registered {len(self.hooks)} hooks")
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        
        self.hooks = {}
        self.hook_handles = []
    
    def _pool_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Pool activations based on configured strategy.
        
        Args:
            activations: Shape [batch, seq_len, hidden]
        
        Returns:
            Pooled activations: Shape [batch, hidden]
        """
        if self.pooling == 'last_token':
            # Take the last non-padding token
            return activations[:, -1, :]
        elif self.pooling == 'mean':
            return activations.mean(dim=1)
        elif self.pooling == 'all':
            return activations.view(-1, activations.shape[-1])
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def collect_single(
        self, 
        prompt: str, 
        max_new_tokens: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activations for a single prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate (0 for just forward pass)
        
        Returns:
            Dictionary of layer_name -> activations
        """
        # Register hooks
        self._register_hooks()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run model
        with torch.no_grad():
            if max_new_tokens > 0:
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                outputs = self.model(**inputs)
        
        # Extract activations from hooks
        result = {}
        for layer_name, hook in self.hooks.items():
            acts = hook.get_activations()
            if acts.numel() > 0:
                # Pool to get [batch, hidden]
                pooled = self._pool_activations(acts)
                result[layer_name] = pooled
        
        # Clear hooks
        self._clear_hooks()
        
        return result
    
    def collect_from_dataset(
        self,
        dataset: Any,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Collect activations from a dataset.
        
        Args:
            dataset: Dataset with 'prompt' field
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with:
                - activations: List of layer activations
                - labels: Labels if available
                - hidden_size: Model hidden size
                - n_layers: Number of layers
        """
        logger.info(f"Collecting activations from {len(dataset)} samples...")
        
        # Register hooks
        self._register_hooks()
        
        all_activations = []
        all_labels = []
        
        for idx, item in enumerate(dataset):
            prompt = item['prompt']
            label = item.get('label', None)
            
            # Collect activations
            layer_activations = self.collect_single(prompt)
            
            if layer_activations:
                # Stack across layers - take last layer for simplicity
                last_layer = list(layer_activations.keys())[-1]
                activations = layer_activations[last_layer]
                
                all_activations.append(activations.squeeze(0).cpu())
                
                if label is not None:
                    all_labels.append(label)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples")
        
        # Clear hooks
        self._clear_hooks()
        
        # Stack activations
        if all_activations:
            stacked_activations = torch.stack(all_activations)
        else:
            stacked_activations = torch.tensor([])
        
        result = {
            'activations': stacked_activations,
            'labels': all_labels if all_labels else None,
            'hidden_size': self._model_info['hidden_size'],
            'n_layers': self._model_info['num_layers']
        }
        
        logger.info(f"Collected activations shape: {stacked_activations.shape}")
        
        return result
    
    def collect_all_layers(
        self,
        dataset: Any,
    ) -> Dict[str, Any]:
        """
        Collect per-layer activations from a dataset.

        Unlike collect_from_dataset (which only keeps the last layer),
        this stores activations from every hooked layer, enabling
        layer-wise probing analysis.

        Args:
            dataset: Iterable of dicts with 'prompt' and optionally 'label'.

        Returns:
            Dictionary with:
                - per_layer: Dict of layer_name -> tensor [n_samples, hidden_dim]
                - labels: List of labels
                - hidden_size: int
                - n_layers: int
        """
        logger.info(f"Collecting all-layer activations from {len(dataset)} samples...")

        per_layer_collections: Dict[str, List[torch.Tensor]] = defaultdict(list)
        all_labels = []

        for idx, item in enumerate(dataset):
            prompt = item['prompt']
            label = item.get('label', None)

            # collect_single handles hook registration/cleanup
            layer_activations = self.collect_single(prompt)

            for layer_name, acts in layer_activations.items():
                # acts shape: [1, hidden] after pooling
                per_layer_collections[layer_name].append(acts.squeeze(0).cpu())

            if label is not None:
                all_labels.append(label)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples")

        # Stack each layer's activations into [n_samples, hidden_dim]
        per_layer = {}
        for layer_name, acts_list in per_layer_collections.items():
            if acts_list:
                per_layer[layer_name] = torch.stack(acts_list)

        logger.info(f"Collected activations from {len(per_layer)} layers, "
                     f"{len(all_labels)} samples")

        return {
            'per_layer': per_layer,
            'labels': all_labels if all_labels else None,
            'hidden_size': self._model_info['hidden_size'],
            'n_layers': self._model_info['num_layers'],
        }

    def collect_for_probing(
        self,
        prompts: List[str],
        labels: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect activations specifically for probe training.

        Args:
            prompts: List of prompts
            labels: List of binary labels

        Returns:
            Tuple of (activations, labels)
        """
        dataset = [{'prompt': p, 'label': l} for p, l in zip(prompts, labels)]

        result = self.collect_from_dataset(dataset)

        return result['activations'], torch.tensor(labels)


class ResidualStreamCollector:
    """
    Specialized collector for residual stream activations.
    
    The residual stream is the main pathway through transformer layers,
    computed as: h_{l+1} = h_l + F(h_l)
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.residual_activations = []
        self.hook_handles = []
    
    def _register_hooks(self):
        """Register hooks on each layer's output."""
        num_layers = self.model.config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            try:
                # Try different naming conventions
                if hasattr(self.model, 'model'):
                    layer = self.model.model.layers[layer_idx]
                elif hasattr(self.model, 'transformer'):
                    layer = self.model.transformer.h[layer_idx]
                else:
                    layer = self.model.layers[layer_idx]
                
                # Hook the layer output (after add & norm)
                hook_fn = self._make_hook(layer_idx)
                handle = layer.register_forward_hook(hook_fn)
                self.hook_handles.append(handle)
                
            except Exception as e:
                logger.warning(f"Could not register hook for layer {layer_idx}: {e}")
    
    def _make_hook(self, layer_idx: int) -> Callable:
        """Create hook function for specific layer."""
        def hook(module, input, output):
            # Output is typically a tuple (hidden_states,)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store the residual (last token)
            self.residual_activations.append({
                'layer': layer_idx,
                'activations': hidden_states[:, -1, :].detach().clone()
            })
        
        return hook
    
    def collect(self, prompt: str) -> Dict[int, torch.Tensor]:
        """Collect residual stream for a prompt."""
        self.residual_activations = []
        self._register_hooks()
        
        # Tokenize and run
        inputs = self.tokenizer(prompt, return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.model(**inputs)
        
        # Clean up hooks
        for handle in self.hook_handles:
            handle.remove()
        
        # Organize by layer
        result = {}
        for item in self.residual_activations:
            result[item['layer']] = item['activations']
        
        return result


# Example usage
if __name__ == '__main__':
    print("Activation collector module ready")
