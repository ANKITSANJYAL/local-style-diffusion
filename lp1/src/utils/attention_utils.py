"""
Attention utilities for Local Prompt Adaptation (LPA).

This module provides functionality for handling cross-attention maps,
attention injection, and spatial attention localization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AttentionMap:
    """Data class for storing attention map information."""
    attention_weights: torch.Tensor  # [batch_size, num_heads, seq_len, seq_len]
    token_positions: Dict[str, List[int]]  # Mapping from token to positions
    spatial_resolution: Tuple[int, int]  # Height, width
    timestep: int
    layer_name: str
    
    def to_numpy(self) -> np.ndarray:
        """Convert attention weights to numpy array."""
        return self.attention_weights.detach().cpu().numpy()
    
    def get_token_attention(self, token: str) -> torch.Tensor:
        """Get attention weights for a specific token."""
        if token not in self.token_positions:
            return torch.zeros_like(self.attention_weights[0, 0])
        
        positions = self.token_positions[token]
        # Average attention across positions for this token
        token_attention = self.attention_weights[:, :, positions, :].mean(dim=2)
        return token_attention.mean(dim=0)  # Average across heads
    
    def get_spatial_attention(self, token: str) -> torch.Tensor:
        """Get spatial attention map for a specific token."""
        token_attention = self.get_token_attention(token)
        # Reshape to spatial dimensions (assuming square attention)
        seq_len = token_attention.shape[0]
        spatial_size = int(np.sqrt(seq_len))
        
        if spatial_size * spatial_size == seq_len:
            return token_attention.reshape(spatial_size, spatial_size)
        else:
            # Handle non-square attention maps
            return token_attention.reshape(self.spatial_resolution)


class AttentionUtils:
    """
    Utilities for handling attention maps and injection in LPA.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize attention utilities.
        
        Args:
            device: Device to use for computations
        """
        self.device = device
        self.attention_maps = {}  # Store attention maps during generation
        
    def register_attention_hooks(self, model):
        """
        Register hooks to capture attention maps during generation.
        
        Args:
            model: The diffusion model to hook
        """
        self.attention_maps.clear()
        
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn2'):  # Cross-attention
                    attention_weights = module.attn2.get_attention_weights()
                    if attention_weights is not None:
                        self.attention_maps[name] = attention_weights
            return hook
        
        # Register hooks for cross-attention layers
        for name, module in model.named_modules():
            if 'attn2' in name and hasattr(module, 'attn2'):
                module.register_forward_hook(get_attention_hook(name))
    
    def inject_attention(self, 
                        attention_weights: torch.Tensor,
                        token_positions: Dict[str, List[int]],
                        injection_schedule: Dict[str, float],
                        timestep: int) -> torch.Tensor:
        """
        Inject attention weights based on token positions and schedule.
        
        Args:
            attention_weights: Current attention weights
            token_positions: Mapping from token to positions
            injection_schedule: Schedule for injection strength per token
            timestep: Current denoising timestep
            
        Returns:
            Modified attention weights
        """
        modified_attention = attention_weights.clone()
        
        for token, strength in injection_schedule.items():
            if token in token_positions and strength > 0:
                positions = token_positions[token]
                
                # Apply injection to specific token positions
                for pos in positions:
                    if pos < attention_weights.shape[-1]:
                        # Increase attention to this token
                        modified_attention[:, :, :, pos] *= (1 + strength)
        
        # Renormalize attention weights
        modified_attention = F.softmax(modified_attention, dim=-1)
        
        return modified_attention
    
    def create_injection_schedule(self,
                                 object_tokens: List[str],
                                 style_tokens: List[str],
                                 timesteps: List[int],
                                 object_layers: List[str],
                                 style_layers: List[str],
                                 injection_strength: float = 1.0) -> Dict[str, Dict[str, float]]:
        """
        Create injection schedule for different tokens across timesteps and layers.
        
        Args:
            object_tokens: List of object tokens
            style_tokens: List of style tokens
            timesteps: List of timesteps for injection
            object_layers: List of layer names for object injection
            style_layers: List of layer names for style injection
            injection_strength: Base injection strength
            
        Returns:
            Dictionary mapping timestep -> layer -> token -> strength
        """
        schedule = {}
        
        for timestep in timesteps:
            schedule[timestep] = {}
            
            # Object tokens in early layers
            for layer in object_layers:
                schedule[timestep][layer] = {}
                for token in object_tokens:
                    # Higher strength in early timesteps
                    strength = injection_strength * (timestep / max(timesteps))
                    schedule[timestep][layer][token] = strength
            
            # Style tokens in later layers
            for layer in style_layers:
                if layer not in schedule[timestep]:
                    schedule[timestep][layer] = {}
                for token in style_tokens:
                    # Higher strength in later timesteps
                    strength = injection_strength * (1 - timestep / max(timesteps))
                    schedule[timestep][layer][token] = strength
        
        return schedule
    
    def get_token_positions(self, 
                           tokenizer,
                           prompt: str,
                           object_tokens: List[str],
                           style_tokens: List[str]) -> Dict[str, List[int]]:
        """
        Get token positions for objects and styles in the prompt.
        
        Args:
            tokenizer: Tokenizer for the model
            prompt: Input prompt
            object_tokens: List of object tokens
            style_tokens: List of style tokens
            
        Returns:
            Dictionary mapping token to list of positions
        """
        # Tokenize the prompt
        tokens = tokenizer.tokenize(prompt)
        token_positions = {}
        
        # Find positions for each token
        for token in object_tokens + style_tokens:
            positions = []
            token_lower = token.lower()
            
            for i, t in enumerate(tokens):
                if token_lower in t.lower():
                    positions.append(i)
            
            if positions:
                token_positions[token] = positions
        
        return token_positions
    
    def visualize_attention_map(self,
                               attention_map: torch.Tensor,
                               token: str,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Visualize attention map for a specific token.
        
        Args:
            attention_map: Attention map tensor
            token: Token name for the visualization
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        attention_np = attention_map.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(attention_np, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        # Set title and labels
        ax.set_title(f'Attention Map for Token: "{token}"')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Spatial Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_attention_overlay(self,
                                image: torch.Tensor,
                                attention_map: torch.Tensor,
                                alpha: float = 0.7) -> torch.Tensor:
        """
        Create attention overlay on an image.
        
        Args:
            image: Input image tensor [C, H, W]
            attention_map: Attention map tensor [H, W]
            alpha: Overlay transparency
            
        Returns:
            Image with attention overlay
        """
        # Resize attention map to match image size
        if attention_map.shape != image.shape[1:]:
            attention_map = F.interpolate(
                attention_map.unsqueeze(0).unsqueeze(0),
                size=image.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Create colored attention map
        attention_colored = torch.zeros_like(image)
        attention_colored[0] = attention_map  # Red channel
        attention_colored[1] = attention_map * 0.5  # Green channel
        
        # Blend with original image
        overlay = alpha * attention_colored + (1 - alpha) * image
        
        return overlay
    
    def compute_attention_statistics(self, attention_maps: Dict[str, AttentionMap]) -> Dict[str, Any]:
        """
        Compute statistics from attention maps.
        
        Args:
            attention_maps: Dictionary of attention maps
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_maps': len(attention_maps),
            'layers': list(attention_maps.keys()),
            'timesteps': [],
            'max_attention': 0,
            'min_attention': float('inf'),
            'avg_attention': 0
        }
        
        total_attention = 0
        count = 0
        
        for name, attn_map in attention_maps.items():
            stats['timesteps'].append(attn_map.timestep)
            
            attention_np = attn_map.attention_weights.detach().cpu().numpy()
            
            stats['max_attention'] = max(stats['max_attention'], attention_np.max())
            stats['min_attention'] = min(stats['min_attention'], attention_np.min())
            
            total_attention += attention_np.mean()
            count += 1
        
        if count > 0:
            stats['avg_attention'] = total_attention / count
        
        return stats
    
    def save_attention_maps(self, 
                           attention_maps: Dict[str, AttentionMap],
                           save_dir: str,
                           prefix: str = "attention"):
        """
        Save attention maps to files.
        
        Args:
            attention_maps: Dictionary of attention maps
            save_dir: Directory to save maps
            prefix: Prefix for filenames
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, attn_map in attention_maps.items():
            # Save attention weights
            weights_path = os.path.join(save_dir, f"{prefix}_{name}_weights.pt")
            torch.save(attn_map.attention_weights, weights_path)
            
            # Save metadata
            metadata = {
                'token_positions': attn_map.token_positions,
                'spatial_resolution': attn_map.spatial_resolution,
                'timestep': attn_map.timestep,
                'layer_name': attn_map.layer_name
            }
            
            metadata_path = os.path.join(save_dir, f"{prefix}_{name}_metadata.json")
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_attention_maps(self, load_dir: str, prefix: str = "attention") -> Dict[str, AttentionMap]:
        """
        Load attention maps from files.
        
        Args:
            load_dir: Directory containing saved maps
            prefix: Prefix for filenames
            
        Returns:
            Dictionary of loaded attention maps
        """
        import os
        import json
        import glob
        
        attention_maps = {}
        
        # Find all weight files
        weight_files = glob.glob(os.path.join(load_dir, f"{prefix}_*_weights.pt"))
        
        for weight_file in weight_files:
            # Extract name from filename
            name = weight_file.split(f"{prefix}_")[1].split("_weights")[0]
            
            # Load weights
            weights = torch.load(weight_file, map_location=self.device)
            
            # Load metadata
            metadata_file = weight_file.replace("_weights.pt", "_metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Create AttentionMap object
            attn_map = AttentionMap(
                attention_weights=weights,
                token_positions=metadata['token_positions'],
                spatial_resolution=tuple(metadata['spatial_resolution']),
                timestep=metadata['timestep'],
                layer_name=metadata['layer_name']
            )
            
            attention_maps[name] = attn_map
        
        return attention_maps 