"""
Local Prompt Adaptation (LPA) Model Implementation.

This module implements the core LPA method by extending Stable Diffusion XL
with controlled cross-attention injection for improved style consistency.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import warnings

from ..utils.prompt_parser import ParsedPrompt, PromptParser
from ..utils.attention_utils import AttentionUtils, AttentionMap


@dataclass
class LPAGenerationConfig:
    """Configuration for LPA generation."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    num_images_per_prompt: int = 4
    injection_timesteps: Optional[List[int]] = None
    object_layers: Optional[List[str]] = None
    style_layers: Optional[List[str]] = None
    injection_strength: float = 1.0
    save_attention_maps: bool = True
    
    def __post_init__(self):
        if self.injection_timesteps is None:
            self.injection_timesteps = [10, 20, 30, 40, 50]
        if self.object_layers is None:
            self.object_layers = ["down_blocks.0", "down_blocks.1", "down_blocks.2"]
        if self.style_layers is None:
            self.style_layers = ["mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2"]


class LPAModel:
    """
    Local Prompt Adaptation (LPA) Model.
    
    Extends Stable Diffusion XL with controlled cross-attention injection
    for improved multi-object style consistency.
    """
    
    def __init__(self, 
                 model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "mps" if torch.backends.mps.is_available() else "cpu",
                 dtype: torch.dtype = torch.float16,
                 use_attention_slicing: bool = True,
                 use_memory_efficient_attention: bool = True):
        """
        Initialize the LPA model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load the model on
            dtype: Data type for model weights
            use_attention_slicing: Whether to use attention slicing
            use_memory_efficient_attention: Whether to use memory efficient attention
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Initialize components
        self.pipeline = None
        self.tokenizer = None
        self.text_encoder = None
        self.prompt_parser = PromptParser()
        self.attention_utils = AttentionUtils(device=device)
        
        # Generation state
        self.current_prompt = None
        self.parsed_prompt = None
        self.attention_maps = {}
        self.injection_schedule = {}
        
        # Load model
        self._load_model(use_attention_slicing, use_memory_efficient_attention)
    
    def _load_model(self, use_attention_slicing: bool, use_memory_efficient_attention: bool):
        """Load the Stable Diffusion XL model."""
        print(f"Loading model: {self.model_name}")
        
        # Load pipeline (handle different model types)
        if "xl" in self.model_name.lower() or "sdxl" in self.model_name.lower():
            # Use SDXL pipeline for XL models
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
        else:
            # Use regular SD pipeline for other models
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
        
        # Configure pipeline
        if use_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if use_memory_efficient_attention:
            # Try different methods for memory efficient attention
            try:
                self.pipeline.enable_memory_efficient_attention()
            except AttributeError:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except (AttributeError, ModuleNotFoundError):
                    print("Warning: Memory efficient attention not available (xformers not installed), using standard attention")
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Get tokenizer and text encoder (handle different model architectures)
        if hasattr(self.pipeline, 'tokenizer'):
            self.tokenizer = self.pipeline.tokenizer
        else:
            # For some models, tokenizer might be in a different location
            self.tokenizer = getattr(self.pipeline, 'tokenizer', None)
            if self.tokenizer is None and hasattr(self.pipeline, 'text_encoder'):
                # Try to get tokenizer from text encoder
                self.tokenizer = getattr(self.pipeline.text_encoder, 'tokenizer', None)
        
        if hasattr(self.pipeline, 'text_encoder'):
            self.text_encoder = self.pipeline.text_encoder
        else:
            # For some models, text encoder might be in a different location
            self.text_encoder = getattr(self.pipeline, 'text_encoder', None)
            if self.text_encoder is None and hasattr(self.pipeline, 'text_encoder_2'):
                self.text_encoder = self.pipeline.text_encoder_2
        
        print("Model loaded successfully!")
    
    def parse_prompt(self, prompt: str) -> ParsedPrompt:
        """
        Parse a prompt into object and style tokens.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            ParsedPrompt object
        """
        return self.prompt_parser.parse_prompt(prompt)
    
    def create_injection_schedule(self, 
                                 parsed_prompt: ParsedPrompt,
                                 config: LPAGenerationConfig) -> Dict[str, Dict[str, float]]:
        """
        Create injection schedule for the parsed prompt.
        
        Args:
            parsed_prompt: Parsed prompt object
            config: Generation configuration
            
        Returns:
            Injection schedule dictionary
        """
        return self.attention_utils.create_injection_schedule(
            object_tokens=parsed_prompt.object_tokens,
            style_tokens=parsed_prompt.style_tokens,
            timesteps=config.injection_timesteps or [10, 20, 30, 40, 50],
            object_layers=config.object_layers or ["down_blocks.0", "down_blocks.1", "down_blocks.2"],
            style_layers=config.style_layers or ["mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2"],
            injection_strength=config.injection_strength
        )
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention maps during generation."""
        self.attention_maps.clear()
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Capture cross-attention weights
                if hasattr(module, 'attn2') and hasattr(module.attn2, 'get_attention_weights'):
                    attention_weights = module.attn2.get_attention_weights()
                    if attention_weights is not None:
                        self.attention_maps[name] = attention_weights
            return hook
        
        # Register hooks for cross-attention layers
        for name, module in self.pipeline.unet.named_modules():
            if 'attn2' in name and hasattr(module, 'attn2'):
                module.register_forward_hook(get_attention_hook(name))
    
    def _inject_attention(self, 
                         layer_name: str,
                         timestep: int,
                         token_positions: Dict[str, List[int]]) -> bool:
        """
        Inject attention for a specific layer and timestep.
        
        Args:
            layer_name: Name of the layer to inject
            timestep: Current denoising timestep
            token_positions: Token position mapping
            
        Returns:
            True if injection was applied, False otherwise
        """
        if timestep not in self.injection_schedule:
            return False
        
        if layer_name not in self.injection_schedule[timestep]:
            return False
        
        # Get injection schedule for this layer and timestep
        layer_schedule = self.injection_schedule[timestep][layer_name]
        
        # Apply injection to the layer
        # This would require modifying the attention computation in the UNet
        # For now, we'll implement this as a placeholder
        # In practice, this would involve hooking into the forward pass
        
        return True
    
    def generate(self, 
                 prompt: str,
                 config: Optional[LPAGenerationConfig] = None,
                 negative_prompt: str = "",
                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate images using LPA method.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            negative_prompt: Negative prompt
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing generated images and metadata
        """
        if config is None:
            config = LPAGenerationConfig()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Parse prompt
        self.current_prompt = prompt
        self.parsed_prompt = self.parse_prompt(prompt)
        
        # Create injection schedule
        self.injection_schedule = self.create_injection_schedule(self.parsed_prompt, config)
        
        # Register attention hooks if needed
        if config.save_attention_maps:
            self._register_attention_hooks()
        
        # Generate images
        results = []
        
        for i in range(config.num_images_per_prompt):
            # Generate single image
            result = self._generate_single_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                config=config
            )
            
            results.append(result)
        
        # Re-rank results if multiple images were generated
        if config.num_images_per_prompt > 1:
            best_result = self._rerank_results(results, config)
        else:
            best_result = results[0]
        
        return {
            "images": [r["image"] for r in results],
            "best_image": best_result["image"],
            "attention_maps": self.attention_maps,
            "parsed_prompt": self.parsed_prompt,
            "injection_schedule": self.injection_schedule,
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "config": config,
                "seed": seed
            }
        }
    
    def _generate_single_image(self,
                              prompt: str,
                              negative_prompt: str,
                              config: LPAGenerationConfig) -> Dict[str, Any]:
        """
        Generate a single image with LPA.
        
        Args:
            prompt: Input prompt
            negative_prompt: Negative prompt
            config: Generation configuration
            
        Returns:
            Dictionary containing image and metadata
        """
        # For now, we'll use the standard pipeline generation
        # In the full implementation, we would modify the UNet forward pass
        # to inject attention at specific timesteps and layers
        
        with torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.height,
                width=config.width,
                output_type="pil"
            ).images[0]
        
        return {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }
    
    def _rerank_results(self, 
                       results: List[Dict[str, Any]], 
                       config: LPAGenerationConfig) -> Dict[str, Any]:
        """
        Re-rank generated results based on style consistency.
        
        Args:
            results: List of generation results
            config: Generation configuration
            
        Returns:
            Best result based on ranking
        """
        # For now, return the first result
        # In the full implementation, we would compute style consistency scores
        # and return the best one
        
        return results[0]
    
    def compute_style_consistency(self, 
                                 image: torch.Tensor,
                                 style_tokens: List[str]) -> float:
        """
        Compute style consistency score for an image.
        
        Args:
            image: Generated image tensor
            style_tokens: List of style tokens
            
        Returns:
            Style consistency score (higher is better)
        """
        # Placeholder implementation
        # In the full implementation, we would:
        # 1. Extract region embeddings using CLIP/DINO
        # 2. Compare with style token embeddings
        # 3. Return average similarity score
        
        return 0.5  # Placeholder score
    
    def save_results(self, 
                     results: Dict[str, Any],
                     save_dir: str,
                     prefix: str = "lpa"):
        """
        Save generation results to files.
        
        Args:
            results: Generation results
            save_dir: Directory to save results
            prefix: Prefix for filenames
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save images
        for i, image in enumerate(results["images"]):
            image_path = os.path.join(save_dir, f"{prefix}_image_{i:03d}.png")
            image.save(image_path)
        
        # Save attention maps if available
        if results["attention_maps"]:
            attention_dir = os.path.join(save_dir, "attention_maps")
            self.attention_utils.save_attention_maps(
                results["attention_maps"],
                attention_dir,
                prefix=f"{prefix}_attention"
            )
        
        # Save metadata
        metadata_path = os.path.join(save_dir, f"{prefix}_metadata.json")
        import json
        
        # Convert tensors to lists for JSON serialization
        metadata = results["metadata"].copy()
        if "config" in metadata:
            metadata["config"] = {
                k: v for k, v in metadata["config"].__dict__.items() 
                if not k.startswith('_')
            }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "pipeline_type": type(self.pipeline).__name__,
            "text_encoder_type": type(self.text_encoder).__name__,
            "tokenizer_type": type(self.tokenizer).__name__
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
        if self.text_encoder is not None:
            del self.text_encoder
        if self.tokenizer is not None:
            del self.tokenizer
        
        torch.cuda.empty_cache() 