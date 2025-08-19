"""
Baseline models for comparison with Local Prompt Adaptation (LPA).

This module implements baseline methods for generating images with
multi-object style consistency for comparison with our LPA method.
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import numpy as np


@dataclass
class BaselineConfig:
    """Configuration for baseline generation."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    num_images_per_prompt: int = 4


class BaselineModels:
    """
    Collection of baseline models for comparison.
    
    Includes:
    - Vanilla SDXL: Standard Stable Diffusion XL
    - High CFG SDXL: SDXL with high guidance scale
    """
    
    def __init__(self, 
                 model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "mps" if torch.backends.mps.is_available() else "cpu",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize baseline models.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load models on
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Load pipeline (handle different model types)
        if "xl" in model_name.lower() or "sdxl" in model_name.lower():
            # Use SDXL pipeline for XL models
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                use_safetensors=True
            )
        else:
            # Use regular SD pipeline for other models
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                use_safetensors=True
            )
        self.pipeline = self.pipeline.to(device)
        
        # Enable optimizations
        self.pipeline.enable_attention_slicing()
        
        # Try different methods for memory efficient attention
        try:
            self.pipeline.enable_memory_efficient_attention()
        except AttributeError:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except (AttributeError, ModuleNotFoundError):
                print("Warning: Memory efficient attention not available (xformers not installed), using standard attention")
    
    def generate_vanilla_sdxl(self,
                             prompt: str,
                             config: Optional[BaselineConfig] = None,
                             negative_prompt: str = "",
                             seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate images using vanilla SDXL.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            negative_prompt: Negative prompt
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing generated images and metadata
        """
        if config is None:
            config = BaselineConfig()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate images
        results = []
        
        for i in range(config.num_images_per_prompt):
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
            
            results.append({
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "method": "vanilla_sdxl"
            })
        
        return {
            "images": [r["image"] for r in results],
            "best_image": results[0]["image"],  # No re-ranking for baseline
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "config": config,
                "seed": seed,
                "method": "vanilla_sdxl"
            }
        }
    
    def generate_high_cfg_sdxl(self,
                               prompt: str,
                               config: Optional[BaselineConfig] = None,
                               negative_prompt: str = "",
                               seed: Optional[int] = None,
                               high_guidance_scale: float = 15.0) -> Dict[str, Any]:
        """
        Generate images using SDXL with high guidance scale.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            negative_prompt: Negative prompt
            seed: Random seed for reproducibility
            high_guidance_scale: High guidance scale value
            
        Returns:
            Dictionary containing generated images and metadata
        """
        if config is None:
            config = BaselineConfig()
        
        # Override guidance scale
        config.guidance_scale = high_guidance_scale
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate images
        results = []
        
        for i in range(config.num_images_per_prompt):
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
            
            results.append({
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "method": "high_cfg_sdxl"
            })
        
        return {
            "images": [r["image"] for r in results],
            "best_image": results[0]["image"],  # No re-ranking for baseline
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "config": config,
                "seed": seed,
                "method": "high_cfg_sdxl",
                "high_guidance_scale": high_guidance_scale
            }
        }
    
    def generate_all_baselines(self,
                              prompt: str,
                              config: Optional[BaselineConfig] = None,
                              negative_prompt: str = "",
                              seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate images using all baseline methods.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            negative_prompt: Negative prompt
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing results from all baselines
        """
        if config is None:
            config = BaselineConfig()
        
        results = {}
        
        # Vanilla SDXL
        results["vanilla_sdxl"] = self.generate_vanilla_sdxl(
            prompt=prompt,
            config=config,
            negative_prompt=negative_prompt,
            seed=seed
        )
        
        # High CFG SDXL
        results["high_cfg_sdxl"] = self.generate_high_cfg_sdxl(
            prompt=prompt,
            config=config,
            negative_prompt=negative_prompt,
            seed=seed
        )
        
        return results
    
    def save_baseline_results(self,
                             results: Dict[str, Any],
                             save_dir: str,
                             prefix: str = "baseline"):
        """
        Save baseline generation results to files.
        
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
        
        # Save metadata
        metadata_path = os.path.join(save_dir, f"{prefix}_metadata.json")
        import json
        
        # Convert config to dict for JSON serialization
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
            "pipeline_type": type(self.pipeline).__name__
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
        
        torch.cuda.empty_cache() 