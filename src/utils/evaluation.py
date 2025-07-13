"""
Evaluation utilities for Local Prompt Adaptation (LPA).

This module provides functionality for computing various evaluation metrics
including style consistency, LPIPS, CLIP scores, and other quality measures.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image
import warnings

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS not available. Install with: pip install lpips")

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("OpenCLIP not available. Install with: pip install open-clip-torch")


class Evaluator:
    """
    Evaluator for computing various metrics on generated images.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            device: Device to use for computations
        """
        self.device = device
        self.lpips_model = None
        self.clip_model = None
        self.clip_preprocess = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize evaluation models."""
        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        
        # Initialize OpenCLIP
        if CLIP_AVAILABLE:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.device
            )
    
    def compute_style_consistency(self,
                                 image: Union[torch.Tensor, Image.Image],
                                 style_tokens: List[str],
                                 method: str = "clip") -> float:
        """
        Compute style consistency score for an image.
        
        Args:
            image: Generated image
            style_tokens: List of style tokens
            method: Method to use ("clip" or "dino")
            
        Returns:
            Style consistency score (higher is better)
        """
        if method == "clip" and CLIP_AVAILABLE:
            return self._compute_clip_style_consistency(image, style_tokens)
        else:
            # Placeholder implementation
            return 0.5
    
    def _compute_clip_style_consistency(self,
                                       image: Union[torch.Tensor, Image.Image],
                                       style_tokens: List[str]) -> float:
        """Compute style consistency using CLIP embeddings."""
        if not CLIP_AVAILABLE:
            return 0.5
        
        # Preprocess image
        if isinstance(image, Image.Image):
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
        
        # Get text embeddings for style tokens
        text_features = []
        for token in style_tokens:
            text = open_clip.tokenize([f"in {token} style"]).to(self.device)
            with torch.no_grad():
                token_features = self.clip_model.encode_text(text)
                token_features = F.normalize(token_features, dim=-1)
                text_features.append(token_features)
        
        # Compute average similarity
        similarities = []
        for text_feat in text_features:
            similarity = torch.cosine_similarity(image_features, text_feat, dim=-1)
            similarities.append(similarity.item())
        
        return np.mean(similarities)
    
    def compute_lpips(self,
                      image1: Union[torch.Tensor, Image.Image],
                      image2: Union[torch.Tensor, Image.Image]) -> float:
        """
        Compute LPIPS distance between two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            LPIPS distance (lower is better)
        """
        if not LPIPS_AVAILABLE:
            return 0.0
        
        # Convert to tensors
        if isinstance(image1, Image.Image):
            image1_tensor = self._pil_to_tensor(image1)
        else:
            image1_tensor = image1
        
        if isinstance(image2, Image.Image):
            image2_tensor = self._pil_to_tensor(image2)
        else:
            image2_tensor = image2
        
        # Ensure tensors are on correct device
        image1_tensor = image1_tensor.to(self.device)
        image2_tensor = image2_tensor.to(self.device)
        
        # Compute LPIPS
        with torch.no_grad():
            distance = self.lpips_model(image1_tensor, image2_tensor)
        
        return distance.item()
    
    def compute_clip_score(self,
                          image: Union[torch.Tensor, Image.Image],
                          text: str) -> float:
        """
        Compute CLIP score between image and text.
        
        Args:
            image: Input image
            text: Input text
            
        Returns:
            CLIP score (higher is better)
        """
        if not CLIP_AVAILABLE:
            return 0.5
        
        # Preprocess image
        if isinstance(image, Image.Image):
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        # Tokenize text
        text_tensor = open_clip.tokenize([text]).to(self.device)
        
        # Compute embeddings
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_tensor)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        
        return similarity.item()
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor for LPIPS."""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transform(image).unsqueeze(0)
    
    def evaluate_generation_results(self,
                                   results: Dict[str, Any],
                                   reference_images: Optional[List[Image.Image]] = None) -> Dict[str, float]:
        """
        Evaluate generation results with multiple metrics.
        
        Args:
            results: Generation results dictionary
            reference_images: Optional reference images for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Extract images and metadata
        images = results.get("images", [])
        prompt = results.get("metadata", {}).get("prompt", "")
        parsed_prompt = results.get("parsed_prompt", {})
        
        # Handle both ParsedPrompt objects and dictionaries
        if hasattr(parsed_prompt, 'style_tokens'):
            style_tokens = parsed_prompt.style_tokens
        elif isinstance(parsed_prompt, dict):
            style_tokens = parsed_prompt.get("style_tokens", [])
        else:
            style_tokens = []
        
        if not images:
            return metrics
        
        # Compute metrics for best image
        best_image = results.get("best_image", images[0])
        
        # Style consistency
        if style_tokens:
            metrics["style_consistency"] = self.compute_style_consistency(
                best_image, style_tokens
            )
        
        # CLIP score
        if prompt:
            metrics["clip_score"] = self.compute_clip_score(best_image, prompt)
        
        # LPIPS between generated images (diversity measure)
        if len(images) > 1:
            lpips_scores = []
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    lpips_score = self.compute_lpips(images[i], images[j])
                    lpips_scores.append(lpips_score)
            
            metrics["lpips_diversity"] = np.mean(lpips_scores)
        
        # LPIPS to reference images if provided
        if reference_images and len(reference_images) > 0:
            lpips_ref_scores = []
            for ref_img in reference_images:
                for gen_img in images:
                    lpips_score = self.compute_lpips(gen_img, ref_img)
                    lpips_ref_scores.append(lpips_score)
            
            metrics["lpips_to_reference"] = np.mean(lpips_ref_scores)
        
        return metrics
    
    def compare_methods(self,
                       lpa_results: Dict[str, Any],
                       baseline_results: Dict[str, Any],
                       reference_images: Optional[List[Image.Image]] = None) -> Dict[str, Any]:
        """
        Compare LPA results with baseline results.
        
        Args:
            lpa_results: Results from LPA method
            baseline_results: Results from baseline methods
            reference_images: Optional reference images
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {}
        
        # Evaluate LPA results
        if "lpa" in lpa_results and "error" not in lpa_results["lpa"]:
            lpa_metrics = self.evaluate_generation_results(
                lpa_results["lpa"], reference_images
            )
            comparison["lpa"] = lpa_metrics
        
        # Evaluate baseline results
        if "baselines" in baseline_results and "error" not in baseline_results["baselines"]:
            baseline_metrics = {}
            for method, method_results in baseline_results["baselines"].items():
                method_metrics = self.evaluate_generation_results(
                    method_results, reference_images
                )
                baseline_metrics[method] = method_metrics
            
            comparison["baselines"] = baseline_metrics
        
        # Compute relative improvements
        if "lpa" in comparison and "baselines" in comparison:
            improvements = {}
            for baseline_method, baseline_metric in comparison["baselines"].items():
                method_improvements = {}
                for metric_name, baseline_value in baseline_metric.items():
                    if metric_name in comparison["lpa"]:
                        lpa_value = comparison["lpa"][metric_name]
                        if baseline_value != 0:
                            improvement = (lpa_value - baseline_value) / baseline_value
                            method_improvements[metric_name] = improvement
                
                improvements[baseline_method] = method_improvements
            
            comparison["improvements"] = improvements
        
        return comparison
    
    def save_evaluation_results(self,
                               evaluation_results: Dict[str, Any],
                               save_path: str):
        """
        Save evaluation results to file.
        
        Args:
            evaluation_results: Evaluation results dictionary
            save_path: Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(evaluation_results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def generate_evaluation_report(self,
                                  evaluation_results: Dict[str, Any],
                                  save_path: str):
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation_results: Evaluation results dictionary
            save_path: Path to save report
        """
        with open(save_path, 'w') as f:
            f.write("LPA Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # LPA Results
            if "lpa" in evaluation_results:
                f.write("LPA Method Results:\n")
                f.write("-" * 20 + "\n")
                for metric, value in evaluation_results["lpa"].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            # Baseline Results
            if "baselines" in evaluation_results:
                f.write("Baseline Results:\n")
                f.write("-" * 20 + "\n")
                for method, metrics in evaluation_results["baselines"].items():
                    f.write(f"\n{method}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
            
            # Improvements
            if "improvements" in evaluation_results:
                f.write("Relative Improvements (LPA vs Baselines):\n")
                f.write("-" * 40 + "\n")
                for method, improvements in evaluation_results["improvements"].items():
                    f.write(f"\n{method}:\n")
                    for metric, improvement in improvements.items():
                        f.write(f"  {metric}: {improvement:+.2%}\n") 