"""
Main experiment runner for Local Prompt Adaptation (LPA).

This module orchestrates experiments comparing LPA with baseline methods
and handles result storage, evaluation, and visualization.
"""

import os
import json
import yaml
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.lpa_model import LPAModel, LPAGenerationConfig
from src.models.baselines import BaselineModels, BaselineConfig
from src.utils.prompt_parser import PromptParser
from src.utils.evaluation import Evaluator
from src.utils.visualization import Visualizer


class ExperimentRunner:
    """
    Main experiment runner for LPA evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to experiment configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        device_config = self.config.get("model", {}).get("device", "auto")
        if device_config == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device_config
        self.prompt_parser = PromptParser(
            method=self.config.get("lpa", {}).get("parser", {}).get("method", "rule_based")
        )
        
        # Initialize evaluator for metrics computation
        self.evaluator = Evaluator(device=self.device)
        
        # Models will be loaded on demand
        self.lpa_model = None
        self.baseline_models = None
        
        # Results storage
        self.results = {}
        self.experiment_dir = None
        
        # Set up experiment directory
        self._setup_experiment_dir()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_experiment_dir(self):
        """Set up experiment output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config.get("experiment", {}).get("name", "lpa_experiment")
        
        self.experiment_dir = Path(f"experiments/{experiment_name}_{timestamp}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "images").mkdir(exist_ok=True)
        (self.experiment_dir / "attention_maps").mkdir(exist_ok=True)
        (self.experiment_dir / "metrics").mkdir(exist_ok=True)
        (self.experiment_dir / "tables").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        
        # Save config
        with open(self.experiment_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"Experiment directory: {self.experiment_dir}")
    
    def _load_prompts(self) -> List[Dict[str, Any]]:
        """Load test prompts from file."""
        prompts_file = self.config.get("dataset", {}).get("prompts_file", "data/prompts/test_prompts.json")
        
        with open(prompts_file, 'r') as f:
            data = json.load(f)
        
        return data.get("prompts", [])
    
    def _load_models(self):
        """Load LPA and baseline models."""
        model_config = self.config.get("model", {})
        
        # Load LPA model
        print("Loading LPA model...")
        dtype_str = model_config.get("dtype", "float32")
        # Use float32 for CPU/MPS, float16 for CUDA
        if self.device in ["cpu", "mps"] and dtype_str == "float16":
            dtype_str = "float32"
            print(f"Warning: Using float32 instead of float16 for {self.device} device")
        
        self.lpa_model = LPAModel(
            model_name=model_config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0"),
            device=self.device,
            dtype=getattr(torch, dtype_str),
            use_attention_slicing=model_config.get("use_attention_slicing", True),
            use_memory_efficient_attention=model_config.get("use_memory_efficient_attention", True)
        )
        
        # Load baseline models
        print("Loading baseline models...")
        self.baseline_models = BaselineModels(
            model_name=model_config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0"),
            device=self.device,
            dtype=getattr(torch, dtype_str)
        )
        
        print("Models loaded successfully!")
    
    def _create_lpa_config(self) -> LPAGenerationConfig:
        """Create LPA generation configuration from config file."""
        lpa_config = self.config.get("lpa", {})
        generation_config = self.config.get("generation", {})
        
        return LPAGenerationConfig(
            num_inference_steps=generation_config.get("num_inference_steps", 50),
            guidance_scale=generation_config.get("guidance_scale", 7.5),
            height=generation_config.get("height", 1024),
            width=generation_config.get("width", 1024),
            num_images_per_prompt=generation_config.get("num_images_per_prompt", 4),
            injection_timesteps=lpa_config.get("injection", {}).get("timesteps", [10, 20, 30, 40, 50]),
            object_layers=lpa_config.get("injection", {}).get("object_layers", ["down_blocks.0", "down_blocks.1", "down_blocks.2"]),
            style_layers=lpa_config.get("injection", {}).get("style_layers", ["mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2"]),
            injection_strength=lpa_config.get("injection", {}).get("injection_strength", 1.0),
            save_attention_maps=lpa_config.get("attention", {}).get("save_attention_maps", True)
        )
    
    def _create_baseline_config(self) -> BaselineConfig:
        """Create baseline generation configuration from config file."""
        generation_config = self.config.get("generation", {})
        
        return BaselineConfig(
            num_inference_steps=generation_config.get("num_inference_steps", 50),
            guidance_scale=generation_config.get("guidance_scale", 7.5),
            height=generation_config.get("height", 1024),
            width=generation_config.get("width", 1024),
            num_images_per_prompt=generation_config.get("num_images_per_prompt", 4)
        )
    
    def _compute_evaluation_metrics(self, results: Dict[str, Any], parsed_prompt) -> Dict[str, Any]:
        """
        Compute evaluation metrics for all methods.
        
        Args:
            results: Experiment results dictionary
            parsed_prompt: Parsed prompt object
            
        Returns:
            Dictionary containing metrics for all methods
        """
        metrics = {}
        
        # Extract style tokens for evaluation
        style_tokens = parsed_prompt.style_tokens if hasattr(parsed_prompt, 'style_tokens') else []
        
        # Evaluate LPA results
        if "lpa" in results and "error" not in results["lpa"]:
            try:
                lpa_metrics = self.evaluator.evaluate_generation_results(
                    results["lpa"], 
                    reference_images=None
                )
                metrics["lpa"] = lpa_metrics
            except Exception as e:
                print(f"    Error evaluating LPA: {e}")
                metrics["lpa"] = {"error": str(e)}
        
        # Evaluate baseline results
        if "baselines" in results and "error" not in results["baselines"]:
            baseline_metrics = {}
            for method, method_results in results["baselines"].items():
                try:
                    method_metrics = self.evaluator.evaluate_generation_results(
                        method_results, 
                        reference_images=None
                    )
                    baseline_metrics[method] = method_metrics
                except Exception as e:
                    print(f"    Error evaluating {method}: {e}")
                    baseline_metrics[method] = {"error": str(e)}
            
            metrics["baselines"] = baseline_metrics
        
        return metrics
    
    def run_single_experiment(self, prompt_data: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run experiment for a single prompt.
        
        Args:
            prompt_data: Prompt data from dataset
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing results from all methods
        """
        prompt = prompt_data["prompt"]
        prompt_id = prompt_data["id"]
        
        print(f"\nRunning experiment for prompt {prompt_id}: {prompt}")
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        results = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "category": prompt_data.get("category", "unknown"),
            "complexity": prompt_data.get("complexity", "medium"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Parse prompt
        parsed_prompt = self.prompt_parser.parse_prompt(prompt)
        results["parsed_prompt"] = parsed_prompt.to_dict()
        
        # Generate with LPA
        if self.lpa_model is not None:
            print("  Generating with LPA...")
            try:
                lpa_config = self._create_lpa_config()
                lpa_results = self.lpa_model.generate(
                    prompt=prompt,
                    config=lpa_config,
                    seed=seed
                )
                results["lpa"] = lpa_results
                
                # Save LPA results
                lpa_dir = self.experiment_dir / "images" / "lpa" / prompt_id
                self.lpa_model.save_results(lpa_results, str(lpa_dir), f"lpa_{prompt_id}")
                
            except Exception as e:
                print(f"  Error in LPA generation: {e}")
                results["lpa"] = {"error": str(e)}
        
        # Generate with baselines
        if self.baseline_models is not None:
            print("  Generating with baselines...")
            try:
                baseline_config = self._create_baseline_config()
                baseline_results = self.baseline_models.generate_all_baselines(
                    prompt=prompt,
                    config=baseline_config,
                    seed=seed
                )
                results["baselines"] = baseline_results
                
                # Save baseline results
                for method, method_results in baseline_results.items():
                    baseline_dir = self.experiment_dir / "images" / method / prompt_id
                    self.baseline_models.save_baseline_results(
                        method_results, 
                        str(baseline_dir), 
                        f"{method}_{prompt_id}"
                    )
                
            except Exception as e:
                print(f"  Error in baseline generation: {e}")
                results["baselines"] = {"error": str(e)}
        
        # Compute evaluation metrics
        print("  Computing evaluation metrics...")
        try:
            evaluation_metrics = self._compute_evaluation_metrics(results, parsed_prompt)
            results["evaluation_metrics"] = evaluation_metrics
            
            # Save metrics to file
            metrics_file = self.experiment_dir / "metrics" / f"{prompt_id}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(evaluation_metrics, f, indent=2)
                
        except Exception as e:
            print(f"  Error computing metrics: {e}")
            results["evaluation_metrics"] = {"error": str(e)}
        
        return results
    
    def run_full_experiment(self, max_prompts: Optional[int] = None, seed: Optional[int] = None):
        """
        Run full experiment on all prompts.
        
        Args:
            max_prompts: Maximum number of prompts to process (for testing)
            seed: Random seed for reproducibility
        """
        print("Starting full experiment...")
        
        # Load prompts
        prompts = self._load_prompts()
        if max_prompts is not None:
            prompts = prompts[:max_prompts]
        
        print(f"Processing {len(prompts)} prompts...")
        
        # Load models
        self._load_models()
        
        # Run experiments
        all_results = []
        
        for i, prompt_data in enumerate(prompts):
            print(f"\nProgress: {i+1}/{len(prompts)}")
            
            try:
                result = self.run_single_experiment(prompt_data, seed=seed)
                all_results.append(result)
                
                # Save intermediate results (JSON-serializable)
                json_results = self._make_json_serializable(all_results)
                with open(self.experiment_dir / "results.json", 'w') as f:
                    json.dump(json_results, f, indent=2)
                
            except Exception as e:
                print(f"Error processing prompt {prompt_data.get('id', i)}: {e}")
                continue
        
        # Save final results
        self.results = all_results
        self._save_final_results()
        
        # Generate summary
        self._generate_summary()
        
        print(f"\nExperiment completed! Results saved to: {self.experiment_dir}")
    
    def _save_final_results(self):
        """Save final experiment results."""
        # Create JSON-serializable version of results (without PIL Images)
        json_results = self._make_json_serializable(self.results)
        
        # Save complete results
        with open(self.experiment_dir / "final_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save summary statistics
        summary = self._compute_summary_statistics()
        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Recursively convert results to JSON-serializable format."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif "PIL" in str(type(obj)):
            return "PIL_Image"
        else:
            return str(obj)
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        if not self.results:
            return {}
        
        total_prompts = len(self.results)
        successful_lpa = sum(1 for r in self.results if "lpa" in r and "error" not in r.get("lpa", {}))
        successful_baselines = sum(1 for r in self.results if "baselines" in r and "error" not in r.get("baselines", {}))
        
        # Category distribution
        categories = {}
        complexities = {}
        
        for result in self.results:
            category = result.get("category", "unknown")
            complexity = result.get("complexity", "medium")
            
            categories[category] = categories.get(category, 0) + 1
            complexities[complexity] = complexities.get(complexity, 0) + 1
        
        return {
            "total_prompts": total_prompts,
            "successful_lpa": successful_lpa,
            "successful_baselines": successful_baselines,
            "lpa_success_rate": successful_lpa / total_prompts if total_prompts > 0 else 0,
            "baseline_success_rate": successful_baselines / total_prompts if total_prompts > 0 else 0,
            "category_distribution": categories,
            "complexity_distribution": complexities,
            "experiment_timestamp": datetime.now().isoformat()
        }
    
    def _generate_summary(self):
        """Generate summary report."""
        summary_path = self.experiment_dir / "summary_report.txt"
        
        with open(summary_path, 'w') as f:
            f.write("LPA Experiment Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment Directory: {self.experiment_dir}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            # Load summary statistics
            summary_file = self.experiment_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as sf:
                    summary = json.load(sf)
                
                f.write(f"Total Prompts: {summary.get('total_prompts', 0)}\n")
                f.write(f"LPA Success Rate: {summary.get('lpa_success_rate', 0):.2%}\n")
                f.write(f"Baseline Success Rate: {summary.get('baseline_success_rate', 0):.2%}\n\n")
                
                f.write("Category Distribution:\n")
                for category, count in summary.get("category_distribution", {}).items():
                    f.write(f"  {category}: {count}\n")
                
                f.write("\nComplexity Distribution:\n")
                for complexity, count in summary.get("complexity_distribution", {}).items():
                    f.write(f"  {complexity}: {count}\n")
    
    def cleanup(self):
        """Clean up resources."""
        if self.lpa_model is not None:
            self.lpa_model.cleanup()
        if self.baseline_models is not None:
            self.baseline_models.cleanup()


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run LPA experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to process (for testing)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run experiment
    runner = ExperimentRunner(args.config)
    
    try:
        runner.run_full_experiment(max_prompts=args.max_prompts, seed=args.seed)
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main() 