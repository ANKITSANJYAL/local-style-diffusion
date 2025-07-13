#!/usr/bin/env python3
"""
Paper Experiment Runner for Local Prompt Adaptation (LPA).

This script runs experiments with both fast (SD v1.5) and SDXL models
to provide comprehensive results for the paper.
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.experiments.run_experiments import ExperimentRunner


class PaperExperimentRunner:
    """
    Runner for paper experiments comparing fast and SDXL models.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize paper experiment runner.
        
        Args:
            config_path: Path to paper experiment configuration
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.base_dir = Path(self.config.get("output", {}).get("base_dir", "experiments"))
        
        # Create paper experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paper_dir = self.base_dir / f"paper_experiments_{timestamp}"
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Paper experiment directory: {self.paper_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run_fast_experiments(self, max_prompts: int = None) -> str:
        """
        Run experiments with fast model (SD v1.5).
        
        Args:
            max_prompts: Maximum number of prompts to process
            
        Returns:
            Path to fast experiment results
        """
        print("\n" + "="*60)
        print("RUNNING FAST EXPERIMENTS (SD v1.5)")
        print("="*60)
        
        # Create fast config
        fast_config = self._create_fast_config()
        fast_config_path = self.paper_dir / "fast_config.yaml"
        
        with open(fast_config_path, 'w') as f:
            yaml.dump(fast_config, f, default_flow_style=False, indent=2)
        
        # Run fast experiments
        runner = ExperimentRunner(str(fast_config_path))
        
        try:
            runner.run_full_experiment(max_prompts=max_prompts, seed=self.config.get("experiment", {}).get("seed", 42))
            return str(runner.experiment_dir)
        finally:
            runner.cleanup()
    
    def run_sdxl_experiments(self, max_prompts: int = None) -> str:
        """
        Run experiments with SDXL model.
        
        Args:
            max_prompts: Maximum number of prompts to process
            
        Returns:
            Path to SDXL experiment results
        """
        print("\n" + "="*60)
        print("RUNNING SDXL EXPERIMENTS")
        print("="*60)
        
        # Create SDXL config
        sdxl_config = self._create_sdxl_config()
        sdxl_config_path = self.paper_dir / "sdxl_config.yaml"
        
        with open(sdxl_config_path, 'w') as f:
            yaml.dump(sdxl_config, f, default_flow_style=False, indent=2)
        
        # Run SDXL experiments
        runner = ExperimentRunner(str(sdxl_config_path))
        
        try:
            runner.run_full_experiment(max_prompts=max_prompts, seed=self.config.get("experiment", {}).get("seed", 42))
            return str(runner.experiment_dir)
        finally:
            runner.cleanup()
    
    def _create_fast_config(self) -> Dict[str, Any]:
        """Create configuration for fast experiments."""
        fast_config = self.config.copy()
        
        # Update model settings
        fast_config["model"]["base_model"] = self.config["model"]["fast_model"]
        fast_config["model"]["device"] = self.config["model"]["fast_device"]
        fast_config["model"]["dtype"] = self.config["model"]["fast_dtype"]
        
        # Update generation settings
        generation = fast_config["generation"]
        generation["num_inference_steps"] = generation["fast_num_inference_steps"]
        generation["num_images_per_prompt"] = generation["fast_num_images_per_prompt"]
        generation["height"] = generation["fast_height"]
        generation["width"] = generation["fast_width"]
        
        # Update LPA settings
        fast_config["lpa"]["injection"] = fast_config["lpa"]["fast_injection"]
        
        # Update experiment name
        fast_config["experiment"]["name"] = "lpa_fast_paper"
        fast_config["experiment"]["description"] = "Fast Model Experiments for Paper"
        
        return fast_config
    
    def _create_sdxl_config(self) -> Dict[str, Any]:
        """Create configuration for SDXL experiments."""
        sdxl_config = self.config.copy()
        
        # Update experiment name
        sdxl_config["experiment"]["name"] = "lpa_sdxl_paper"
        sdxl_config["experiment"]["description"] = "SDXL Model Experiments for Paper"
        
        return sdxl_config
    
    def generate_comparison_report(self, fast_results_dir: str, sdxl_results_dir: str):
        """
        Generate comparison report between fast and SDXL results.
        
        Args:
            fast_results_dir: Directory containing fast experiment results
            sdxl_results_dir: Directory containing SDXL experiment results
        """
        print("\n" + "="*60)
        print("GENERATING COMPARISON REPORT")
        print("="*60)
        
        comparison_data = {
            "fast_results_dir": fast_results_dir,
            "sdxl_results_dir": sdxl_results_dir,
            "comparison_timestamp": datetime.now().isoformat(),
            "config": self.config
        }
        
        # Save comparison metadata
        comparison_file = self.paper_dir / "comparison_metadata.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(fast_results_dir, sdxl_results_dir)
        
        print(f"Comparison report saved to: {self.paper_dir}")
    
    def _generate_summary_report(self, fast_results_dir: str, sdxl_results_dir: str):
        """Generate summary report comparing both experiments."""
        report_path = self.paper_dir / "paper_experiment_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Local Prompt Adaptation (LPA) Paper Experiments Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Paper Experiment Directory: {self.paper_dir}\n\n")
            
            f.write("Model Configurations:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Fast Model (SD v1.5): {self.config['model']['fast_model']}\n")
            f.write(f"SDXL Model: {self.config['model']['base_model']}\n\n")
            
            f.write("Results Directories:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Fast Results: {fast_results_dir}\n")
            f.write(f"SDXL Results: {sdxl_results_dir}\n\n")
            
            f.write("Generation Parameters:\n")
            f.write("-" * 30 + "\n")
            f.write("Fast Model:\n")
            f.write(f"  - Inference Steps: {self.config['generation']['fast_num_inference_steps']}\n")
            f.write(f"  - Image Size: {self.config['generation']['fast_width']}x{self.config['generation']['fast_height']}\n")
            f.write(f"  - Images per Prompt: {self.config['generation']['fast_num_images_per_prompt']}\n\n")
            
            f.write("SDXL Model:\n")
            f.write(f"  - Inference Steps: {self.config['generation']['num_inference_steps']}\n")
            f.write(f"  - Image Size: {self.config['generation']['width']}x{self.config['generation']['height']}\n")
            f.write(f"  - Images per Prompt: {self.config['generation']['num_images_per_prompt']}\n\n")
            
            f.write("Next Steps for Paper:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Analyze results in both experiment directories\n")
            f.write("2. Compare image quality and style consistency\n")
            f.write("3. Generate comparison visualizations\n")
            f.write("4. Calculate quantitative metrics\n")
            f.write("5. Create paper figures and tables\n")
    
    def run_full_paper_experiments(self, max_prompts: int = None):
        """
        Run complete paper experiments with both models.
        
        Args:
            max_prompts: Maximum number of prompts to process
        """
        print("Starting Paper Experiments for Local Prompt Adaptation")
        print(f"Configuration: {self.config_path}")
        print(f"Output Directory: {self.paper_dir}")
        
        # Run fast experiments
        fast_results_dir = self.run_fast_experiments(max_prompts)
        
        # Run SDXL experiments
        sdxl_results_dir = self.run_sdxl_experiments(max_prompts)
        
        # Generate comparison report
        self.generate_comparison_report(fast_results_dir, sdxl_results_dir)
        
        print(f"\n🎉 Paper experiments completed!")
        print(f"Results saved to: {self.paper_dir}")
        print(f"Fast results: {fast_results_dir}")
        print(f"SDXL results: {sdxl_results_dir}")


def main():
    """Main function for running paper experiments."""
    parser = argparse.ArgumentParser(description="Run LPA paper experiments")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to paper experiment config file")
    parser.add_argument("--max-prompts", type=int, 
                       help="Maximum number of prompts to process (for testing)")
    parser.add_argument("--fast-only", action="store_true",
                       help="Run only fast experiments")
    parser.add_argument("--sdxl-only", action="store_true",
                       help="Run only SDXL experiments")
    
    args = parser.parse_args()
    
    # Run experiments
    runner = PaperExperimentRunner(args.config)
    
    if args.fast_only:
        runner.run_fast_experiments(args.max_prompts)
    elif args.sdxl_only:
        runner.run_sdxl_experiments(args.max_prompts)
    else:
        runner.run_full_paper_experiments(args.max_prompts)


if __name__ == "__main__":
    main() 