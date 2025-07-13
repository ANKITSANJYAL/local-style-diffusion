#!/usr/bin/env python3
"""
Test script for evaluation metrics on recent experiment results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.evaluation_summary import EvaluationSummary
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test evaluation on experiment results")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    
    args = parser.parse_args()
    
    print(f"Testing evaluation on: {args.experiment_dir}")
    
    # Create evaluation summary
    summary = EvaluationSummary(args.experiment_dir)
    
    # Load metrics
    print("Loading metrics...")
    summary.load_metrics()
    
    if not summary.metrics_data:
        print("No metrics found. Running a test experiment first...")
        print("Please run: python src/experiments/run_experiments.py --config configs/experiment_config_fast.yaml --max-prompts 1")
        return
    
    # Compute summary statistics
    print("Computing summary statistics...")
    stats = summary.compute_summary_statistics()
    
    # Generate report
    print("Generating report...")
    report = summary.generate_summary_report()
    print("\n" + "="*60)
    print(report)
    
    # Create visualizations
    print("\nCreating visualizations...")
    summary.create_visualizations()
    
    # Save summary data
    print("Saving summary data...")
    summary.save_summary_data()
    
    print(f"\n✅ Evaluation test completed!")
    print(f"Check the visualizations and summary files in: {args.experiment_dir}")

if __name__ == "__main__":
    main() 