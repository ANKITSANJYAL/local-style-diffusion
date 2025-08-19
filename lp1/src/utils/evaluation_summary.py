"""
Evaluation Summary Utilities for Local Prompt Adaptation (LPA).

This module provides functionality for aggregating evaluation metrics
across experiments and generating paper-ready statistics and visualizations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

class EvaluationSummary:
    """
    Summary and analysis of evaluation metrics across experiments.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize evaluation summary.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.metrics_data = {}
        self.summary_stats = {}
        
    def load_metrics(self):
        """Load all metrics from experiment directory."""
        metrics_dir = self.experiment_dir / "metrics"
        if not metrics_dir.exists():
            print(f"Warning: Metrics directory not found: {metrics_dir}")
            return
        
        # Load individual prompt metrics
        for metrics_file in metrics_dir.glob("*_metrics.json"):
            prompt_id = metrics_file.stem.replace("_metrics", "")
            with open(metrics_file, 'r') as f:
                self.metrics_data[prompt_id] = json.load(f)
        
        # Load final results if available
        final_results_file = self.experiment_dir / "final_results.json"
        if final_results_file.exists():
            with open(final_results_file, 'r') as f:
                self.final_results = json.load(f)
    
    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all prompts.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.metrics_data:
            return {}
        
        summary = {
            "total_prompts": len(self.metrics_data),
            "methods": {},
            "comparisons": {},
            "category_analysis": {},
            "complexity_analysis": {}
        }
        
        # Aggregate metrics by method
        method_metrics = {}
        for prompt_id, metrics in self.metrics_data.items():
            for method, method_metrics_dict in metrics.items():
                if method not in method_metrics:
                    method_metrics[method] = {}
                
                for metric_name, metric_value in method_metrics_dict.items():
                    if metric_name == "error":
                        continue
                    
                    if metric_name not in method_metrics[method]:
                        method_metrics[method][metric_name] = []
                    
                    if isinstance(metric_value, (int, float)):
                        method_metrics[method][metric_name].append(metric_value)
        
        # Compute statistics for each method
        for method, metrics in method_metrics.items():
            summary["methods"][method] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary["methods"][method][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }
        
        # Compute method comparisons
        if "lpa" in method_metrics and len(method_metrics) > 1:
            summary["comparisons"] = self._compute_method_comparisons(method_metrics)
        
        self.summary_stats = summary
        return summary
    
    def _compute_method_comparisons(self, method_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Compute statistical comparisons between methods."""
        comparisons = {}
        
        # Get common metrics across methods
        all_metrics = set()
        for method_metrics_dict in method_metrics.values():
            all_metrics.update(method_metrics_dict.keys())
        
        for metric in all_metrics:
            metric_data = {}
            for method, metrics_dict in method_metrics.items():
                if metric in metrics_dict:
                    metric_data[method] = metrics_dict[metric]
            
            if len(metric_data) >= 2:
                comparisons[metric] = {}
                
                # Perform t-tests between LPA and each baseline
                if "lpa" in metric_data:
                    lpa_values = metric_data["lpa"]
                    for method, values in metric_data.items():
                        if method != "lpa" and len(values) > 0:
                            try:
                                t_stat, p_value = stats.ttest_ind(lpa_values, values)
                                comparisons[metric][f"lpa_vs_{method}"] = {
                                    "t_statistic": float(t_stat),
                                    "p_value": float(p_value),
                                    "significant": p_value < 0.05,
                                    "lpa_mean": float(np.mean(lpa_values)),
                                    f"{method}_mean": float(np.mean(values)),
                                    "improvement": float(np.mean(lpa_values) - np.mean(values))
                                }
                            except Exception as e:
                                comparisons[metric][f"lpa_vs_{method}"] = {"error": str(e)}
        
        return comparisons
    
    def generate_summary_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report content as string
        """
        if not self.summary_stats:
            self.compute_summary_statistics()
        
        report_lines = []
        report_lines.append("Local Prompt Adaptation (LPA) Evaluation Summary")
        report_lines.append("=" * 60)
        report_lines.append(f"Experiment Directory: {self.experiment_dir}")
        report_lines.append(f"Total Prompts: {self.summary_stats.get('total_prompts', 0)}")
        report_lines.append("")
        
        # Method performance summary
        report_lines.append("METHOD PERFORMANCE SUMMARY")
        report_lines.append("-" * 30)
        
        for method, metrics in self.summary_stats.get("methods", {}).items():
            report_lines.append(f"\n{method.upper()}:")
            for metric, stats_dict in metrics.items():
                mean_val = stats_dict.get("mean", 0)
                std_val = stats_dict.get("std", 0)
                report_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Statistical comparisons
        comparisons = self.summary_stats.get("comparisons", {})
        if comparisons:
            report_lines.append("\n\nSTATISTICAL COMPARISONS")
            report_lines.append("-" * 30)
            
            for metric, method_comparisons in comparisons.items():
                report_lines.append(f"\n{metric.upper()}:")
                for comparison, stats_dict in method_comparisons.items():
                    if "error" not in stats_dict:
                        p_val = stats_dict.get("p_value", 1.0)
                        improvement = stats_dict.get("improvement", 0)
                        significant = stats_dict.get("significant", False)
                        
                        sig_marker = "***" if significant else ""
                        report_lines.append(f"  {comparison}: {improvement:+.4f} (p={p_val:.4f}) {sig_marker}")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
        
        return report_content
    
    def create_visualizations(self, save_dir: Optional[str] = None):
        """
        Create visualization plots for the evaluation results.
        
        Args:
            save_dir: Directory to save plots
        """
        if not self.summary_stats:
            self.compute_summary_statistics()
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.experiment_dir / "visualizations"
            save_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Method comparison bar chart
        self._create_method_comparison_plot(save_path)
        
        # 2. Metric distribution plots
        self._create_metric_distribution_plots(save_path)
        
        # 3. Statistical significance heatmap
        self._create_significance_heatmap(save_path)
        
        print(f"Visualizations saved to: {save_path}")
    
    def _create_method_comparison_plot(self, save_path: Path):
        """Create bar chart comparing methods across metrics."""
        methods = self.summary_stats.get("methods", {})
        if not methods:
            return
        
        # Get all metrics
        all_metrics = set()
        for method_metrics in methods.values():
            all_metrics.update(method_metrics.keys())
        
        # Prepare data
        metric_names = sorted(list(all_metrics))
        method_names = list(methods.keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metric_names))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            means = []
            stds = []
            for metric in metric_names:
                if metric in methods[method]:
                    means.append(methods[method][metric]["mean"])
                    stds.append(methods[method][metric]["std"])
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i * width, means, width, label=method, 
                   yerr=stds, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Method Comparison Across Metrics')
        ax.set_xticks(x + width * (len(method_names) - 1) / 2)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "method_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_distribution_plots(self, save_path: Path):
        """Create distribution plots for each metric."""
        methods = self.summary_stats.get("methods", {})
        if not methods:
            return
        
        # Get all metrics
        all_metrics = set()
        for method_metrics in methods.values():
            all_metrics.update(method_metrics.keys())
        
        for metric in all_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for method in methods.keys():
                if metric in methods[method]:
                    # We need to reconstruct the original values for plotting
                    # This is a limitation of the current summary approach
                    # In a full implementation, we'd store the raw values
                    mean_val = methods[method][metric]["mean"]
                    std_val = methods[method][metric]["std"]
                    
                    # Create a normal distribution approximation
                    x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
                    y = stats.norm.pdf(x, mean_val, std_val)
                    
                    ax.plot(x, y, label=method, linewidth=2)
            
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path / f"{metric}_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_significance_heatmap(self, save_path: Path):
        """Create heatmap of statistical significance."""
        comparisons = self.summary_stats.get("comparisons", {})
        if not comparisons:
            return
        
        # Prepare data for heatmap
        metrics = list(comparisons.keys())
        methods = set()
        for metric_comparisons in comparisons.values():
            for comparison in metric_comparisons.keys():
                if "lpa_vs_" in comparison:
                    method = comparison.replace("lpa_vs_", "")
                    methods.add(method)
        
        methods = sorted(list(methods))
        
        # Create p-value matrix
        p_values = np.full((len(metrics), len(methods)), np.nan)
        
        for i, metric in enumerate(metrics):
            for j, method in enumerate(methods):
                comparison_key = f"lpa_vs_{method}"
                if comparison_key in comparisons[metric]:
                    stats_dict = comparisons[metric][comparison_key]
                    if "p_value" in stats_dict:
                        p_values[i, j] = stats_dict["p_value"]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for NaN values
        mask = np.isnan(p_values)
        
        # Create heatmap
        sns.heatmap(p_values, 
                   mask=mask,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0.05,
                   cbar_kws={'label': 'p-value'},
                   xticklabels=methods,
                   yticklabels=metrics,
                   ax=ax)
        
        ax.set_title('Statistical Significance (p-values)\nLPA vs Baselines')
        ax.set_xlabel('Baseline Methods')
        ax.set_ylabel('Metrics')
        
        plt.tight_layout()
        plt.savefig(save_path / "statistical_significance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_summary_data(self, save_path: Optional[str] = None):
        """
        Save summary data to JSON file.
        
        Args:
            save_path: Path to save summary data
        """
        if not self.summary_stats:
            self.compute_summary_statistics()
        
        if save_path is None:
            save_path = self.experiment_dir / "evaluation_summary.json"
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        serializable_summary = convert_numpy(self.summary_stats)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
        print(f"Summary data saved to: {save_path}")


def main():
    """Example usage of EvaluationSummary."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation summary")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    parser.add_argument("--save-report", help="Path to save summary report")
    
    args = parser.parse_args()
    
    # Create summary
    summary = EvaluationSummary(args.experiment_dir)
    summary.load_metrics()
    
    # Generate report
    report = summary.generate_summary_report(args.save_report)
    print(report)
    
    # Create visualizations
    summary.create_visualizations(args.output_dir)
    
    # Save summary data
    summary.save_summary_data()


if __name__ == "__main__":
    main() 