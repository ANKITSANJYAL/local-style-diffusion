#!/usr/bin/env python3
"""
Comprehensive analysis script to address research questions for LPA paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import re

class ResearchQuestionAnalyzer:
    """
    Analyzer for addressing research questions in LPA paper.
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize analyzer.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.results = {}
        self.prompts_data = {}
        
    def load_experiment_data(self):
        """Load experiment results and prompts data."""
        # Load final results
        final_results_file = self.experiment_dir / "final_results.json"
        if final_results_file.exists():
            with open(final_results_file, 'r') as f:
                results_list = json.load(f)
                # Convert list to dictionary with prompt_id as key
                self.results = {result["prompt_id"]: result for result in results_list}
        
        # Load prompts data
        prompts_file = Path("data/prompts/test_prompts.json")
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                prompts_data = json.load(f)
                self.prompts_data = {p["id"]: p for p in prompts_data.get("prompts", [])}
    
    def analyze_complexity_scaling(self) -> Dict[str, Any]:
        """
        Research Question 2: How does the method scale with prompt complexity?
        """
        print("🔍 Analyzing complexity scaling...")
        
        complexity_analysis = {
            "object_count": {},
            "complexity_levels": {},
            "correlations": {}
        }
        
        # Analyze by object count
        object_counts = {}
        for prompt_id, prompt_data in self.prompts_data.items():
            if prompt_id in self.results:
                # Count objects in prompt
                prompt_text = prompt_data.get("prompt", "")
                objects = prompt_data.get("objects", [])
                object_count = len(objects)
                
                if object_count not in object_counts:
                    object_counts[object_count] = {"style_consistency": [], "clip_score": []}
                
                # Get metrics for this prompt
                if "lpa" in self.results[prompt_id]:
                    lpa_results = self.results[prompt_id]["lpa"]
                    if "evaluation_metrics" in lpa_results:
                        metrics = lpa_results["evaluation_metrics"]["lpa"]
                        if "style_consistency" in metrics:
                            object_counts[object_count]["style_consistency"].append(metrics["style_consistency"])
                        if "clip_score" in metrics:
                            object_counts[object_count]["clip_score"].append(metrics["clip_score"])
        
        # Compute statistics for each object count
        for obj_count, metrics in object_counts.items():
            complexity_analysis["object_count"][obj_count] = {}
            for metric_name, values in metrics.items():
                if values:
                    complexity_analysis["object_count"][obj_count][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
        
        # Analyze by complexity levels
        complexity_levels = {}
        for prompt_id, prompt_data in self.prompts_data.items():
            if prompt_id in self.results:
                complexity = prompt_data.get("complexity", "medium")
                
                if complexity not in complexity_levels:
                    complexity_levels[complexity] = {"style_consistency": [], "clip_score": []}
                
                # Get metrics
                if "lpa" in self.results[prompt_id]:
                    lpa_results = self.results[prompt_id]["lpa"]
                    if "evaluation_metrics" in lpa_results:
                        metrics = lpa_results["evaluation_metrics"]["lpa"]
                        if "style_consistency" in metrics:
                            complexity_levels[complexity]["style_consistency"].append(metrics["style_consistency"])
                        if "clip_score" in metrics:
                            complexity_levels[complexity]["clip_score"].append(metrics["clip_score"])
        
        # Compute statistics for complexity levels
        for complexity, metrics in complexity_levels.items():
            complexity_analysis["complexity_levels"][complexity] = {}
            for metric_name, values in metrics.items():
                if values:
                    complexity_analysis["complexity_levels"][complexity][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
        
        return complexity_analysis
    
    def analyze_style_content_tradeoff(self) -> Dict[str, Any]:
        """
        Research Question 3: What's the trade-off between style consistency and content fidelity?
        """
        print("🔍 Analyzing style vs content trade-off...")
        
        tradeoff_data = {
            "correlation": {},
            "scatter_data": [],
            "analysis": {}
        }
        
        # Collect style consistency and CLIP scores
        style_scores = []
        clip_scores = []
        prompt_ids = []
        
        for prompt_id, result in self.results.items():
            if "lpa" in result and "evaluation_metrics" in result["lpa"]:
                metrics = result["lpa"]["evaluation_metrics"]["lpa"]
                if "style_consistency" in metrics and "clip_score" in metrics:
                    style_scores.append(metrics["style_consistency"])
                    clip_scores.append(metrics["clip_score"])
                    prompt_ids.append(prompt_id)
        
        if len(style_scores) > 1:
            # Compute correlation
            correlation, p_value = stats.pearsonr(style_scores, clip_scores)
            tradeoff_data["correlation"] = {
                "pearson_r": correlation,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
            
            # Store scatter data
            tradeoff_data["scatter_data"] = list(zip(style_scores, clip_scores, prompt_ids))
            
            # Analyze trade-off patterns
            tradeoff_data["analysis"] = {
                "high_style_low_clip": len([(s, c) for s, c in zip(style_scores, clip_scores) if s > np.mean(style_scores) and c < np.mean(clip_scores)]),
                "low_style_high_clip": len([(s, c) for s, c in zip(style_scores, clip_scores) if s < np.mean(style_scores) and c > np.mean(clip_scores)]),
                "both_high": len([(s, c) for s, c in zip(style_scores, clip_scores) if s > np.mean(style_scores) and c > np.mean(clip_scores)]),
                "both_low": len([(s, c) for s, c in zip(style_scores, clip_scores) if s < np.mean(style_scores) and c < np.mean(clip_scores)])
            }
        
        return tradeoff_data
    
    def analyze_style_robustness(self) -> Dict[str, Any]:
        """
        Research Question 4: How robust is the method to different style definitions?
        """
        print("🔍 Analyzing style robustness...")
        
        style_analysis = {
            "style_categories": {},
            "style_performance": {},
            "robustness_metrics": {}
        }
        
        # Define style categories
        style_categories = {
            "artistic": ["watercolor", "oil painting", "sketch", "impressionist", "cubist", "surrealist", "minimalist", "abstract"],
            "photographic": ["realistic", "photographic", "cinematic", "portrait", "landscape", "street photography"],
            "digital": ["cyberpunk", "synthwave", "steampunk", "pixel art", "digital art", "3d render"],
            "traditional": ["classical", "renaissance", "baroque", "romantic", "neoclassical"]
        }
        
        # Analyze performance by style category
        for category, style_keywords in style_categories.items():
            style_analysis["style_categories"][category] = {
                "style_consistency": [],
                "clip_score": [],
                "prompts": []
            }
            
            for prompt_id, prompt_data in self.prompts_data.items():
                if prompt_id in self.results:
                    prompt_text = prompt_data.get("prompt", "").lower()
                    style_tokens = prompt_data.get("style", [])
                    
                    # Check if prompt contains style keywords from this category
                    matches_category = any(keyword in prompt_text for keyword in style_keywords) or \
                                     any(keyword in " ".join(style_tokens).lower() for keyword in style_keywords)
                    
                    if matches_category:
                        # Get metrics
                        if "lpa" in self.results[prompt_id]:
                            lpa_results = self.results[prompt_id]["lpa"]
                            if "evaluation_metrics" in lpa_results:
                                metrics = lpa_results["evaluation_metrics"]["lpa"]
                                if "style_consistency" in metrics:
                                    style_analysis["style_categories"][category]["style_consistency"].append(metrics["style_consistency"])
                                if "clip_score" in metrics:
                                    style_analysis["style_categories"][category]["clip_score"].append(metrics["clip_score"])
                                style_analysis["style_categories"][category]["prompts"].append(prompt_id)
        
        # Compute statistics for each style category
        for category, data in style_analysis["style_categories"].items():
            style_analysis["style_performance"][category] = {}
            for metric_name, values in data.items():
                if metric_name != "prompts" and values:
                    style_analysis["style_performance"][category][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values)
                    }
        
        # Compute robustness metrics
        if style_analysis["style_performance"]:
            style_consistency_means = [data.get("style_consistency", {}).get("mean", 0) for data in style_analysis["style_performance"].values()]
            clip_score_means = [data.get("clip_score", {}).get("mean", 0) for data in style_analysis["style_performance"].values()]
            
            style_analysis["robustness_metrics"] = {
                "style_consistency_variance": np.var(style_consistency_means) if len(style_consistency_means) > 1 else 0,
                "clip_score_variance": np.var(clip_score_means) if len(clip_score_means) > 1 else 0,
                "style_consistency_range": max(style_consistency_means) - min(style_consistency_means) if style_consistency_means else 0,
                "clip_score_range": max(clip_score_means) - min(clip_score_means) if clip_score_means else 0
            }
        
        return style_analysis
    
    def create_visualizations(self, analysis_results: Dict[str, Any]):
        """Create visualizations for research questions."""
        print("📊 Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        viz_dir = self.experiment_dir / "research_analysis"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Complexity scaling plot
        if "object_count" in analysis_results.get("complexity_scaling", {}):
            self._plot_complexity_scaling(analysis_results["complexity_scaling"], viz_dir)
        
        # 2. Style vs content trade-off plot
        if "scatter_data" in analysis_results.get("style_content_tradeoff", {}):
            self._plot_style_content_tradeoff(analysis_results["style_content_tradeoff"], viz_dir)
        
        # 3. Style robustness plot
        if "style_performance" in analysis_results.get("style_robustness", {}):
            self._plot_style_robustness(analysis_results["style_robustness"], viz_dir)
        
        print(f"Visualizations saved to: {viz_dir}")
    
    def _plot_complexity_scaling(self, complexity_data: Dict[str, Any], save_dir: Path):
        """Plot complexity scaling analysis."""
        object_counts = complexity_data.get("object_count", {})
        
        if not object_counts:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Style consistency by object count
        obj_counts = sorted(object_counts.keys())
        style_means = [object_counts[count].get("style_consistency", {}).get("mean", 0) for count in obj_counts]
        style_stds = [object_counts[count].get("style_consistency", {}).get("std", 0) for count in obj_counts]
        
        ax1.errorbar(obj_counts, style_means, yerr=style_stds, marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Number of Objects')
        ax1.set_ylabel('Style Consistency Score')
        ax1.set_title('Style Consistency vs Object Count')
        ax1.grid(True, alpha=0.3)
        
        # CLIP score by object count
        clip_means = [object_counts[count].get("clip_score", {}).get("mean", 0) for count in obj_counts]
        clip_stds = [object_counts[count].get("clip_score", {}).get("std", 0) for count in obj_counts]
        
        ax2.errorbar(obj_counts, clip_means, yerr=clip_stds, marker='s', capsize=5, capthick=2, color='orange')
        ax2.set_xlabel('Number of Objects')
        ax2.set_ylabel('CLIP Score')
        ax2.set_title('CLIP Score vs Object Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "complexity_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_style_content_tradeoff(self, tradeoff_data: Dict[str, Any], save_dir: Path):
        """Plot style vs content trade-off."""
        scatter_data = tradeoff_data.get("scatter_data", [])
        
        if not scatter_data:
            return
        
        style_scores, clip_scores, _ = zip(*scatter_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(style_scores, clip_scores, alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(style_scores, clip_scores, 1)
        p = np.poly1d(z)
        ax.plot(style_scores, p(style_scores), "r--", alpha=0.8)
        
        # Add correlation info
        correlation = tradeoff_data.get("correlation", {})
        if correlation:
            r_value = correlation.get("pearson_r", 0)
            p_value = correlation.get("p_value", 1)
            ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Style Consistency Score')
        ax.set_ylabel('CLIP Score')
        ax.set_title('Style Consistency vs Content Fidelity Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "style_content_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_style_robustness(self, robustness_data: Dict[str, Any], save_dir: Path):
        """Plot style robustness analysis."""
        style_performance = robustness_data.get("style_performance", {})
        
        if not style_performance:
            return
        
        categories = list(style_performance.keys())
        style_means = [style_performance[cat].get("style_consistency", {}).get("mean", 0) for cat in categories]
        clip_means = [style_performance[cat].get("clip_score", {}).get("mean", 0) for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Style consistency by category
        bars1 = ax1.bar(categories, style_means, alpha=0.8)
        ax1.set_ylabel('Style Consistency Score')
        ax1.set_title('Style Consistency by Style Category')
        ax1.tick_params(axis='x', rotation=45)
        
        # CLIP score by category
        bars2 = ax2.bar(categories, clip_means, alpha=0.8, color='orange')
        ax2.set_ylabel('CLIP Score')
        ax2.set_title('CLIP Score by Style Category')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / "style_robustness.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_research_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive research report."""
        print("📝 Generating research report...")
        
        report_lines = []
        report_lines.append("LPA RESEARCH QUESTIONS ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Experiment Directory: {self.experiment_dir}")
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Question 2: Complexity Scaling
        complexity_data = analysis_results.get("complexity_scaling", {})
        if complexity_data:
            report_lines.append("RESEARCH QUESTION 2: COMPLEXITY SCALING")
            report_lines.append("-" * 40)
            
            object_counts = complexity_data.get("object_count", {})
            for obj_count, metrics in object_counts.items():
                report_lines.append(f"\n{obj_count} Objects:")
                for metric, stats in metrics.items():
                    mean_val = stats.get("mean", 0)
                    std_val = stats.get("std", 0)
                    count = stats.get("count", 0)
                    report_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f} (n={count})")
        
        # Question 3: Style vs Content Trade-off
        tradeoff_data = analysis_results.get("style_content_tradeoff", {})
        if tradeoff_data:
            report_lines.append("\n\nRESEARCH QUESTION 3: STYLE VS CONTENT TRADE-OFF")
            report_lines.append("-" * 40)
            
            correlation = tradeoff_data.get("correlation", {})
            if correlation:
                r_value = correlation.get("pearson_r", 0)
                p_value = correlation.get("p_value", 1)
                significant = correlation.get("significant", False)
                
                report_lines.append(f"Correlation (Pearson's r): {r_value:.4f}")
                report_lines.append(f"P-value: {p_value:.4f}")
                report_lines.append(f"Statistically significant: {'Yes' if significant else 'No'}")
                
                if abs(r_value) < 0.1:
                    report_lines.append("Interpretation: No strong trade-off between style and content")
                elif r_value > 0:
                    report_lines.append("Interpretation: Higher style consistency correlates with better content fidelity")
                else:
                    report_lines.append("Interpretation: Trade-off exists - higher style consistency may reduce content fidelity")
        
        # Question 4: Style Robustness
        robustness_data = analysis_results.get("style_robustness", {})
        if robustness_data:
            report_lines.append("\n\nRESEARCH QUESTION 4: STYLE ROBUSTNESS")
            report_lines.append("-" * 40)
            
            style_performance = robustness_data.get("style_performance", {})
            for category, metrics in style_performance.items():
                report_lines.append(f"\n{category.upper()} Styles:")
                for metric, stats in metrics.items():
                    mean_val = stats.get("mean", 0)
                    std_val = stats.get("std", 0)
                    count = stats.get("count", 0)
                    report_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f} (n={count})")
            
            robustness_metrics = robustness_data.get("robustness_metrics", {})
            if robustness_metrics:
                report_lines.append(f"\nRobustness Metrics:")
                report_lines.append(f"  Style consistency variance: {robustness_metrics.get('style_consistency_variance', 0):.6f}")
                report_lines.append(f"  CLIP score variance: {robustness_metrics.get('clip_score_variance', 0):.6f}")
                report_lines.append(f"  Style consistency range: {robustness_metrics.get('style_consistency_range', 0):.4f}")
                report_lines.append(f"  CLIP score range: {robustness_metrics.get('clip_score_range', 0):.4f}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.experiment_dir / "research_questions_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return report_content
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete analysis for all research questions."""
        print("🚀 Starting comprehensive research questions analysis...")
        
        # Load data
        self.load_experiment_data()
        
        # Run analyses
        analysis_results = {
            "complexity_scaling": self.analyze_complexity_scaling(),
            "style_content_tradeoff": self.analyze_style_content_tradeoff(),
            "style_robustness": self.analyze_style_robustness()
        }
        
        # Create visualizations
        self.create_visualizations(analysis_results)
        
        # Generate report
        report = self.generate_research_report(analysis_results)
        
        # Save analysis results
        results_file = self.experiment_dir / "research_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"✅ Analysis complete! Results saved to: {results_file}")
        print("\n" + "="*60)
        print(report)
        
        return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze research questions for LPA paper")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    
    args = parser.parse_args()
    
    analyzer = ResearchQuestionAnalyzer(args.experiment_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 