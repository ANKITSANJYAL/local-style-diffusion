#!/usr/bin/env python3
"""
Script to run ablation studies for LPA injection order analysis.
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import sys
import time

class AblationStudyRunner:
    """
    Run ablation studies for LPA injection order analysis.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize ablation study runner.
        
        Args:
            config_file: Path to ablation config file
        """
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load ablation study configuration."""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_variant_experiment(self, variant_name: str, variant_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run experiment for a specific variant.
        
        Args:
            variant_name: Name of the variant
            variant_config: Configuration for the variant
            
        Returns:
            Experiment results
        """
        print(f"🔬 Running experiment for variant: {variant_name}")
        print(f"   Description: {variant_config.get('description', 'N/A')}")
        
        # Create temporary config for this variant
        temp_config = self.config.copy()
        temp_config['lpa'] = variant_config
        
        # Save temporary config
        temp_config_file = Path(f"temp_ablation_{variant_name}.yaml")
        with open(temp_config_file, 'w') as f:
            yaml.dump(temp_config, f)
        
        try:
            # Run experiment
            cmd = [
                sys.executable, "scripts/run_experiment.py",
                "--config", str(temp_config_file),
                "--output-dir", f"experiments/ablation_{variant_name}",
                "--max-prompts", str(self.config['dataset']['num_prompts'])
            ]
            
            print(f"   Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Experiment completed successfully")
                
                # Load results
                results_file = Path(f"experiments/ablation_{variant_name}/final_results.json")
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        variant_results = json.load(f)
                    
                    # Extract key metrics
                    metrics = {
                        "variant_name": variant_name,
                        "description": variant_config.get("description", ""),
                        "config": variant_config,
                        "results": variant_results
                    }
                    
                    return metrics
                else:
                    print(f"   ⚠️  Results file not found: {results_file}")
                    return {"variant_name": variant_name, "error": "Results file not found"}
            else:
                print(f"   ❌ Experiment failed")
                print(f"   Error: {result.stderr}")
                return {"variant_name": variant_name, "error": result.stderr}
                
        finally:
            # Clean up temporary config
            if temp_config_file.exists():
                temp_config_file.unlink()
    
    def run_all_ablation_studies(self) -> Dict[str, Any]:
        """Run all ablation study variants."""
        print("🚀 Starting ablation studies...")
        print(f"Config file: {self.config_file}")
        print(f"Number of variants: {len(self.config['lpa_variants'])}")
        print()
        
        all_results = {
            "ablation_study_id": f"ablation_{int(time.time())}",
            "config_file": str(self.config_file),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "variants": {},
            "summary": {}
        }
        
        # Run each variant
        for variant_name, variant_config in self.config['lpa_variants'].items():
            print(f"📊 Variant {list(self.config['lpa_variants'].keys()).index(variant_name) + 1}/{len(self.config['lpa_variants'])}")
            
            variant_results = self.run_variant_experiment(variant_name, variant_config)
            all_results["variants"][variant_name] = variant_results
            
            print(f"   Results: {variant_results.get('error', 'Success')}")
            print()
            
            # Small delay between experiments
            time.sleep(2)
        
        # Generate summary
        all_results["summary"] = self.generate_ablation_summary(all_results["variants"])
        
        # Save all results
        self.save_ablation_results(all_results)
        
        return all_results
    
    def generate_ablation_summary(self, variants_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of ablation study results."""
        print("📝 Generating ablation study summary...")
        
        summary = {
            "total_variants": len(variants_results),
            "successful_variants": 0,
            "failed_variants": 0,
            "metrics_comparison": {},
            "best_variant": None,
            "worst_variant": None
        }
        
        # Collect metrics for comparison
        metrics_data = {}
        
        for variant_name, variant_data in variants_results.items():
            if "error" not in variant_data:
                summary["successful_variants"] += 1
                
                # Extract metrics
                results = variant_data.get("results", {})
                if "methods" in results and "lpa" in results["methods"]:
                    lpa_metrics = results["methods"]["lpa"]
                    
                    metrics_data[variant_name] = {
                        "style_consistency": lpa_metrics.get("style_consistency", {}).get("mean", 0),
                        "clip_score": lpa_metrics.get("clip_score", {}).get("mean", 0),
                        "lpips": lpa_metrics.get("lpips", {}).get("mean", 0)
                    }
            else:
                summary["failed_variants"] += 1
        
        # Find best and worst variants
        if metrics_data:
            # Best by style consistency
            best_style = max(metrics_data.items(), key=lambda x: x[1]["style_consistency"])
            worst_style = min(metrics_data.items(), key=lambda x: x[1]["style_consistency"])
            
            # Best by CLIP score
            best_clip = max(metrics_data.items(), key=lambda x: x[1]["clip_score"])
            worst_clip = min(metrics_data.items(), key=lambda x: x[1]["clip_score"])
            
            summary["metrics_comparison"] = {
                "style_consistency": {
                    "best": {"variant": best_style[0], "score": best_style[1]["style_consistency"]},
                    "worst": {"variant": worst_style[0], "score": worst_style[1]["style_consistency"]},
                    "range": best_style[1]["style_consistency"] - worst_style[1]["style_consistency"]
                },
                "clip_score": {
                    "best": {"variant": best_clip[0], "score": best_clip[1]["clip_score"]},
                    "worst": {"variant": worst_clip[0], "score": worst_clip[1]["clip_score"]},
                    "range": best_clip[1]["clip_score"] - worst_clip[1]["clip_score"]
                }
            }
            
            # Overall best (balanced score)
            balanced_scores = {}
            for variant, metrics in metrics_data.items():
                # Normalize scores (assuming 0-1 range) and combine
                style_norm = metrics["style_consistency"]
                clip_norm = metrics["clip_score"]
                balanced_score = (style_norm + clip_norm) / 2
                balanced_scores[variant] = balanced_score
            
            best_overall = max(balanced_scores.items(), key=lambda x: x[1])
            worst_overall = min(balanced_scores.items(), key=lambda x: x[1])
            
            summary["best_variant"] = {
                "name": best_overall[0],
                "balanced_score": best_overall[1],
                "metrics": metrics_data[best_overall[0]]
            }
            
            summary["worst_variant"] = {
                "name": worst_overall[0],
                "balanced_score": worst_overall[1],
                "metrics": metrics_data[worst_overall[0]]
            }
        
        return summary
    
    def save_ablation_results(self, all_results: Dict[str, Any]):
        """Save ablation study results."""
        print("💾 Saving ablation study results...")
        
        # Create ablation results directory
        ablation_dir = Path("experiments/ablation_studies")
        ablation_dir.mkdir(exist_ok=True)
        
        # Save complete results
        results_file = ablation_dir / f"ablation_results_{all_results['ablation_study_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save summary
        summary_file = ablation_dir / f"ablation_summary_{all_results['ablation_study_id']}.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results["summary"], f, indent=2)
        
        # Generate ablation report
        report = self.generate_ablation_report(all_results)
        report_file = ablation_dir / f"ablation_report_{all_results['ablation_study_id']}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"✅ Ablation results saved to: {ablation_dir}")
        print(f"   Complete results: {results_file}")
        print(f"   Summary: {summary_file}")
        print(f"   Report: {report_file}")
    
    def generate_ablation_report(self, all_results: Dict[str, Any]) -> str:
        """Generate ablation study report."""
        print("📄 Generating ablation study report...")
        
        report_lines = []
        report_lines.append("LPA ABLATION STUDY REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Study ID: {all_results['ablation_study_id']}")
        report_lines.append(f"Timestamp: {all_results['timestamp']}")
        report_lines.append(f"Config File: {all_results['config_file']}")
        report_lines.append("")
        
        summary = all_results["summary"]
        report_lines.append("EXECUTION SUMMARY:")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Variants: {summary['total_variants']}")
        report_lines.append(f"Successful: {summary['successful_variants']}")
        report_lines.append(f"Failed: {summary['failed_variants']}")
        report_lines.append("")
        
        # Variant results
        report_lines.append("VARIANT RESULTS:")
        report_lines.append("-" * 20)
        
        for variant_name, variant_data in all_results["variants"].items():
            report_lines.append(f"\n{variant_name.upper()}:")
            
            if "error" in variant_data:
                report_lines.append(f"  Status: ❌ Failed")
                report_lines.append(f"  Error: {variant_data['error']}")
            else:
                report_lines.append(f"  Status: ✅ Success")
                report_lines.append(f"  Description: {variant_data.get('description', 'N/A')}")
                
                # Extract metrics
                results = variant_data.get("results", {})
                if "methods" in results and "lpa" in results["methods"]:
                    lpa_metrics = results["methods"]["lpa"]
                    style_score = lpa_metrics.get("style_consistency", {}).get("mean", 0)
                    clip_score = lpa_metrics.get("clip_score", {}).get("mean", 0)
                    lpips_score = lpa_metrics.get("lpips", {}).get("mean", 0)
                    
                    report_lines.append(f"  Style Consistency: {style_score:.4f}")
                    report_lines.append(f"  CLIP Score: {clip_score:.4f}")
                    report_lines.append(f"  LPIPS: {lpips_score:.4f}")
        
        # Key findings
        if summary["metrics_comparison"]:
            report_lines.append("\n\nKEY FINDINGS:")
            report_lines.append("-" * 20)
            
            # Style consistency findings
            style_comp = summary["metrics_comparison"]["style_consistency"]
            report_lines.append(f"Style Consistency:")
            report_lines.append(f"  Best: {style_comp['best']['variant']} ({style_comp['best']['score']:.4f})")
            report_lines.append(f"  Worst: {style_comp['worst']['variant']} ({style_comp['worst']['score']:.4f})")
            report_lines.append(f"  Range: {style_comp['range']:.4f}")
            
            # CLIP score findings
            clip_comp = summary["metrics_comparison"]["clip_score"]
            report_lines.append(f"\nCLIP Score:")
            report_lines.append(f"  Best: {clip_comp['best']['variant']} ({clip_comp['best']['score']:.4f})")
            report_lines.append(f"  Worst: {clip_comp['worst']['variant']} ({clip_comp['worst']['score']:.4f})")
            report_lines.append(f"  Range: {clip_comp['range']:.4f}")
            
            # Overall best
            if summary["best_variant"]:
                report_lines.append(f"\nOverall Best Variant: {summary['best_variant']['name']}")
                report_lines.append(f"  Balanced Score: {summary['best_variant']['balanced_score']:.4f}")
                report_lines.append(f"  Style Consistency: {summary['best_variant']['metrics']['style_consistency']:.4f}")
                report_lines.append(f"  CLIP Score: {summary['best_variant']['metrics']['clip_score']:.4f}")
        
        # Research question insights
        report_lines.append("\n\nRESEARCH QUESTION INSIGHTS:")
        report_lines.append("-" * 30)
        report_lines.append("Q1: Does injection order matter?")
        
        if summary["metrics_comparison"]:
            style_range = summary["metrics_comparison"]["style_consistency"]["range"]
            clip_range = summary["metrics_comparison"]["clip_score"]["range"]
            
            if style_range > 0.05 or clip_range > 0.05:
                report_lines.append("  ✅ YES - Injection order significantly affects performance")
                report_lines.append(f"  Style consistency varies by {style_range:.4f}")
                report_lines.append(f"  CLIP score varies by {clip_range:.4f}")
            else:
                report_lines.append("  ❌ NO - Injection order has minimal impact")
                report_lines.append("  All variants perform similarly")
        else:
            report_lines.append("  ⚠️  Insufficient data to determine")
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS:")
        report_lines.append("-" * 20)
        if summary["best_variant"]:
            report_lines.append(f"• Use {summary['best_variant']['name']} as the optimal configuration")
            report_lines.append(f"• This variant achieves the best balance of style and content")
        
        if summary["failed_variants"] > 0:
            report_lines.append(f"• Investigate {summary['failed_variants']} failed variants")
            report_lines.append("• Consider adjusting parameters for failed variants")
        
        report_lines.append("• Run additional experiments with different injection schedules")
        report_lines.append("• Conduct user studies to validate findings")
        
        return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for LPA injection order analysis")
    parser.add_argument("config_file", help="Path to ablation study config file")
    
    args = parser.parse_args()
    
    runner = AblationStudyRunner(args.config_file)
    results = runner.run_all_ablation_studies()
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE!")
    print("="*60)
    print(f"Study ID: {results['ablation_study_id']}")
    print(f"Successful variants: {results['summary']['successful_variants']}/{results['summary']['total_variants']}")
    
    if results['summary']['best_variant']:
        print(f"Best variant: {results['summary']['best_variant']['name']}")
        print(f"Balanced score: {results['summary']['best_variant']['balanced_score']:.4f}")

if __name__ == "__main__":
    main() 