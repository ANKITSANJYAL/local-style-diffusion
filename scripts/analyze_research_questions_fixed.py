#!/usr/bin/env python3
"""
Fixed Research Questions Analysis for LPA
Properly processes the actual experiment data structure
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_experiment_data(experiment_dir: str) -> List[Dict]:
    """Load experiment results from the correct file"""
    results_file = os.path.join(experiment_dir, "final_results.json")
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return []
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} results from {results_file}")
    return data

def analyze_complexity_scaling(data: List[Dict]) -> Dict:
    """Analyze how performance scales with prompt complexity"""
    print("\n🔍 Analyzing complexity scaling...")
    
    # Group by complexity level
    complexity_groups = defaultdict(list)
    object_count_groups = defaultdict(list)
    
    for result in data:
        complexity = result.get('complexity', 'unknown')
        complexity_groups[complexity].append(result)
        
        # Count objects from parsed prompt
        parsed = result.get('parsed_prompt', {})
        object_tokens = parsed.get('object_tokens', [])
        object_count = len([t for t in object_tokens if not t.startswith('a ') and not t.startswith('an ')])
        object_count_groups[object_count].append(result)
    
    # Analyze by complexity level
    complexity_analysis = {}
    for complexity, results in complexity_groups.items():
        if not results:
            continue
            
        lpa_scores = [r['evaluation_metrics']['lpa']['clip_score'] for r in results]
        lpa_style = [r['evaluation_metrics']['lpa']['style_consistency'] for r in results]
        
        complexity_analysis[complexity] = {
            'count': len(results),
            'clip_score_mean': np.mean(lpa_scores),
            'clip_score_std': np.std(lpa_scores),
            'style_consistency_mean': np.mean(lpa_style),
            'style_consistency_std': np.std(lpa_style)
        }
    
    # Analyze by object count
    object_analysis = {}
    for obj_count, results in object_count_groups.items():
        if not results:
            continue
            
        lpa_scores = [r['evaluation_metrics']['lpa']['clip_score'] for r in results]
        lpa_style = [r['evaluation_metrics']['lpa']['style_consistency'] for r in results]
        
        object_analysis[obj_count] = {
            'count': len(results),
            'clip_score_mean': np.mean(lpa_scores),
            'clip_score_std': np.std(lpa_scores),
            'style_consistency_mean': np.mean(lpa_style),
            'style_consistency_std': np.std(lpa_style)
        }
    
    return {
        'complexity_levels': complexity_analysis,
        'object_count': object_analysis
    }

def analyze_style_content_tradeoff(data: List[Dict]) -> Dict:
    """Analyze the trade-off between style consistency and content fidelity"""
    print("🔍 Analyzing style vs content trade-off...")
    
    # Extract style consistency vs CLIP score pairs
    style_scores = []
    clip_scores = []
    prompts = []
    
    for result in data:
        metrics = result['evaluation_metrics']['lpa']
        style_consistency = metrics['style_consistency']
        clip_score = metrics['clip_score']
        
        style_scores.append(style_consistency)
        clip_scores.append(clip_score)
        prompts.append(result['prompt'])
    
    # Calculate correlation
    correlation = np.corrcoef(style_scores, clip_scores)[0, 1] if len(style_scores) > 1 else 0
    
    # Create scatter data
    scatter_data = list(zip(style_scores, clip_scores, prompts))
    
    return {
        'correlation': correlation,
        'scatter_data': scatter_data,
        'style_scores': style_scores,
        'clip_scores': clip_scores,
        'analysis': {
            'positive_correlation': correlation > 0.1,
            'negative_correlation': correlation < -0.1,
            'no_correlation': abs(correlation) <= 0.1
        }
    }

def analyze_style_robustness(data: List[Dict]) -> Dict:
    """Analyze robustness across different style categories"""
    print("🔍 Analyzing style robustness...")
    
    # Define style categories based on prompt content
    style_categories = {
        'artistic': ['painting', 'artistic', 'oil painting', 'watercolor', 'acrylic', 'canvas'],
        'photographic': ['photographic', 'photo', 'realistic', 'photography', 'camera'],
        'digital': ['digital', 'cyberpunk', 'synthwave', 'neon', 'futuristic', '3d'],
        'traditional': ['traditional', 'classical', 'vintage', 'retro', 'antique']
    }
    
    # Categorize prompts
    categorized_data = defaultdict(list)
    uncategorized = []
    
    for result in data:
        prompt = result['prompt'].lower()
        categorized = False
        
        for category, keywords in style_categories.items():
            if any(keyword in prompt for keyword in keywords):
                categorized_data[category].append(result)
                categorized = True
                break
        
        if not categorized:
            uncategorized.append(result)
    
    # Analyze each category
    style_performance = {}
    for category, results in categorized_data.items():
        if not results:
            continue
            
        style_scores = [r['evaluation_metrics']['lpa']['style_consistency'] for r in results]
        clip_scores = [r['evaluation_metrics']['lpa']['clip_score'] for r in results]
        
        style_performance[category] = {
            'count': len(results),
            'style_consistency_mean': np.mean(style_scores),
            'style_consistency_std': np.std(style_scores),
            'clip_score_mean': np.mean(clip_scores),
            'clip_score_std': np.std(clip_scores),
            'prompts': [r['prompt'] for r in results]
        }
    
    # Calculate overall robustness metrics
    all_style_scores = [r['evaluation_metrics']['lpa']['style_consistency'] for r in data]
    all_clip_scores = [r['evaluation_metrics']['lpa']['clip_score'] for r in data]
    
    robustness_metrics = {
        'style_consistency_variance': np.var(all_style_scores),
        'clip_score_variance': np.var(all_clip_scores),
        'style_consistency_range': max(all_style_scores) - min(all_style_scores),
        'clip_score_range': max(all_clip_scores) - min(all_clip_scores)
    }
    
    return {
        'style_categories': categorized_data,
        'style_performance': style_performance,
        'uncategorized_count': len(uncategorized),
        'robustness_metrics': robustness_metrics
    }

def generate_visualizations(analysis_results: Dict, output_dir: str):
    """Generate visualization plots"""
    print("📊 Generating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Complexity scaling plot
    complexity_data = analysis_results['complexity_scaling']['complexity_levels']
    if complexity_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        complexities = list(complexity_data.keys())
        clip_means = [complexity_data[c]['clip_score_mean'] for c in complexities]
        style_means = [complexity_data[c]['style_consistency_mean'] for c in complexities]
        
        ax1.bar(complexities, clip_means, alpha=0.7)
        ax1.set_title('CLIP Score by Complexity')
        ax1.set_ylabel('CLIP Score')
        
        ax2.bar(complexities, style_means, alpha=0.7, color='orange')
        ax2.set_title('Style Consistency by Complexity')
        ax2.set_ylabel('Style Consistency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'complexity_scaling.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Style vs Content trade-off
    tradeoff_data = analysis_results['style_content_tradeoff']
    if tradeoff_data['scatter_data']:
        plt.figure(figsize=(10, 6))
        style_scores, clip_scores, _ = zip(*tradeoff_data['scatter_data'])
        
        plt.scatter(style_scores, clip_scores, alpha=0.6)
        plt.xlabel('Style Consistency')
        plt.ylabel('CLIP Score')
        plt.title(f'Style vs Content Trade-off (r={tradeoff_data["correlation"]:.3f})')
        
        # Add trend line
        z = np.polyfit(style_scores, clip_scores, 1)
        p = np.poly1d(z)
        plt.plot(style_scores, p(style_scores), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'style_content_tradeoff.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Style robustness
    style_data = analysis_results['style_robustness']['style_performance']
    if style_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        categories = list(style_data.keys())
        style_means = [style_data[c]['style_consistency_mean'] for c in categories]
        clip_means = [style_data[c]['clip_score_mean'] for c in categories]
        
        ax1.bar(categories, style_means, alpha=0.7, color='green')
        ax1.set_title('Style Consistency by Category')
        ax1.set_ylabel('Style Consistency')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(categories, clip_means, alpha=0.7, color='purple')
        ax2.set_title('CLIP Score by Category')
        ax2.set_ylabel('CLIP Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'style_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(analysis_results: Dict, experiment_dir: str, output_dir: str):
    """Generate comprehensive analysis report"""
    print("📝 Generating analysis report...")
    
    report_lines = [
        "LPA RESEARCH QUESTIONS ANALYSIS REPORT (FIXED)",
        "=" * 60,
        f"Experiment Directory: {experiment_dir}",
        f"Analysis Date: {os.popen('date').read().strip()}",
        "",
        "RESEARCH QUESTION 1: COMPLEXITY SCALING",
        "-" * 40,
        ""
    ]
    
    # Complexity scaling
    complexity_data = analysis_results['complexity_scaling']['complexity_levels']
    for complexity, stats in complexity_data.items():
        report_lines.extend([
            f"{complexity.upper()} Complexity ({stats['count']} prompts):",
            f"  CLIP Score: {stats['clip_score_mean']:.4f} ± {stats['clip_score_std']:.4f}",
            f"  Style Consistency: {stats['style_consistency_mean']:.4f} ± {stats['style_consistency_std']:.4f}",
            ""
        ])
    
    # Object count analysis
    object_data = analysis_results['complexity_scaling']['object_count']
    report_lines.extend([
        "Object Count Analysis:",
        "-" * 20
    ])
    for obj_count, stats in object_data.items():
        report_lines.extend([
            f"{obj_count} Objects ({stats['count']} prompts):",
            f"  CLIP Score: {stats['clip_score_mean']:.4f} ± {stats['clip_score_std']:.4f}",
            f"  Style Consistency: {stats['style_consistency_mean']:.4f} ± {stats['style_consistency_std']:.4f}",
            ""
        ])
    
    # Style vs Content trade-off
    tradeoff_data = analysis_results['style_content_tradeoff']
    report_lines.extend([
        "RESEARCH QUESTION 2: STYLE VS CONTENT TRADE-OFF",
        "-" * 40,
        f"Correlation coefficient: {tradeoff_data['correlation']:.4f}",
        ""
    ])
    
    if tradeoff_data['correlation'] > 0.1:
        report_lines.append("✅ Positive correlation: Better style consistency correlates with better content fidelity")
    elif tradeoff_data['correlation'] < -0.1:
        report_lines.append("⚠️  Negative correlation: Trade-off between style and content")
    else:
        report_lines.append("➖ No significant correlation: Style and content are independent")
    
    report_lines.extend([
        "",
        "RESEARCH QUESTION 3: STYLE ROBUSTNESS",
        "-" * 40,
        ""
    ])
    
    # Style robustness
    style_data = analysis_results['style_robustness']['style_performance']
    for category, stats in style_data.items():
        report_lines.extend([
            f"{category.upper()} Styles ({stats['count']} prompts):",
            f"  Style Consistency: {stats['style_consistency_mean']:.4f} ± {stats['style_consistency_std']:.4f}",
            f"  CLIP Score: {stats['clip_score_mean']:.4f} ± {stats['clip_score_std']:.4f}",
            ""
        ])
    
    # Robustness metrics
    robustness = analysis_results['style_robustness']['robustness_metrics']
    report_lines.extend([
        "Overall Robustness Metrics:",
        f"  Style consistency variance: {robustness['style_consistency_variance']:.6f}",
        f"  CLIP score variance: {robustness['clip_score_variance']:.6f}",
        f"  Style consistency range: {robustness['style_consistency_range']:.4f}",
        f"  CLIP score range: {robustness['clip_score_range']:.4f}",
        ""
    ])
    
    # Write report
    report_path = os.path.join(output_dir, 'research_questions_report_fixed.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Report saved to: {report_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_research_questions_fixed.py <experiment_dir>")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    if not os.path.exists(experiment_dir):
        print(f"❌ Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    print(f"🔍 Analyzing research questions for: {experiment_dir}")
    
    # Load data
    data = load_experiment_data(experiment_dir)
    if not data:
        print("❌ No data found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(experiment_dir, "research_analysis_fixed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analyses
    analysis_results = {
        'complexity_scaling': analyze_complexity_scaling(data),
        'style_content_tradeoff': analyze_style_content_tradeoff(data),
        'style_robustness': analyze_style_robustness(data)
    }
    
    # Save results (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, defaultdict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Debug: print analysis results structure
    print("Debug: Analysis results keys:", list(analysis_results.keys()))
    for key, value in analysis_results.items():
        print(f"Debug: {key} type: {type(value)}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {type(subvalue)}")
    
    analysis_results_serializable = convert_numpy_types(analysis_results)
    results_path = os.path.join(output_dir, 'research_analysis_results_fixed.json')
    with open(results_path, 'w') as f:
        json.dump(analysis_results_serializable, f, indent=2)
    
    print(f"✅ Analysis results saved to: {results_path}")
    
    # Generate visualizations
    generate_visualizations(analysis_results, output_dir)
    
    # Generate report
    generate_report(analysis_results, experiment_dir, output_dir)
    
    print("\n🎉 Research questions analysis completed!")
    print(f"📁 Results in: {output_dir}")

if __name__ == "__main__":
    main() 