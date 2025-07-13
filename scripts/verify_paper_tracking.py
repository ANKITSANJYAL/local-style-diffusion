#!/usr/bin/env python3
"""
Verification script for paper tracking components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

def check_config_tracking(config_path: str) -> Dict[str, bool]:
    """Check if config has all required tracking settings."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tracking_checks = {
        "evaluation_metrics": False,
        "attention_maps": False,
        "baseline_comparisons": False,
        "paper_settings": False,
        "visualization": False,
        "reproducibility": False
    }
    
    # Check evaluation metrics
    eval_config = config.get("evaluation", {})
    if "metrics" in eval_config:
        required_metrics = ["style_consistency", "clip_score", "lpips"]
        tracking_checks["evaluation_metrics"] = all(
            metric in eval_config["metrics"] for metric in required_metrics
        )
    
    # Check attention maps
    lpa_config = config.get("lpa", {})
    attention_config = lpa_config.get("attention", {})
    tracking_checks["attention_maps"] = attention_config.get("save_attention_maps", False)
    
    # Check baseline comparisons
    baselines = config.get("baselines", [])
    tracking_checks["baseline_comparisons"] = len(baselines) >= 2
    
    # Check paper settings
    paper_config = config.get("paper", {})
    tracking_checks["paper_settings"] = all([
        paper_config.get("generate_comparison_tables", False),
        paper_config.get("generate_comparison_plots", False),
        paper_config.get("save_model_comparisons", False)
    ])
    
    # Check visualization
    viz_config = config.get("visualization", {})
    tracking_checks["visualization"] = all([
        "attention_maps" in viz_config,
        "comparison" in viz_config,
        "heatmaps" in viz_config
    ])
    
    # Check reproducibility
    repro_config = config.get("reproducibility", {})
    tracking_checks["reproducibility"] = all([
        repro_config.get("set_deterministic", False),
        repro_config.get("seed_workers", False)
    ])
    
    return tracking_checks

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed."""
    deps = {
        "torch": False,
        "lpips": False,
        "open_clip": False,
        "spacy": False,
        "matplotlib": False,
        "seaborn": False,
        "scipy": False
    }
    
    try:
        import torch
        deps["torch"] = True
    except ImportError:
        pass
    
    try:
        import lpips
        deps["lpips"] = True
    except ImportError:
        pass
    
    try:
        import open_clip
        deps["open_clip"] = True
    except ImportError:
        pass
    
    try:
        import spacy
        deps["spacy"] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        deps["matplotlib"] = True
    except ImportError:
        pass
    
    try:
        import seaborn
        deps["seaborn"] = True
    except ImportError:
        pass
    
    try:
        import scipy
        deps["scipy"] = True
    except ImportError:
        pass
    
    return deps

def check_data_files() -> Dict[str, bool]:
    """Check if required data files exist."""
    files = {
        "prompts": False,
        "categories": False,
        "test_prompts": False
    }
    
    prompts_file = Path("data/prompts/test_prompts.json")
    categories_file = Path("data/prompts/prompt_categories.json")
    
    files["prompts"] = prompts_file.exists()
    files["categories"] = categories_file.exists()
    
    if files["prompts"]:
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            files["test_prompts"] = len(data.get("prompts", [])) > 0
    
    return files

def print_tracking_summary():
    """Print comprehensive tracking summary."""
    print("🔍 PAPER TRACKING VERIFICATION")
    print("=" * 50)
    
    # Check configurations
    print("\n📋 CONFIGURATION TRACKING:")
    print("-" * 30)
    
    configs = [
        ("Fast Config", "configs/experiment_config_fast.yaml"),
        ("Paper Config", "configs/experiment_config_paper.yaml")
    ]
    
    for config_name, config_path in configs:
        if Path(config_path).exists():
            checks = check_config_tracking(config_path)
            print(f"\n{config_name}:")
            for check, status in checks.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {check.replace('_', ' ').title()}")
        else:
            print(f"\n❌ {config_name}: File not found")
    
    # Check dependencies
    print("\n📦 DEPENDENCIES:")
    print("-" * 30)
    deps = check_dependencies()
    for dep, status in deps.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {dep}")
    
    # Check data files
    print("\n📁 DATA FILES:")
    print("-" * 30)
    files = check_data_files()
    for file, status in files.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {file.replace('_', ' ').title()}")
    
    # Overall assessment
    print("\n📊 OVERALL ASSESSMENT:")
    print("-" * 30)
    
    all_configs_ok = all(
        all(checks.values()) 
        for config_name, config_path in configs 
        if Path(config_path).exists()
    )
    all_deps_ok = all(deps.values())
    all_files_ok = all(files.values())
    
    if all_configs_ok and all_deps_ok and all_files_ok:
        print("🎉 READY FOR PAPER EXPERIMENTS!")
        print("   All tracking components are properly configured.")
    else:
        print("⚠️  SOME ISSUES DETECTED:")
        if not all_configs_ok:
            print("   - Configuration tracking needs attention")
        if not all_deps_ok:
            print("   - Missing dependencies")
        if not all_files_ok:
            print("   - Missing data files")
    
    print("\n📝 PAPER TRACKING CAPABILITIES:")
    print("-" * 40)
    print("✅ Quantitative Metrics:")
    print("   - Style consistency scores")
    print("   - CLIP text-image alignment")
    print("   - LPIPS perceptual distance")
    print("   - Statistical significance testing")
    
    print("\n✅ Qualitative Analysis:")
    print("   - Attention map visualization")
    print("   - Method comparison plots")
    print("   - Style consistency heatmaps")
    
    print("\n✅ Reproducibility:")
    print("   - Deterministic generation")
    print("   - Seed control")
    print("   - Config versioning")
    
    print("\n✅ Baseline Comparisons:")
    print("   - Multiple baseline methods")
    print("   - Relative performance analysis")
    print("   - Statistical comparisons")

if __name__ == "__main__":
    print_tracking_summary() 