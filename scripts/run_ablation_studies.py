import os
import sys
import yaml
import json
import copy
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.experiments.run_experiments import ExperimentRunner

# 1. Define ablation variants (edit/add as needed)
ABLATION_VARIANTS = [
    {
        "name": "injection_early",
        "description": "Inject style at early step (step 1)",
        "lpa": {"injection_step": 1, "injection_order": "early"}
    },
    {
        "name": "injection_middle",
        "description": "Inject style at middle step (step 10)",
        "lpa": {"injection_step": 10, "injection_order": "middle"}
    },
    {
        "name": "injection_late",
        "description": "Inject style at late step (step 25)",
        "lpa": {"injection_step": 25, "injection_order": "late"}
    },
    {
        "name": "injection_random",
        "description": "Inject style at random step",
        "lpa": {"injection_step": "random", "injection_order": "random"}
    },
    {
        "name": "no_injection",
        "description": "No style injection (baseline)",
        "lpa": {"injection_step": None, "injection_order": None, "disable_style": True}
    },
    {
        "name": "full_style_parsing",
        "description": "Full style prompt parsing",
        "lpa": {"style_parsing": "full"}
    },
    {
        "name": "partial_style_parsing",
        "description": "Partial style prompt parsing",
        "lpa": {"style_parsing": "partial"}
    },
    {
        "name": "random_style_parsing",
        "description": "Random style prompt parsing",
        "lpa": {"style_parsing": "random"}
    },
    {
        "name": "no_style_parsing",
        "description": "No style prompt parsing",
        "lpa": {"style_parsing": None}
    },
    {
        "name": "low_guidance",
        "description": "Low guidance scale (trade-off)",
        "generation": {"guidance_scale": 3.5}
    },
    {
        "name": "high_guidance",
        "description": "High guidance scale (trade-off)",
        "generation": {"guidance_scale": 12.0}
    },
]

# 2. Load base config (edit path as needed)
BASE_CONFIG_PATH = "configs/experiment_config_paper.yaml"
assert os.path.exists(BASE_CONFIG_PATH), f"Base config not found: {BASE_CONFIG_PATH}"
with open(BASE_CONFIG_PATH, 'r') as f:
    base_config = yaml.safe_load(f)

# 3. Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("experiments/ablation_studies_" + timestamp)
output_dir.mkdir(parents=True, exist_ok=True)

# 4. Run ablation experiments
summary = []
for variant in ABLATION_VARIANTS:
    print(f"\n=== Running ablation: {variant['name']} ===")
    # Deep copy base config and update with ablation settings
    config = copy.deepcopy(base_config)
    config['experiment']['name'] = f"ablation_{variant['name']}"
    config['experiment']['description'] = variant['description']
    # Update LPA and generation settings
    if 'lpa' in variant:
        config['lpa'].update(variant['lpa'])
    if 'generation' in variant:
        config['generation'].update(variant['generation'])
    # Save config
    config_path = output_dir / f"config_{variant['name']}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    # Run experiment
    runner = ExperimentRunner(str(config_path))
    try:
        runner.run_full_experiment(seed=config.get('experiment', {}).get('seed', 42))
        # Load results
        results_file = runner.experiment_dir / "final_results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)
        clip_scores = [r['evaluation_metrics']['lpa']['clip_score'] for r in results]
        style_scores = [r['evaluation_metrics']['lpa']['style_consistency'] for r in results]
        summary.append({
            'variant': variant['name'],
            'description': variant['description'],
            'clip_mean': float(np.mean(clip_scores)),
            'clip_std': float(np.std(clip_scores)),
            'style_mean': float(np.mean(style_scores)),
            'style_std': float(np.std(style_scores)),
            'results_dir': str(runner.experiment_dir)
        })
    finally:
        runner.cleanup()

# 5. Save summary table
summary_path = output_dir / "ablation_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

# 6. Save markdown summary for paper
md_path = output_dir / "ablation_summary.md"
with open(md_path, 'w') as f:
    f.write("# LPA Ablation Study Summary\n\n")
    f.write("| Variant | Description | CLIP Mean | CLIP Std | Style Mean | Style Std |\n")
    f.write("|---------|-------------|-----------|----------|------------|-----------|\n")
    for row in summary:
        f.write(f"| {row['variant']} | {row['description']} | {row['clip_mean']:.4f} | {row['clip_std']:.4f} | {row['style_mean']:.4f} | {row['style_std']:.4f} |\n")

def plot_bar(summary, metric, ylabel, out_path):
    names = [row['variant'] for row in summary]
    means = [row[f'{metric}_mean'] for row in summary]
    stds = [row[f'{metric}_std'] for row in summary]
    plt.figure(figsize=(10, 5))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# 7. Generate and save plots
plot_bar(summary, 'clip', 'CLIP Score', output_dir / 'ablation_clip_scores.png')
plot_bar(summary, 'style', 'Style Consistency', output_dir / 'ablation_style_consistency.png')

# 8. Create image grid for a fixed prompt (e.g., first prompt)
try:
    prompt_id = None
    # Find a prompt id present in all results
    for row in summary:
        results_file = Path(row['results_dir']) / 'final_results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)
        if results:
            prompt_id = results[0]['prompt_id'] if 'prompt_id' in results[0] else results[0].get('id')
            break
    if prompt_id:
        images = []
        labels = []
        for row in summary:
            results_file = Path(row['results_dir']) / 'final_results.json'
            with open(results_file, 'r') as f:
                results = json.load(f)
            for r in results:
                pid = r.get('prompt_id', r.get('id'))
                if pid == prompt_id:
                    img_path = r['image_path'] if 'image_path' in r else r.get('img_path')
                    if img_path and os.path.exists(img_path):
                        images.append(Image.open(img_path).resize((256, 256)))
                        labels.append(row['variant'])
                    break
        if images:
            grid_width = len(images)
            grid_img = Image.new('RGB', (256 * grid_width, 256))
            for i, img in enumerate(images):
                grid_img.paste(img, (i * 256, 0))
            grid_img.save(output_dir / 'ablation_image_grid.png')
            print(f"Image grid saved: {output_dir / 'ablation_image_grid.png'}")
except Exception as e:
    print(f"[WARN] Could not create image grid: {e}")

print(f"\n✅ Ablation studies complete. Results, plots, and grids in: {output_dir}\n") 