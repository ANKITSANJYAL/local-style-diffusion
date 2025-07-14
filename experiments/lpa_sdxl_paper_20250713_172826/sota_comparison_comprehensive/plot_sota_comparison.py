import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
lpa_results_path = os.path.abspath(os.path.join(base_dir, '..', 'final_results.json'))

# 1. Load LPA (our model) results
with open(lpa_results_path, 'r') as f:
    lpa_data = json.load(f)
lpa_clip_scores = [r['evaluation_metrics']['lpa']['clip_score'] for r in lpa_data]
lpa_style_scores = [r['evaluation_metrics']['lpa']['style_consistency'] for r in lpa_data]

methods = ['LPA']
clip_means = [np.mean(lpa_clip_scores)]
clip_stds = [np.std(lpa_clip_scores)]
style_means = [np.mean(lpa_style_scores)]
style_stds = [np.std(lpa_style_scores)]

# 2. Load SOTA results
for fname in os.listdir(base_dir):
    if fname.endswith('_results.json'):
        method = fname.replace('_results.json', '').replace('_', ' ').title()
        with open(os.path.join(base_dir, fname), 'r') as f:
            data = json.load(f)
        # Handle list-of-dict format
        if isinstance(data, list):
            # LoRA and similar: only clip_score present
            clip_scores = [r.get('clip_score', r.get('clip_mean')) for r in data if isinstance(r, dict) and (r.get('clip_score') is not None or r.get('clip_mean') is not None)]
            style_scores = [r.get('style_consistency', r.get('style_mean')) for r in data if isinstance(r, dict) and (r.get('style_consistency') is not None or r.get('style_mean') is not None)]
            # If only one dict with means, use those
            if len(data) == 1 and isinstance(data[0], dict) and ('clip_mean' in data[0] or 'style_mean' in data[0]):
                clip_scores = [data[0].get('clip_mean')]
                style_scores = [data[0].get('style_mean')]
        elif isinstance(data, dict):
            clip_scores = data.get('clip_scores')
            if clip_scores is None and 'clip_mean' in data:
                clip_scores = [data['clip_mean']]
            style_scores = data.get('style_scores')
            if style_scores is None and 'style_mean' in data:
                style_scores = [data['style_mean']]
        else:
            continue
        # Remove None values
        clip_scores = [float(x) for x in clip_scores if x is not None]
        style_scores = [float(x) for x in style_scores if x is not None]
        if not clip_scores:
            continue
        # If style_scores is empty, set to 0.0 and warn
        if not style_scores:
            print(f"[WARN] No style consistency scores for {method}, setting to 0.0 for plot.")
            style_scores = [0.0 for _ in clip_scores]
        methods.append(method)
        clip_means.append(float(np.mean(clip_scores)))
        clip_stds.append(float(np.std(clip_scores)))
        style_means.append(float(np.mean(style_scores)))
        style_stds.append(float(np.std(style_scores)))

# 3. Plot CLIP score comparison
plt.figure(figsize=(10, 6))
plt.bar(methods, clip_means, yerr=clip_stds, capsize=5, color='skyblue')
plt.ylabel('CLIP Score')
plt.title('CLIP Score Comparison (Mean ± Std)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'clip_score_comparison.png'))
plt.close()

# 4. Plot Style Consistency comparison
plt.figure(figsize=(10, 6))
plt.bar(methods, style_means, yerr=style_stds, capsize=5, color='lightgreen')
plt.ylabel('Style Consistency')
plt.title('Style Consistency Comparison (Mean ± Std)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'style_consistency_comparison.png'))
plt.close()

print('Saved: clip_score_comparison.png, style_consistency_comparison.png') 