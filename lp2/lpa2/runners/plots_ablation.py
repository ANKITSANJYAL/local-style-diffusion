
#!/usr/bin/env python3
"""
Plot ablations for LPA paper from a single CSV.

Usage:
  python3 plots_ablation.py \
      --csv artifacts/runs/ablation_summary.csv \
      --outdir artifacts/plots

This produces:
  - abl_heatmap_clip_prompt_mean_over_parsers.(png|pdf)
  - abl_heatmap_clip_style_mean_over_parsers.(png|pdf)
  - abl_heatmap_clip_prompt_mean_parser=<parser>.(png|pdf)    # one per parser
  - abl_heatmap_clip_style_mean_parser=<parser>.(png|pdf)     # one per parser
  - abl_lines_windows_clip_prompt_mean.(png|pdf)               # window sweep per preset
  - abl_lines_windows_clip_style_mean.(png|pdf)
  - abl_groupedbars_parser_effect_clip_prompt_mean_(window=...).(png|pdf)
  - abl_groupedbars_parser_effect_clip_style_mean_(window=...).(png|pdf)
  - ablation_best_by_preset_window.csv                         # helper table

Notes:
- No seaborn. Pure matplotlib.
- One chart per figure (no subplots).
- No explicit color settings.
"""

import argparse
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

METRICS = [
    ("clip_prompt_mean", "CLIP-Prompt (↑)"),
    ("clip_style_mean", "Style-CLIP (↑)"),
]

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_window_str(w: str):
    # window strings like "300-650" -> (300, 650, midpoint)
    a, b = w.split("-")
    a, b = int(a), int(b)
    mid = 0.5 * (a + b)
    return a, b, mid

def ordered_windows(unique_windows):
    # Order windows by their midpoint; return list[str]
    ws = []
    for w in unique_windows:
        a, b, mid = parse_window_str(w)
        ws.append((w, mid))
    ws.sort(key=lambda x: x[1])
    return [w for w, _ in ws]

def df_read(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity
    required_cols = {"run_dir","preset","window","parser",
                     "clip_prompt_mean","clip_prompt_std",
                     "clip_style_mean","clip_style_std",
                     "div_lpips_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def agg_over_parsers(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Average metric across parsers for each preset × window
    g = df.groupby(["preset","window"], as_index=False)[metric].mean()
    return g

def pivot_for_heatmap(df: pd.DataFrame, metric: str, windows_order) -> pd.DataFrame:
    p = df.pivot_table(index="preset", columns="window", values=metric, aggfunc="mean")
    # Reindex windows in desired order if present
    cols_present = [w for w in windows_order if w in p.columns]
    p = p.reindex(columns=cols_present)
    return p

def draw_heatmap(matrix: pd.DataFrame, title: str, outbase: str) -> None:
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.gca()
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if isinstance(val, (int, float)) and not (val != val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=200)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)

def draw_lines_window_sweep(df_avg: pd.DataFrame, metric: str, windows_order, ylabel: str, outbase: str) -> None:
    # df_avg has columns: preset, window, metric
    # convert window to midpoint on x-axis for clean spacing
    x_mid = [parse_window_str(w)[2] for w in windows_order]
    fig = plt.figure(figsize=(8, 5.0))
    ax = fig.gca()
    presets = sorted(df_avg["preset"].unique())
    for preset in presets:
        sub = df_avg[df_avg["preset"] == preset]
        # align to windows_order
        vals = []
        for w in windows_order:
            row = sub[sub["window"] == w]
            vals.append(row[metric].values[0] if len(row) else np.nan)
        ax.plot(x_mid, vals, marker="o", label=preset)
    ax.set_xlabel("Window midpoint (timestep)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across timestep windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=200)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)

def draw_grouped_bars_parser_effect(df: pd.DataFrame, metric: str, window_choice: str, ylabel: str, outbase: str) -> None:
    # Bars grouped by preset; bars within group correspond to parsers
    parsers = sorted(df["parser"].unique())
    presets = sorted(df["preset"].unique())
    # Filter to chosen window
    sub = df[df["window"] == window_choice].copy()
    # Build value matrix (len(presets) × len(parsers))
    val_mat = np.zeros((len(presets), len(parsers)))
    std_mat = np.zeros_like(val_mat)
    std_col = metric.replace("_mean", "_std") if metric.endswith("_mean") else None
    for i, preset in enumerate(presets):
        for j, parser in enumerate(parsers):
            r = sub[(sub["preset"] == preset) & (sub["parser"] == parser)]
            if len(r) == 0:
                val_mat[i, j] = np.nan
                std_mat[i, j] = 0.0
            else:
                val_mat[i, j] = float(r[metric].values[0])
                if std_col and std_col in r.columns:
                    std_mat[i, j] = float(r[std_col].values[0])
                else:
                    std_mat[i, j] = 0.0

    x = np.arange(len(presets))
    width = 0.8 / max(1, len(parsers))

    fig = plt.figure(figsize=(9, 5.0))
    ax = fig.gca()
    for j, parser in enumerate(parsers):
        ax.bar(x + j*width, val_mat[:, j], width=width, yerr=std_mat[:, j], capsize=3, label=parser)
    ax.set_xticks(x + (len(parsers)-1)*width/2)
    ax.set_xticklabels(presets, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by parser at window={window_choice}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=200)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)

def export_best_by_preset_window(df: pd.DataFrame, out_csv: str) -> None:
    # For each (preset, window), pick parser with highest clip_prompt_mean; write full row
    parts = []
    for (preset, window), g in df.groupby(["preset","window"]):
        idx = g["clip_prompt_mean"].idxmax()
        parts.append(df.loc[idx])
    best = pd.DataFrame(parts).reset_index(drop=True)
    best.to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to ablation_summary.csv")
    ap.add_argument("--outdir", default="artifacts/plots", help="Output directory for figures")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = df_read(args.csv)

    # Window ordering
    win_order = ordered_windows(sorted(df["window"].unique()))

    # 0) Export helper table (best parser per preset×window)
    export_best_by_preset_window(df, os.path.join(args.outdir, "ablation_best_by_preset_window.csv"))

    # 1) Heatmaps averaged over parsers
    for metric, ylabel in METRICS:
        avg = agg_over_parsers(df, metric)
        mat = pivot_for_heatmap(avg, metric, win_order)
        outbase = os.path.join(args.outdir, f"abl_heatmap_{metric}_over_parsers")
        draw_heatmap(mat, f"{ylabel} — averaged over parsers", outbase)

    # 2) Heatmaps per parser
    for metric, ylabel in METRICS:
        for parser in sorted(df["parser"].unique()):
            sub = df[df["parser"] == parser]
            mat = pivot_for_heatmap(sub, metric, win_order)
            outbase = os.path.join(args.outdir, f"abl_heatmap_{metric}_parser={parser}")
            draw_heatmap(mat, f"{ylabel} — parser={parser}", outbase)

    # 3) Line plots: window sweep per preset (averaged over parsers)
    for metric, ylabel in METRICS:
        avg = agg_over_parsers(df, metric)  # preset × window mean over parsers
        outbase = os.path.join(args.outdir, f"abl_lines_windows_{metric}")
        draw_lines_window_sweep(avg, metric, win_order, ylabel, outbase)

    # 4) Grouped bars: parser effect at a representative window (middle window by midpoint)
    # Pick middle window by midpoint
    mids = [parse_window_str(w)[2] for w in win_order]
    if len(win_order) > 0:
        mid_idx = len(win_order) // 2
        window_choice = win_order[mid_idx]
        for metric, ylabel in METRICS:
            outbase = os.path.join(args.outdir, f"abl_groupedbars_parser_effect_{metric}_window={window_choice}")
            draw_grouped_bars_parser_effect(df, metric, window_choice, ylabel, outbase)

    print("[OK] Plots written to:", args.outdir)

if __name__ == "__main__":
    main()
