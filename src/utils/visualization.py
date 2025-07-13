"""
Visualization utilities for Local Prompt Adaptation (LPA).

This module provides functionality for creating visualizations including
attention maps, comparison plots, and result summaries.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import torch
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")


class Visualizer:
    """
    Visualizer for creating plots and visualizations for LPA results.
    """
    
    def __init__(self, style: str = "default"):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        plt.style.use(style)
        
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        
        # Color palette
        self.colors = {
            'lpa': '#1f77b4',
            'vanilla_sdxl': '#ff7f0e',
            'high_cfg_sdxl': '#2ca02c',
            'baseline': '#d62728'
        }
    
    def create_attention_visualization(self,
                                      attention_map: np.ndarray,
                                      token: str,
                                      image: Optional[Image.Image] = None,
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create attention map visualization.
        
        Args:
            attention_map: Attention map array
            token: Token name for the attention map
            image: Optional original image for overlay
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=figsize)
        
        if image is not None:
            # Original image
            axes[0].imshow(image)
            axes[0].set_title(f"Original Image")
            axes[0].axis('off')
            
            # Attention map
            im = axes[1].imshow(attention_map, cmap='viridis', alpha=0.8)
            axes[1].set_title(f"Attention Map for '{token}'")
            axes[1].axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1])
            cbar.set_label('Attention Weight')
            
        else:
            # Just attention map
            im = axes.imshow(attention_map, cmap='viridis')
            axes.set_title(f"Attention Map for '{token}'")
            axes.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes)
            cbar.set_label('Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_grid(self,
                              images: List[Image.Image],
                              titles: List[str],
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create a grid comparison of multiple images.
        
        Args:
            images: List of images to compare
            titles: List of titles for each image
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_images = len(images)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_images == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for i, (image, title) in enumerate(zip(images, titles)):
            if i < len(axes):
                axes[i].imshow(image)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metrics_comparison(self,
                                 metrics_data: Dict[str, Dict[str, float]],
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create bar chart comparison of metrics across methods.
        
        Args:
            metrics_data: Dictionary mapping method names to metrics
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics and methods
        methods = list(metrics_data.keys())
        metrics = set()
        for method_metrics in metrics_data.values():
            metrics.update(method_metrics.keys())
        metrics = sorted(list(metrics))
        
        # Prepare data for plotting
        x = np.arange(len(metrics))
        width = 0.8 / len(methods)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, method in enumerate(methods):
            values = [metrics_data[method].get(metric, 0) for metric in metrics]
            color = self.colors.get(method, f'C{i}')
            
            ax.bar(x + i * width, values, width, label=method, color=color, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Method Comparison by Metrics')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_improvement_heatmap(self,
                                  improvements: Dict[str, Dict[str, float]],
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create heatmap showing improvements of LPA over baselines.
        
        Args:
            improvements: Dictionary mapping baseline methods to metric improvements
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        baseline_methods = list(improvements.keys())
        metrics = set()
        for method_improvements in improvements.values():
            metrics.update(method_improvements.keys())
        metrics = sorted(list(metrics))
        
        # Create data matrix
        data = []
        for method in baseline_methods:
            row = [improvements[method].get(metric, 0) for metric in metrics]
            data.append(row)
        
        data = np.array(data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')
        
        # Add text annotations
        for i in range(len(baseline_methods)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        # Set labels
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(baseline_methods)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(baseline_methods)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Improvement Ratio')
        
        ax.set_title('LPA Improvements Over Baselines')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_prompt_analysis_plot(self,
                                   parsed_prompts: List[Dict[str, Any]],
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create analysis plots for parsed prompts.
        
        Args:
            parsed_prompts: List of parsed prompt dictionaries
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract data
        object_counts = [len(p.get('object_tokens', [])) for p in parsed_prompts]
        style_counts = [len(p.get('style_tokens', [])) for p in parsed_prompts]
        complexities = [p.get('complexity', 'medium') for p in parsed_prompts]
        categories = [p.get('category', 'unknown') for p in parsed_prompts]
        
        # Plot 1: Object count distribution
        axes[0, 0].hist(object_counts, bins=range(max(object_counts) + 2), alpha=0.7, color=self.colors['lpa'])
        axes[0, 0].set_xlabel('Number of Objects')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Object Count Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Style count distribution
        axes[0, 1].hist(style_counts, bins=range(max(style_counts) + 2), alpha=0.7, color=self.colors['baseline'])
        axes[0, 1].set_xlabel('Number of Styles')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Style Count Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Complexity distribution
        complexity_counts = {}
        for comp in complexities:
            complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
        
        axes[1, 0].bar(complexity_counts.keys(), complexity_counts.values(), 
                      color=[self.colors['lpa'], self.colors['vanilla_sdxl'], 
                             self.colors['high_cfg_sdxl'], self.colors['baseline']][:len(complexity_counts)])
        axes[1, 0].set_xlabel('Complexity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Complexity Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Category distribution
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        axes[1, 1].bar(category_counts.keys(), category_counts.values(), 
                      color=[self.colors['lpa'], self.colors['vanilla_sdxl'], 
                             self.colors['high_cfg_sdxl'], self.colors['baseline']][:len(category_counts)])
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Category Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self,
                                    results: Dict[str, Any],
                                    save_path: Optional[str] = None) -> Optional[Any]:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            results: Experiment results dictionary
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure object or None
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available for interactive dashboard")
            return None
        
        try:
            import plotly.graph_objects as go
            
            # Create subplots
            fig = go.Figure()
            
            # Add traces for different metrics
            if "lpa" in results and "baselines" in results:
                methods = ["LPA"] + list(results["baselines"].keys())
                
                # Example: Style consistency comparison
                style_scores = []
                for method in methods:
                    if method == "LPA":
                        score = results["lpa"].get("style_consistency", 0)
                    else:
                        score = results["baselines"][method].get("style_consistency", 0)
                    style_scores.append(score)
                
                fig.add_trace(go.Bar(
                    x=methods,
                    y=style_scores,
                    name="Style Consistency",
                    marker_color=[self.colors.get(method.lower(), 'gray') for method in methods]
                ))
            
            fig.update_layout(
                title="LPA vs Baselines Comparison",
                xaxis_title="Methods",
                yaxis_title="Score",
                barmode='group'
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
            
        except ImportError:
            warnings.warn("Plotly not available for interactive dashboard")
            return None
    
    def save_all_visualizations(self,
                               results: Dict[str, Any],
                               save_dir: str,
                               prefix: str = "viz"):
        """
        Save all visualizations for experiment results.
        
        Args:
            results: Experiment results
            save_dir: Directory to save visualizations
            prefix: Prefix for filenames
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Create comparison grid if images are available
        if "lpa" in results and "baselines" in results:
            images = []
            titles = []
            
            # Add LPA image
            if "best_image" in results["lpa"]:
                images.append(results["lpa"]["best_image"])
                titles.append("LPA")
            
            # Add baseline images
            for method, method_results in results["baselines"].items():
                if "best_image" in method_results:
                    images.append(method_results["best_image"])
                    titles.append(method.replace("_", " ").title())
            
            if images:
                comparison_path = os.path.join(save_dir, f"{prefix}_comparison.png")
                self.create_comparison_grid(images, titles, comparison_path)
        
        # Create metrics comparison if metrics are available
        # This would require evaluation results to be computed first
        
        # Create prompt analysis if parsed prompts are available
        if "parsed_prompt" in results:
            analysis_path = os.path.join(save_dir, f"{prefix}_prompt_analysis.png")
            self.create_prompt_analysis_plot([results["parsed_prompt"]], analysis_path)
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all') 