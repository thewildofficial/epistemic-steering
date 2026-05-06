"""Reusable plotting functions for epistemic steering evaluation.

Provides publication-quality visualizations for probe performance analysis,
including ROC curves, confusion matrices, threshold tradeoffs, calibration curves,
and dataset comparisons.

All functions use consistent styling:
- Seaborn whitegrid theme
- Font sizes ≥ 12pt
- Colorblind-friendly palette
- Save as both PNG (300 dpi) and PDF
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple


def _setup_style():
    """Apply consistent plotting style."""
    sns.set_theme(style='whitegrid', font_scale=1.2)
    sns.set_palette('colorblind')


def _save_figure(fig: plt.Figure, save_path: str):
    """Save figure as both PNG (300 dpi) and PDF.
    
    Args:
        fig: Matplotlib figure object.
        save_path: Base path without extension.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    
    plt.close(fig)


def plot_auroc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    title: str,
    save_path: str
) -> None:
    """Plot ROC curve with AUROC annotation.
    
    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auroc: Area under ROC curve.
        title: Plot title.
        save_path: Base path for saving (without extension).
    """
    _setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC (AUROC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.15, color='blue')
    
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    textstr = f'AUROC = {auroc:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    _save_figure(fig, save_path)


def plot_confusion_matrix_heatmap(
    cm: dict,
    labels: List[str],
    title: str,
    save_path: str
) -> None:
    """Plot confusion matrix heatmap.
    
    Args:
        cm: Dict with TP, FP, TN, FN counts.
        labels: Class labels ['Negative', 'Positive'].
        title: Plot title.
        save_path: Base path for saving (without extension).
    """
    _setup_style()
    
    cm_matrix = np.array([
        [cm['TN'], cm['FP']],
        [cm['FN'], cm['TP']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 16, 'weight': 'bold'},
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    _save_figure(fig, save_path)


def plot_threshold_tradeoff(
    thresholds: np.ndarray,
    prevention: np.ndarray,
    unnecessary_block: np.ndarray,
    optimal_threshold: float,
    save_path: str
) -> None:
    """Plot prevention rate vs unnecessary block rate tradeoff curve.
    
    Args:
        thresholds: Threshold values.
        prevention: Prevention rates at each threshold.
        unnecessary_block: Unnecessary block rates at each threshold.
        optimal_threshold: Optimal threshold to mark.
        save_path: Base path for saving (without extension).
    """
    _setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(
        unnecessary_block,
        prevention,
        c=thresholds,
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )
    
    opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.scatter(
        unnecessary_block[opt_idx],
        prevention[opt_idx],
        c='red',
        s=200,
        marker='*',
        edgecolors='black',
        linewidth=1.5,
        label=f'Optimal (t={optimal_threshold:.2f})',
        zorder=5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Unnecessary Block Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prevention Rate', fontsize=14, fontweight='bold')
    ax.set_title('Threshold Tradeoff: Prevention vs Unnecessary Block', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.05, max(unnecessary_block) * 1.1])
    ax.set_ylim([min(prevention) * 0.9, 1.05])
    
    _save_figure(fig, save_path)


def plot_calibration_curve(
    bin_centers: np.ndarray,
    observed_accuracy: np.ndarray,
    title: str,
    save_path: str,
    bin_counts: Optional[np.ndarray] = None
) -> None:
    """Plot calibration curve (reliability diagram) with diagonal.
    
    Args:
        bin_centers: Center of each confidence bin.
        observed_accuracy: Actual accuracy in each bin.
        title: Plot title.
        save_path: Base path for saving (without extension).
        bin_counts: Optional sample counts per bin for sizing points.
    """
    _setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect calibration')
    
    if bin_counts is not None:
        sizes = 50 + (bin_counts / bin_counts.max()) * 300
        scatter = ax.scatter(
            bin_centers,
            observed_accuracy,
            s=sizes,
            c='steelblue',
            alpha=0.8,
            edgecolors='black',
            linewidth=1,
            label='Model calibration'
        )
    else:
        ax.plot(
            bin_centers,
            observed_accuracy,
            'o-',
            color='steelblue',
            markersize=10,
            linewidth=2.5,
            markeredgecolor='black',
            markeredgewidth=1,
            label='Model calibration'
        )
    
    ax.set_xlabel('Mean Predicted Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Observed Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_aspect('equal')
    
    _save_figure(fig, save_path)


def plot_dataset_comparison(
    mmlu_metrics: dict,
    gsm8k_metrics: dict,
    save_path: str
) -> None:
    """Plot side-by-side MMLU vs GSM8K bar chart.
    
    Args:
        mmlu_metrics: Dict with 'auroc', 'prevention_rate', 'best_threshold'.
        gsm8k_metrics: Dict with 'auroc', 'prevention_rate', 'best_threshold'.
        save_path: Base path for saving (without extension).
    """
    _setup_style()
    
    metrics = ['AUROC', 'Prevention Rate', 'Best Threshold']
    mmlu_values = [
        mmlu_metrics.get('auroc', 0),
        mmlu_metrics.get('prevention_rate', 0),
        mmlu_metrics.get('best_threshold', 0)
    ]
    gsm8k_values = [
        gsm8k_metrics.get('auroc', 0),
        gsm8k_metrics.get('prevention_rate', 0),
        gsm8k_metrics.get('best_threshold', 0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, mmlu_values, width, label='MMLU', 
                   color='steelblue', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, gsm8k_values, width, label='GSM8K', 
                   color='coral', edgecolor='black', linewidth=1)
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_title('MMLU vs GSM8K Performance Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, max(max(mmlu_values), max(gsm8k_values)) * 1.15])
    
    _save_figure(fig, save_path)


def plot_prevention_rate_curve(
    thresholds: np.ndarray,
    prevention_rates: np.ndarray,
    optimal_threshold: float,
    save_path: str,
    dataset_name: str = ""
) -> None:
    """Plot prevention rate vs threshold with optimal threshold marker.
    
    Args:
        thresholds: Threshold values.
        prevention_rates: Prevention rates at each threshold.
        optimal_threshold: Optimal threshold to mark.
        save_path: Base path for saving (without extension).
        dataset_name: Optional dataset name for title.
    """
    _setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(
        thresholds,
        prevention_rates,
        'o-',
        color='forestgreen',
        markersize=8,
        linewidth=2.5,
        markeredgecolor='black',
        markeredgewidth=1,
        label='Prevention Rate'
    )
    
    opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.axvline(
        x=optimal_threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label=f'Optimal (t={optimal_threshold:.2f})'
    )
    ax.scatter(
        optimal_threshold,
        prevention_rates[opt_idx],
        c='red',
        s=150,
        marker='*',
        edgecolors='black',
        linewidth=1.5,
        zorder=5
    )
    
    ax.set_xlabel('Confidence Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prevention Rate', fontsize=14, fontweight='bold')
    
    title = 'Prevention Rate vs Threshold'
    if dataset_name:
        title += f' ({dataset_name})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([min(thresholds) * 0.9, max(thresholds) * 1.1])
    ax.set_ylim([0, 1.05])
    
    _save_figure(fig, save_path)