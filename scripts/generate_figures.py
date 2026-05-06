"""Generate publication-quality figures for epistemic steering evaluation.

Produces 5 figures from verification_results.json:
1. AUROC curves for MMLU and GSM8K
2. Threshold tradeoff curves (prevention rate vs unnecessary block rate)
3. Calibration curves (reliability diagrams)
4. Prevention rate vs threshold with optimal threshold marked
5. MMLU vs GSM8K comparison bar chart

All figures saved as both PNG (300 dpi) and PDF in figures/ directory.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting import (
    plot_auroc_curve,
    plot_confusion_matrix_heatmap,
    plot_threshold_tradeoff,
    plot_calibration_curve,
    plot_dataset_comparison,
    plot_prevention_rate_curve
)


def load_verification_data(data_path: str = "data/verification_results.json") -> dict:
    """Load verification results from JSON file.
    
    Args:
        data_path: Path to verification_results.json.
        
    Returns:
        Dict with verification results.
    """
    with open(data_path, 'r') as f:
        return json.load(f)


def compute_roc_from_threshold_sweep(threshold_sweep: list) -> tuple:
    """Compute approximate ROC curve from threshold sweep data.
    
    Since we don't have raw predictions, we approximate FPR/TPR
    from the confusion matrix components at each threshold.
    
    Args:
        threshold_sweep: List of threshold sweep results.
        
    Returns:
        Tuple of (fpr, tpr) arrays.
    """
    fprs = []
    tprs = []
    
    for entry in threshold_sweep:
        prevention_rate = entry['prevention_rate']
        unnecessary_block_rate = entry['unnecessary_block_rate']
        
        fpr = 1.0 - prevention_rate
        tpr = 1.0 - unnecessary_block_rate
        
        fprs.append(fpr)
        tprs.append(tpr)
    
    sorted_indices = np.argsort(fprs)
    fpr_array = np.array(fprs)[sorted_indices]
    tpr_array = np.array(tprs)[sorted_indices]
    
    fpr_array = np.concatenate([[0], fpr_array, [1]])
    tpr_array = np.concatenate([[0], tpr_array, [1]])
    
    return fpr_array, tpr_array


def find_optimal_threshold(threshold_sweep: list) -> float:
    """Find optimal threshold maximizing F1 score.
    
    Args:
        threshold_sweep: List of threshold sweep results.
        
    Returns:
        Optimal threshold value.
    """
    f1_scores = [entry['f1'] for entry in threshold_sweep]
    optimal_idx = np.argmax(f1_scores)
    return threshold_sweep[optimal_idx]['threshold']


def generate_figure1_auroc(data: dict, output_dir: Path) -> None:
    """Figure 1: AUROC curves for MMLU and GSM8K (side-by-side).
    
    Args:
        data: Verification results dict.
        output_dir: Output directory for figures.
    """
    print("Generating Figure 1: AUROC curves...")
    
    mmlu_fpr, mmlu_tpr = compute_roc_from_threshold_sweep(data['mmlu']['threshold_sweep'])
    plot_auroc_curve(
        fpr=mmlu_fpr,
        tpr=mmlu_tpr,
        auroc=data['mmlu']['auroc'],
        title='MMLU ROC Curve',
        save_path=str(output_dir / 'fig1_mmlu_auroc')
    )
    
    gsm8k_fpr, gsm8k_tpr = compute_roc_from_threshold_sweep(data['gsm8k']['threshold_sweep'])
    plot_auroc_curve(
        fpr=gsm8k_fpr,
        tpr=gsm8k_tpr,
        auroc=data['gsm8k']['auroc'],
        title='GSM8K ROC Curve (In-Sample, Overfit)',
        save_path=str(output_dir / 'fig1_gsm8k_auroc')
    )
    
    print("  ✓ Figure 1 saved")


def generate_figure2_threshold_tradeoff(data: dict, output_dir: Path) -> None:
    """Figure 2: Threshold tradeoff curves with optimal threshold markers.
    
    Args:
        data: Verification results dict.
        output_dir: Output directory for figures.
    """
    print("Generating Figure 2: Threshold tradeoff curves...")
    
    mmlu_sweep = data['mmlu']['threshold_sweep']
    mmlu_thresholds = np.array([e['threshold'] for e in mmlu_sweep])
    mmlu_prevention = np.array([e['prevention_rate'] for e in mmlu_sweep])
    mmlu_unnecessary = np.array([e['unnecessary_block_rate'] for e in mmlu_sweep])
    mmlu_optimal = find_optimal_threshold(mmlu_sweep)
    
    plot_threshold_tradeoff(
        thresholds=mmlu_thresholds,
        prevention=mmlu_prevention,
        unnecessary_block=mmlu_unnecessary,
        optimal_threshold=mmlu_optimal,
        save_path=str(output_dir / 'fig2_mmlu_tradeoff')
    )
    
    gsm8k_sweep = data['gsm8k']['threshold_sweep']
    gsm8k_thresholds = np.array([e['threshold'] for e in gsm8k_sweep])
    gsm8k_prevention = np.array([e['prevention_rate'] for e in gsm8k_sweep])
    gsm8k_unnecessary = np.array([e['unnecessary_block_rate'] for e in gsm8k_sweep])
    gsm8k_optimal = find_optimal_threshold(gsm8k_sweep)
    
    plot_threshold_tradeoff(
        thresholds=gsm8k_thresholds,
        prevention=gsm8k_prevention,
        unnecessary_block=gsm8k_unnecessary,
        optimal_threshold=gsm8k_optimal,
        save_path=str(output_dir / 'fig2_gsm8k_tradeoff')
    )
    
    print("  ✓ Figure 2 saved")


def generate_figure3_calibration(data: dict, output_dir: Path) -> None:
    """Figure 3: Calibration curves (reliability diagrams) for both datasets.
    
    Args:
        data: Verification results dict.
        output_dir: Output directory for figures.
    """
    print("Generating Figure 3: Calibration curves...")
    
    mmlu_calib = data['mmlu']['calibration']
    plot_calibration_curve(
        bin_centers=np.array(mmlu_calib['bin_centers']),
        observed_accuracy=np.array(mmlu_calib['observed_accuracy']),
        title='MMLU Calibration Curve',
        save_path=str(output_dir / 'fig3_mmlu_calibration'),
        bin_counts=np.array(mmlu_calib['bin_counts'])
    )
    
    gsm8k_calib = data['gsm8k']['calibration']
    plot_calibration_curve(
        bin_centers=np.array(gsm8k_calib['bin_centers']),
        observed_accuracy=np.array(gsm8k_calib['observed_accuracy']),
        title='GSM8K Calibration Curve (In-Sample)',
        save_path=str(output_dir / 'fig3_gsm8k_calibration'),
        bin_counts=np.array(gsm8k_calib['bin_counts'])
    )
    
    print("  ✓ Figure 3 saved")


def generate_figure4_prevention_rate(data: dict, output_dir: Path) -> None:
    """Figure 4: Prevention rate vs threshold with optimal threshold marked.
    
    Args:
        data: Verification results dict.
        output_dir: Output directory for figures.
    """
    print("Generating Figure 4: Prevention rate curves...")
    
    mmlu_sweep = data['mmlu']['threshold_sweep']
    mmlu_thresholds = np.array([e['threshold'] for e in mmlu_sweep])
    mmlu_prevention = np.array([e['prevention_rate'] for e in mmlu_sweep])
    mmlu_optimal = find_optimal_threshold(mmlu_sweep)
    
    plot_prevention_rate_curve(
        thresholds=mmlu_thresholds,
        prevention_rates=mmlu_prevention,
        optimal_threshold=mmlu_optimal,
        save_path=str(output_dir / 'fig4_mmlu_prevention'),
        dataset_name='MMLU'
    )
    
    gsm8k_sweep = data['gsm8k']['threshold_sweep']
    gsm8k_thresholds = np.array([e['threshold'] for e in gsm8k_sweep])
    gsm8k_prevention = np.array([e['prevention_rate'] for e in gsm8k_sweep])
    gsm8k_optimal = find_optimal_threshold(gsm8k_sweep)
    
    plot_prevention_rate_curve(
        thresholds=gsm8k_thresholds,
        prevention_rates=gsm8k_prevention,
        optimal_threshold=gsm8k_optimal,
        save_path=str(output_dir / 'fig4_gsm8k_prevention'),
        dataset_name='GSM8K'
    )
    
    print("  ✓ Figure 4 saved")


def generate_figure5_comparison(data: dict, output_dir: Path) -> None:
    """Figure 5: MMLU vs GSM8K comparison bar chart.
    
    Args:
        data: Verification results dict.
        output_dir: Output directory for figures.
    """
    print("Generating Figure 5: Dataset comparison...")
    
    mmlu_sweep = data['mmlu']['threshold_sweep']
    mmlu_optimal = find_optimal_threshold(mmlu_sweep)
    mmlu_optimal_entry = next(e for e in mmlu_sweep if e['threshold'] == mmlu_optimal)
    
    mmlu_metrics = {
        'auroc': data['mmlu']['auroc'],
        'prevention_rate': mmlu_optimal_entry['prevention_rate'],
        'best_threshold': mmlu_optimal
    }
    
    gsm8k_sweep = data['gsm8k']['threshold_sweep']
    gsm8k_optimal = find_optimal_threshold(gsm8k_sweep)
    gsm8k_optimal_entry = next(e for e in gsm8k_sweep if e['threshold'] == gsm8k_optimal)
    
    gsm8k_metrics = {
        'auroc': data['gsm8k']['auroc'],
        'prevention_rate': gsm8k_optimal_entry['prevention_rate'],
        'best_threshold': gsm8k_optimal
    }
    
    plot_dataset_comparison(
        mmlu_metrics=mmlu_metrics,
        gsm8k_metrics=gsm8k_metrics,
        save_path=str(output_dir / 'fig5_dataset_comparison')
    )
    
    print("  ✓ Figure 5 saved")


def main():
    """Generate all 5 figures from verification results."""
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    
    data_path = Path('data/verification_results.json')
    output_dir = Path('figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        print("Run verify_insamp.py first to generate verification results.")
        sys.exit(1)
    
    print(f"\nLoading data from {data_path}...")
    data = load_verification_data(str(data_path))
    print(f"  MMLU: {data['mmlu']['n_total']} questions, AUROC = {data['mmlu']['auroc']:.3f}")
    print(f"  GSM8K: {data['gsm8k']['n_total']} questions, AUROC = {data['gsm8k']['auroc']:.3f}")
    
    print(f"\nGenerating figures in {output_dir}/...")
    
    generate_figure1_auroc(data, output_dir)
    generate_figure2_threshold_tradeoff(data, output_dir)
    generate_figure3_calibration(data, output_dir)
    generate_figure4_prevention_rate(data, output_dir)
    generate_figure5_comparison(data, output_dir)
    
    print("\n" + "=" * 70)
    print("GENERATED FILES:")
    print("=" * 70)
    
    png_files = sorted(output_dir.glob('*.png'))
    pdf_files = sorted(output_dir.glob('*.pdf'))
    
    print(f"\nPNG files ({len(png_files)}):")
    for f in png_files:
        print(f"  {f}")
    
    print(f"\nPDF files ({len(pdf_files)}):")
    for f in pdf_files:
        print(f"  {f}")
    
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    
    print("\nFIGURE DESCRIPTIONS FOR PAPER:")
    print("-" * 70)
    print("Figure 1: ROC curves showing probe discriminability.")
    print(f"  MMLU AUROC = {data['mmlu']['auroc']:.3f}")
    print(f"  GSM8K AUROC = {data['gsm8k']['auroc']:.3f} (in-sample, overfit)")
    print("\nFigure 2: Threshold tradeoff curves (prevention rate vs unnecessary block rate).")
    print("  Shows the cost-benefit tradeoff of different confidence thresholds.")
    print("\nFigure 3: Calibration curves (reliability diagrams).")
    print("  Compares predicted confidence to observed accuracy.")
    print("\nFigure 4: Prevention rate vs threshold curves.")
    print("  Shows how hallucination prevention improves with higher thresholds.")
    print("\nFigure 5: MMLU vs GSM8K performance comparison.")
    print("  Side-by-side comparison of key metrics at optimal thresholds.")
    print("-" * 70)


if __name__ == '__main__':
    main()