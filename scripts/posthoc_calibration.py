"""Post-hoc calibration of probe confidence scores.

Applies Platt scaling and isotonic regression to existing baseline probe scores.
Compares raw vs calibrated scores on held-out set using 5-seed protocol.

Decision rule:
- KEEP if ECE reduction >= 30% AND AUROC drop < 0.01
- DISCARD if ECE reduction < 10%
- INVESTIGATE otherwise
"""

import json
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.stats import levene
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_DIR = Path("/Users/aban/drive/Projects/epistemic-steering")
HELD_OUT_RESULTS = PROJECT_DIR / "data/heldout_eval/final_summary.json"
OUTPUT_DIR = PROJECT_DIR / "data/ablation_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_FILE = OUTPUT_DIR / "calibration_5seed.json"

# 5 seeds for protocol
SEEDS = [42, 123, 456, 789, 2024]
N_BINS = 10  # ECE bins


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        # Samples in this bin
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        # Last bin: include right edge
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        bin_count = np.sum(mask)
        if bin_count == 0:
            continue

        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += (bin_count / total) * abs(bin_acc - bin_conf)

    return ece


def compute_brier(y_true, y_prob):
    """Compute Brier score."""
    return brier_score_loss(y_true, y_prob)


def load_heldout_data():
    """Load held-out evaluation results."""
    with open(HELD_OUT_RESULTS, 'r') as f:
        data = json.load(f)

    # Extract probe scores and labels from results
    results = data.get('results', [])

    scores = np.array([r['probe_score'] for r in results])
    labels = np.array([r['correct'] for r in results], dtype=bool)

    print(f"Loaded {len(scores)} held-out samples")
    print(f"  Positive (correct): {np.sum(labels)}, Negative (incorrect): {np.sum(~labels)}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Raw AUROC: {roc_auc_score(labels, scores):.4f}")

    return scores, labels


def calibrate_platt(scores_cal, labels_cal, scores_test):
    """Apply Platt scaling (sigmoid calibration)."""
    # Reshape for sklearn
    X_cal = scores_cal.reshape(-1, 1)
    X_test = scores_test.reshape(-1, 1)

    # Fit logistic regression on calibration set
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(X_cal, labels_cal)

    # Predict probabilities on test set
    calibrated = lr.predict_proba(X_test)[:, 1]

    return calibrated


def calibrate_isotonic(scores_cal, labels_cal, scores_test):
    """Apply isotonic regression calibration."""
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    ir.fit(scores_cal, labels_cal)

    # Predict on test set
    calibrated = ir.transform(scores_test)

    return calibrated


def run_single_seed(scores, labels, seed, cal_ratio=0.2):
    """Run calibration experiment for one seed."""
    np.random.seed(seed)

    n = len(scores)
    indices = np.random.permutation(n)

    # Split: 20% calibration, 80% test
    cal_size = int(n * cal_ratio)
    cal_idx = indices[:cal_size]
    test_idx = indices[cal_size:]

    scores_cal = scores[cal_idx]
    labels_cal = labels[cal_idx]
    scores_test = scores[test_idx]
    labels_test = labels[test_idx]

    # Raw scores (just clip to [0.001, 0.999] to avoid edge issues for ECE computation)
    raw_calibrated_test = np.clip(scores_test, 0.001, 0.999)

    # Platt scaling
    platt_test = calibrate_platt(scores_cal, labels_cal, scores_test)

    # Isotonic regression
    isotonic_test = calibrate_isotonic(scores_cal, labels_cal, scores_test)

    # Compute metrics on test set
    metrics = {}

    # Raw
    metrics['raw'] = {
        'ece': compute_ece(labels_test, np.clip(scores_test, 0.001, 0.999), N_BINS),
        'brier': compute_brier(labels_test, scores_test),
        'auroc': roc_auc_score(labels_test, scores_test),
    }

    # Platt
    metrics['platt'] = {
        'ece': compute_ece(labels_test, platt_test, N_BINS),
        'brier': compute_brier(labels_test, platt_test),
        'auroc': roc_auc_score(labels_test, platt_test),
    }

    # Isotonic
    metrics['isotonic'] = {
        'ece': compute_ece(labels_test, isotonic_test, N_BINS),
        'brier': compute_brier(labels_test, isotonic_test),
        'auroc': roc_auc_score(labels_test, isotonic_test),
    }

    return metrics, cal_idx, test_idx


def run_5seed_protocol(scores, labels):
    """Run 5-seed calibration protocol."""
    all_metrics = []
    cal_indices = []
    test_indices = []

    print("\n" + "="*60)
    print("5-SEED PROTOCOL")
    print("="*60)

    for i, seed in enumerate(SEEDS):
        print(f"\nSeed {i+1}/5: {seed}")
        metrics, cal_idx, test_idx = run_single_seed(scores, labels, seed)
        all_metrics.append(metrics)
        cal_indices.append(cal_idx)
        test_indices.append(test_idx)

        # Print per-seed results
        for method in ['raw', 'platt', 'isotonic']:
            m = metrics[method]
            print(f"  {method:10s} | ECE: {m['ece']:.4f} | Brier: {m['brier']:.4f} | AUROC: {m['auroc']:.4f}")

    return all_metrics, cal_indices, test_indices


def aggregate_results(all_metrics, baseline_auroc):
    """Aggregate 5-seed results and apply decision rule."""
    # Average metrics across seeds
    avg_results = {}

    for method in ['raw', 'platt', 'isotonic']:
        ece_vals = [m[method]['ece'] for m in all_metrics]
        brier_vals = [m[method]['brier'] for m in all_metrics]
        auroc_vals = [m[method]['auroc'] for m in all_metrics]

        avg_results[method] = {
            'ece_mean': np.mean(ece_vals),
            'ece_std': np.std(ece_vals),
            'ece_vals': ece_vals,
            'brier_mean': np.mean(brier_vals),
            'brier_std': np.std(brier_vals),
            'brier_vals': brier_vals,
            'auroc_mean': np.mean(auroc_vals),
            'auroc_std': np.std(auroc_vals),
            'auroc_vals': auroc_vals,
        }

    # Levene's test for variance homogeneity (before aggregation)
    raw_ece = [m['raw']['ece'] for m in all_metrics]
    platt_ece = [m['platt']['ece'] for m in all_metrics]
    isotonic_ece = [m['isotonic']['ece'] for m in all_metrics]

    levene_raw_platt = levene(raw_ece, platt_ece)
    levene_raw_isotonic = levene(raw_ece, isotonic_ece)

    # ECE reductions
    raw_ece_mean = avg_results['raw']['ece_mean']
    platt_ece_reduction = (raw_ece_mean - avg_results['platt']['ece_mean']) / raw_ece_mean * 100
    isotonic_ece_reduction = (raw_ece_mean - avg_results['isotonic']['ece_mean']) / raw_ece_mean * 100

    # AUROC drops from baseline
    platt_auroc_drop = baseline_auroc - avg_results['platt']['auroc_mean']
    isotonic_auroc_drop = baseline_auroc - avg_results['isotonic']['auroc_mean']

    print("\n" + "="*60)
    print("AGGREGATED RESULTS (5-seed average)")
    print("="*60)

    print(f"\n{'Method':<12} {'ECE Mean±Std':<18} {'Brier Mean±Std':<18} {'AUROC Mean±Std':<18}")
    print("-" * 66)
    for method in ['raw', 'platt', 'isotonic']:
        r = avg_results[method]
        print(f"{method:<12} {r['ece_mean']:.4f}±{r['ece_std']:.4f}     {r['brier_mean']:.4f}±{r['brier_std']:.4f}     {r['auroc_mean']:.4f}±{r['auroc_std']:.4f}")

    print(f"\nLevene's test (ECE variance):")
    print(f"  Raw vs Platt:    statistic={levene_raw_platt.statistic:.4f}, p={levene_raw_platt.pvalue:.4f}")
    print(f"  Raw vs Isotonic: statistic={levene_raw_isotonic.statistic:.4f}, p={levene_raw_isotonic.pvalue:.4f}")

    print(f"\nECE Reduction from baseline:")
    print(f"  Platt:    {platt_ece_reduction:.1f}%")
    print(f"  Isotonic: {isotonic_ece_reduction:.1f}%")

    print(f"\nAUROC drop from baseline ({baseline_auroc:.4f}):")
    print(f"  Platt:    {platt_auroc_drop:.4f}")
    print(f"  Isotonic: {isotonic_auroc_drop:.4f}")

    # Decision rule
    print("\n" + "="*60)
    print("DECISION RULE")
    print("="*60)
    print(f"Pre-registered thresholds:")
    print(f"  ECE reduction >= 30% → KEEP")
    print(f"  ECE reduction < 10%  → DISCARD")
    print(f"  Otherwise            → INVESTIGATE")
    print(f"  AUROC drop must be < 0.01 for KEEP")

    decisions = {}

    # Platt decision
    platt_keep = platt_ece_reduction >= 30 and platt_auroc_drop < 0.01
    platt_discard = platt_ece_reduction < 10
    if platt_keep:
        platt_decision = "KEEP"
    elif platt_discard:
        platt_decision = "DISCARD"
    else:
        platt_decision = "INVESTIGATE"
    decisions['platt'] = platt_decision

    # Isotonic decision
    iso_keep = isotonic_ece_reduction >= 30 and isotonic_auroc_drop < 0.01
    iso_discard = isotonic_ece_reduction < 10
    if iso_keep:
        iso_decision = "KEEP"
    elif iso_discard:
        iso_decision = "DISCARD"
    else:
        iso_decision = "INVESTIGATE"
    decisions['isotonic'] = iso_decision

    print(f"\nPlatt scaling: {platt_decision}")
    print(f"  (ECE reduction={platt_ece_reduction:.1f}%, AUROC drop={platt_auroc_drop:.4f})")
    print(f"\nIsotonic regression: {iso_decision}")
    print(f"  (ECE reduction={isotonic_ece_reduction:.1f}%, AUROC drop={isotonic_auroc_drop:.4f})")

    return avg_results, decisions, {
        'levene_raw_platt_stat': levene_raw_platt.statistic,
        'levene_raw_platt_p': levene_raw_platt.pvalue,
        'levene_raw_isotonic_stat': levene_raw_isotonic.statistic,
        'levene_raw_isotonic_p': levene_raw_isotonic.pvalue,
        'platt_ece_reduction_pct': platt_ece_reduction,
        'isotonic_ece_reduction_pct': isotonic_ece_reduction,
        'platt_auroc_drop': platt_auroc_drop,
        'isotonic_auroc_drop': isotonic_auroc_drop,
    }


def main():
    print("="*60)
    print("POST-HOC CALIBRATION: Platt scaling vs Isotonic regression")
    print("="*60)

    # Load baseline AUROC from context
    baseline_auroc = 0.9678  # From context / final_summary.json

    # Load data
    scores, labels = load_heldout_data()

    # Run 5-seed protocol
    all_metrics, cal_indices, test_indices = run_5seed_protocol(scores, labels)

    # Aggregate and decide
    avg_results, decisions, stats = aggregate_results(all_metrics, baseline_auroc)

    # Compile final output
    output = {
        'metadata': {
            'task': 'posthoc_calibration',
            'num_samples': len(scores),
            'n_bins': N_BINS,
            'cal_ratio': 0.2,
            'seeds': SEEDS,
            'baseline_auroc': baseline_auroc,
        },
        'per_seed_results': [
            {
                'seed': SEEDS[i],
                'metrics': all_metrics[i],
                'cal_size': len(cal_indices[i]),
                'test_size': len(test_indices[i]),
            }
            for i in range(5)
        ],
        'aggregated': {
            'raw': avg_results['raw'],
            'platt': avg_results['platt'],
            'isotonic': avg_results['isotonic'],
        },
        'statistics': stats,
        'decisions': decisions,
        'decision_rule': {
            'keep_threshold_ece_reduction': 30,
            'keep_threshold_auroc_drop': 0.01,
            'discard_threshold_ece_reduction': 10,
        },
    }

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n" + "="*60)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("="*60)

    return output


if __name__ == "__main__":
    main()