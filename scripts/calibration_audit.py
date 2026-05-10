"""Calibration Audit of Existing Probe — 5-Seed Protocol.

Audits g(h_l(x)) = σ(w^T h_30(x) + b) for calibration quality using 5-seed protocol.
Outputs: data/ablation_results/baseline_5seed.json and calibration curve PNG.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import levene
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LAYER = 30
N_SEEDS = 5
N_BINS = 10
ECE_THRESHOLD = 0.05


def load_training_data(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df['correct'] = df['correct'].astype(bool)

    mask = (df['dataset'] == 'mmlu') & (df['model_answer'] == '?')
    df.loc[mask, 'correct'] = False
    if mask.sum() > 0:
        print(f"  Marked {mask.sum()} MMLU '?' as incorrect")

    print(f"  Loaded {len(df)} training questions")
    return df


def load_activations_for_df(df: pd.DataFrame, activations_dir: Path, layer: int):
    activations, labels, matched_indices = [], [], []

    for idx, row in df.iterrows():
        qid = row['question_id']
        for candidate in [
            activations_dir / f"{qid}__layer_{layer}.npy",
            activations_dir / f"q{qid}_layer_{layer}.npy",
        ]:
            if candidate.exists():
                act = np.load(candidate)
                activations.append(act.ravel())
                labels.append(row['correct'])
                matched_indices.append(idx)
                break

    if not activations:
        raise ValueError("No activations loaded!")

    return np.array(activations), np.array(labels, dtype=bool), matched_indices


def load_held_out_data(held_out_json_path: str, results_jsonl_path: str):
    with open(held_out_json_path, 'r') as f:
        summary = json.load(f)

    results = summary.get('results', [])
    scores, labels, metadata = [], [], []

    question_texts = {}
    if Path(results_jsonl_path).exists():
        with open(results_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    qid = rec.get('question_id')
                    question = rec.get('question', rec.get('prompt', ''))
                    if qid:
                        question_texts[qid] = question[:200]

    for item in results:
        ps = item.get('probe_score')
        if ps is None:
            continue
        scores.append(ps)
        labels.append(item.get('correct', False))
        qid = item.get('question_id')
        metadata.append({
            'question_id': qid,
            'dataset': item.get('dataset'),
            'probe_score': float(ps),
            'label': bool(item.get('correct')),
            'model_answer': item.get('model_answer', ''),
            'correct_answer': item.get('correct_answer', ''),
            'question_text': question_texts.get(qid, ''),
        })

    print(f"  Loaded {len(scores)} held-out predictions")
    return np.array(scores), np.array(labels, dtype=bool), metadata


def compute_ece(scores, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(scores)

    for i in range(n_bins):
        in_bin = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])

        count = np.sum(in_bin)
        if count == 0:
            continue

        avg_conf = np.mean(scores[in_bin])
        avg_acc = np.mean(labels[in_bin])
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


def compute_calibration_metrics(scores, labels, n_bins=10):
    labels_int = labels.astype(int)

    brier = float(brier_score_loss(labels_int, scores))
    ece = compute_ece(scores, labels, n_bins)
    auroc = float(roc_auc_score(labels_int, scores))
    auprc = float(average_precision_score(labels_int, scores))

    frac_pos, bin_center = calibration_curve(
        labels_int, scores, n_bins=n_bins, strategy='uniform'
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(scores, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(int)

    return {
        'brier_score': brier,
        'ece': ece,
        'auroc': auroc,
        'auprc': auprc,
        'calibration_curve': {
            'bin_centers': bin_center.tolist(),
            'observed_accuracy': frac_pos.tolist(),
            'bin_counts': bin_counts.tolist(),
        },
    }


def run_5seed_audit(train_activations, train_labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    seed_results = []

    for seed_idx, (train_idx, val_idx) in enumerate(skf.split(train_activations, train_labels)):
        print(f"\n  Seed {seed_idx}")

        X_train, y_train = train_activations[train_idx], train_labels[train_idx]
        X_val, y_val = train_activations[val_idx], train_labels[val_idx]

        print(f"    Train: {len(X_train)} ({y_train.sum()} correct), Val: {len(X_val)} ({y_val.sum()} correct)")

        correct_mask = y_train
        direction = X_train[correct_mask].mean(axis=0) - X_train[~correct_mask].mean(axis=0)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        midpoint = (X_train[correct_mask].mean(axis=0) + X_train[~correct_mask].mean(axis=0)) / 2.0

        projections_val = np.dot(X_val - midpoint, direction)
        scores_val = expit(projections_val)

        metrics_val = compute_calibration_metrics(scores_val, y_val, N_BINS)

        print(f"    Val Brier: {metrics_val['brier_score']:.4f}, ECE: {metrics_val['ece']:.4f}, "
              f"AUROC: {metrics_val['auroc']:.4f}")

        seed_results.append({
            'seed': seed_idx,
            'train_size': int(len(X_train)),
            'val_size': int(len(X_val)),
            'metrics': metrics_val,
        })

    return seed_results


def run_heldout_audit(held_out_scores, held_out_labels, held_out_metadata):
    print(f"\n  Held-Out ({len(held_out_scores)} samples)")

    labels_int = held_out_labels.astype(int)
    brier = float(brier_score_loss(labels_int, held_out_scores))
    ece = compute_ece(held_out_scores, held_out_labels, N_BINS)
    auroc = float(roc_auc_score(labels_int, held_out_scores))
    auprc = float(average_precision_score(labels_int, held_out_scores))

    frac_pos, bin_center = calibration_curve(
        labels_int, held_out_scores, n_bins=N_BINS, strategy='uniform'
    )
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    bin_indices = np.digitize(held_out_scores, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, N_BINS - 1)
    bin_counts = np.bincount(bin_indices, minlength=N_BINS).astype(int)

    print(f"    Brier: {brier:.4f}, ECE: {ece:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    return {
        'brier_score': brier,
        'ece': ece,
        'auroc': auroc,
        'auprc': auprc,
        'n_total': int(len(held_out_scores)),
        'n_correct': int(held_out_labels.sum()),
        'n_incorrect': int((~held_out_labels).sum()),
        'calibration_curve': {
            'bin_centers': bin_center.tolist(),
            'observed_accuracy': frac_pos.tolist(),
            'bin_counts': bin_counts.tolist(),
        },
        'per_sample_predictions': held_out_metadata,
    }


def levene_test(seed_results, metric_key):
    key_map = {
        'brier_score': lambda m: m['brier_score'],
        'ece': lambda m: m['ece'],
        'auroc': lambda m: m['auroc'],
        'auprc': lambda m: m['auprc'],
    }
    values = np.array([key_map[metric_key](sr['metrics']) for sr in seed_results])
    if np.std(values) < 1e-10 or np.any(np.isnan(values)):
        return {'statistic': np.nan, 'p_value': np.nan, 'n_seeds': len(values),
                'values': [float(v) for v in values], 'note': 'zero variance or NaN'}
    stat, p_value = levene(*values.tolist())
    return {'statistic': float(stat), 'p_value': float(p_value), 'n_seeds': len(values), 'values': [float(v) for v in values]}


def aggregate_seed_results(seed_results):
    metrics_list = [sr['metrics'] for sr in seed_results]

    def stats(key):
        vals = [m[key] for m in metrics_list]
        return {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'values': [float(v) for v in vals],
        }

    return {k: stats(k) for k in ['brier_score', 'ece', 'auroc', 'auprc']}


def plot_calibration_curve(held_out_metrics, seed_aggregates, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)

    cc = held_out_metrics['calibration_curve']
    ax.plot(cc['bin_centers'], cc['observed_accuracy'], 'bo-',
            label=f"Held-out (ECE={held_out_metrics['ece']:.3f})",
            linewidth=2, markersize=6)
    ax.fill_between(cc['bin_centers'], cc['bin_centers'], cc['observed_accuracy'],
                    alpha=0.1, color='blue')

    ax.set_xlabel('Confidence (predicted probability)', fontsize=12)
    ax.set_ylabel('Fraction of positives (accuracy)', fontsize=12)
    ax.set_title('Calibration Curve — Probe g(h₃₀(x)) = σ(w^T h₃₀(x) + b)', fontsize=13)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    textstr = (f"Held-out\n"
               f"Brier: {held_out_metrics['brier_score']:.3f}\n"
               f"ECE: {held_out_metrics['ece']:.3f}\n"
               f"AUROC: {held_out_metrics['auroc']:.3f}\n"
               f"5-seed CV ECE: {seed_aggregates['ece']['mean']:.3f} ± {seed_aggregates['ece']['std']:.3f}")
    ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved calibration curve to {output_path}")


def main():
    print("=" * 70)
    print("CALIBRATION AUDIT — 5-SEED PROTOCOL")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    activations_dir = data_dir / 'activations'
    output_dir = data_dir / 'ablation_results'
    output_path = output_dir / 'baseline_5seed.json'
    plot_path = project_root / '.sisyphus' / 'evidence' / 'task-1-calibration-curve.png'

    train_jsonl = data_dir / 'probe_extract_results.jsonl'
    held_out_json = data_dir / 'heldout_eval' / 'final_summary.json'
    held_out_results_jsonl = data_dir / 'heldout_eval' / 'final_results.jsonl'

    print("\nLoading training data...")
    train_df = load_training_data(str(train_jsonl))

    print(f"\nLoading training activations (layer {LAYER})...")
    train_activations, train_labels, matched_indices = load_activations_for_df(
        train_df, activations_dir, LAYER
    )
    print(f"  Loaded {len(train_activations)} activations, shape: {train_activations.shape}")

    print("\nLoading held-out data...")
    held_out_scores, held_out_labels, held_out_metadata = load_held_out_data(
        str(held_out_json), str(held_out_results_jsonl)
    )

    print("\n5-seed CV on training data...")
    seed_results = run_5seed_audit(train_activations, train_labels)

    print("\nAggregating seed results...")
    seed_aggregates = aggregate_seed_results(seed_results)
    print(f"  Brier:  {seed_aggregates['brier_score']['mean']:.4f} ± {seed_aggregates['brier_score']['std']:.4f}")
    print(f"  ECE:    {seed_aggregates['ece']['mean']:.4f} ± {seed_aggregates['ece']['std']:.4f}")
    print(f"  AUROC:  {seed_aggregates['auroc']['mean']:.4f} ± {seed_aggregates['auroc']['std']:.4f}")
    print(f"  AUPRC:  {seed_aggregates['auprc']['mean']:.4f} ± {seed_aggregates['auprc']['std']:.4f}")

    print("\nLevene's test for seed homogeneity...")
    levene_results = {}
    for metric_key in ['brier_score', 'ece', 'auroc', 'auprc']:
        lr = levene_test(seed_results, metric_key)
        levene_results[metric_key] = lr
        sig = "SIGNIFICANT" if lr['p_value'] < 0.05 else "not significant"
        print(f"  {metric_key}: W={lr['statistic']:.4f}, p={lr['p_value']:.4f} ({sig})")

    print("\nHeld-out set audit...")
    held_out_metrics = run_heldout_audit(held_out_scores, held_out_labels, held_out_metadata)

    decision = "Calibration is FINE (ECE < 0.05)" if held_out_metrics['ece'] < ECE_THRESHOLD else "Calibration NEEDS FIXING (ECE >= 0.05)"
    print(f"\n{'=' * 70}")
    print(f"DECISION: {decision}")
    print(f"{'=' * 70}")

    print("\nPlotting calibration curve...")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_calibration_curve(held_out_metrics, seed_aggregates, plot_path)

    print(f"\nSaving results to {output_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'type': 'calibration_audit_5seed',
            'probe': 'g(h_30(x)) = sigma(w^T h_30(x) + b)',
            'layer': LAYER,
            'n_seeds': N_SEEDS,
            'n_bins': N_BINS,
            'ece_threshold': ECE_THRESHOLD,
            'training_samples': int(len(train_activations)),
            'held_out_samples': int(len(held_out_scores)),
        },
        'decision': decision,
        'held_out': held_out_metrics,
        'cv_5seed': {
            'per_seed': seed_results,
            'aggregated': seed_aggregates,
        },
        'levene_test': levene_results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  Saved {os.path.getsize(output_path)} bytes")

    print(f"\n{'=' * 70}")
    print("CALIBRATION AUDIT COMPLETE")
    print(f"  Held-out ECE: {held_out_metrics['ece']:.4f}")
    print(f"  Held-out Brier: {held_out_metrics['brier_score']:.4f}")
    print(f"  Held-out AUROC: {held_out_metrics['auroc']:.4f}")
    print(f"  Held-out AUPRC: {held_out_metrics['auprc']:.4f}")
    print(f"  5-seed CV ECE: {seed_aggregates['ece']['mean']:.4f} +/- {seed_aggregates['ece']['std']:.4f}")
    print(f"  {decision}")
    print(f"{'=' * 70}")

    return output


if __name__ == '__main__':
    main()
    sys.exit(0)