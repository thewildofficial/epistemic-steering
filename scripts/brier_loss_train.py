#!/usr/bin/env uv run python
"""
Brier Loss vs BCE Loss Comparison for Epistemic Probes
======================================================
Tests whether Brier score loss (via HistGradientBoostingRegressor with squared_error)
produces better calibrated probes than BCE loss (LogisticRegressionCV).

Layer: 25 (confirmed optimal from T2: AUROC 0.928)
Baseline BCE: held-out AUROC 0.9678, ECE 0.1011, Brier 0.076
Pre-registered decision: KEEP if ECE improves >= 30% AND AUROC drop < 0.01
"""

import json
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import levene

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations_allpos"
LABELS_FILE = PROJECT_ROOT / "data" / "probe_extract_allpos_results.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ablation_results" / "brier_comparison_5seed.json"

LAYER = 25
N_SEEDS = 5
N_FOLDS = 5


def load_labels():
    labels = {}
    with open(LABELS_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item['question_id']
            labels[qid] = {
                'correct': item['correct'],
                'dataset': item.get('dataset', 'mmlu')
            }
    return labels


def get_valid_question_ids(activations_dir, labels):
    act_files = list(activations_dir.glob("*.npy"))
    all_qids = set()
    for f in act_files:
        parts = f.stem.split("__")
        if len(parts) >= 1:
            all_qids.add(parts[0])
    valid_qids = all_qids & set(labels.keys())
    return valid_qids


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin)
    return ece / len(y_true)


def get_last_token(arr):
    return arr[-1, :] if arr.ndim == 2 else arr


def run_seed_bce(X_train, y_train, X_test, y_test, X_train_full, y_train_full, seed):
    """Train BCE baseline probe (LogisticRegressionCV)."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    model = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        scoring='roc_auc',
        max_iter=1000,
        random_state=seed
    )
    model.fit(X_train, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train_full)[:, 1]

    auroc_test = roc_auc_score(y_test, y_prob_test)
    auroc_train = roc_auc_score(y_train_full, y_prob_train)
    brier_test = brier_score_loss(y_test, y_prob_test)
    brier_train = brier_score_loss(y_train_full, y_prob_train)
    ece_test = compute_ece(y_test, y_prob_test)
    ece_train = compute_ece(y_train_full, y_prob_train)

    return {
        'auroc_test': auroc_test,
        'auroc_train': auroc_train,
        'brier_test': brier_test,
        'brier_train': brier_train,
        'ece_test': ece_test,
        'ece_train': ece_train,
        'y_prob_test': y_prob_test.tolist(),
        'y_true_test': y_test.tolist()
    }


def run_seed_brier(X_train, y_train, X_test, y_test, X_train_full, y_train_full, seed):
    """Train Brier loss probe (HistGradientBoostingRegressor with squared_error)."""
    model = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=100,
        random_state=seed
    )
    model.fit(X_train, y_train)

    y_prob_test = model.predict(X_test)
    y_prob_train = model.predict(X_train_full)

    # Clip probabilities to [0, 1] for Brier/ECE computation
    y_prob_test = np.clip(y_prob_test, 0, 1)
    y_prob_train = np.clip(y_prob_train, 0, 1)

    auroc_test = roc_auc_score(y_test, y_prob_test)
    auroc_train = roc_auc_score(y_train_full, y_prob_train)
    brier_test = brier_score_loss(y_test, y_prob_test)
    brier_train = brier_score_loss(y_train_full, y_prob_train)
    ece_test = compute_ece(y_test, y_prob_test)
    ece_train = compute_ece(y_train_full, y_prob_train)

    return {
        'auroc_test': auroc_test,
        'auroc_train': auroc_train,
        'brier_test': brier_test,
        'brier_train': brier_train,
        'ece_test': ece_test,
        'ece_train': ece_train,
        'y_prob_test': y_prob_test.tolist(),
        'y_true_test': y_test.tolist()
    }


def main():
    print("=" * 70)
    print("Brier Loss vs BCE Loss Comparison")
    print("Layer 25 | 5-Seed Protocol")
    print("=" * 70)

    print("\n[1] Loading labels...")
    labels = load_labels()
    print(f"    Loaded {len(labels)} question labels")

    print("\n[2] Scanning activations directory...")
    all_question_ids = get_valid_question_ids(ACTIVATIONS_DIR, labels)
    print(f"    Found {len(all_question_ids)} valid question IDs")

    print("\n[3] Loading layer 25 activations...")
    activations = {}
    for qid in all_question_ids:
        filepath = ACTIVATIONS_DIR / f"{qid}__layer_{LAYER}.npy"
        if filepath.exists():
            activations[qid] = np.load(filepath)
    print(f"    Loaded {len(activations)} activations for layer {LAYER}")

    print("\n[4] Creating held-out split (20%)...")
    np.random.seed(42)
    all_qids_list = list(all_question_ids)
    np.random.shuffle(all_qids_list)
    held_out_size = int(len(all_qids_list) * 0.20)
    held_out_ids = set(all_qids_list[:held_out_size])
    print(f"    Held-out: {len(held_out_ids)} questions ({held_out_size/len(all_qids_list)*100:.0f}%)")

    train_ids = [qid for qid in all_question_ids if qid not in held_out_ids]
    test_ids = [qid for qid in all_question_ids if qid in held_out_ids]

    # Full train set (used for final model trained on all train data)
    X_train_full = np.array([get_last_token(activations[qid]) for qid in train_ids])
    y_train_full = np.array([labels[qid]['correct'] for qid in train_ids])

    # Test set
    X_test = np.array([get_last_token(activations[qid]) for qid in test_ids])
    y_test = np.array([labels[qid]['correct'] for qid in test_ids])

    print(f"    Train size: {len(train_ids)}, Test size: {len(test_ids)}")

    print("\n[5] Running 5-seed comparison...")
    print("-" * 70)

    results = {
        'layer': LAYER,
        'bce_baseline': {'seeds': []},
        'brier_probe': {'seeds': []}
    }

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed}:")

        # BCE baseline
        bce_result = run_seed_bce(
            X_train_full, y_train_full, X_test, y_test, X_train_full, y_train_full, seed
        )
        results['bce_baseline']['seeds'].append(bce_result)

        auroc_gap_bce = bce_result['auroc_train'] - bce_result['auroc_test']
        print(f"    BCE: AUROC={bce_result['auroc_test']:.4f} (train-test gap: {auroc_gap_bce:.4f}), "
              f"Brier={bce_result['brier_test']:.4f}, ECE={bce_result['ece_test']:.4f}")

        # Brier probe
        brier_result = run_seed_brier(
            X_train_full, y_train_full, X_test, y_test, X_train_full, y_train_full, seed
        )
        results['brier_probe']['seeds'].append(brier_result)

        auroc_gap_brier = brier_result['auroc_train'] - brier_result['auroc_test']
        print(f"    Brier: AUROC={brier_result['auroc_test']:.4f} (train-test gap: {auroc_gap_brier:.4f}), "
              f"Brier={brier_result['brier_test']:.4f}, ECE={brier_result['ece_test']:.4f}")

        # Delta
        delta_auroc = bce_result['auroc_test'] - brier_result['auroc_test']
        delta_ece = bce_result['ece_test'] - brier_result['ece_test']
        print(f"    ΔAUROC={delta_auroc:+.4f}, ΔECE={delta_ece:+.4f}")

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    # Aggregate
    bce_aurocs = [s['auroc_test'] for s in results['bce_baseline']['seeds']]
    bce_briers = [s['brier_test'] for s in results['bce_baseline']['seeds']]
    bce_eces = [s['ece_test'] for s in results['bce_baseline']['seeds']]
    bce_auroc_gaps = [s['auroc_train'] - s['auroc_test'] for s in results['bce_baseline']['seeds']]

    brier_aurocs = [s['auroc_test'] for s in results['brier_probe']['seeds']]
    brier_briers = [s['brier_test'] for s in results['brier_probe']['seeds']]
    brier_eces = [s['ece_test'] for s in results['brier_probe']['seeds']]
    brier_auroc_gaps = [s['auroc_train'] - s['auroc_test'] for s in results['brier_probe']['seeds']]

    results['bce_baseline']['aggregated'] = {
        'auroc_mean': np.mean(bce_aurocs),
        'auroc_std': np.std(bce_aurocs),
        'brier_mean': np.mean(bce_briers),
        'brier_std': np.std(bce_briers),
        'ece_mean': np.mean(bce_eces),
        'ece_std': np.std(bce_eces),
        'auroc_gap_mean': np.mean(bce_auroc_gaps),
    }

    results['brier_probe']['aggregated'] = {
        'auroc_mean': np.mean(brier_aurocs),
        'auroc_std': np.std(brier_aurocs),
        'brier_mean': np.mean(brier_briers),
        'brier_std': np.std(brier_briers),
        'ece_mean': np.mean(brier_eces),
        'ece_std': np.std(brier_eces),
        'auroc_gap_mean': np.mean(brier_auroc_gaps),
    }

    # Delta
    bce_agg = results['bce_baseline']['aggregated']
    brier_agg = results['brier_probe']['aggregated']

    delta_auroc = bce_agg['auroc_mean'] - brier_agg['auroc_mean']
    delta_ece = bce_agg['ece_mean'] - brier_agg['ece_mean']
    delta_brier = brier_agg['brier_mean'] - bce_agg['brier_mean']

    # ECE improvement percentage
    ece_improvement_pct = (delta_ece / bce_agg['ece_mean']) * 100 if bce_agg['ece_mean'] > 0 else 0
    auroc_drop = -delta_auroc if delta_auroc < 0 else 0  # Positive if AUROC dropped

    results['deltas'] = {
        'delta_auroc': delta_auroc,
        'delta_ece': delta_ece,
        'delta_brier': delta_brier,
        'ece_improvement_pct': ece_improvement_pct,
        'auroc_drop': auroc_drop
    }

    print(f"\nBCE Baseline:")
    print(f"  AUROC: {bce_agg['auroc_mean']:.4f} ± {bce_agg['auroc_std']:.4f}")
    print(f"  Brier: {bce_agg['brier_mean']:.4f} ± {bce_agg['brier_std']:.4f}")
    print(f"  ECE:   {bce_agg['ece_mean']:.4f} ± {bce_agg['ece_std']:.4f}")
    print(f"  Train-test gap: {bce_agg['auroc_gap_mean']:.4f}")

    print(f"\nBrier Probe:")
    print(f"  AUROC: {brier_agg['auroc_mean']:.4f} ± {brier_agg['auroc_std']:.4f}")
    print(f"  Brier: {brier_agg['brier_mean']:.4f} ± {brier_agg['brier_std']:.4f}")
    print(f"  ECE:   {brier_agg['ece_mean']:.4f} ± {brier_agg['ece_std']:.4f}")
    print(f"  Train-test gap: {brier_agg['auroc_gap_mean']:.4f}")

    print(f"\nDeltas (BCE - Brier):")
    print(f"  ΔAUROC: {delta_auroc:+.4f} (AUROC drop: {auroc_drop:.4f})")
    print(f"  ΔBrier: {delta_brier:+.4f}")
    print(f"  ΔECE:   {delta_ece:+.4f} (ECE improvement: {ece_improvement_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("OVERFITTING CHECK")
    print("=" * 70)
    max_gap = 0.15
    bce_ok = bce_agg['auroc_gap_mean'] < max_gap
    brier_ok = brier_agg['auroc_gap_mean'] < max_gap
    print(f"  BCE train-test gap: {bce_agg['auroc_gap_mean']:.4f} {'OK' if bce_ok else 'FAIL'} (threshold < {max_gap})")
    print(f"  Brier train-test gap: {brier_agg['auroc_gap_mean']:.4f} {'OK' if brier_ok else 'FAIL'} (threshold < {max_gap})")

    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    ece_threshold_pct = 30
    auroc_drop_threshold = 0.01

    decision = "DISCARD"
    rationale = []

    if ece_improvement_pct >= ece_threshold_pct and auroc_drop < auroc_drop_threshold:
        decision = "KEEP"
        rationale.append(f"ECE improved by {ece_improvement_pct:.1f}% (>= {ece_threshold_pct}% threshold)")
        rationale.append(f"AUROC drop {auroc_drop:.4f} < {auroc_drop_threshold} threshold")
    elif ece_improvement_pct < 10:
        decision = "DISCARD"
        rationale.append(f"ECE improvement {ece_improvement_pct:.1f}% < 10% minimum")
    else:
        rationale.append(f"ECE improvement {ece_improvement_pct:.1f}% insufficient (need >= {ece_threshold_pct}%)")
        rationale.append(f"OR AUROC drop {auroc_drop:.4f} >= {auroc_drop_threshold}")
        rationale.append("Pre-registered: KEEP if ECE improves >= 30% AND AUROC drop < 0.01")

    results['decision'] = decision
    results['rationale'] = rationale

    print(f"\n  Decision: {decision}")
    for r in rationale:
        print(f"    - {r}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\n  Results written to: {OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    main()