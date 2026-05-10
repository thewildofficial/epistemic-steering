#!/usr/bin/env uv run python
"""
Multi-Layer Ensemble Probe
Train per-layer probes at layers 5, 10, 15, 20, 25, 30.
Combine via majority vote, mean probability, and logistic regression stacking.
Compare against single-layer baseline at LAYER 25 (NOT layer 30).
5-seed protocol.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import t as t_dist
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations_allpos"
LABELS_FILE = PROJECT_ROOT / "data" / "probe_extract_allpos_results.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ablation_results" / "ensemble_5seed.json"

ENSEMBLE_LAYERS = [5, 10, 15, 20, 25, 30]
BASELINE_LAYER = 25  # IMPORTANT: baseline is layer 25, NOT layer 30
N_SEEDS = 5
N_FOLDS = 5
DECISION_AUROC_DELTA = 0.02
DECISION_COHENS_D = 0.5


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


def load_activations_for_layer(question_ids, layer):
    """Load activations for a specific layer."""
    activations = {}
    for qid in question_ids:
        filepath = ACTIVATIONS_DIR / f"{qid}__layer_{layer}.npy"
        if filepath.exists():
            activations[qid] = np.load(filepath)
    return activations


def get_last_token(arr):
    return arr[-1, :] if arr.ndim == 2 else arr


def cohen_d(x1, x2):
    """Compute Cohen's d between two groups of values."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std


def bootstrap_ci(x1, x2, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap CI for difference in means."""
    diffs = []
    for _ in range(n_bootstrap):
        boot1 = np.random.choice(x1, size=len(x1), replace=True)
        boot2 = np.random.choice(x2, size=len(x2), replace=True)
        diffs.append(np.mean(boot1) - np.mean(boot2))
    alpha = (1 - ci) / 2
    return np.percentile(diffs, [alpha * 100, (1 - alpha) * 100])


def run_single_layer_probe(X_train, y_train, X_test, seed):
    """Train a single layer probe and return predictions."""
    model = LogisticRegressionCV(
        Cs=10,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
        scoring='roc_auc',
        max_iter=1000,
        random_state=seed
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_prob, model


def run_ensemble_seed(question_ids, labels, held_out_ids, activations_by_layer, seed):
    """
    Run ensemble experiment for one seed.
    Returns per-layer AUROCs, and three ensemble AUROCs.
    """
    train_ids = [qid for qid in question_ids if qid not in held_out_ids]
    test_ids = [qid for qid in question_ids if qid in held_out_ids]

    # Prepare data
    y_train = np.array([labels[qid]['correct'] for qid in train_ids])
    y_test = np.array([labels[qid]['correct'] for qid in test_ids])

    # Train per-layer probes and collect predictions
    layer_probs = {}  # layer -> array of probs for test set
    layer_models = {}  # layer -> trained model

    for layer in ENSEMBLE_LAYERS:
        activations = activations_by_layer[layer]
        X_train = np.array([get_last_token(activations[qid]) for qid in train_ids])
        X_test = np.array([get_last_token(activations[qid]) for qid in test_ids])

        y_prob, model = run_single_layer_probe(X_train, y_train, X_test, seed)
        layer_probs[layer] = y_prob
        layer_models[layer] = model

    # Compute per-layer AUROCs
    per_layer_results = {}
    for layer in ENSEMBLE_LAYERS:
        per_layer_results[layer] = {
            'auroc': roc_auc_score(y_test, layer_probs[layer]),
            'brier': brier_score_loss(y_test, layer_probs[layer]),
            'ece': compute_ece(y_test, layer_probs[layer]),
        }

    # === ENSEMBLE METHODS ===

    # (a) Majority vote: each probe votes correct/incorrect (threshold 0.5), final = mode
    votes_per_layer = {}
    for layer in ENSEMBLE_LAYERS:
        votes_per_layer[layer] = (layer_probs[layer] >= 0.5).astype(int)

    # Stack votes: shape = (n_test, n_layers)
    vote_matrix = np.column_stack([votes_per_layer[layer] for layer in ENSEMBLE_LAYERS])
    # Majority vote: count votes, threshold = n_layers/2
    majority_pred = (np.sum(vote_matrix, axis=1) >= len(ENSEMBLE_LAYERS) / 2).astype(int)
    majority_auroc = roc_auc_score(y_test, majority_pred)
    majority_brier = brier_score_loss(y_test, majority_pred)
    majority_ece = compute_ece(y_test, majority_pred)

    # (b) Mean probability: average probe scores across layers, threshold 0.5
    prob_matrix = np.column_stack([layer_probs[layer] for layer in ENSEMBLE_LAYERS])
    mean_prob = np.mean(prob_matrix, axis=1)
    mean_auroc = roc_auc_score(y_test, mean_prob)
    mean_brier = brier_score_loss(y_test, mean_prob)
    mean_ece = compute_ece(y_test, mean_prob)

    # (c) Stacking: train a meta-classifier on per-layer probe scores
    # Use training data to get out-of-fold predictions for meta-classifier
    meta_features_train = np.column_stack([
        layer_models[layer].predict_proba(
            np.array([get_last_token(activations_by_layer[layer][qid]) for qid in train_ids])
        )[:, 1]
        for layer in ENSEMBLE_LAYERS
    ])
    meta_features_test = prob_matrix  # already have test predictions

    # Train meta-classifier with inner CV on training data
    meta_model = LogisticRegressionCV(
        Cs=10,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
        scoring='roc_auc',
        max_iter=1000,
        random_state=seed
    )
    meta_model.fit(meta_features_train, y_train)
    stacking_prob = meta_model.predict_proba(meta_features_test)[:, 1]
    stacking_auroc = roc_auc_score(y_test, stacking_prob)
    stacking_brier = brier_score_loss(y_test, stacking_prob)
    stacking_ece = compute_ece(y_test, stacking_prob)

    # Baseline: Layer 25 single probe (same as layer_probs[25])
    baseline_auroc = per_layer_results[BASELINE_LAYER]['auroc']
    baseline_brier = per_layer_results[BASELINE_LAYER]['brier']
    baseline_ece = per_layer_results[BASELINE_LAYER]['ece']

    return {
        'baseline': {
            'layer': BASELINE_LAYER,
            'auroc': baseline_auroc,
            'brier': baseline_brier,
            'ece': baseline_ece,
        },
        'per_layer': per_layer_results,
        'majority_vote': {
            'auroc': majority_auroc,
            'brier': majority_brier,
            'ece': majority_ece,
        },
        'mean_probability': {
            'auroc': mean_auroc,
            'brier': mean_brier,
            'ece': mean_ece,
        },
        'stacking': {
            'auroc': stacking_auroc,
            'brier': stacking_brier,
            'ece': stacking_ece,
        },
        'y_test': y_test.tolist(),
        'layer_probs': {layer: layer_probs[layer].tolist() for layer in ENSEMBLE_LAYERS},
    }


def run_experiment():
    print("=" * 60)
    print("Multi-Layer Ensemble Probe")
    print("=" * 60)
    print(f"Ensemble layers: {ENSEMBLE_LAYERS}")
    print(f"Baseline layer: {BASELINE_LAYER}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Decision rule: AUROC Δ >= {DECISION_AUROC_DELTA} AND Cohen's d >= {DECISION_COHENS_D}")

    # Load labels
    print("\nLoading labels...")
    labels = load_labels()
    print(f"  Loaded {len(labels)} question labels")

    # Get valid question IDs
    print("\nScanning activations directory...")
    all_question_ids = get_valid_question_ids(ACTIVATIONS_DIR, labels)
    print(f"  Found {len(all_question_ids)} valid question IDs")

    # Create held-out split (20%)
    np.random.seed(42)
    all_qids_list = list(all_question_ids)
    np.random.shuffle(all_qids_list)
    held_out_size = int(len(all_qids_list) * 0.20)
    held_out_ids = set(all_qids_list[:held_out_size])
    print(f"  Held-out split: {len(held_out_ids)} questions ({held_out_size/len(all_qids_list)*100:.0f}%)")

    # Load activations for all ensemble layers
    print("\nLoading activations for all layers...")
    activations_by_layer = {}
    for layer in ENSEMBLE_LAYERS:
        print(f"  Layer {layer}...", end=" ", flush=True)
        activations_by_layer[layer] = load_activations_for_layer(all_question_ids, layer)
        print(f"loaded {len(activations_by_layer[layer])} activations")

    # Run 5-seed experiment
    print("\n" + "=" * 60)
    print("Running 5-seed ensemble experiments...")
    print("=" * 60)

    all_seed_results = []
    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        seed_result = run_ensemble_seed(
            all_question_ids, labels, held_out_ids, activations_by_layer, seed
        )
        all_seed_results.append(seed_result)

        # Print per-layer and ensemble results for this seed
        print(f"  Per-layer AUROCs:")
        for layer in ENSEMBLE_LAYERS:
            print(f"    L{layer}: AUROC={seed_result['per_layer'][layer]['auroc']:.4f}", end="")
        print()

        print(f"  Ensemble AUROCs:")
        print(f"    Majority vote: {seed_result['majority_vote']['auroc']:.4f}")
        print(f"    Mean prob:     {seed_result['mean_probability']['auroc']:.4f}")
        print(f"    Stacking:     {seed_result['stacking']['auroc']:.4f}")
        print(f"    Baseline (L{BASELINE_LAYER}): {seed_result['baseline']['auroc']:.4f}")

    # Aggregate results across seeds
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS (across 5 seeds)")
    print("=" * 60)

    # Per-layer aggregation
    per_layer_agg = {}
    for layer in ENSEMBLE_LAYERS:
        aurocs = [r['per_layer'][layer]['auroc'] for r in all_seed_results]
        briers = [r['per_layer'][layer]['brier'] for r in all_seed_results]
        eces = [r['per_layer'][layer]['ece'] for r in all_seed_results]
        per_layer_agg[layer] = {
            'auroc_mean': np.mean(aurocs),
            'auroc_std': np.std(aurocs),
            'brier_mean': np.mean(briers),
            'brier_std': np.std(briers),
            'ece_mean': np.mean(eces),
            'ece_std': np.std(eces),
        }
        print(f"  Layer {layer}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}, "
              f"Brier={np.mean(briers):.4f}±{np.std(briers):.4f}")

    # Ensemble method aggregation
    ensemble_methods = ['majority_vote', 'mean_probability', 'stacking']
    ensemble_agg = {}
    for method in ensemble_methods:
        aurocs = [r[method]['auroc'] for r in all_seed_results]
        briers = [r[method]['brier'] for r in all_seed_results]
        eces = [r[method]['ece'] for r in all_seed_results]
        ensemble_agg[method] = {
            'auroc_mean': np.mean(aurocs),
            'auroc_std': np.std(aurocs),
            'brier_mean': np.mean(briers),
            'brier_std': np.std(briers),
            'ece_mean': np.mean(eces),
            'ece_std': np.std(eces),
        }
        print(f"  {method}: AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}, "
              f"Brier={np.mean(briers):.4f}±{np.std(briers):.4f}")

    # Baseline aggregation (layer 25)
    baseline_aurocs = [r['baseline']['auroc'] for r in all_seed_results]
    baseline_briers = [r['baseline']['brier'] for r in all_seed_results]
    baseline_eces = [r['baseline']['ece'] for r in all_seed_results]
    baseline_agg = {
        'auroc_mean': np.mean(baseline_aurocs),
        'auroc_std': np.std(baseline_aurocs),
        'brier_mean': np.mean(baseline_briers),
        'brier_std': np.std(baseline_briers),
        'ece_mean': np.mean(baseline_eces),
        'ece_std': np.std(baseline_eces),
    }
    print(f"  Baseline (L{BASELINE_LAYER}): AUROC={np.mean(baseline_aurocs):.4f}±{np.std(baseline_aurocs):.4f}, "
          f"Brier={np.mean(baseline_briers):.4f}±{np.std(baseline_briers):.4f}")

    # Compute deltas and effect sizes
    print("\n" + "=" * 60)
    print("COMPARISON VS BASELINE (Layer 25)")
    print("=" * 60)

    best_method = None
    best_delta = -999

    comparison_results = {}
    for method in ensemble_methods:
        delta_auroc = ensemble_agg[method]['auroc_mean'] - baseline_agg['auroc_mean']
        delta_brier = ensemble_agg[method]['brier_mean'] - baseline_agg['brier_mean']
        delta_ece = ensemble_agg[method]['ece_mean'] - baseline_agg['ece_mean']

        # Cohen's d using per-seed AUROC differences
        method_aurocs = [r[method]['auroc'] for r in all_seed_results]
        d = cohen_d(method_aurocs, baseline_aurocs)

        # 95% CI for AUROC difference (bootstrap)
        ci_low, ci_high = bootstrap_ci(method_aurocs, baseline_aurocs)

        comparison_results[method] = {
            'delta_auroc': delta_auroc,
            'delta_brier': delta_brier,
            'delta_ece': delta_ece,
            'cohen_d': d,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'auroc_mean': ensemble_agg[method]['auroc_mean'],
            'brier_mean': ensemble_agg[method]['brier_mean'],
            'ece_mean': ensemble_agg[method]['ece_mean'],
        }

        print(f"\n  {method}:")
        print(f"    AUROC Δ: {delta_auroc:+.4f} (ensemble={ensemble_agg[method]['auroc_mean']:.4f}, "
              f"baseline={baseline_agg['auroc_mean']:.4f})")
        print(f"    Brier Δ: {delta_brier:+.4f}")
        print(f"    ECE Δ:   {delta_ece:+.4f}")
        print(f"    Cohen's d: {d:.4f}")
        print(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        # Track best method
        if delta_auroc > best_delta:
            best_delta = delta_auroc
            best_method = method

    # Apply decision rule
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)

    decision = "REJECT"
    if best_method is not None:
        best_result = comparison_results[best_method]
        decision_auroc = best_result['delta_auroc'] >= DECISION_AUROC_DELTA
        decision_d = best_result['cohen_d'] >= DECISION_COHENS_D

        if decision_auroc and decision_d:
            decision = "KEEP"
        elif not decision_auroc:
            decision = "REJECT (AUROC Δ < threshold)"
        elif not decision_d:
            decision = "REJECT (Cohen's d < threshold)"

        print(f"  Best method: {best_method}")
        print(f"  AUROC Δ = {best_result['delta_auroc']:+.4f} (threshold: >={DECISION_AUROC_DELTA})")
        print(f"  Cohen's d = {best_result['cohen_d']:.4f} (threshold: >={DECISION_COHENS_D})")
        print(f"  95% CI: [{best_result['ci_95_low']:.4f}, {best_result['ci_95_high']:.4f}]")
    print(f"  DECISION: {decision}")

    # Prepare output
    output = {
        'ensemble_layers': ENSEMBLE_LAYERS,
        'baseline_layer': BASELINE_LAYER,
        'n_seeds': N_SEEDS,
        'decision_rule': {
            'aurocc_delta_threshold': DECISION_AUROC_DELTA,
            'cohens_d_threshold': DECISION_COHENS_D,
        },
        'per_seed': all_seed_results,
        'per_layer_aggregated': per_layer_agg,
        'baseline_aggregated': baseline_agg,
        'ensemble_aggregated': ensemble_agg,
        'comparison_vs_baseline': comparison_results,
        'best_method': best_method,
        'best_delta_auroc': best_delta,
        'decision': decision,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\nResults written to: {OUTPUT_FILE}")
    return output


if __name__ == "__main__":
    run_experiment()