#!/usr/bin/env uv run python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import levene
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations_allpos"
LABELS_FILE = PROJECT_ROOT / "data" / "probe_extract_allpos_results.jsonl"
HELD_OUT_FILE = PROJECT_ROOT / "data" / "heldout_questions.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ablation_results" / "layer_sweep_5seed.json"

LAYERS = [5, 10, 15, 20, 25, 30, 31]
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


def load_held_out_question_ids():
    held_out_ids = set()
    with open(HELD_OUT_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            held_out_ids.add(item['question_id'])
    return held_out_ids


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


def run_seed(activations_dict, labels, question_ids, held_out_ids, layer, seed):
    train_ids = [qid for qid in question_ids if qid not in held_out_ids]
    test_ids = [qid for qid in question_ids if qid in held_out_ids]

    def get_last_token(arr):
        return arr[-1, :] if arr.ndim == 2 else arr

    X_train = np.array([get_last_token(activations_dict[qid]) for qid in train_ids])
    y_train = np.array([labels[qid]['correct'] for qid in train_ids])
    X_test = np.array([get_last_token(activations_dict[qid]) for qid in test_ids])
    y_test = np.array([labels[qid]['correct'] for qid in test_ids])

    model = LogisticRegressionCV(
        Cs=10,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
        scoring='roc_auc',
        max_iter=1000,
        random_state=seed
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_test, y_prob)

    return {
        'auroc': auroc,
        'brier': brier,
        'ece': ece,
        'y_prob': y_prob.tolist(),
        'y_true': y_test.tolist()
    }


def run_layer_experiment(layer, labels, held_out_ids, all_question_ids, activations_by_layer):
    print(f"\n{'='*60}")
    print(f"Layer {layer}")
    print(f"{'='*60}")

    activations_dict = activations_by_layer[layer]

    results = {
        'layer': layer,
        'seeds': [],
        'per_dataset': {}
    }

    for seed in range(N_SEEDS):
        seed_result = run_seed(
            activations_dict, labels, all_question_ids, held_out_ids, layer, seed
        )
        results['seeds'].append(seed_result)
        print(f"  Seed {seed}: AUROC={seed_result['auroc']:.4f}, Brier={seed_result['brier']:.4f}, ECE={seed_result['ece']:.4f}")

    auroc_values = [s['auroc'] for s in results['seeds']]
    brier_values = [s['brier'] for s in results['seeds']]
    ece_values = [s['ece'] for s in results['seeds']]

    levene_auroc = levene(*[[s['auroc']] for s in results['seeds']])
    levene_brier = levene(*[[s['brier']] for s in results['seeds']])

    results['aggregated'] = {
        'auroc_mean': np.mean(auroc_values),
        'auroc_std': np.std(auroc_values),
        'brier_mean': np.mean(brier_values),
        'brier_std': np.std(brier_values),
        'ece_mean': np.mean(ece_values),
        'ece_std': np.std(ece_values),
        'levene_auroc_p': levene_auroc.pvalue,
        'levene_brier_p': levene_brier.pvalue
    }

    for dataset in ['mmlu', 'gsm8k']:
        dataset_test_ids = [qid for qid in held_out_ids if labels[qid]['dataset'] == dataset]
        if len(dataset_test_ids) == 0:
            continue

        y_true = results['seeds'][0]['y_true']
        y_prob = results['seeds'][0]['y_prob']

        dataset_mask = [labels[qid]['dataset'] == dataset for qid in held_out_ids]
        dataset_y_true = [y for y, m in zip(y_true, dataset_mask) if m]
        dataset_y_prob = [y for y, m in zip(y_prob, dataset_mask) if m]

        if len(dataset_y_true) > 0 and len(np.unique(dataset_y_true)) > 1:
            dataset_auroc = roc_auc_score(dataset_y_true, dataset_y_prob)
            dataset_brier = brier_score_loss(dataset_y_true, dataset_y_prob)
            dataset_ece = compute_ece(dataset_y_true, dataset_y_prob)

            results['per_dataset'][dataset] = {
                'auroc': dataset_auroc,
                'brier': dataset_brier,
                'ece': dataset_ece,
                'n_samples': len(dataset_y_true)
            }

    return results


def main():
    print("Layer Sweep Baseline Validation")
    print("="*60)

    print("\nLoading labels...")
    labels = load_labels()
    print(f"  Loaded {len(labels)} question labels")

    print("\nScanning activations directory...")
    all_question_ids = get_valid_question_ids(ACTIVATIONS_DIR, labels)
    print(f"  Found {len(all_question_ids)} valid question IDs (in both activations and labels)")

    # Create new stratified held-out split (20% of data)
    # Use seed=42 for reproducibility
    np.random.seed(42)
    all_qids_list = list(all_question_ids)
    np.random.shuffle(all_qids_list)

    held_out_size = int(len(all_qids_list) * 0.20)
    held_out_ids = set(all_qids_list[:held_out_size])
    print(f"  Created held-out split: {len(held_out_ids)} questions ({held_out_size/len(all_qids_list)*100:.0f}%)")

    print("\nLoading activations for all layers...")
    activations_by_layer = {}
    for layer in LAYERS:
        print(f"  Layer {layer}...", end=" ", flush=True)
        activations = {}
        for qid in all_question_ids:
            filepath = ACTIVATIONS_DIR / f"{qid}__layer_{layer}.npy"
            if filepath.exists():
                activations[qid] = np.load(filepath)
        activations_by_layer[layer] = activations
        print(f"loaded {len(activations)} activations")

    print("\n" + "="*60)
    print("Running layer sweep experiments...")
    print("="*60)

    all_results = []
    for layer in LAYERS:
        layer_result = run_layer_experiment(
            layer, labels, held_out_ids, all_question_ids, activations_by_layer
        )
        all_results.append(layer_result)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary_rows = []
    for result in all_results:
        agg = result['aggregated']
        summary_rows.append({
            'layer': result['layer'],
            'auroc_mean': agg['auroc_mean'],
            'auroc_std': agg['auroc_std'],
            'brier_mean': agg['brier_mean'],
            'brier_std': agg['brier_std'],
            'ece_mean': agg['ece_mean'],
            'ece_std': agg['ece_std'],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nPer-Layer Results (Held-Out):")
    print(summary_df.to_string(index=False))

    optimal_auroc_idx = summary_df['auroc_mean'].idxmax()
    optimal_by_auroc = summary_df.loc[optimal_auroc_idx, 'layer']
    optimal_auroc_value = summary_df.loc[optimal_auroc_idx, 'auroc_mean']

    optimal_brier_idx = summary_df['brier_mean'].idxmin()
    optimal_by_brier = summary_df.loc[optimal_brier_idx, 'layer']
    optimal_brier_value = summary_df.loc[optimal_brier_idx, 'brier_mean']

    print(f"\nOptimal by AUROC: Layer {optimal_by_auroc} ({optimal_auroc_value:.4f})")
    print(f"Optimal by Brier: Layer {optimal_by_brier} ({optimal_brier_value:.4f})")

    print("\nPer-Dataset Results (MMLU vs GSM8K):")
    for result in all_results:
        layer = result['layer']
        print(f"\nLayer {layer}:")
        for dataset, metrics in result['per_dataset'].items():
            print(f"  {dataset}: AUROC={metrics['auroc']:.4f}, "
                  f"Brier={metrics['brier']:.4f}, ECE={metrics['ece']:.4f} "
                  f"(n={metrics['n_samples']})")

    print("\nVariance Stability (Levene's test p-values):")
    for result in all_results:
        agg = result['aggregated']
        print(f"  Layer {result['layer']}: auroc_p={agg['levene_auroc_p']:.4f}, "
              f"brier_p={agg['levene_brier_p']:.4f}")

    output = {
        'layers': {},
        'optimal_by_auroc': int(optimal_by_auroc),
        'optimal_by_brier': int(optimal_by_brier),
        'decision': None,
        'summary_df': summary_df.to_dict(orient='records')
    }

    for result in all_results:
        layer = result['layer']
        output['layers'][f'layer_{layer}'] = {
            'per_seed': result['seeds'],
            'aggregated': result['aggregated'],
            'per_dataset': result['per_dataset']
        }

    if optimal_by_auroc == optimal_by_brier:
        output['decision'] = f"Layer {optimal_by_auroc} confirmed optimal"
    else:
        output['decision'] = f"Layer {optimal_by_brier} is optimal (changing baseline)"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\nResults written to: {OUTPUT_FILE}")
    print(f"\nFINAL DECISION: {output['decision']}")

    return output


if __name__ == "__main__":
    main()
