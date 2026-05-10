#!/usr/bin/env uv run python
"""
Conformal Prediction with Class-Conditional Nonconformity Score

Key insight: Standard s = 1 - g for ALL examples fails with overconfident probes.
Class-conditional fix:
    s = 1 - g  for CORRECT examples (label=1)  → low when confident+correct
    s = g      for INCORRECT examples (label=0) → high when confident+wrong

Layer: 25 (AUROC 0.928, optimal from layer sweep)
Probe: Single-layer LR with Platt calibration
Coverage: α ∈ {0.05, 0.10, 0.20} via 5-seed protocol
Decision: Viable if |actual_coverage - nominal| ≤ 0.05 for all α
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations_allpos"
LABELS_FILE = PROJECT_ROOT / "data" / "probe_extract_allpos_results.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ablation_results"
OUTPUT_FILE = OUTPUT_DIR / "conformal_5seed.json"

LAYER = 25
SEEDS = [42, 123, 456, 789, 2024]
ALPHAS = [0.05, 0.10, 0.20]
CAL_RATIO = 0.2
PLATT_C = 1e10
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
    return all_qids & set(labels.keys())


def load_activations(qids, activations_dir, layer):
    activations = {}
    for qid in qids:
        filepath = activations_dir / f"{qid}__layer_{layer}.npy"
        if filepath.exists():
            act = np.load(filepath)
            if act.ndim == 2:
                act = act[-1]
            activations[qid] = act.ravel()
    return activations


def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin)
    return ece / len(y_true)


def platt_calibrate(scores_cal, labels_cal, scores_test):
    X_cal = scores_cal.reshape(-1, 1)
    X_test = scores_test.reshape(-1, 1)
    lr = LogisticRegression(C=PLATT_C, solver='lbfgs', max_iter=1000)
    lr.fit(X_cal, labels_cal)
    return lr.predict_proba(X_test)[:, 1]


def class_conditional_nonconformity(scores, labels):
    return np.where(labels == 1, 1.0 - scores, scores)


def standard_nonconformity(scores):
    return 1.0 - scores


def conformal_quantile(cal_scores, alpha):
    n_cal = len(cal_scores)
    k = min(int(np.ceil((n_cal + 1) * (1 - alpha))), n_cal)
    return np.sort(cal_scores)[k - 1]


def conformal_predict(scores, q_alpha):
    threshold_high = 1.0 - q_alpha
    threshold_low = q_alpha

    answer_correct = scores >= threshold_high
    answer_incorrect = scores <= threshold_low
    abstain = ~answer_correct & ~answer_incorrect

    return answer_correct, answer_incorrect, abstain


def compute_coverage(answer_correct, answer_incorrect, labels):
    both_predicted = answer_correct & answer_incorrect
    correctly_predicted = (answer_correct & (labels == 1)) | (answer_incorrect & (labels == 0))
    return float(np.mean(both_predicted | correctly_predicted))


def run_single_seed(scores_cal, labels_cal, scores_test, labels_test, alpha, use_class_cond):
    if use_class_cond:
        cal_nc = class_conditional_nonconformity(scores_cal, labels_cal)
    else:
        cal_nc = standard_nonconformity(scores_cal)

    q_alpha = conformal_quantile(cal_nc, alpha)
    answer_correct, answer_incorrect, abstain = conformal_predict(scores_test, q_alpha)

    coverage = compute_coverage(answer_correct, answer_incorrect, labels_test)
    answer_rate = float(np.mean(answer_correct | answer_incorrect))
    answered_mask = answer_correct | answer_incorrect
    answer_acc = float(labels_test[answer_correct].mean()) if answer_correct.sum() > 0 else 0.0

    threshold_g = 1.0 - q_alpha

    return {
        'alpha': alpha,
        'nominal_coverage': 1 - alpha,
        'q_alpha': float(q_alpha),
        'threshold_g': float(threshold_g),
        'actual_coverage': coverage,
        'coverage_gap': float(coverage - (1 - alpha)),
        'answer_rate': answer_rate,
        'answer_accuracy': answer_acc,
        'n_test': len(labels_test),
        'n_cal': len(labels_cal),
    }


def threshold_baseline(scores, labels, threshold=0.5):
    answer_correct = scores >= threshold
    answer_incorrect = scores <= threshold
    coverage = compute_coverage(answer_correct, answer_incorrect, labels)
    answer_rate = float(np.mean(answer_correct | answer_incorrect))
    answer_acc = float(labels[answer_correct].mean()) if answer_correct.sum() > 0 else 0.0

    return {
        'threshold': threshold,
        'coverage': coverage,
        'answer_rate': answer_rate,
        'answer_accuracy': answer_acc,
    }


def main():
    print("=" * 70)
    print("CONFORMAL PREDICTION WITH CLASS-CONDITIONAL NONCONFORMITY")
    print(f"Layer: {LAYER} | 5-seed protocol | α ∈ {ALPHAS}")
    print("=" * 70)

    print("\n[1] Loading labels...")
    labels = load_labels()
    print(f"    Loaded {len(labels)} question labels")

    print("\n[2] Scanning activations directory...")
    valid_qids = get_valid_question_ids(ACTIVATIONS_DIR, labels)
    print(f"    Found {len(valid_qids)} valid question IDs")

    print(f"\n[3] Loading layer {LAYER} activations...")
    activations = load_activations(valid_qids, ACTIVATIONS_DIR, LAYER)
    print(f"    Loaded {len(activations)} activations")

    qids_list = list(activations.keys())
    X = np.array([activations[qid] for qid in qids_list])
    y = np.array([labels[qid]['correct'] for qid in qids_list])

    print(f"\n[4] Dataset summary:")
    print(f"    Total samples: {len(X)}")
    print(f"    Correct: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"    Incorrect: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    print("\n[5] Computing probe scores via 5-fold CV...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    raw_scores = np.zeros(len(y))
    platt_scores = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        correct_mask = y_train.astype(bool)
        direction = X_train[correct_mask].mean(axis=0) - X_train[~correct_mask].mean(axis=0)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        midpoint = (X_train[correct_mask].mean(axis=0) + X_train[~correct_mask].mean(axis=0)) / 2.0

        raw_scores_val = expit(np.dot(X_val - midpoint, direction))
        train_raw = expit(np.dot(X_train - midpoint, direction))
        platt_scores_val = platt_calibrate(train_raw, y_train, raw_scores_val)

        raw_scores[val_idx] = raw_scores_val
        platt_scores[val_idx] = platt_scores_val

    auroc_raw = roc_auc_score(y, raw_scores)
    auroc_platt = roc_auc_score(y, platt_scores)
    ece_raw = compute_ece(y, raw_scores)
    ece_platt = compute_ece(y, platt_scores)

    print(f"\n    Raw probe — AUROC: {auroc_raw:.4f}, ECE: {ece_raw:.4f}")
    print(f"    Platt — AUROC: {auroc_platt:.4f}, ECE: {ece_platt:.4f}")

    print("\n[6] Running 5-seed conformal prediction protocol...")
    print("-" * 70)

    all_results = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n── Seed {seed_idx + 1}/5: {seed} ──")

        rng = np.random.RandomState(seed)
        n = len(platt_scores)
        indices = rng.permutation(n)

        cal_size = int(n * CAL_RATIO)
        cal_idx = indices[:cal_size]
        test_idx = indices[cal_size:]

        scores_cal = platt_scores[cal_idx]
        labels_cal = y[cal_idx]
        scores_test = platt_scores[test_idx]
        labels_test = y[test_idx]

        print(f"    Cal size: {cal_size}, Test size: {len(test_idx)}")

        seed_result = {
            'seed': seed,
            'cal_size': cal_size,
            'test_size': len(test_idx),
            'class_conditional': {},
            'standard': {},
            'threshold_baseline': threshold_baseline(scores_test, labels_test, 0.5),
            'test_auroc_platt': float(roc_auc_score(labels_test, scores_test)),
        }

        for alpha in ALPHAS:
            cc_result = run_single_seed(scores_cal, labels_cal, scores_test, labels_test, alpha, True)
            std_result = run_single_seed(scores_cal, labels_cal, scores_test, labels_test, alpha, False)

            cc_key = f'alpha_{alpha}'
            seed_result['class_conditional'][cc_key] = cc_result
            seed_result['standard'][cc_key] = std_result

            status_cc = "✓" if abs(cc_result['coverage_gap']) <= 0.05 else "✗"
            status_std = "✓" if abs(std_result['coverage_gap']) <= 0.05 else "✗"

            print(
                f"  α={alpha:.2f} | "
                f"nominal={(1-alpha):.2f} | "
                f"CC actual={cc_result['actual_coverage']:.4f} (gap={cc_result['coverage_gap']:+.4f}) [{status_cc}] | "
                f"Std actual={std_result['actual_coverage']:.4f} (gap={std_result['coverage_gap']:+.4f}) [{status_std}]"
            )

        all_results.append(seed_result)

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)

    aggregated = {}
    decisions = {}

    for method_name, method_key in [('class_conditional', 'class_conditional'), ('standard', 'standard')]:
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        print("-" * 60)

        method_agg = {}
        method_decisions = {}

        for alpha in ALPHAS:
            key = f'alpha_{alpha}'
            coverages = [r[method_key][key]['actual_coverage'] for r in all_results]
            gaps = [r[method_key][key]['coverage_gap'] for r in all_results]
            answer_rates = [r[method_key][key]['answer_rate'] for r in all_results]
            answer_accs = [r[method_key][key]['answer_accuracy'] for r in all_results]
            thresholds = [r[method_key][key]['threshold_g'] for r in all_results]

            agg = {
                'alpha': alpha,
                'nominal_coverage': 1 - alpha,
                'actual_coverage_mean': float(np.mean(coverages)),
                'actual_coverage_std': float(np.std(coverages)),
                'coverage_gap_mean': float(np.mean(gaps)),
                'coverage_gap_std': float(np.std(gaps)),
                'answer_rate_mean': float(np.mean(answer_rates)),
                'answer_accuracy_mean': float(np.mean(answer_accs)),
                'threshold_g_mean': float(np.mean(thresholds)),
            }

            mean_gap = np.mean(gaps)
            max_gap = max(abs(g) for g in gaps)

            if all(abs(g) <= 0.05 for g in gaps):
                decision = "PASS — all seeds within 5% of nominal"
            elif abs(mean_gap) <= 0.05:
                decision = "PASS (mean) — mean within 5%, some seed variation"
            elif abs(mean_gap) <= 0.10:
                decision = "INVESTIGATE — mean gap > 5% but ≤ 10%"
            else:
                decision = "REJECT — mean gap > 10%"

            dec = {
                'decision': decision,
                'mean_coverage_gap': float(mean_gap),
                'max_per_seed_gap': float(max_gap),
                'all_seeds_pass': all(abs(g) <= 0.05 for g in gaps),
            }

            print(
                f"  α={alpha:.2f} (nominal={(1-alpha):.2f}): "
                f"coverage={agg['actual_coverage_mean']:.4f}±{agg['actual_coverage_std']:.4f}, "
                f"gap={agg['coverage_gap_mean']:+.4f}±{agg['coverage_gap_std']:.4f}, "
                f"threshold_g={agg['threshold_g_mean']:.3f}, "
                f"→ {decision}"
            )

            method_agg[key] = agg
            method_decisions[key] = dec

        aggregated[method_key] = method_agg
        decisions[method_key] = method_decisions

    tb_covs = [r['threshold_baseline']['coverage'] for r in all_results]
    tb_accs = [r['threshold_baseline']['answer_accuracy'] for r in all_results]
    tb_rates = [r['threshold_baseline']['answer_rate'] for r in all_results]

    aggregated['threshold_baseline'] = {
        'coverage_mean': float(np.mean(tb_covs)),
        'coverage_std': float(np.std(tb_covs)),
        'answer_accuracy_mean': float(np.mean(tb_accs)),
        'answer_accuracy_std': float(np.std(tb_accs)),
        'answer_rate_mean': float(np.mean(tb_rates)),
        'answer_rate_std': float(np.std(tb_rates)),
    }

    print(f"\nTHRESHOLD BASELINE (g ≥ 0.5):")
    print(f"  Coverage: {aggregated['threshold_baseline']['coverage_mean']:.4f} ± "
          f"{aggregated['threshold_baseline']['coverage_std']:.4f}")
    print(f"  Answer accuracy: {aggregated['threshold_baseline']['answer_accuracy_mean']:.4f} ± "
          f"{aggregated['threshold_baseline']['answer_accuracy_std']:.4f}")
    print(f"  Answer rate: {aggregated['threshold_baseline']['answer_rate_mean']:.3f} ± "
          f"{aggregated['threshold_baseline']['answer_rate_std']:.3f}")

    print("\n" + "=" * 70)
    print("PRODUCTION READINESS DECISION")
    print("=" * 70)

    cc_decisions = decisions['class_conditional']
    cc_all_pass = all(cc_decisions[f'alpha_{a}']['all_seeds_pass'] for a in ALPHAS)
    cc_max_gap = max(abs(cc_decisions[f'alpha_{a}']['mean_coverage_gap']) for a in ALPHAS)

    std_decisions = decisions['standard']
    std_all_pass = all(std_decisions[f'alpha_{a}']['all_seeds_pass'] for a in ALPHAS)
    std_max_gap = max(abs(std_decisions[f'alpha_{a}']['mean_coverage_gap']) for a in ALPHAS)

    if cc_all_pass:
        cc_final = "VIABLE — class-conditional CP meets coverage guarantees at all α levels"
    elif cc_max_gap <= 0.05:
        cc_final = "VIABLE (mean-level) — mean coverage within 5% at all α"
    elif cc_max_gap <= 0.10:
        cc_final = "INVESTIGATE — coverage gaps exceed 5% target"
    else:
        cc_final = "NOT VIABLE — coverage gaps exceed 10%"

    if std_all_pass:
        std_final = "VIABLE — standard CP meets coverage guarantees at all α levels"
    elif std_max_gap <= 0.05:
        std_final = "VIABLE (mean-level)"
    elif std_max_gap <= 0.10:
        std_final = "INVESTIGATE"
    else:
        std_final = "NOT VIABLE"

    print(f"\nClass-Conditional Nonconformity:")
    print(f"  All seeds pass: {cc_all_pass}")
    print(f"  Max mean gap: {cc_max_gap:.4f}")
    print(f"  → {cc_final}")

    print(f"\nStandard Nonconformity (s = 1 - g):")
    print(f"  All seeds pass: {std_all_pass}")
    print(f"  Max mean gap: {std_max_gap:.4f}")
    print(f"  → {std_final}")

    print(f"\nComparison:")
    print(f"  Class-conditional gap: {cc_max_gap:.4f} vs Standard gap: {std_max_gap:.4f}")
    print(f"  {'Class-conditional' if cc_max_gap < std_max_gap else 'Standard'} has smaller maximum gap")

    final_decision = cc_final

    print(f"\n[7] Saving results to {OUTPUT_FILE}...")

    output = {
        'metadata': {
            'task': 'conformal_prediction_class_conditional',
            'probe': f'layer_{LAYER}_analytical_platt',
            'nonconformity': 'class_conditional: s=1-g for correct, s=g for incorrect',
            'comparison': 'standard: s=1-g for all',
            'num_samples': len(y),
            'cal_ratio': CAL_RATIO,
            'alphas': ALPHAS,
            'seeds': SEEDS,
            'decision_rule': 'viable if |actual_coverage - nominal| ≤ 0.05 across all α and seeds',
            'probe_metrics': {
                'auroc_raw': float(auroc_raw),
                'auroc_platt': float(auroc_platt),
                'ece_raw': float(ece_raw),
                'ece_platt': float(ece_platt),
            },
        },
        'per_seed_results': all_results,
        'aggregated': aggregated,
        'decisions': decisions,
        'final_decision': final_decision,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print(f"    Saved {OUTPUT_FILE}")

    print(f"\n{'=' * 70}")
    print(f"FINAL DECISION: {final_decision}")
    print(f"{'=' * 70}")

    return output


if __name__ == '__main__':
    result = main()
    sys.exit(0)