import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path("/Users/aban/drive/Projects/epistemic-steering")
DATA_DIR = PROJECT_DIR / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations_allpos"
PROBE_EXTRACT_FILE = DATA_DIR / "probe_extract_allpos_results.jsonl"
RESULTS_DIR = DATA_DIR / "ablation_results"
RESULTS_FILE = RESULTS_DIR / "feature_aug_5seed.json"

np.random.seed(42)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def compute_entropy_from_hidden(logits_vector):
    probs = softmax(logits_vector)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

def get_last_token_activation(act_array):
    return act_array[-1, :] if act_array.ndim == 2 else act_array

def load_data():
    print("Loading probe_extract_allpos_results.jsonl...")
    data_records = []
    with open(PROBE_EXTRACT_FILE) as f:
        for line in f:
            data_records.append(json.loads(line))
    
    qid_info = {r['question_id']: {'dataset': r['dataset'], 'correct': int(r['correct']), 'top_token_prob': r['top_token_prob']} for r in data_records}
    all_qids = list(qid_info.keys())
    print(f"Total questions: {len(all_qids)}")
    
    mmlu_qids = [q for q in all_qids if qid_info[q]['dataset'] == 'mmlu']
    gsm_qids = [q for q in all_qids if qid_info[q]['dataset'] == 'gsm8k']
    print(f"MMLU: {len(mmlu_qids)}, GSM8K: {len(gsm_qids)}")
    
    print("\nLoading layer 25 activations and computing entropy from logits...")
    activations = {}
    entropies = {}
    max_probs = {}
    missing_entropy = 0
    
    for qid in all_qids:
        act_file = ACTIVATIONS_DIR / f"{qid}__layer_25.npy"
        logits_file = ACTIVATIONS_DIR / f"{qid}__logits.npy"
        
        if not act_file.exists():
            continue
        
        act = np.load(act_file)
        h = get_last_token_activation(act)
        activations[qid] = h
        
        max_probs[qid] = qid_info[qid]['top_token_prob']
        
        if logits_file.exists():
            logits = np.load(logits_file)
            last_token_logits = logits[-(2560):]
            ent = compute_entropy_from_hidden(last_token_logits)
            entropies[qid] = ent
        else:
            entropies[qid] = np.nan
            missing_entropy += 1
    
    valid_qids = list(activations.keys())
    print(f"Valid qids: {len(valid_qids)}, missing entropy (no logits file): {missing_entropy}")
    
    return qid_info, valid_qids, mmlu_qids, gsm_qids, activations, entropies, max_probs

def build_X(h, ent, mp, feature_type):
    h = np.array(h)
    ent = np.array(ent)
    mp = np.array(mp)
    
    if feature_type == 'h_only':
        return h.reshape(1, -1)
    elif feature_type == 'h+entropy':
        if np.isnan(ent):
            return None
        return np.hstack([h.reshape(1, -1), ent.reshape(1, -1)])
    elif feature_type == 'h+maxprob':
        return np.hstack([h.reshape(1, -1), mp.reshape(1, -1)])
    elif feature_type == 'h+both':
        if np.isnan(ent):
            return None
        return np.hstack([h.reshape(1, -1), ent.reshape(1, -1), mp.reshape(1, -1)])

def evaluate_probe(X_train, y_train, X_test, y_test, seed):
    model = LogisticRegressionCV(Cs=10, cv=5, solver='lbfgs', max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    return auroc, brier

def cohen_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std

def run_experiment():
    print("=" * 60)
    print("Feature Augmentation — Layer 25 + Entropy/MaxProb")
    print("=" * 60)
    
    qid_info, valid_qids, mmlu_qids, gsm_qids, activations, entropies, max_probs = load_data()
    
    feature_combos = ['h_only', 'h+entropy', 'h+maxprob', 'h+both']
    seeds = [42, 123, 456, 789, 2024]
    n_seeds = len(seeds)
    
    results = {combo: {'auroc': {'indomain': [], 'cross_mmlu2gsm': [], 'cross_gsm2mmlu': []},
                       'brier': {'indomain': [], 'cross_mmlu2gsm': [], 'cross_gsm2mmlu': []}}
               for combo in feature_combos}
    
    print(f"\nRunning 5-seed protocol with {n_seeds} seeds...")
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{n_seeds}) ---")
        np.random.seed(seed)
        np.random.shuffle(valid_qids)
        n_total = len(valid_qids)
        n_train = int(0.8 * n_total)
        train_qids = valid_qids[:n_train]
        test_qids = valid_qids[n_train:]
        
        train_mmlu = [q for q in train_qids if qid_info[q]['dataset'] == 'mmlu']
        train_gsm = [q for q in train_qids if qid_info[q]['dataset'] == 'gsm8k']
        test_mmlu = [q for q in test_qids if qid_info[q]['dataset'] == 'mmlu']
        test_gsm = [q for q in test_qids if qid_info[q]['dataset'] == 'gsm8k']
        
        print(f"  Train: {len(train_mmlu)} mmlu + {len(train_gsm)} gsm | Test: {len(test_mmlu)} mmlu + {len(test_gsm)} gsm")
        
        for combo in feature_combos:
            auroc_indomain, brier_indomain, auroc_cross1, auroc_cross2 = run_combo(
                combo, train_qids, test_qids, train_mmlu, train_gsm, test_mmlu, test_gsm,
                activations, entropies, max_probs, qid_info, seed
            )
            results[combo]['auroc']['indomain'].append(auroc_indomain)
            results[combo]['brier']['indomain'].append(brier_indomain)
            results[combo]['auroc']['cross_mmlu2gsm'].append(auroc_cross1)
            results[combo]['auroc']['cross_gsm2mmlu'].append(auroc_cross2)
            
            print(f"  {combo}: indomain={auroc_indomain:.4f}, mmlu2gsm={auroc_cross1:.4f}, gsm2mmlu={auroc_cross2:.4f}")
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    summary = {}
    for combo in feature_combos:
        summary[combo] = {}
        for metric in ['auroc', 'brier']:
            for eval_type in ['indomain', 'cross_mmlu2gsm', 'cross_gsm2mmlu']:
                vals = [v for v in results[combo][metric][eval_type] if not np.isnan(v)]
                if vals:
                    summary[combo][f"{metric}_{eval_type}_mean"] = float(np.mean(vals))
                    summary[combo][f"{metric}_{eval_type}_std"] = float(np.std(vals))
                    summary[combo][f"{metric}_{eval_type}_seeds"] = [float(v) for v in results[combo][metric][eval_type]]
                else:
                    summary[combo][f"{metric}_{eval_type}_mean"] = np.nan
                    summary[combo][f"{metric}_{eval_type}_std"] = np.nan
                    summary[combo][f"{metric}_{eval_type}_seeds"] = []
    
    print("\nCross-domain effect sizes (Cohen's d + 95% CI):")
    effect_sizes = {}
    for combo in feature_combos:
        if combo == 'h_only':
            continue
        baseline_cross = results['h_only']['auroc']['cross_mmlu2gsm'] + results['h_only']['auroc']['cross_gsm2mmlu']
        combo_cross = results[combo]['auroc']['cross_mmlu2gsm'] + results[combo]['auroc']['cross_gsm2mmlu']
        
        baseline_mean = np.nanmean(baseline_cross)
        combo_mean = np.nanmean(combo_cross)
        delta = combo_mean - baseline_mean
        
        all_baseline = np.array([v for v in baseline_cross if not np.isnan(v)])
        all_combo = np.array([v for v in combo_cross if not np.isnan(v)])
        d = cohen_d(all_combo, all_baseline) if len(all_combo) > 1 and len(all_baseline) > 1 else 0.0
        
        deltas_boot = []
        for _ in range(1000):
            b_sample = np.random.choice(all_baseline, size=len(all_baseline), replace=True)
            c_sample = np.random.choice(all_combo, size=len(all_combo), replace=True)
            deltas_boot.append(np.mean(c_sample) - np.mean(b_sample))
        ci_low, ci_high = np.percentile(deltas_boot, [2.5, 97.5])
        
        effect_sizes[combo] = {
            'delta': float(delta),
            'cohen_d': float(d),
            'ci_95_low': float(ci_low),
            'ci_95_high': float(ci_high),
            'baseline_cross_mean': float(baseline_mean),
            'combo_cross_mean': float(combo_mean)
        }
        print(f"  {combo} vs h_only: delta={delta:.4f}, d={d:.3f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
    
    print("\nPre-registered decisions (cross-domain delta threshold):")
    decisions = {}
    for combo in feature_combos:
        if combo == 'h_only':
            decisions[combo] = 'BASELINE'
            continue
        delta = effect_sizes[combo]['delta']
        if delta >= 0.03:
            decision = 'KEEP'
        elif delta < 0.01:
            decision = 'DISCARD'
        else:
            decision = 'INCONCLUSIVE'
        decisions[combo] = decision
        print(f"  {combo}: delta={delta:.4f} -> {decision}")
    
    output = {
        'experiment': 'feature_augmentation_layer25',
        'layer': 25,
        'feature_dimensions': {'h_only': 2560, 'h+entropy': 2561, 'h+maxprob': 2561, 'h+both': 2562},
        'seeds': seeds,
        'n_train': int(0.8 * len(valid_qids)),
        'n_test': len(valid_qids) - int(0.8 * len(valid_qids)),
        'n_mmlu': len(mmlu_qids),
        'n_gsm8k': len(gsm_qids),
        'per_seed_results': results,
        'summary': summary,
        'effect_sizes': effect_sizes,
        'decisions': decisions
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    return output

def run_combo(combo, train_qids, test_qids, train_mmlu, train_gsm, test_mmlu, test_gsm,
              activations, entropies, max_probs, qid_info, seed):
    def get_X(qs):
        X_list, y_list = [], []
        for q in qs:
            h = activations[q]
            ent = entropies.get(q, np.nan)
            mp = max_probs[q]
            X = build_X(h, ent, mp, combo)
            if X is not None:
                X_list.append(X)
                y_list.append(qid_info[q]['correct'])
        if not X_list:
            return None, None
        return np.vstack(X_list), np.array(y_list)
    
    X_train, y_train = get_X(train_qids)
    X_test, y_test = get_X(test_qids)
    auroc_indomain, brier_indomain = evaluate_probe(X_train, y_train, X_test, y_test, seed)
    
    X_mmlu, y_mmlu = get_X(train_mmlu)
    X_gsm_test, y_gsm_test = get_X(test_gsm)
    if X_mmlu is not None and X_gsm_test is not None and len(X_mmlu) > 10 and len(X_gsm_test) > 10:
        auroc_cross1, _ = evaluate_probe(X_mmlu, y_mmlu, X_gsm_test, y_gsm_test, seed)
    else:
        auroc_cross1 = np.nan
    
    X_gsm, y_gsm = get_X(train_gsm)
    X_mmlu_test, y_mmlu_test = get_X(test_mmlu)
    if X_gsm is not None and X_mmlu_test is not None and len(X_gsm) > 10 and len(X_mmlu_test) > 10:
        auroc_cross2, _ = evaluate_probe(X_gsm, y_gsm, X_mmlu_test, y_mmlu_test, seed)
    else:
        auroc_cross2 = np.nan
    
    return auroc_indomain, brier_indomain, auroc_cross1, auroc_cross2

if __name__ == '__main__':
    run_experiment()