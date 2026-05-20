"""Cross-domain probe transfer: MMLU prefill <-> GSM8K gen-time.
Tests whether proper prompting enables cross-domain generalization.
"""
import json, pickle, os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT = Path("/Users/aban/drive/Projects/epistemic-steering")
DATA = PROJECT / "data"
LAYER = 25

# Load MMLU prefill activations
def load_mmlu_prefill():
    """Load MMLU prefill activations and labels from local data."""
    activations_dir = DATA / "activations_allpos"
    results_path = DATA / "probe_extract_results.jsonl"
    
    records = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("dataset") == "mmlu":
                    records.append(r)
    
    X, y = [], []
    for r in records:
        qid = r["question_id"]
        npy_path = activations_dir / f"{qid}__layer_{LAYER}.npy"
        if not npy_path.exists():
            npy_path = activations_dir / f"q{qid}_layer_{LAYER}.npy"
        if not npy_path.exists():
            continue
        act = np.load(npy_path).ravel()
        X.append(act)
        y.append(r["correct"])
    
    return np.array(X), np.array(y, dtype=bool)


def load_gsm8k_gen_time():
    """Load GSM8K gen-time hidden states from all_results.pkl (position 0)."""
    pkl_path = DATA / "gen_time_gsm8k_layer25_qwen_prompt" / "all_results.pkl"
    if not pkl_path.exists():
        print(f"ERROR: {pkl_path} not found. Download first.")
        return None, None
    
    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)
    if isinstance(all_results, list):
        all_results = {r.get('question_id', str(i)): r for i, r in enumerate(all_results)}
    
    X, y = [], []
    for qid, r in all_results.items():
        if not isinstance(r, dict):
            continue
        hs = r.get('hidden_states')
        if hs is None:
            continue
        
        correct = r.get('correct', False)
        
        # Use position 0 (first generated token, closest to prefill)
        if isinstance(hs, np.ndarray) and hs.ndim == 2:
            x = hs[0, :].copy()
        elif isinstance(hs, list) and len(hs) > 0:
            x = np.asarray(hs[0]).squeeze()
        else:
            continue
        X.append(x)
        y.append(correct)
    
    return np.array(X), np.array(y, dtype=bool)


def compute_ece(scores, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(scores)) * abs(labels[mask].mean() - scores[mask].mean())
    return ece


def train_and_eval(X_train, y_train, X_test, y_test, name):
    """Train LR probe and evaluate."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"error": "Single class in train or test"}
    
    clf = LogisticRegressionCV(Cs=10, cv=3, max_iter=1000, scoring="roc_auc", solver="lbfgs")
    clf.fit(X_train, y_train)
    
    test_preds = clf.predict_proba(X_test)[:, 1]
    train_preds = clf.predict_proba(X_train)[:, 1]
    
    test_auroc = float(roc_auc_score(y_test, test_preds))
    train_auroc = float(roc_auc_score(y_train, train_preds))
    brier = float(brier_score_loss(y_test, test_preds))
    ece = float(compute_ece(test_preds, y_test))
    
    result = {
        "name": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_auroc": round(train_auroc, 4),
        "test_auroc": round(test_auroc, 4),
        "overfitting_gap": round(train_auroc - test_auroc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "test_accuracy": round(y_test.mean(), 4),
    }
    return result


def main():
    print("=" * 60)
    print("CROSS-DOMAIN PROBE TRANSFER ANALYSIS")
    print("=" * 60)
    
    # Load MMLU prefill data
    print("\nLoading MMLU prefill...")
    X_mmlu, y_mmlu = load_mmlu_prefill()
    print(f"  MMLU: {len(X_mmlu)} samples, {y_mmlu.sum()} correct ({y_mmlu.mean():.1%})")
    
    # Load GSM8K gen-time data (position 0)
    print("Loading GSM8K gen-time (position 0)...")
    X_gsm8k, y_gsm8k = load_gsm8k_gen_time()
    if X_gsm8k is None:
        return
    print(f"  GSM8K: {len(X_gsm8k)} samples, {y_gsm8k.sum()} correct ({y_gsm8k.mean():.1%})")
    
    results = []
    
    # 1. MMLU -> MMLU (in-domain baseline)
    X_tr, X_te, y_tr, y_te = train_test_split(X_mmlu, y_mmlu, test_size=0.2, random_state=42, stratify=y_mmlu)
    r = train_and_eval(X_tr, y_tr, X_te, y_te, "MMLU→MMLU (in-domain)")
    results.append(r)
    print(f"\n  1. {r['name']}: AUROC={r['test_auroc']}")
    
    # 2. GSM8K -> GSM8K (in-domain baseline)
    X_tr, X_te, y_tr, y_te = train_test_split(X_gsm8k, y_gsm8k, test_size=0.2, random_state=42, stratify=y_gsm8k)
    r = train_and_eval(X_tr, y_tr, X_te, y_te, "GSM8K→GSM8K (in-domain)")
    results.append(r)
    print(f"  2. {r['name']}: AUROC={r['test_auroc']}")
    
    # 3. MMLU -> GSM8K (cross-domain, KEY TEST)
    r = train_and_eval(X_mmlu, y_mmlu, X_gsm8k, y_gsm8k, "MMLU→GSM8K (cross-domain)")
    results.append(r)
    print(f"  3. {r['name']}: AUROC={r['test_auroc']}{' ★ CROSS-DOMAIN TRANSFER!' if r.get('test_auroc', 0) > 0.7 else ''}")
    
    # 4. GSM8K -> MMLU (reverse cross-domain)
    r = train_and_eval(X_gsm8k, y_gsm8k, X_mmlu, y_mmlu, "GSM8K→MMLU (cross-domain)")
    results.append(r)
    print(f"  4. {r['name']}: AUROC={r['test_auroc']}{' ★' if r.get('test_auroc', 0) > 0.7 else ''}")
    
    # 5. Combined -> held-out
    X_combined = np.vstack([X_mmlu, X_gsm8k])
    y_combined = np.concatenate([y_mmlu, y_gsm8k])
    X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
    r = train_and_eval(X_tr, y_tr, X_te, y_te, "Combined→Held-out")
    results.append(r)
    print(f"  5. {r['name']}: AUROC={r['test_auroc']}")
    
    # 6. Combined -> MMLU held-out only
    mmlu_test_mask = np.array([i >= len(X_mmlu) for i in range(len(X_te))])  # simplified
    # Better: do proper split
    mmlu_idx = np.arange(len(X_mmlu))
    gsm8k_idx = np.arange(len(X_mmlu), len(X_combined))
    train_idx = np.concatenate([
        np.random.choice(mmlu_idx, size=int(0.8 * len(mmlu_idx)), replace=False),
        np.random.choice(gsm8k_idx, size=int(0.8 * len(gsm8k_idx)), replace=False),
    ])
    test_idx_mmlu = np.setdiff1d(mmlu_idx, train_idx[train_idx < len(mmlu_idx)])
    test_idx_gsm8k = np.setdiff1d(gsm8k_idx, train_idx[train_idx >= len(mmlu_idx)] - len(mmlu_idx))
    
    X_train_c = X_combined[train_idx]
    y_train_c = y_combined[train_idx]
    
    r = train_and_eval(X_train_c, y_train_c, X_mmlu[test_idx_mmlu], y_mmlu[test_idx_mmlu], "Combined→MMLU held-out")
    results.append(r)
    print(f"  6. {r['name']}: AUROC={r['test_auroc']}")
    
    r = train_and_eval(X_train_c, y_train_c, X_gsm8k[test_idx_gsm8k], y_gsm8k[test_idx_gsm8k], "Combined→GSM8K held-out")
    results.append(r)
    print(f"  7. {r['name']}: AUROC={r['test_auroc']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    for r in results:
        star = " ★" if r.get('test_auroc', 0) > 0.7 else ""
        print(f"  {r['name']:<30} AUROC={r['test_auroc']:.4f}  gap={r['overfitting_gap']:.4f}  ECE={r['ece']:.4f}{star}")
    
    # Verdict
    cross_auroc = next(r['test_auroc'] for r in results if 'MMLU→GSM8K' in r['name'])
    if cross_auroc > 0.7:
        print(f"\n  VERDICT: CROSS-DOMAIN TRANSFER ACHIEVED ({cross_auroc:.4f} > 0.70)")
    elif cross_auroc > 0.6:
        print(f"\n  VERDICT: PARTIAL CROSS-DOMAIN ({cross_auroc:.4f}, marginal improvement)")
    else:
        print(f"\n  VERDICT: NO CROSS-DOMAIN TRANSFER ({cross_auroc:.4f})")
    
    # Save
    out_path = DATA / "ablation_results" / "cross_domain_transfer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
