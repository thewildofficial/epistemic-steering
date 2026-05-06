"""Threshold Sensitivity & Prevention Rate Analysis.

Loads verification data (JSONL + activations), runs a comprehensive threshold
sweep (19 points from 0.05 to 0.95), identifies optimal thresholds per dataset
via Youden's J, computes calibration curves, and analyses "?" answer impact.

CRITICAL: ALL RESULTS ARE IN-SAMPLE UPPER BOUNDS.
The training-free probe uses per-dataset mean-diff directions computed from
the same data, so metrics represent theoretical ceilings, not generalizable.

Output: data/threshold_analysis.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import calibration_curve
from src.probe import compute_auroc

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
JSONL_PATH = DATA_DIR / 'probe_extract_results.jsonl'
ACTIVATIONS_DIR = DATA_DIR / 'activations'
OUTPUT_PATH = DATA_DIR / 'threshold_analysis.json'
LAYER = 30

# Thresholds from 0.05 to 0.95 in 0.05 steps (19 points)
THRESHOLDS = np.arange(0.05, 1.0, 0.05)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_raw_data(jsonl_path: str) -> pd.DataFrame:
    """Load JSONL probe results without coercing question_id to int.

    Returns DataFrame with columns: question_id (str), dataset, model_answer,
    correct (bool), top_token_prob.
    """
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df['correct'] = df['correct'].astype(bool)
    print(f"  Loaded {len(df)} questions from JSONL")
    return df


def mark_qmark_incorrect(df: pd.DataFrame) -> pd.DataFrame:
    """Mark MMLU ``?`` model_answer rows as incorrect."""
    df = df.copy()
    mask = (df['dataset'] == 'mmlu') & (df['model_answer'] == '?')
    n = mask.sum()
    df.loc[mask, 'correct'] = False
    print(f"  Marked {n} MMLU '?' answers as incorrect")
    return df


def compute_training_free_scores(
    df: pd.DataFrame,
    activations_dir: str,
    layer: int = 30,
    per_dataset: bool = True,
) -> np.ndarray:
    """Compute training-free probe scores via per-dataset mean-diff directions.

    Returns
    -------
    np.ndarray of length == len(df) with scores in [0, 1].
    """
    activations_dir = Path(activations_dir)
    acts_list, lbls_list, dsets_list = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading activations"):
        qid = row['question_id']
        # Try naming patterns: {qid}__layer_{layer}.npy or q{qid}_layer_{layer}.npy
        for candidate in [
            activations_dir / f"{qid}__layer_{layer}.npy",
            activations_dir / f"q{qid}_layer_{layer}.npy",
        ]:
            if candidate.exists():
                act = np.load(candidate)
                acts_list.append(act.ravel())
                lbls_list.append(row['correct'])
                dsets_list.append(row['dataset'])
                break
        else:
            print(f"  WARNING: No activation for {qid}")
            continue

    acts = np.array(acts_list)
    labels = np.array(lbls_list, dtype=bool)
    dsets = np.array(dsets_list)
    print(f"  Loaded {len(acts)} activations (shape: {acts.shape})")

    def _direction(acts_sub, lbls_sub):
        c_acts = acts_sub[lbls_sub]
        i_acts = acts_sub[~lbls_sub]
        direction = c_acts.mean(axis=0) - i_acts.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        midpoint = (c_acts.mean(axis=0) + i_acts.mean(axis=0)) / 2.0
        return direction, midpoint

    projections = np.zeros(len(acts))
    if per_dataset and 'mmlu' in dsets and 'gsm8k' in dsets:
        for ds_name in ['mmlu', 'gsm8k']:
            mask = dsets == ds_name
            d, m = _direction(acts[mask], labels[mask])
            projections[mask] = np.dot(acts[mask] - m, d)
            print(f"  {ds_name.upper()}: {labels[mask].sum()} correct / {len(labels[mask])} total")
    else:
        d, m = _direction(acts, labels)
        projections = np.dot(acts - m, d)

    scores = expit(projections)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return np.array(scores)


# ── Metric Helpers ───────────────────────────────────────────────────────────

def _cm(scores, labels, thresh):
    """Return (TP, FP, TN, FN) as ints at given threshold."""
    c = scores >= thresh
    tp = int(np.sum(labels & c))
    fp = int(np.sum(~labels & c))
    tn = int(np.sum(~labels & ~c))
    fn = int(np.sum(labels & ~c))
    return tp, fp, tn, fn


def _metrics_at_threshold(scores, labels, thresh):
    """Compute all per-threshold metrics.

    Returns dict with:
    - threshold, prevention_rate, unnecessary_block_rate
    - selective_accuracy (=1 - FP/N, upper bound assuming CoT fixes all)
    - precision, recall, f1_score, sensitivity, specificity, youden_j
    """
    tp, fp, tn, fn = _cm(scores, labels, thresh)
    n = len(scores)
    n_correct = int(np.sum(labels))
    n_wrong = int(np.sum(~labels))

    # Prevention rate = fraction of hallucinations caught = TN / (TN + FP)
    prevention_rate = float(tn / n_wrong) if n_wrong > 0 else 0.0

    # Unnecessary block rate = fraction of correct answers blocked = FN / (FN + TP)
    unnecessary_block_rate = float(fn / n_correct) if n_correct > 0 else 0.0

    # Selective accuracy (upper bound):
    # Direct = TP_correct + CoT(TN+FN)_correct = TP + TN + FN = N - FP
    # selective_accuracy = (N - FP) / N
    selective_acc = float((n - fp) / n) if n > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Recall (sensitivity) = TP / (TP + FN)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # F1 = 2 * P * R / (P + R)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Youden's J = sensitivity + specificity - 1 = recall + specificity - 1
    youden_j = float(recall + specificity - 1)

    return {
        'threshold': round(float(thresh), 4),
        'prevention_rate': prevention_rate,
        'unnecessary_block_rate': unnecessary_block_rate,
        'selective_accuracy': selective_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': recall,
        'specificity': specificity,
        'youden_j': youden_j,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_total': n, 'n_correct': n_correct, 'n_wrong': n_wrong,
    }


# ── Dataset-Level Analysis ───────────────────────────────────────────────────

def analyse_dataset(scores: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """Run full threshold sweep + calibration for one dataset."""
    print(f"\n── Analysing {name} ({len(scores)} questions) ──")

    auroc = compute_auroc(scores, labels)

    # Threshold sweep
    results = [_metrics_at_threshold(scores, labels, t) for t in THRESHOLDS]

    # Extract arrays in threshold order
    thr_list       = [r['threshold'] for r in results]
    prev_list      = [r['prevention_rate'] for r in results]
    block_list     = [r['unnecessary_block_rate'] for r in results]
    sel_acc_list   = [r['selective_accuracy'] for r in results]
    f1_list        = [r['f1_score'] for r in results]
    youden_list    = [r['youden_j'] for r in results]

    # Optimal threshold via Youden's J
    best_idx = int(np.argmax(youden_list))
    best_thresh = thr_list[best_idx]
    best_youden = youden_list[best_idx]

    print(f"  AUROC: {auroc:.4f}")
    print(f"  Optimal threshold (Youden's J): {best_thresh} (J = {best_youden:.4f})")

    # Calibration curve
    calib = calibration_curve(scores, labels, n_bins=10)
    calib = {k: v.tolist() if isinstance(v, np.ndarray) else v
             for k, v in calib.items()}

    return {
        'auroc': float(auroc) if not np.isnan(auroc) else None,
        'n_total': int(len(scores)),
        'n_correct': int(np.sum(labels)),
        'n_wrong': int(np.sum(~labels)),
        'optimal_threshold': best_thresh,
        'optimal_youden_j': best_youden,
        'thresholds': thr_list,
        'prevention_rates': prev_list,
        'unnecessary_block_rates': block_list,
        'selective_accuracies': sel_acc_list,
        'f1_scores': f1_list,
        'youden_j_values': youden_list,
        'calibration': calib,
        # Full sweep details
        'threshold_sweep': results,
    }


# ── "?" Answer Impact Analysis ────────────────────────────────────────────────

def qmark_impact_analysis(
    df_raw: pd.DataFrame,
    scores_clean: np.ndarray,
    df_clean: pd.DataFrame,
    scores_raw: np.ndarray,
) -> dict:
    """Compare metrics with and without treating '?' answers as incorrect.

    Returns comparison dict with counts and metric differences.
    """
    # Locate MMLU "?" rows in cleaned data
    mmlu_mask_clean = df_clean['dataset'] == 'mmlu'
    qmask_clean = (df_clean['model_answer'] == '?') & mmlu_mask_clean

    n_qmark = qmask_clean.sum()
    if n_qmark == 0:
        return {'n_qmark': 0, 'note': 'No ? answers found'}

    # In the clean dataset, "?" answers already have correct=False
    # We want WITH ? (as incorrect) vs WITHOUT ? (remove these rows entirely)
    mmlu_labels_clean = df_clean.loc[mmlu_mask_clean, 'correct'].values.astype(bool)
    mmlu_scores_clean = scores_clean[mmlu_mask_clean.values]

    # Mask excluding "?" rows
    non_qmask = mmlu_mask_clean & ~qmask_clean
    mmlu_labels_noq = df_clean.loc[non_qmask, 'correct'].values.astype(bool)
    mmlu_scores_noq = scores_clean[non_qmask.values]

    def _summary(s, l):
        """Quick metric summary at 19 thresholds."""
        thrs = []
        prevs = []
        blocks = []
        youdens = []
        for t in THRESHOLDS:
            m = _metrics_at_threshold(s, l, t)
            thrs.append(m['threshold'])
            prevs.append(m['prevention_rate'])
            blocks.append(m['unnecessary_block_rate'])
            youdens.append(m['youden_j'])
        best_j = max(youdens)
        best_t = thrs[youdens.index(best_j)]
        return {
            'n_total': int(len(s)),
            'n_correct': int(np.sum(l)),
            'n_wrong': int(np.sum(~l)),
            'auroc': compute_auroc(s, l),
            'best_youden_j': best_j,
            'best_threshold': best_t,
        }

    with_q = _summary(mmlu_scores_clean, mmlu_labels_clean)
    without_q = _summary(mmlu_scores_noq, mmlu_labels_noq)

    auroc_with = with_q['auroc']
    auroc_without = without_q['auroc']
    auroc_delta = auroc_with - auroc_without if (not np.isnan(auroc_with) and not np.isnan(auroc_without)) else None

    print(f"\n── '?' Answer Impact Analysis (MMLU) ──")
    print(f"  MMLU total: {len(df_raw)}→{len(df_clean)} after cleaning")
    print(f"  '?' answers: {n_qmark}")
    print(f"  With ? (as incorrect):   n={with_q['n_total']}, AUROC={auroc_with:.4f}, "
          f"Youden={with_q['best_youden_j']:.4f}, opt_thresh={with_q['best_threshold']}")
    print(f"  Without ? (excluded):    n={without_q['n_total']}, AUROC={auroc_without:.4f}, "
          f"Youden={without_q['best_youden_j']:.4f}, opt_thresh={without_q['best_threshold']}")
    if auroc_delta is not None:
        print(f"  AUROC delta (with - without): {auroc_delta:+.4f}")
    print(f"  Interpretation: '?' answers are automatically wrong under our "
          f"framework. Including them as 'incorrect' gives the model credit for "
          f"expressing uncertainty (the probe sees these as low-confidence). "
          f"This inflates the AUROC.")

    return {
        'n_qmark': int(n_qmark),
        'dataset': 'mmlu',
        'with_q_treated_as_incorrect': with_q,
        'without_q_excluded': without_q,
        'auroc_delta': float(auroc_delta) if auroc_delta is not None else None,
        'note': (
            "Treating '?' answers as incorrect inflates AUROC because the probe "
            "tends to give them low confidence scores, making them easy to classify "
            "as hallucinations—even though the model might have been 'right to be uncertain.' "
            "These are IN-SAMPLE UPPER BOUNDS; real-world ? answers may behave differently."
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("THRESHOLD SENSITIVITY & PREVENTION RATE ANALYSIS")
    print("=" * 70)

    # 1. Load data
    print("\n── Step 1: Loading data ──")
    df_raw = load_raw_data(str(JSONL_PATH))
    mmlu_df = df_raw[df_raw['dataset'] == 'mmlu']
    gsm8k_df = df_raw[df_raw['dataset'] == 'gsm8k']
    print(f"  MMLU: {len(mmlu_df)}, GSM8K: {len(gsm8k_df)}")

    # 2. Build two variants: with "?" as incorrect (clean) + raw
    print("\n── Step 2: Preparing data variants ──")
    df_clean = mark_qmark_incorrect(df_raw)

    # 3. Compute scores for BOTH variants
    print("\n── Step 3: Computing training-free probe scores (clean variant) ──")
    scores_clean = compute_training_free_scores(
        df_clean, str(ACTIVATIONS_DIR), LAYER, per_dataset=True
    )

    print("\n── Step 3b: Computing training-free probe scores (raw variant) ──")
    scores_raw = compute_training_free_scores(
        df_raw, str(ACTIVATIONS_DIR), LAYER, per_dataset=True
    )

    # 4. Split by dataset for analysis (clean variant)
    mmlu_mask = (df_clean['dataset'] == 'mmlu').values
    gsm8k_mask = (df_clean['dataset'] == 'gsm8k').values
    mmlu_scores = scores_clean[mmlu_mask]
    mmlu_labels = df_clean.loc[mmlu_mask, 'correct'].values.astype(bool)
    gsm8k_scores = scores_clean[gsm8k_mask]
    gsm8k_labels = df_clean.loc[gsm8k_mask, 'correct'].values.astype(bool)

    # 5. Analyse each dataset
    print("\n── Step 4: Threshold sweep analysis ──")
    mmlu_analysis = analyse_dataset(mmlu_scores, mmlu_labels, 'MMLU')
    gsm8k_analysis = analyse_dataset(gsm8k_scores, gsm8k_labels, 'GSM8K')

    # 6. "?" impact
    print("\n── Step 5: '?' answer impact analysis ──")
    impact = qmark_impact_analysis(df_raw, scores_clean, df_clean, scores_raw)

    # 7. Assemble output
    print("\n── Step 6: Assembling output ──")
    output = {
        'metadata': {
            'type': 'threshold_analysis',
            'in_sample': True,
            'warning': 'ALL RESULTS ARE IN-SAMPLE UPPER BOUNDS — NOT GENERALIZABLE',
            'method': 'training_free_probe_layer_30',
            'model': 'Qwen3.5-4B',
            'n_questions': int(len(df_clean)),
            'n_matched_activations': int(len(scores_clean)),
            'thresholds_count': len(THRESHOLDS),
            'thresholds_min': float(THRESHOLDS[0]),
            'thresholds_max': float(THRESHOLDS[-1]),
            'threshold_step': float(THRESHOLDS[1] - THRESHOLDS[0]),
            'youden_j_formula': 'sensitivity + specificity - 1 = recall + specificity - 1',
            'selective_accuracy_definition': 'Upper bound: (N - FP)/N — assumes CoT fixes all blocked errors',
        },
        'mmlu': mmlu_analysis,
        'gsm8k': gsm8k_analysis,
        'qmark_impact': impact,
    }

    # 8. Save
    print(f"\n── Step 7: Saving to {OUTPUT_PATH} ──")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  Saved {OUTPUT_PATH.stat().st_size} bytes")

    # 9. Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"  MMLU  AUROC: {mmlu_analysis['auroc']:.4f}  "
          f"opt_thresh={mmlu_analysis['optimal_threshold']}  "
          f"J={mmlu_analysis['optimal_youden_j']:.4f}")
    print(f"  GSM8K AUROC: {gsm8k_analysis['auroc']:.4f}  "
          f"opt_thresh={gsm8k_analysis['optimal_threshold']}  "
          f"J={gsm8k_analysis['optimal_youden_j']:.4f}")
    print(f"  ? impact: {impact.get('n_qmark', 0)} question(s)")
    if gsm8k_analysis.get('auroc', 0) and gsm8k_analysis['auroc'] > 0.9:
        print("  ⚠️  GSM8K AUROC is inflated (in-sample overfitting with 7/200 correct)")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
