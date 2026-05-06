"""In-sample verification experiment.

Loads probe data from Modal volume, computes training-free probe confidence scores
for each question using layer 30 activations, then builds confusion matrices
and computes prevention rates.

CRITICAL: These are IN-SAMPLE results. The probe was trained on this exact data.
These numbers are THEORETICAL UPPER BOUNDS, not generalizable.
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from tqdm import tqdm

sys.path.insert(0, '.')
from src.data import split_by_dataset, clean_mmlu_answers
from src.probe import (
    compute_auroc, compute_confusion_matrix, compute_prevention_rate,
    compute_unnecessary_block_rate, threshold_sweep
)
from src.evaluate import compute_all_metrics, confusion_matrix_at_threshold

import modal


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = Path('data')
ACTIVATIONS_DIR = DATA_DIR / 'activations'
JSONL_PATH = DATA_DIR / 'probe_extract_results.jsonl'
OUTPUT_PATH = DATA_DIR / 'verification_results.json'
LAYER = 30
VOLUME_NAME = 'epistemic-model-cache'
REMOTE_PREFIX = 'results/activations'


# ── Step 1: Download activations from Modal ─────────────────────────────────

def download_activations(force: bool = False) -> int:
    """Download all layer 30 activations from Modal volume.

    Args:
        force: Re-download even if files exist.

    Returns:
        Number of files downloaded.
    """
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Modal volume: {VOLUME_NAME}...")
    vol = modal.Volume.from_name(VOLUME_NAME)

    print(f"Listing {REMOTE_PREFIX}/ ...")
    entries = list(vol.listdir(REMOTE_PREFIX))
    layer_30_entries = [e for e in entries if f'layer_{LAYER}.npy' in e.path]
    print(f"Found {len(layer_30_entries)} layer {LAYER} activation files")

    downloaded = 0
    skipped = 0
    for entry in tqdm(layer_30_entries, desc=f"Downloading layer {LAYER} activations"):
        filename = entry.path.split('/')[-1]
        local_path = ACTIVATIONS_DIR / filename

        if local_path.exists() and not force:
            skipped += 1
            continue

        try:
            chunks = list(vol.read_file(entry.path))
            data = b''.join(chunks)
            with open(local_path, 'wb') as f:
                f.write(data)
            downloaded += 1
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")

    print(f"Downloaded {downloaded} files, skipped {skipped} existing")
    return downloaded + skipped


# ── Step 2: Compute training-free probe scores ──────────────────────────────

def compute_training_free_scores(
    df,
    activations_dir: str,
    layer: int = 30,
    per_dataset: bool = True
) -> np.ndarray:
    """Compute training-free probe scores using mean-diff direction.

    The training-free probe computes the mean-difference direction between
    correct and incorrect activations, then projects each activation onto
    this direction. The score is sigmoid(projection / direction_norm).

    When per_dataset=True, separate directions are computed for MMLU and GSM8K
    to avoid dataset-specific patterns diluting the separation.

    Args:
        df: DataFrame with question_id, dataset, and correct columns.
        activations_dir: Directory with .npy activation files.
        layer: Layer number.
        per_dataset: If True, compute separate directions per dataset.

    Returns:
        np.ndarray of confidence scores in [0, 1], one per row in df.
    """
    activations_dir = Path(activations_dir)

    # Collect activations and labels
    activations = []
    labels = []
    datasets_loaded = []
    matched_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading activations"):
        qid = row['question_id']
        npy_path = activations_dir / f"{qid}__layer_{layer}.npy"

        if not npy_path.exists():
            # Try alternative naming patterns
            alt_path = activations_dir / f"q{qid}_layer_{layer}.npy"
            if alt_path.exists():
                npy_path = alt_path
            else:
                print(f"  WARNING: No activation file for {qid}")
                continue

        try:
            act = np.load(npy_path)
            activations.append(act.ravel())  # ensure 1D
            labels.append(row['correct'])
            datasets_loaded.append(row['dataset'])
            matched_ids.append(qid)
        except Exception as e:
            print(f"  ERROR loading {npy_path}: {e}")

    activations = np.array(activations)
    labels = np.array(labels, dtype=bool)
    datasets_arr = np.array(datasets_loaded)
    print(f"Loaded {len(activations)} activations (shape: {activations.shape})")

    def train_free_direction(acts, lbls):
        correct_acts = acts[lbls]
        incorrect_acts = acts[~lbls]
        direction = correct_acts.mean(axis=0) - incorrect_acts.mean(axis=0)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        midpoint = (correct_acts.mean(axis=0) + incorrect_acts.mean(axis=0)) / 2.0
        return direction, midpoint

    if per_dataset and 'mmlu' in datasets_arr and 'gsm8k' in datasets_arr:
        mmlu_mask = datasets_arr == 'mmlu'
        gsm8k_mask = datasets_arr == 'gsm8k'

        dir_mmlu, mid_mmlu = train_free_direction(activations[mmlu_mask], labels[mmlu_mask])
        dir_gsm8k, mid_gsm8k = train_free_direction(activations[gsm8k_mask], labels[gsm8k_mask])

        print(f"Correct (MMLU): {labels[mmlu_mask].sum()}, Incorrect (MMLU): {(~labels[mmlu_mask]).sum()}")
        print(f"Correct (GSM8K): {labels[gsm8k_mask].sum()}, Incorrect (GSM8K): {(~labels[gsm8k_mask]).sum()}")

        projections = np.zeros(len(activations))
        projections[mmlu_mask] = np.dot(activations[mmlu_mask] - mid_mmlu, dir_mmlu)
        projections[gsm8k_mask] = np.dot(activations[gsm8k_mask] - mid_gsm8k, dir_gsm8k)
    else:
        direction, midpoint = train_free_direction(activations, labels)
        print(f"Correct: {labels.sum()}, Incorrect: {(~labels).sum()}")
        projections = np.dot(activations - midpoint, direction)

    scores = expit(projections)
    return np.array(scores)


# ── Step 3: Compute token-prob baseline (fallback) ──────────────────────────

def compute_token_prob_scores(df) -> np.ndarray:
    """Use top_token_prob as a proxy confidence score.

    Low top-token probability indicates uncertainty/hallucination.
    This is a baseline, not the true probe confidence.

    Returns:
        np.ndarray of scores in [0, 1] (1 = confident, 0 = uncertain)
    """
    scores = df['top_token_prob'].values.astype(float)
    return scores


# ── Step 4: Build verification results ──────────────────────────────────────

def build_verification_results(
    df,
    scores: np.ndarray,
    method: str
) -> dict:
    """Build the verification results JSON.

    Args:
        df: Full DataFrame (already cleaned for MMLU ? answers).
        scores: Confidence scores array, aligned with df.
        method: Label for the scoring method used.

    Returns:
        Dict matching the output spec.
    """
    mmlu_df, gsm8k_df = split_by_dataset(df)
    mmlu_labels = mmlu_df['correct'].values.astype(bool)
    gsm8k_labels = gsm8k_df['correct'].values.astype(bool)

    # Map scores back to each dataset
    mmlu_mask = (df['dataset'] == 'mmlu').values
    gsm8k_mask = (df['dataset'] == 'gsm8k').values
    mmlu_scores = scores[mmlu_mask]
    gsm8k_scores = scores[gsm8k_mask]

    print(f"\nMMLU: {len(mmlu_scores)} questions, {mmlu_labels.sum()} correct, {len(mmlu_scores) - mmlu_labels.sum()} incorrect")
    print(f"GSM8K: {len(gsm8k_scores)} questions, {gsm8k_labels.sum()} correct, {len(gsm8k_scores) - gsm8k_labels.sum()} incorrect")

    def compute_dataset_metrics(scores_ds, labels_ds, name):
        """Compute all metrics for one dataset."""
        # AUROC
        auroc = compute_auroc(scores_ds, labels_ds)

        # Confusion matrix at threshold 0.5
        cm_05 = confusion_matrix_at_threshold(scores_ds, labels_ds, threshold=0.5)

        # Prevention rates at multiple thresholds
        thresholds_to_check = [0.3, 0.4, 0.5, 0.6, 0.7]
        prevention_rates = {}
        unnecessary_block_rates = {}
        for t in thresholds_to_check:
            prevention_rates[str(t)] = float(compute_prevention_rate(scores_ds, labels_ds, t))
            unnecessary_block_rates[str(t)] = float(compute_unnecessary_block_rate(scores_ds, labels_ds, t))

        # Threshold sweep
        sweep = threshold_sweep(scores_ds, labels_ds)

        # Full metrics at threshold 0.5
        all_metrics = compute_all_metrics(scores_ds, labels_ds, threshold=0.5)

        return {
            'n_total': int(len(scores_ds)),
            'n_correct': int(labels_ds.sum()),
            'n_incorrect': int((~labels_ds).sum()),
            'auroc': float(auroc) if not np.isnan(auroc) else None,
            'confusion_matrix_at_threshold_0.5': cm_05,
            'prevention_rates': prevention_rates,
            'unnecessary_block_rates': unnecessary_block_rates,
            'threshold_sweep': [
                {
                    'threshold': float(row['threshold']),
                    'precision': float(row['precision']),
                    'recall': float(row['recall']),
                    'f1': float(row['f1']),
                    'prevention_rate': float(row['prevention_rate']),
                    'unnecessary_block_rate': float(row['unnecessary_block_rate']),
                }
                for _, row in sweep.iterrows()
            ],
            'calibration': {
                'bin_centers': all_metrics['calibration']['bin_centers'].tolist(),
                'observed_accuracy': all_metrics['calibration']['observed_accuracy'].tolist(),
                'bin_counts': all_metrics['calibration']['bin_counts'].tolist(),
            },
        }

    return {
        'metadata': {
            'type': 'in_sample_verification',
            'warning': 'THESE ARE IN-SAMPLE UPPER BOUNDS. PROBE WAS TRAINED ON THIS DATA.',
            'method': method,
            'model': 'Qwen3.5-4B',
            'n_questions': len(df),
            'probe_layer': LAYER,
            'n_matched_activations': len(scores),
        },
        'mmlu': compute_dataset_metrics(mmlu_scores, mmlu_labels, 'MMLU'),
        'gsm8k': compute_dataset_metrics(gsm8k_scores, gsm8k_labels, 'GSM8K'),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("IN-SAMPLE VERIFICATION EXPERIMENT")
    print("=" * 70)

    # Step 1: Download activations
    print("\n── Step 1: Downloading activations from Modal ──")
    n_files = download_activations(force=False)
    print(f"  Available activation files: {n_files}")

    # Step 2: Load and clean data
    print("\n── Step 2: Loading probe results ──")
    import pandas as pd
    records = []
    with open(str(JSONL_PATH), 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df['correct'] = df['correct'].astype(bool)
    print(f"  Loaded {len(df)} questions")
    print(f"  MMLU: {(df['dataset'] == 'mmlu').sum()}, GSM8K: {(df['dataset'] == 'gsm8k').sum()}")

    # Clean MMLU ? answers (mark as incorrect)
    df_clean = clean_mmlu_answers(df)
    n_qmark = (
        (df['dataset'] == 'mmlu') & (df['model_answer'] == '?')
    ).sum()
    print(f"  MMLU ? answers marked incorrect: {n_qmark}")

    # Step 3: Choose scoring method
    print("\n── Step 3: Computing probe scores ──")

    # Check how many activation files match the DataFrame
    activations_available = len(list(ACTIVATIONS_DIR.glob(f'*layer_{LAYER}.npy')))
    print(f"  Activation files in data/activations/: {activations_available}")

    if activations_available > 0:
        print("  Using TRAINING-FREE probe scores (per-dataset directions)")
        scores = compute_training_free_scores(df_clean, str(ACTIVATIONS_DIR), LAYER, per_dataset=True)
        method = 'training_free_probe_layer_30'
    else:
        print("  WARNING: No activation files. Falling back to top_token_prob baseline.")
        print("  These scores WILL NOT match the known AUROC of 0.87.")
        scores = compute_token_prob_scores(df_clean)
        method = 'top_token_prob_baseline'

    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean score: {scores.mean():.4f}")

    # Step 4: Build results
    print("\n── Step 4: Computing verification metrics ──")
    results = build_verification_results(df_clean, scores, method)

    # Step 5: Validate against known values
    print("\n── Step 5: Validation ──")
    mmlu_auroc = results['mmlu']['auroc']
    gsm8k_auroc = results['gsm8k']['auroc']

    print(f"  MMLU AUROC: {mmlu_auroc:.4f}" + (f" (cf. cross-validated LR: 0.87)" if mmlu_auroc else ""))
    print(f"  GSM8K AUROC: {gsm8k_auroc:.4f}" + (f" (cf. cross-validated LR: 0.64)" if gsm8k_auroc else ""))

    if mmlu_auroc and abs(mmlu_auroc - 0.87) <= 0.05:
        print(f"  ✓ MMLU AUROC within 0.05 of expected (training-free vs cross-validated LR)")
    elif mmlu_auroc:
        print(f"  NOTE: MMLU AUROC difference from 0.87: {abs(mmlu_auroc - 0.87):.4f}")
        print("    Training-free probe gives lower AUROC than cross-validated logistic regression.")

    if gsm8k_auroc and gsm8k_auroc > 0.8:
        print(f"  NOTE: GSM8K AUROC is inflated (in-sample overfitting). Only 7/200 correct.")

    mmlu_prev_rate = results['mmlu']['prevention_rates'].get('0.5', 0)
    print(f"  MMLU Prevention Rate at 0.5: {mmlu_prev_rate:.4f}")
    if mmlu_prev_rate > 0.5:
        print("  ✓ Prevention rate > 50% at threshold 0.5")
    else:
        print("  ✗ Prevention rate <= 50% at threshold 0.5")

    # Step 6: Save results
    print(f"\n── Step 6: Saving to {OUTPUT_PATH} ──")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved {os.path.getsize(OUTPUT_PATH)} bytes")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("⚠️  ALL RESULTS ARE IN-SAMPLE UPPER BOUNDS — NOT GENERALIZABLE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
