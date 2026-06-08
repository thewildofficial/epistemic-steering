#!/usr/bin/env python3
"""
Train a within-domain logistic regression probe for a given benchmark.

Loads activations (.npy) and labels (.json) from
data/benchmark_activations_v2/{benchmark}/, trains a LogisticRegressionCV
with Platt scaling, and saves probe weights to data/probes/probe_{benchmark}.npz.

Usage:
    uv run python scripts/train_transfer_probes.py --benchmark arc_challenge
    uv run python scripts/train_transfer_probes.py --benchmark math --mock_data
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN_DIM = 2560
RANDOM_SEED = 42


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (bin_count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_selective_accuracy(scores, labels, threshold):
    """Accuracy on the subset where score >= threshold."""
    mask = scores >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(labels[mask].mean())


def compute_coverage(scores, threshold):
    """Fraction of questions with score >= threshold."""
    return float((scores >= threshold).mean())


def find_threshold_for_selective_accuracy(scores, labels, target_acc=0.95):
    """Find the lowest threshold achieving at least target_acc selective accuracy."""
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_thresh = 1.0
    for thresh in thresholds:
        acc = compute_selective_accuracy(scores, labels, thresh)
        if acc >= target_acc:
            best_thresh = thresh
            break
    return best_thresh


def load_v2_data(benchmark_dir: Path):
    """Load activations and labels from a v2-format benchmark directory.

    Expects .npy files (activations) and .json files (labels with 'correct' field)
    sharing the same stem (e.g. q0.npy + q0.json).

    Returns:
        X: np.ndarray of shape (n_questions, HIDDEN_DIM)
        y: np.ndarray of shape (n_questions,) bool
    """
    npy_files = sorted(benchmark_dir.glob("*.npy"))
    if not npy_files:
        return None, None

    X_list = []
    y_list = []
    for npy_path in npy_files:
        json_path = npy_path.with_suffix(".json")
        if not json_path.exists():
            continue
        activation = np.load(npy_path).astype(np.float64)
        # If 2D (seq_len, hidden), take last token
        if activation.ndim == 2:
            activation = activation[-1]
        if activation.shape[0] != HIDDEN_DIM:
            continue
        with open(json_path) as f:
            meta = json.load(f)
        X_list.append(activation)
        y_list.append(bool(meta["correct"]))

    if not X_list:
        return None, None

    return np.stack(X_list, axis=0), np.array(y_list, dtype=bool)


def load_legacy_data(benchmark: str):
    """Fallback: load from data/benchmark_activations/{benchmark}/ + {benchmark}_results.jsonl.

    Returns:
        X: np.ndarray of shape (n_questions, HIDDEN_DIM)
        y: np.ndarray of shape (n_questions,) bool
    """
    acts_dir = PROJECT_ROOT / "data" / "benchmark_activations" / benchmark
    results_file = PROJECT_ROOT / "data" / "benchmark_activations" / f"{benchmark}_results.jsonl"

    if not acts_dir.is_dir() or not results_file.is_file():
        return None, None

    # Load results JSONL to get question_id -> correct mapping
    label_map = {}
    with open(results_file) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                label_map[rec["question_id"]] = bool(rec["correct"])

    # Load activations matching the benchmark prefix
    prefix = benchmark.replace("_challenge", "").replace("_qa", "")
    npy_files = sorted(acts_dir.glob(f"{prefix}_*.npy"))
    if not npy_files:
        # Try without prefix
        npy_files = sorted(acts_dir.glob("*.npy"))

    X_list = []
    y_list = []
    for npy_path in npy_files:
        # Extract question_id from filename: e.g. "arc_ACTAAP_2009_7_8__layer_25.npy"
        # -> stem is "arc_ACTAAP_2009_7_8__layer_25"
        stem = npy_path.stem
        # Remove __layer_N suffix to get question_id
        qid = stem.rsplit("__layer_", 1)[0] if "__layer_" in stem else stem
        if qid not in label_map:
            continue
        activation = np.load(npy_path).astype(np.float64)
        if activation.ndim == 2:
            activation = activation[-1]
        if activation.shape[0] != HIDDEN_DIM:
            continue
        X_list.append(activation)
        y_list.append(label_map[qid])

    if not X_list:
        return None, None

    return np.stack(X_list, axis=0), np.array(y_list, dtype=bool)


def generate_mock_data(n_samples=500):
    """Generate synthetic data for testing.

    Creates a 2D array with strong signal dimensions that correlate with labels,
    ensuring a trained probe can achieve AUROC > 0.80.

    Returns:
        X: np.ndarray of shape (n_samples, HIDDEN_DIM)
        y: np.ndarray of shape (n_samples,) bool
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n_features_signal = 100
    X_noise = rng.randn(n_samples, HIDDEN_DIM - n_features_signal).astype(np.float64)
    # Generate latent variable that determines label
    latent = rng.randn(n_samples)
    y = latent > 0.0
    # Signal features: 50 dims with positive correlation, 50 with negative
    X_signal_pos = latent[:, None] * 2.0 + rng.randn(n_samples, n_features_signal // 2) * 0.3
    X_signal_neg = -latent[:, None] * 2.0 + rng.randn(n_samples, n_features_signal // 2) * 0.3
    X = np.concatenate([X_signal_pos, X_signal_neg, X_noise], axis=1).astype(np.float64)
    return X, y


def apply_platt_calibration(raw_scores, y_true, cal_mask):
    """Apply Platt scaling on a calibration subset.

    Args:
        raw_scores: Raw sigmoid outputs from the probe.
        y_true: Ground truth labels.
        cal_mask: Boolean mask for calibration set indices.

    Returns:
        np.ndarray of calibrated probabilities (clipped to [1e-12, 1-1e-12]).
    """
    logits = np.log(np.maximum(raw_scores, 1e-12) / np.maximum(1 - raw_scores, 1e-12))
    cal_logits = logits[cal_mask]
    cal_labels = y_true[cal_mask]
    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(cal_logits.reshape(-1, 1), cal_labels)
    test_logits = logits[~cal_mask]
    calibrated = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 1e-12, 1 - 1e-12)


def main():
    parser = argparse.ArgumentParser(
        description="Train a within-domain logistic regression probe for a benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name (e.g. arc_challenge, math, humaneval, triviaqa)",
    )
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Use synthetic mock data instead of real activations (for testing)",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default=None,
        help="Override the activations base directory (default: data/benchmark_activations_v2)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=25,
        help="Layer number for probe metadata (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    benchmark = args.benchmark
    seed = args.seed

    # --- Load data ---
    if args.mock_data:
        print(f"[train_transfer_probes] Using mock data for benchmark '{benchmark}'")
        X, y = generate_mock_data()
    else:
        # Try v2 format first
        if args.activations_dir:
            base_dir = Path(args.activations_dir)
        else:
            base_dir = PROJECT_ROOT / "data" / "benchmark_activations_v2"
        benchmark_dir = base_dir / benchmark

        X, y = load_v2_data(benchmark_dir)

        # Fallback to legacy format
        if X is None:
            print(f"[train_transfer_probes] v2 data not found at {benchmark_dir}, trying legacy format...")
            X, y = load_legacy_data(benchmark)

        if X is None:
            print(
                f"[train_transfer_probes] ERROR: No data found for benchmark '{benchmark}'. "
                f"Use --mock_data for testing.",
                file=sys.stderr,
            )
            sys.exit(1)

    n_samples = X.shape[0]
    print(f"[train_transfer_probes] Loaded {n_samples} samples, X shape: {X.shape}")
    print(f"[train_transfer_probes] Label balance: {y.mean():.3f} correct")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    print(f"[train_transfer_probes] Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Split training data: 70% fit, 30% calibration (stratified) ---
    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=seed
    )
    print(f"[train_transfer_probes] Fit: {len(X_train_fit)}, Calibration: {len(X_cal)}")

    # --- Train LogisticRegressionCV on the fit subset ---
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    model = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        scoring="roc_auc",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(X_train_fit, y_train_fit)

    weights = {
        "coef": model.coef_[0].astype(np.float64),
        "intercept": float(model.intercept_[0]),
    }

    # --- Raw probe scores on test and calibration sets ---
    raw_scores = expit(np.dot(X_test, weights["coef"]) + weights["intercept"])
    cal_raw = expit(np.dot(X_cal, weights["coef"]) + weights["intercept"])

    # --- Platt scaling (fit on calibration set, apply to test) ---
    cal_logits = np.log(
        np.maximum(cal_raw, 1e-12) / np.maximum(1 - cal_raw, 1e-12)
    )
    test_logits = np.log(
        np.maximum(raw_scores, 1e-12) / np.maximum(1 - raw_scores, 1e-12)
    )
    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(cal_logits.reshape(-1, 1), y_cal)
    platt_scores = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    platt_scores = np.clip(platt_scores, 1e-12, 1 - 1e-12)

    # --- Raw metrics on test set ---
    raw_auroc = float(roc_auc_score(y_test, raw_scores))
    raw_ece = compute_ece(y_test, raw_scores)
    raw_brier = float(brier_score_loss(y_test, raw_scores))

    # --- Platt-calibrated metrics on test set ---
    platt_auroc = float(roc_auc_score(y_test, platt_scores))
    platt_ece = compute_ece(y_test, platt_scores)
    platt_brier = float(brier_score_loss(y_test, platt_scores))

    # Selective accuracy / coverage
    thresh_95 = find_threshold_for_selective_accuracy(platt_scores, y_test, target_acc=0.95)
    coverage_95 = compute_coverage(platt_scores, thresh_95)
    selective_acc_95 = compute_selective_accuracy(platt_scores, y_test, thresh_95)

    # --- Print results ---
    print()
    print("=" * 60)
    print(f"  Benchmark: {benchmark}")
    print(f"  Samples:   {n_samples}")
    print(f"  Layer:     {args.layer}")
    print(f"  Seed:      {seed}")
    print("-" * 60)
    print(f"  Raw AUROC:          {raw_auroc:.4f}")
    print(f"  Raw ECE:            {raw_ece:.4f}")
    print(f"  Raw Brier:          {raw_brier:.4f}")
    print(f"  Platt AUROC:        {platt_auroc:.4f}")
    print(f"  Platt ECE:          {platt_ece:.4f}")
    print(f"  Platt Brier:        {platt_brier:.4f}")
    print(f"  Coverage @ 95%:     {coverage_95:.4f}")
    print(f"  Selective Acc @ 95%: {selective_acc_95:.4f}")
    print("=" * 60)

    # --- Sanity check for mock data ---
    if args.mock_data:
        if platt_auroc > 0.80:
            print(f"[train_transfer_probes] MOCK DATA CHECK PASSED: AUROC={platt_auroc:.4f} > 0.80")
        else:
            print(
                f"[train_transfer_probes] MOCK DATA CHECK FAILED: AUROC={platt_auroc:.4f} <= 0.80",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Save probe weights ---
    probes_dir = PROJECT_ROOT / "data" / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    save_path = probes_dir / f"probe_{benchmark}.npz"
    np.savez(
        save_path,
        coef=weights["coef"],
        intercept=weights["intercept"],
        layer=args.layer,
        seed=seed,
        benchmark=benchmark,
    )
    print(f"[train_transfer_probes] Saved probe weights to {save_path}")


if __name__ == "__main__":
    main()
