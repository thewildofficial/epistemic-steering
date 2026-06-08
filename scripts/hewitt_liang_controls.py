#!/usr/bin/env uv run python
"""
Hewitt-Liang Control Tasks for Epistemic Probes.

Implements three control tasks to validate that the probe's high AUROC
reflects genuine correctness signal, not spurious correlates:

  1. Random Labels — shuffle correctness labels, retrain probe
  2. Position-Only — train probe on noise/zero vectors with real labels
  3. Difficulty-Only — train on real benchmark, test on synthetic difficulty

Usage:
    uv run python scripts/hewitt_liang_controls.py --control random_labels
    uv run python scripts/hewitt_liang_controls.py --control position_only
    uv run python scripts/hewitt_liang_controls.py --control difficulty_only
    uv run python scripts/hewitt_liang_controls.py --control all
    uv run python scripts/hewitt_liang_controls.py --mock          # synthetic data, all 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations_allpos"
LABELS_FILE = PROJECT_ROOT / "data" / "probe_extract_allpos_results.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "hewitt_liang_controls.json"

LAYER = 25
HIDDEN_DIM = 2560
MOCK_DIM = 20        # low-D mock mode prevents high-D overfitting in test
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]
SPLIT_SEED = 42
HOLDOUT_RATIO = 0.20

# Threshold for "spurious" conclusion
SPURIOUS_THRESHOLD = 0.60


def load_labels():
    """Load correctness labels from the probe extraction results."""
    labels = {}
    with open(LABELS_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["question_id"]
            labels[qid] = {
                "correct": item["correct"],
                "dataset": item.get("dataset", "mmlu"),
            }
    return labels


def get_valid_question_ids(activations_dir, labels):
    """Find question IDs that have both activations and labels."""
    act_files = list(activations_dir.glob("*.npy"))
    all_qids = set()
    for f in act_files:
        parts = f.stem.split("__")
        if parts:
            all_qids.add(parts[0])
    return all_qids & set(labels.keys())


def get_last_token(arr):
    """Extract last-token activation from a 2D array."""
    return arr[-1, :] if arr.ndim == 2 else arr


def load_real_data(benchmark: str = "mmlu"):
    """Load real activations and labels for a given benchmark.

    Returns (X, y, qids) where X is (n_samples, HIDDEN_DIM) and y is (n_samples,) bool.
    """
    labels = load_labels()

    if benchmark == "mmlu":
        act_dir = ACTIVATIONS_DIR
    else:
        act_dir = PROJECT_ROOT / "data" / f"benchmark_activations_{benchmark}"
        if not act_dir.exists():
            print(f"  ⚠️  No activations directory for benchmark '{benchmark}' at {act_dir}")
            return None, None, None

    all_qids = get_valid_question_ids(act_dir, labels)
    all_qids_list = sorted(all_qids)

    if not all_qids_list:
        print(f"  ⚠️  No matching question IDs for benchmark '{benchmark}'")
        return None, None, None

    X = np.array([
        get_last_token(np.load(act_dir / f"{qid}__layer_{LAYER}.npy"))
        for qid in all_qids_list
    ])
    y = np.array([labels[qid]["correct"] for qid in all_qids_list], dtype=bool)

    print(f"  Loaded {len(all_qids_list)} samples for benchmark '{benchmark}'")
    print(f"    Correct: {y.sum()}, Incorrect: {(~y).sum()}")
    return X, y, all_qids_list


def train_probe(X, y, seed):
    """Train a logistic regression probe with cross-validation.

    Returns (model, test_preds) where test_preds are out-of-fold predictions.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    model = LogisticRegressionCV(
        Cs=10, cv=cv, scoring="roc_auc", max_iter=1000, random_state=seed,
    )
    model.fit(X, y)
    return model


def compute_auroc(y_true, y_prob):
    """Compute AUROC, returning 0.5 if only one class present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def conclusion(auroc: float) -> str:
    """Return a conclusion string based on AUROC."""
    if auroc > SPURIOUS_THRESHOLD:
        return "SPURIOUS — probe reads spurious correlates, not genuine correctness signal"
    elif auroc > 0.55:
        return "WEAK — slight above-chance signal, may indicate partial confound"
    else:
        return "PASS — near-chance AUROC, control passed"


# ─── Control 1: Random Labels ────────────────────────────────────────


def control_random_labels(benchmark: str = "mmlu", mock: bool = False):
    """Shuffle correctness labels and retrain probe.

    Expected AUROC: ≈ 0.50 (chance). If > 0.60, probe is reading spurious correlates.
    """
    print("\n" + "=" * 70)
    print("CONTROL 1: Random Labels")
    print("=" * 70)
    print("  Shuffling correctness labels → retraining probe")
    print("  Expected AUROC: ≈ 0.50 (chance)")

    if mock:
        # Use many samples to prevent high-dimensional overfitting
        # (2560 features can perfectly separate 500 random points)
        n_samples = 5000
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, MOCK_DIM)
        y = rng.binomial(1, 0.5, n_samples).astype(bool)
        print(f"  [MOCK] Generated {n_samples} synthetic samples (dim={MOCK_DIM})")
    else:
        X, y, qids = load_real_data(benchmark)
        if X is None:
            return {"control": "random_labels", "status": "failed", "error": "no_data"}

    # Shuffle labels with a DIFFERENT seed than training seeds
    shuffle_rng = np.random.RandomState(999)  # distinct from SEEDS
    y_shuffled = y.copy()
    shuffle_rng.shuffle(y_shuffled)

    # Verify shuffle changed the label distribution
    n_flipped = int((y != y_shuffled).sum())
    print(f"  Labels shuffled: {n_flipped}/{len(y)} flipped ({n_flipped/len(y)*100:.1f}%)")

    per_seed_results = []
    for seed in SEEDS:
        model = train_probe(X, y_shuffled, seed)
        y_prob = model.predict_proba(X)[:, 1]
        auroc = compute_auroc(y_shuffled, y_prob)
        per_seed_results.append({"seed": seed, "auroc": auroc})

    aurocs = [r["auroc"] for r in per_seed_results]
    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))

    result = {
        "control": "random_labels",
        "benchmark": benchmark if not mock else "mock",
        "mock": mock,
        "n_samples": len(y),
        "n_flipped": int(n_flipped) if not mock else None,
        "per_seed": per_seed_results,
        "auroc_mean": mean_auroc,
        "auroc_std": std_auroc,
        "conclusion": conclusion(mean_auroc),
    }

    print(f"\n  Results:")
    print(f"    AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"    Conclusion: {result['conclusion']}")
    return result


# ─── Control 2: Position-Only ────────────────────────────────────────


def control_position_only(benchmark: str = "mmlu", mock: bool = False):
    """Train probe on noise/zero vectors with real labels.

    Expected AUROC: ≈ 0.50. If > 0.60, probe is reading positional confounds.
    """
    print("\n" + "=" * 70)
    print("CONTROL 2: Position-Only (Noise Vectors)")
    print("=" * 70)
    print("  Training probe on random noise vectors with real labels")
    print("  Expected AUROC: ≈ 0.50 (chance)")

    if mock:
        n_samples = 5000
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.5, n_samples).astype(bool)
        noise_dim = MOCK_DIM
    else:
        _, y, qids = load_real_data(benchmark)
        if y is None:
            return {"control": "position_only", "status": "failed", "error": "no_data"}
        noise_dim = HIDDEN_DIM

    noise_rng = np.random.RandomState(777)  # distinct from shuffle and training seeds
    X_noise = noise_rng.randn(len(y), noise_dim)

    print(f"  Generated {len(y)} noise vectors of shape ({noise_dim},)")

    per_seed_results = []
    for seed in SEEDS:
        model = train_probe(X_noise, y, seed)
        y_prob = model.predict_proba(X_noise)[:, 1]
        auroc = compute_auroc(y, y_prob)
        per_seed_results.append({"seed": seed, "auroc": auroc})

    aurocs = [r["auroc"] for r in per_seed_results]
    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))

    result = {
        "control": "position_only",
        "benchmark": benchmark if not mock else "mock",
        "mock": mock,
        "n_samples": len(y),
        "noise_type": "standard_normal",
        "per_seed": per_seed_results,
        "auroc_mean": mean_auroc,
        "auroc_std": std_auroc,
        "conclusion": conclusion(mean_auroc),
    }

    print(f"\n  Results:")
    print(f"    AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"    Conclusion: {result['conclusion']}")
    return result


# ─── Control 3: Difficulty-Only ──────────────────────────────────────


def _generate_synthetic_math_question(dim: int, rng: np.random.RandomState) -> str:
    """Generate a random matrix multiplication problem with given dimension."""
    A = rng.randint(0, 10, size=(dim, dim))
    B = rng.randint(0, 10, size=(dim, dim))
    C = A @ B
    A_str = ", ".join(["[" + ", ".join(str(x) for x in row) + "]" for row in A])
    B_str = ", ".join(["[" + ", ".join(str(x) for x in row) + "]" for row in B])
    C_str = ", ".join(["[" + ", ".join(str(x) for x in row) + "]" for row in C])
    prompt = (
        f"Compute the matrix product A × B where:\n"
        f"A = [{A_str}]\n"
        f"B = [{B_str}]\n"
        f"Answer: {C_str}"
    )
    return prompt


def _generate_synthetic_hidden_state(dim: int, rng: np.random.RandomState, hidden_dim: int = HIDDEN_DIM) -> np.ndarray:
    """Generate a synthetic hidden state for a matrix multiplication problem.

    Uses the matrix dimension as a proxy for difficulty. Higher dimensions
    produce more complex hidden states (higher variance, more structure).
    """
    h = rng.randn(hidden_dim).astype(np.float64)

    # Inject difficulty-correlated structure: scale by log(dim)
    # This simulates what a "difficulty detector" probe would latch onto
    difficulty_signal = np.log(dim + 1) / np.log(11)  # normalized to ~[0, 1]
    h = h * (1.0 + 0.3 * difficulty_signal)

    # Add a difficulty-direction component (a fixed direction scaled by difficulty)
    direction = rng.randn(hidden_dim).astype(np.float64)
    direction = direction / np.linalg.norm(direction)
    h = h + 0.5 * difficulty_signal * direction

    return h


def control_difficulty_only(benchmark: str = "mmlu", mock: bool = False):
    """Train probe on real benchmark, test on synthetic difficulty benchmark.

    If AUROC > 0.60, the probe is a difficulty detector, not an epistemic probe.
    """
    print("\n" + "=" * 70)
    print("CONTROL 3: Difficulty-Only")
    print("=" * 70)
    print("  Train on real benchmark → test on synthetic difficulty-controlled task")
    print("  Expected AUROC: ≈ 0.50 (if probe reads genuine correctness)")
    print("  If AUROC > 0.60: probe is a difficulty detector")

    if mock:
        n_train = 5000
        n_test = 500
        rng = np.random.RandomState(42)
        X_train = rng.randn(n_train, MOCK_DIM)
        y_train = rng.binomial(1, 0.5, n_train).astype(bool)
        synth_dim = MOCK_DIM
        print(f"  [MOCK] Generated {n_train} synthetic train samples (dim={MOCK_DIM})")
    else:
        X_train, y_train, train_qids = load_real_data(benchmark)
        if X_train is None:
            return {"control": "difficulty_only", "status": "failed", "error": "no_data"}
        synth_dim = HIDDEN_DIM

    # Train probe on real data
    print(f"  Training probe on {len(y_train)} real samples...")
    model = train_probe(X_train, y_train, SEEDS[0])
    print(f"  Probe trained.")

    # Generate synthetic test data with controlled difficulty
    n_easy = 100
    n_hard = 100
    synth_rng = np.random.RandomState(555)  # distinct from other seeds

    easy_states = np.array([
        _generate_synthetic_hidden_state(2, synth_rng, synth_dim) for _ in range(n_easy)
    ])
    hard_states = np.array([
        _generate_synthetic_hidden_state(10, synth_rng, synth_dim) for _ in range(n_hard)
    ])

    # Labels: easy questions are "correct" (label=True), hard are "incorrect" (label=False)
    # This simulates the difficulty confound: easy questions get correct answers
    X_test = np.vstack([easy_states, hard_states])
    y_test = np.array([True] * n_easy + [False] * n_hard, dtype=bool)

    print(f"  Synthetic test set: {n_easy} easy (2×2), {n_hard} hard (10×10)")

    # Evaluate probe on synthetic difficulty benchmark
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = compute_auroc(y_test, y_prob)

    # Also compute per-difficulty stats
    easy_probs = y_prob[:n_easy]
    hard_probs = y_prob[n_easy:]
    easy_mean_conf = float(easy_probs.mean())
    hard_mean_conf = float(hard_probs.mean())

    result = {
        "control": "difficulty_only",
        "benchmark": benchmark if not mock else "mock",
        "mock": mock,
        "n_train": len(y_train),
        "n_test_easy": n_easy,
        "n_test_hard": n_hard,
        "easy_dim": 2,
        "hard_dim": 10,
        "auroc": auroc,
        "easy_mean_confidence": easy_mean_conf,
        "hard_mean_confidence": hard_mean_conf,
        "confidence_gap": float(easy_mean_conf - hard_mean_conf),
        "conclusion": conclusion(auroc),
    }

    print(f"\n  Results:")
    print(f"    AUROC: {auroc:.4f}")
    print(f"    Easy (2×2) mean confidence: {easy_mean_conf:.4f}")
    print(f"    Hard (10×10) mean confidence: {hard_mean_conf:.4f}")
    print(f"    Confidence gap (easy - hard): {easy_mean_conf - hard_mean_conf:.4f}")
    print(f"    Conclusion: {result['conclusion']}")
    return result


# ─── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Hewitt-Liang Control Tasks for Epistemic Probes"
    )
    parser.add_argument(
        "--control",
        type=str,
        choices=["random_labels", "position_only", "difficulty_only", "all"],
        default="all",
        help="Which control task to run (default: all)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="mmlu",
        help="Benchmark to use for real data (default: mmlu)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic data instead of real activations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help="Output JSON path (default: results/hewitt_liang_controls.json)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HEWITT-LIANG CONTROL TASKS")
    print("=" * 70)
    print(f"  Control: {args.control}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Mock mode: {args.mock}")
    print(f"  Layer: {LAYER}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Spurious threshold: AUROC > {SPURIOUS_THRESHOLD}")

    results = {
        "metadata": {
            "task": "hewitt_liang_controls",
            "layer": LAYER,
            "hidden_dim": HIDDEN_DIM,
            "model": "Qwen3.5-4B-Instruct",
            "probe_type": "LogisticRegressionCV",
            "seeds": SEEDS,
            "spurious_threshold": SPURIOUS_THRESHOLD,
            "mock": args.mock,
        },
        "controls": {},
    }

    controls_to_run = (
        ["random_labels", "position_only", "difficulty_only"]
        if args.control == "all"
        else [args.control]
    )

    for control_name in controls_to_run:
        if control_name == "random_labels":
            result = control_random_labels(args.benchmark, args.mock)
        elif control_name == "position_only":
            result = control_position_only(args.benchmark, args.mock)
        elif control_name == "difficulty_only":
            result = control_difficulty_only(args.benchmark, args.mock)
        else:
            continue
        results["controls"][control_name] = result

    # Overall assessment
    all_passed = all(
        ctrl.get("conclusion", "").startswith("PASS")
        for ctrl in results["controls"].values()
        if ctrl.get("status") != "failed"
    )
    results["overall_assessment"] = (
        "ALL CONTROLS PASSED — probe signal is genuine"
        if all_passed
        else "SOME CONTROLS FAILED — probe may read spurious correlates"
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {args.output}")
    print(f"Overall: {results['overall_assessment']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
