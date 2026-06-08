#!/usr/bin/env uv run python
"""
Build the 6×6 cross-benchmark probe transfer matrix.

Evaluates each trained probe (from data/probes/) on every other benchmark's
activations (from data/benchmark_activations_v2/), producing a 36-cell matrix
of AUROC, ECE, and Brier scores (raw and Platt-calibrated).

Usage:
    uv run python scripts/build_transfer_matrix.py                          # Real data
    uv run python scripts/build_transfer_matrix.py --mock_data             # Synthetic test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
PROBES_DIR = PROJECT_ROOT / "data" / "probes"
ACTS_DIR = PROJECT_ROOT / "data" / "benchmark_activations_v2"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

BENCHMARKS = ["mmlu", "gsm8k", "math", "humaneval", "arc_challenge", "triviaqa"]
N_BENCHMARKS = len(BENCHMARKS)

# ── helpers ──────────────────────────────────────────────────────────────


def get_last_token(arr: np.ndarray) -> np.ndarray:
    """Extract last-token activation from a (seq_len, hidden) array."""
    return arr[-1, :] if arr.ndim == 2 else arr


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
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


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUROC, Brier, ECE."""
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
    }


def apply_probe(X: np.ndarray, weights: dict) -> np.ndarray:
    """Logistic regression forward pass: σ(X @ coef + intercept)."""
    return expit(np.dot(X, weights["coef"]) + weights["intercept"])


def apply_platt_calibration(
    raw_scores: np.ndarray, y_true: np.ndarray, cal_mask: np.ndarray
) -> np.ndarray:
    """Platt scaling on held-out calibration split."""
    logits = np.log(
        np.maximum(raw_scores, 1e-12) / np.maximum(1 - raw_scores, 1e-12)
    )
    cal_logits = logits[cal_mask]
    cal_labels = y_true[cal_mask]
    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(cal_logits.reshape(-1, 1), cal_labels)
    test_logits = logits[~cal_mask]
    calibrated = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 1e-12, 1 - 1e-12)


# ── loading ─────────────────────────────────────────────────────────────


def load_probe(bench_name: str) -> dict | None:
    """Load a trained probe .npz file from data/probes/."""
    path = PROBES_DIR / f"probe_{bench_name}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "coef": data["coef"].astype(np.float64),
        "intercept": float(data["intercept"]),
        "layer": int(data.get("layer", 30)),
        "token_position": str(data.get("token_position", "last")),
    }


def load_activations_and_labels(bench_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load activations and correctness labels for a benchmark.

    Activations: data/benchmark_activations_v2/{bench_name}/*.npy
    Labels:       data/benchmark_activations_v2/{bench_name}_results.jsonl
    """
    act_dir = ACTS_DIR / bench_name
    if not act_dir.exists():
        return None

    # Load labels
    results_path = ACTS_DIR / f"{bench_name}_results.jsonl"
    if not results_path.exists():
        return None
    labels = {}
    with open(results_path) as f:
        for line in f:
            entry = json.loads(line)
            labels[entry["question_id"]] = entry["correct"]

    # Load activations
    activations = {}
    for fpath in sorted(act_dir.glob("*.npy")):
        qid = fpath.stem.split("__")[0]
        try:
            act = get_last_token(np.load(fpath)).astype(np.float64)
            activations[qid] = act
        except Exception:
            continue

    valid_qids = sorted(set(activations.keys()) & set(labels.keys()))
    if not valid_qids:
        return None

    X = np.array([activations[qid] for qid in valid_qids])
    y = np.array([labels[qid] for qid in valid_qids], dtype=bool)
    return X, y


# ── evaluation ──────────────────────────────────────────────────────────


def evaluate_probe_on_benchmark(
    probe_weights: dict, X_test: np.ndarray, y_test: np.ndarray, seed: int = 42
) -> dict:
    """Evaluate a single probe on a single benchmark's activations.

    Uses a 70/30 train/calibration split for Platt scaling (same protocol
    as cross_benchmark_eval.py).
    """
    n_total = len(y_test)
    cal_size = max(int(n_total * 0.3), 20)
    cal_size = min(cal_size, n_total - 1)

    rng = np.random.RandomState(seed)
    cal_mask = np.zeros(n_total, dtype=bool)
    cal_indices = rng.choice(n_total, size=cal_size, replace=False)
    cal_mask[cal_indices] = True
    test_mask = ~cal_mask

    raw_scores = apply_probe(X_test, probe_weights)
    platt_scores = apply_platt_calibration(raw_scores, y_test, cal_mask)

    raw_metrics = compute_all_metrics(y_test[test_mask], raw_scores[test_mask])
    platt_metrics = compute_all_metrics(y_test[test_mask], platt_scores)

    return {
        "n_total": n_total,
        "n_cal": int(cal_mask.sum()),
        "n_eval": int(test_mask.sum()),
        "raw": raw_metrics,
        "platt": platt_metrics,
    }


# ── mock data ───────────────────────────────────────────────────────────


def _make_mock_probe(bench_name: str, rng: np.random.RandomState) -> dict:
    """Synthetic probe with random coefficients."""
    return {
        "coef": rng.randn(2560).astype(np.float64) * 0.1,
        "intercept": float(rng.randn() * 0.1),
        "layer": 30,
        "token_position": "last",
    }


def _make_mock_activations(
    bench_name: str, n_samples: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic activations and labels."""
    X = rng.randn(n_samples, 2560).astype(np.float64)
    y = rng.binomial(1, 0.6, size=n_samples).astype(bool)
    return X, y


def build_mock_data() -> dict:
    """Generate 6 synthetic probes + 6 synthetic benchmark datasets."""
    rng = np.random.RandomState(42)
    probes = {}
    datasets = {}
    for bench in BENCHMARKS:
        probes[bench] = _make_mock_probe(bench, rng)
        datasets[bench] = _make_mock_activations(bench, 200, rng)
    return {"probes": probes, "datasets": datasets}


# ── main ────────────────────────────────────────────────────────────────


def build_transfer_matrix(mock: bool = False) -> dict:
    """Build the 36-cell transfer matrix.

    Returns a dict keyed by "train→test" with per-cell metrics.
    """
    if mock:
        print("Using mock data (synthetic probes + activations)")
        mock_data = build_mock_data()
        probes = mock_data["probes"]
        datasets = mock_data["datasets"]
    else:
        probes = {}
        datasets = {}
        for bench in BENCHMARKS:
            probe = load_probe(bench)
            if probe is None:
                print(f"  WARNING: probe for {bench} not found at {PROBES_DIR / f'probe_{bench}.npz'}")
            probes[bench] = probe

            data = load_activations_and_labels(bench)
            if data is None:
                print(f"  WARNING: activations for {bench} not found in {ACTS_DIR / bench}")
            datasets[bench] = data

    matrix = {}
    for train_bench in BENCHMARKS:
        probe = probes.get(train_bench)
        if probe is None:
            print(f"  SKIP {train_bench}→* : probe missing")
            for test_bench in BENCHMARKS:
                matrix[f"{train_bench}→{test_bench}"] = {"status": "missing_probe"}
            continue

        for test_bench in BENCHMARKS:
            cell_key = f"{train_bench}→{test_bench}"
            data = datasets.get(test_bench)
            if data is None:
                matrix[cell_key] = {"status": "missing_activations"}
                continue

            X_test, y_test = data
            result = evaluate_probe_on_benchmark(probe, X_test, y_test)
            matrix[cell_key] = {
                "status": "completed",
                "train_benchmark": train_bench,
                "test_benchmark": test_bench,
                "n_total": result["n_total"],
                "n_cal": result["n_cal"],
                "n_eval": result["n_eval"],
                "auroc_raw": result["raw"]["auroc"],
                "brier_raw": result["raw"]["brier"],
                "ece_raw": result["raw"]["ece"],
                "auroc_platt": result["platt"]["auroc"],
                "brier_platt": result["platt"]["brier"],
                "ece_platt": result["platt"]["ece"],
            }

    return matrix


def save_csvs(matrix: dict) -> None:
    """Save 4 CSV files, each a 6×6 grid of one metric."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("auroc_raw", "AUROC_raw"),
        ("auroc_platt", "AUROC_Platt"),
        ("ece_raw", "ECE"),
        ("brier_raw", "Brier"),
    ]

    for metric_key, metric_name in metrics:
        grid = np.full((N_BENCHMARKS, N_BENCHMARKS), np.nan)
        for i, train_bench in enumerate(BENCHMARKS):
            for j, test_bench in enumerate(BENCHMARKS):
                cell = matrix.get(f"{train_bench}→{test_bench}", {})
                if cell.get("status") == "completed":
                    grid[i, j] = cell[metric_key]

        df = pd.DataFrame(grid, index=BENCHMARKS, columns=BENCHMARKS)
        df.index.name = "train_benchmark"
        csv_path = RESULTS_DIR / f"transfer_matrix_{metric_name}.csv"
        df.to_csv(csv_path)
        print(f"  Saved {csv_path}")


def save_json(matrix: dict) -> None:
    """Save full 36-cell dict as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "transfer_matrix.json"
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"  Saved {path}")


def generate_heatmap(matrix: dict) -> None:
    """Generate 6×6 AUROC heatmap."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build AUROC_raw grid
    grid = np.full((N_BENCHMARKS, N_BENCHMARKS), np.nan)
    for i, train_bench in enumerate(BENCHMARKS):
        for j, test_bench in enumerate(BENCHMARKS):
            cell = matrix.get(f"{train_bench}→{test_bench}", {})
            if cell.get("status") == "completed":
                grid[i, j] = cell["auroc_raw"]

    # Build annotation grid: show AUROC_raw values on cells
    annot = np.empty_like(grid, dtype=object)
    for i in range(N_BENCHMARKS):
        for j in range(N_BENCHMARKS):
            val = grid[i, j]
            if np.isfinite(val):
                annot[i, j] = f"{val:.3f}"
            else:
                annot[i, j] = ""

    # Short labels for display
    short_labels = {
        "mmlu": "MMLU",
        "gsm8k": "GSM8K",
        "math": "MATH",
        "humaneval": "HumanEval",
        "arc_challenge": "ARC",
        "triviaqa": "TriviaQA",
    }
    display_labels = [short_labels[b] for b in BENCHMARKS]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        grid,
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "AUROC"},
        ax=ax,
    )

    # Highlight diagonal cells with a white border
    for i in range(N_BENCHMARKS):
        ax.add_patch(
            plt.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor="white", lw=3, clip_on=False
            )
        )

    ax.set_title("Cross-Benchmark Probe Transfer Matrix", fontsize=14, pad=16)
    ax.set_xlabel("Test Benchmark", fontsize=12)
    ax.set_ylabel("Train Benchmark", fontsize=12)
    plt.tight_layout()

    path = FIGURES_DIR / "transfer_matrix_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 6×6 cross-benchmark probe transfer matrix"
    )
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Use synthetic probes and activations for testing",
    )
    args = parser.parse_args()

    print("Building transfer matrix...")
    matrix = build_transfer_matrix(mock=args.mock_data)

    # Count completed cells
    completed = sum(
        1 for v in matrix.values() if v.get("status") == "completed"
    )
    print(f"  Completed cells: {completed} / {N_BENCHMARKS * N_BENCHMARKS}")

    save_json(matrix)
    save_csvs(matrix)
    generate_heatmap(matrix)

    print("Done.")


if __name__ == "__main__":
    main()
