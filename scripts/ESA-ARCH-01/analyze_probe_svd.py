#!/usr/bin/env python3
"""
ESA-ARCH-01 T2: SVD spectral diagnostics on per-domain probe weight matrices.

Loads probe_*.npz files from data/probes/, extracts weight vectors (coef),
computes SVD spectra for individual probes and the stacked matrix, and
emits a JSON report plus a singular-value spectrum plot.
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def effective_rank(s: np.ndarray) -> float:
    """Effective rank = exp(H) where H is the Shannon entropy of normalized singular values."""
    s = s[s > 0]
    p = s / s.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def participation_ratio(s: np.ndarray) -> float:
    """Participation ratio = (sum s_i)^2 / sum s_i^2."""
    s = s[s > 0]
    return float(s.sum() ** 2 / np.sum(s ** 2))


def top_k_variance_ratio(s: np.ndarray, k: int = 2) -> float:
    """Return (sum of top-k singular values) / (sum of all singular values)."""
    top = np.sort(s)[-k:]
    return float(top.sum() / s.sum())


def condition_number(s: np.ndarray) -> float:
    """2-norm condition number = sigma_max / sigma_min."""
    s = s[s > 0]
    return float(s.max() / s.min())


def shuffle_trap_test(W: np.ndarray, n_shuffles: int = 100, seed: int = 42) -> Tuple[float, float]:
    """
    Compare actual top2 variance ratio to that of column-shuffled versions of W.
    Returns (actual_top2_ratio, trap_ratio).
    """
    rng = np.random.default_rng(seed)
    _, s_actual, _ = np.linalg.svd(W, full_matrices=False)
    actual = top_k_variance_ratio(s_actual, k=2)
    shuffled_ratios = []
    for _ in range(n_shuffles):
        W_shuf = W.copy()
        for row in W_shuf:
            rng.shuffle(row)
        _, s_shuf, _ = np.linalg.svd(W_shuf, full_matrices=False)
        shuffled_ratios.append(top_k_variance_ratio(s_shuf, k=2))
    mean_shuffled = float(np.mean(shuffled_ratios))
    trap_ratio = actual / mean_shuffled if mean_shuffled > 0 else float("inf")
    return actual, trap_ratio


def load_probes(probe_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load individual probe weight vectors and return stacked matrix."""
    files = sorted(glob.glob(str(probe_dir / "probe_*.npz")))
    probes: Dict[str, np.ndarray] = {}
    for f in files:
        data = np.load(f)
        name = Path(f).stem.replace("probe_", "")
        probes[name] = data["coef"]
    W = np.stack([probes[k] for k in sorted(probes.keys())], axis=0)
    return probes, W


def analyze_probe(name: str, w: np.ndarray) -> Dict:
    """Run SVD diagnostics on a single probe weight vector."""
    W = w.reshape(1, -1)
    _, s, _ = np.linalg.svd(W, full_matrices=False)
    return {
        "name": name,
        "shape": list(W.shape),
        "singular_values": s.tolist(),
        "effective_rank": effective_rank(s),
        "participation_ratio": participation_ratio(s),
        "top2_ratio": top_k_variance_ratio(s, k=2),
        "condition_number": condition_number(s),
    }


def analyze_stacked(W: np.ndarray) -> Dict:
    """Run SVD diagnostics on the stacked probe matrix."""
    _, s, _ = np.linalg.svd(W, full_matrices=False)
    actual_top2, trap = shuffle_trap_test(W)
    return {
        "shape": list(W.shape),
        "singular_values": s.tolist(),
        "effective_rank": effective_rank(s),
        "participation_ratio": participation_ratio(s),
        "top2_ratio": actual_top2,
        "condition_number": condition_number(s),
        "shuffle_trap_ratio": trap,
    }


def make_plot(individual: Dict[str, Dict], stacked: Dict, out_path: Path) -> None:
    """Plot singular-value spectra for all probes and the stacked matrix."""
    n = len(individual) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, metrics) in zip(axes, individual.items()):
        s = np.array(metrics["singular_values"])
        ax.plot(np.arange(1, len(s) + 1), s, marker="o", markersize=2)
        ax.set_title(f"{name}\nER={metrics['effective_rank']:.1f}, PR={metrics['participation_ratio']:.1f}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", lw=0.5)

    s = np.array(stacked["singular_values"])
    ax = axes[len(individual)]
    ax.plot(np.arange(1, len(s) + 1), s, marker="o", markersize=2, color="darkred")
    ax.set_title(f"STACKED\nER={stacked['effective_rank']:.1f}, PR={stacked['participation_ratio']:.1f}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def verdict(stacked: Dict, hidden_dim: int) -> Dict:
    """Produce go/no-go statement based on thresholds."""
    er = stacked["effective_rank"]
    top2 = stacked["top2_ratio"]
    trap = stacked.get("shuffle_trap_ratio", None)
    low_dim_threshold = 0.3 * hidden_dim
    high_dim_threshold = 0.8 * hidden_dim

    reasons = []
    if er < low_dim_threshold:
        reasons.append(f"effective_rank {er:.1f} < {low_dim_threshold:.0f}: low-dimensional, single probe plausible")
        go = True
    elif er > high_dim_threshold:
        reasons.append(f"effective_rank {er:.1f} > {high_dim_threshold:.0f}: overfit/noise, probe bank needed")
        go = False
    else:
        reasons.append(f"effective_rank {er:.1f} in intermediate range")
        go = False

    if trap is not None and top2 < 2 and trap > 3:
        reasons.append(f"top2_ratio {top2:.2f} < 2 and trap_ratio {trap:.2f} > 3: artifact suspected, HALT")
        go = False

    return {
        "go": go,
        "reasons": reasons,
        "verdict": "GO" if go else "NO-GO",
    }


def main():
    base = Path("/home/guest/epistemic-steering")
    probe_dir = base / "data" / "probes"
    out_dir = base / "outputs" / "ESA-ARCH-01" / "svd_probe_weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    probes, W = load_probes(probe_dir)
    hidden_dim = W.shape[1]

    individual: Dict[str, Dict] = {}
    for name, w in sorted(probes.items()):
        individual[name] = analyze_probe(name, w)

    stacked = analyze_stacked(W)
    stacked["verdict"] = verdict(stacked, hidden_dim)

    report = {
        "hidden_dim": hidden_dim,
        "n_probes": len(probes),
        "individual_probes": individual,
        "stacked": stacked,
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    make_plot(individual, stacked, out_dir / "svd_spectrum.png")

    print("=" * 60)
    print("ESA-ARCH-01 T2: SVD Spectral Diagnostics Summary")
    print("=" * 60)
    print(f"Hidden dim: {hidden_dim}")
    print(f"Probes:     {', '.join(sorted(probes.keys()))}")
    print(f"Stacked:    {W.shape}")
    print("-" * 60)
    print(f"Stacked effective_rank:    {stacked['effective_rank']:.2f}")
    print(f"Stacked participation_ratio: {stacked['participation_ratio']:.2f}")
    print(f"Stacked top2_ratio:          {stacked['top2_ratio']:.4f}")
    print(f"Stacked condition_number:    {stacked['condition_number']:.4e}")
    print(f"Shuffle trap ratio:        {stacked.get('shuffle_trap_ratio', None)}")
    print("-" * 60)
    print(f"VERDICT: {stacked['verdict']['verdict']}")
    for r in stacked["verdict"]["reasons"]:
        print(f"  - {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
