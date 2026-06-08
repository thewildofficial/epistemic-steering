#!/usr/bin/env python3
"""
analyze_transfer_matrix.py — Visualization and statistical analysis for D1
cross-benchmark transfer matrix.

Usage:
    python scripts/analyze_transfer_matrix.py [--input results/transfer_matrix.json] [--output_dir figures/]

Outputs:
    - figures/transfer_heatmap.png      — 6×6 AUROC heatmap
    - figures/transfer_dendrogram.png    — hierarchical clustering dendrogram
    - figures/transfer_asymmetry.png     — A→B vs B→A asymmetry scatter
    - stdout                             — statistical summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
BENCHMARK_LABELS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "math": "MATH",
    "humaneval": "HumanEval",
    "arc_challenge": "ARC-Challenge",
    "triviaqa": "TriviaQA",
}

CMAP = "viridis"
AUROC_MIN = 0.5
AUROC_MAX = 1.0


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-benchmark transfer matrix."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/transfer_matrix.json",
        help="Path to transfer_matrix.json (default: results/transfer_matrix.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save figures (default: figures/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Loading & helpers
# ---------------------------------------------------------------------------
def load_matrix(path: str) -> dict:
    """Load and validate the transfer matrix JSON.

    Accepts either nested format {"benchmarks": [...], "cells": {...}}
    or flat format {"train→test": {...}} (auto-detected).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"[ERROR] Input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path_obj, "r") as f:
        data = json.load(f)

    if "benchmarks" not in data or "cells" not in data:
        benchmarks = set()
        for key in data:
            if "→" in key:
                train, test = key.split("→", 1)
                benchmarks.add(train)
                benchmarks.add(test)
        benchmarks = sorted(benchmarks)
        data = {"benchmarks": benchmarks, "cells": data}
        print(f"[INFO] Detected flat JSON format. Inferred {len(benchmarks)} benchmarks.")

    required = {"cells", "benchmarks"}
    if not required.issubset(data.keys()):
        print(
            f"[ERROR] Missing required keys: {required - set(data.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    benchmarks = data["benchmarks"]
    expected_entries = len(benchmarks) ** 2
    if len(data["cells"]) != expected_entries:
        print(
            f"[WARN] Expected {expected_entries} cell entries, found {len(data['cells'])}",
            file=sys.stderr,
        )

    return data


def build_auroc_matrix(data: dict) -> np.ndarray:
    """Build 6×6 AUROC matrix from cell data."""
    benchmarks = data["benchmarks"]
    n = len(benchmarks)
    matrix = np.zeros((n, n))
    for i, train in enumerate(benchmarks):
        for j, test in enumerate(benchmarks):
            key = f"{train}→{test}"
            cell = data["cells"].get(key)
            if cell is None:
                print(f"[WARN] Missing cell: {key}", file=sys.stderr)
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = cell.get("auroc_raw", np.nan)
    return matrix


def make_labels(benchmarks: list[str]) -> list[str]:
    """Convert raw benchmark IDs to display labels."""
    return [BENCHMARK_LABELS.get(b, b.replace("_", " ").title()) for b in benchmarks]


# ---------------------------------------------------------------------------
#  1. Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(
    matrix: np.ndarray,
    benchmarks: list[str],
    output_dir: str,
) -> None:
    """6×6 AUROC heatmap with annotated cells and highlighted diagonal."""
    labels = make_labels(benchmarks)
    n = len(benchmarks)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Mask NaN for display
    mask = np.isnan(matrix)

    # Draw heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=CMAP,
        vmin=AUROC_MIN,
        vmax=AUROC_MAX,
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        cbar_kws={"label": "AUROC", "shrink": 0.8},
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )

    # Highlight diagonal with a border
    for i in range(n):
        ax.add_patch(
            plt.Rectangle(
                (i, i), 1, 1, fill=False, edgecolor="gold", lw=3,
            )
        )

    ax.set_title("Cross-Benchmark Transfer Matrix (AUROC)", fontsize=14, pad=16)
    ax.set_xlabel("Test Benchmark", fontsize=12)
    ax.set_ylabel("Train Benchmark", fontsize=12)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "transfer_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ---------------------------------------------------------------------------
#  2. Dendrogram
# ---------------------------------------------------------------------------
def plot_dendrogram(
    matrix: np.ndarray,
    benchmarks: list[str],
    output_dir: str,
) -> np.ndarray:
    """Hierarchical clustering of benchmarks by cosine distance on AUROC vectors.

    Returns the linkage matrix for later cluster assignment.
    """
    labels = make_labels(benchmarks)

    # Handle any NaN rows by replacing with column means
    row_means = np.nanmean(matrix, axis=1)
    matrix_clean = np.where(np.isnan(matrix), row_means[:, None], matrix)

    # Cosine distance between rows (train benchmarks)
    dist_vec = pdist(matrix_clean, metric="cosine")
    Z = linkage(dist_vec, method="average")

    fig, ax = plt.subplots(figsize=(9, 5))
    dn = dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_font_size=11,
        above_threshold_color="gray",
    )

    ax.set_title("Benchmark Clustering by Transfer Similarity", fontsize=14, pad=16)
    ax.set_ylabel("Cosine Distance", fontsize=12)
    ax.set_xlabel("Benchmark", fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "transfer_dendrogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")

    return Z


# ---------------------------------------------------------------------------
#  3. Asymmetry Scatter
# ---------------------------------------------------------------------------
def plot_asymmetry(
    matrix: np.ndarray,
    benchmarks: list[str],
    output_dir: str,
) -> tuple[str, float]:
    """Scatter of A→B vs B→A AUROC for all off-diagonal pairs.

    Returns the most asymmetric pair (name, delta).
    """
    labels = make_labels(benchmarks)
    n = len(benchmarks)

    pairs: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            ab = matrix[i, j]
            ba = matrix[j, i]
            if np.isnan(ab) or np.isnan(ba):
                continue
            pairs.append({
                "i": i,
                "j": j,
                "ab": ab,
                "ba": ba,
                "label": f"{labels[i]}→{labels[j]}",
                "delta": abs(ab - ba),
            })

    if not pairs:
        print("[WARN] No valid off-diagonal pairs for asymmetry plot.")
        return "", 0.0

    fig, ax = plt.subplots(figsize=(7, 7))

    # Compute axis limits with some padding
    all_vals = [p["ab"] for p in pairs] + [p["ba"] for p in pairs]
    lo = max(AUROC_MIN, min(all_vals) - 0.05)
    hi = min(AUROC_MAX, max(all_vals) + 0.05)

    # y = x reference line
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x")

    # Plot points
    xs = [p["ab"] for p in pairs]
    ys = [p["ba"] for p in pairs]
    sc = ax.scatter(
        xs, ys, c=[p["delta"] for p in pairs],
        cmap="plasma", s=60, edgecolors="k", linewidth=0.5,
        vmin=0, vmax=max(p["delta"] for p in pairs) + 0.02,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("|A→B − B→A|", fontsize=10)

    # Label outliers (top 4 most asymmetric)
    sorted_pairs = sorted(pairs, key=lambda p: p["delta"], reverse=True)
    for p in sorted_pairs[:4]:
        ax.annotate(
            p["label"],
            (p["ab"], p["ba"]),
            fontsize=8,
            xytext=(6, 6),
            textcoords="offset points",
            alpha=0.85,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Transfer Asymmetry (A→B vs B→A)", fontsize=14, pad=16)
    ax.set_xlabel("AUROC (A → B)", fontsize=12)
    ax.set_ylabel("AUROC (B → A)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(output_dir, "transfer_asymmetry.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")

    # Return most asymmetric pair
    worst = max(pairs, key=lambda p: p["delta"])
    return worst["label"], worst["delta"]


# ---------------------------------------------------------------------------
#  Statistical Analysis
# ---------------------------------------------------------------------------
def print_statistics(
    matrix: np.ndarray,
    benchmarks: list[str],
    linkage_matrix: np.ndarray,
    worst_pair: tuple[str, float],
) -> None:
    """Print full statistical report to stdout."""
    labels = make_labels(benchmarks)
    n = len(benchmarks)
    print("=" * 68)
    print("  D1 Cross-Benchmark Transfer — Statistical Analysis")
    print("=" * 68)

    # 1. Within-domain AUROCs (diagonal)
    print("\n[1] Within-Domain AUROCs (Diagonal)")
    print("-" * 40)
    within = []
    for i in range(n):
        val = matrix[i, i]
        within.append(val)
        flag = " ⚠ < 0.80" if not np.isnan(val) and val < 0.80 else ""
        print(f"  {labels[i]:20s}  {val:.4f}{flag}")
    if within:
        mean_within = np.nanmean(within)
        print(f"  {'─' * 32}")
        print(f"  {'Mean':20s}  {mean_within:.4f}")

    # 2. Mean cross-domain transfer per benchmark
    print("\n[2] Mean Cross-Domain Transfer (Off-Diagonal Rows)")
    print("-" * 40)
    cross_means = {}
    for i in range(n):
        row = [matrix[i, j] for j in range(n) if j != i]
        val = np.nanmean(row) if row else np.nan
        cross_means[benchmarks[i]] = val
        print(f"  {labels[i]:20s}  {val:.4f}")

    # 3. Most asymmetric pair
    print("\n[3] Most Asymmetric Pair")
    print("-" * 40)
    print(f"  {worst_pair[0]}  (Δ = {worst_pair[1]:.4f})")

    # 4. Cluster membership
    print("\n[4] Cluster Membership (Dendrogram Cut)")
    print("-" * 40)
    # Cut at distance that yields 2-4 clusters
    if linkage_matrix is not None and len(linkage_matrix) > 0:
        clusters = fcluster(linkage_matrix, t=0.6, criterion="distance")
        unique_clusters = sorted(set(clusters))
        for cid in unique_clusters:
            members = [labels[i] for i in range(n) if clusters[i] == cid]
            print(f"  Cluster {cid}: {', '.join(members)}")
    else:
        print("  (Not enough data for clustering)")

    # 5. Correlation between within-domain and mean cross-domain
    print("\n[5] Within-Domain vs Cross-Domain Correlation")
    print("-" * 40)
    valid = [(within[i], cross_means[b]) for i, b in enumerate(benchmarks)
             if not np.isnan(within[i]) and not np.isnan(cross_means[b])]
    if len(valid) >= 3:
        corr = np.corrcoef([v[0] for v in valid], [v[1] for v in valid])[0, 1]
        print(f"  Pearson r = {corr:.4f}")
        if corr > 0.5:
            print("  → Positive: benchmarks with good within-domain transfer")
            print("    also transfer well to other benchmarks.")
        elif corr < -0.5:
            print("  → Negative: within-domain strength trades off against")
            print("    cross-domain generality.")
        else:
            print("  → Weak/no correlation: within-domain and cross-domain")
            print("    transfer are largely independent.")
    else:
        print("  (Insufficient data)")

    # 6. Range statistics
    print("\n[6] Range Summary")
    print("-" * 40)
    all_vals = matrix[~np.isnan(matrix)]
    if len(all_vals) > 0:
        flat = sorted(all_vals, reverse=True)
        print(f"  Global max:     {flat[0]:.4f}")
        print(f"  Global min:     {flat[-1]:.4f}")
        # Strongest / weakest transfers (excluding diagonal)
        off_diag = [matrix[i, j] for i in range(n) for j in range(n)
                    if i != j and not np.isnan(matrix[i, j])]
        if off_diag:
            max_idx = np.argmax(off_diag)
            min_idx = np.argmin(off_diag)
            # Map back to coordinates
            od_idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j and not np.isnan(matrix[i, j]):
                        if od_idx == max_idx:
                            print(f"  Strongest cross:  {labels[i]}→{labels[j]} = {off_diag[max_idx]:.4f}")
                        if od_idx == min_idx:
                            print(f"  Weakest cross:    {labels[i]}→{labels[j]} = {off_diag[min_idx]:.4f}")
                        od_idx += 1

    print("\n" + "=" * 68)
    print("  Analysis complete.")
    print("=" * 68)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_matrix(args.input)
    benchmarks = data["benchmarks"]
    matrix = build_auroc_matrix(data)

    # Set seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Generate visualizations
    plot_heatmap(matrix, benchmarks, args.output_dir)

    Z = plot_dendrogram(matrix, benchmarks, args.output_dir)

    worst_pair_name, worst_pair_delta = plot_asymmetry(
        matrix, benchmarks, args.output_dir
    )

    # Print statistics
    print_statistics(matrix, benchmarks, Z, (worst_pair_name, worst_pair_delta))


if __name__ == "__main__":
    main()