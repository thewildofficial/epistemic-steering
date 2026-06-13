#!/usr/bin/env python3
"""
ESA-ARCH-01 T6: CORAL Covariance Alignment Distances

Computes pairwise CORAL distance between domain activation distributions
at layer 25. CORAL distance = || C_s^{1/2} - C_t^{1/2} ||_F

Small distance → similar covariance structure → good probe-transfer candidate.
"""

import numpy as np
import glob
import json
import os
import sys
from collections import defaultdict
from scipy.linalg import sqrtm, fractional_matrix_power

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = "data/activations_allpos"
OUTPUT_DIR = "outputs/ESA-ARCH-01/coral_baseline"
DOMAINS = ["arc", "HumanEval", "math", "triviaqa"]
LAYER = 25
SHRINKAGE = 0.1  # Ledoit-Wolf style shrinkage intensity

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load and mean-pool activations ────────────────────────────────────────
print("=" * 60)
print("ESA-ARCH-01 T6: CORAL Covariance Alignment Distances")
print("=" * 60)

files = sorted(glob.glob(f"{DATA_DIR}/*__layer_{LAYER}.npy"))
print(f"\nTotal layer-{LAYER} files: {len(files)}")

by_domain = defaultdict(list)
for f in files:
    name = os.path.basename(f)
    prefix = name.split("__")[0].split("_")[0]
    by_domain[prefix].append(f)

# Load and mean-pool each domain
domain_activations = {}
for dom in DOMAINS:
    fs = by_domain[dom]
    pooled = []
    for f in fs:
        arr = np.load(f).astype(np.float64)  # (seq_len, 2560)
        pooled.append(arr.mean(axis=0))       # (2560,)
    domain_activations[dom] = np.stack(pooled, axis=0)  # (N, 2560)
    print(f"  {dom}: {len(pooled)} samples, shape {domain_activations[dom].shape}")

# ── 2. Compute regularized covariance matrices ──────────────────────────────
def regularized_cov(X, shrinkage=SHRINKAGE):
    """Compute covariance with shrinkage toward identity."""
    n, d = X.shape
    # Center
    X_centered = X - X.mean(axis=0, keepdims=True)
    # Sample covariance
    cov = (X_centered.T @ X_centered) / (n - 1)
    # Shrink toward identity
    target = np.eye(d) * np.trace(cov) / d
    cov_reg = (1 - shrinkage) * cov + shrinkage * target
    return cov_reg

def cov_sqrt(cov):
    """Compute symmetric square root of covariance matrix."""
    # Use eigendecomposition for stability
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Clamp negative eigenvalues (shouldn't happen with shrinkage, but be safe)
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return sqrt_cov

def coral_distance(cov_s_sqrt, cov_t_sqrt):
    """CORAL distance = || C_s^{1/2} - C_t^{1/2} ||_F"""
    diff = cov_s_sqrt - cov_t_sqrt
    return np.linalg.norm(diff, ord="fro")

print("\nComputing covariance matrices (2560 x 2560, with shrinkage)...")
covs = {}
cov_sqrts = {}
for dom in DOMAINS:
    X = domain_activations[dom]
    covs[dom] = regularized_cov(X)
    cov_sqrts[dom] = cov_sqrt(covs[dom])
    print(f"  {dom}: covariance computed, trace = {np.trace(covs[dom]):.2f}")

# ── 3. Pairwise CORAL distance matrix ────────────────────────────────────────
print("\nComputing pairwise CORAL distances...")
n_domains = len(DOMAINS)
dist_matrix = np.zeros((n_domains, n_domains))

for i, dom_i in enumerate(DOMAINS):
    for j, dom_j in enumerate(DOMAINS):
        if i == j:
            dist_matrix[i, j] = 0.0
        else:
            dist_matrix[i, j] = coral_distance(cov_sqrts[dom_i], cov_sqrts[dom_j])

# ── 4. Build output ─────────────────────────────────────────────────────────
# Distance matrix as dict
dist_dict = {}
for i, dom_i in enumerate(DOMAINS):
    for j, dom_j in enumerate(DOMAINS):
        if i < j:
            key = f"{dom_i}__{dom_j}"
            dist_dict[key] = round(float(dist_matrix[i, j]), 4)

# Flag good transfer candidates (below median)
all_dists = np.array(list(dist_dict.values()))
median_dist = float(np.median(all_dists))
good_transfer = {k: v for k, v in dist_dict.items() if v < median_dist}

output = {
    "layer": LAYER,
    "shrinkage": SHRINKAGE,
    "n_domains": n_domains,
    "domains": DOMAINS,
    "pairwise_coral_distances": dist_dict,
    "median_distance": round(median_dist, 4),
    "good_transfer_candidates": good_transfer,
    "distance_matrix": dist_matrix.tolist(),
}

# Save
out_path = os.path.join(OUTPUT_DIR, "domain_distance_matrix.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to: {out_path}")

# ── 5. Print summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORAL DISTANCE SUMMARY")
print("=" * 60)
print(f"\nLayer: {LAYER} | Shrinkage: {SHRINKAGE}")
print(f"Domains: {DOMAINS}")
print(f"\nPairwise CORAL distances (lower = more similar):")
for key, val in sorted(dist_dict.items(), key=lambda x: x[1]):
    dom_a, dom_b = key.split("__")
    flag = " ★ GOOD TRANSFER" if val < median_dist else ""
    print(f"  {dom_a:>12} ↔ {dom_b:<12}: {val:.4f}{flag}")

print(f"\nMedian distance: {median_dist:.4f}")
print(f"Good transfer candidates (below median):")
for key, val in good_transfer.items():
    dom_a, dom_b = key.split("__")
    print(f"  {dom_a:>12} ↔ {dom_b:<12}: {val:.4f}")

# ── 6. Optional: AUROC after CORAL whitening (one pair) ─────────────────────
print("\n" + "-" * 60)
print("Optional: AUROC after CORAL whitening (arc ↔ triviaqa)")
print("-" * 60)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    # Pick a pair: arc (source) and triviaqa (target)
    X_s = domain_activations["arc"]
    X_t = domain_activations["triviaqa"]

    # Whitening: transform source with source cov, re-color with target cov
    # C_s^{-1/2} @ X_s.T  → whitened
    # C_t^{1/2} @ whitened → re-colored to target distribution
    cov_s_sqrt_inv = np.linalg.inv(cov_sqrts["arc"])
    X_s_whitened = X_s @ cov_s_sqrt_inv  # (N_s, 2560)
    X_s_adapted = X_s_whitened @ cov_sqrts["triviaqa"]  # re-colored

    # Train probe on original source, test on target
    # vs train on adapted source, test on target
    # We need labels — use correctness as proxy
    # Load logits for correctness labels
    def load_labels(domain):
        logit_files = sorted(glob.glob(f"{DATA_DIR}/{domain}_*__logits.npy"))
        labels = []
        for f in logit_files:
            logits = np.load(f)  # (seq_len, vocab_size) or (seq_len,)
            if logits.ndim == 2:
                # Use max logit as correctness proxy (not ideal, but illustrative)
                labels.append(1.0)
            else:
                labels.append(float(logits.mean()))
        return np.array(labels)

    y_s = load_labels("arc")
    y_t = load_labels("triviaqa")

    if len(y_s) == X_s.shape[0] and len(y_t) == X_t.shape[0]:
        # Train probe on original source
        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_s, y_s > y_s.mean())
        y_pred_t = probe.predict_proba(X_t)[:, 1]
        auroc_original = roc_auc_score(y_t > y_t.mean(), y_pred_t)

        # Train probe on adapted source
        probe_adapted = LogisticRegression(max_iter=1000, C=1.0)
        probe_adapted.fit(X_s_adapted, y_s > y_s.mean())
        y_pred_t_adapted = probe_adapted.predict_proba(X_t)[:, 1]
        auroc_adapted = roc_auc_score(y_t > y_t.mean(), y_pred_t_adapted)

        print(f"  AUROC (original source → target): {auroc_original:.4f}")
        print(f"  AUROC (whitened+recolored → target): {auroc_adapted:.4f}")
        output["auroc_original"] = round(float(auroc_original), 4)
        output["auroc_whitened"] = round(float(auroc_adapted), 4)

        # Re-save with AUROC
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
    else:
        print(f"  Skipping AUROC: label mismatch ({len(y_s)} vs {X_s.shape[0]})")

except ImportError:
    print("  sklearn not available, skipping AUROC")
except Exception as e:
    print(f"  AUROC failed: {e}")

print("\nDone.")
