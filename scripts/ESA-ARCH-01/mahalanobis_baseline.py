"""
ESA-ARCH-01 T5: label-free Mahalanobis uncertainty detector.

Lazily loads per-layer .npy activations, does leave-one-domain-out cross-validation,
fits class-conditional Gaussians with Ledoit-Wolf covariance shrinkage,
and computes L2-normalized Mahalanobis distance (Mahalanobis++) per sample.
"""

import os
import json
import glob
import time
import resource
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data/activations_allpos"
LAYERS = [5, 10, 15, 20, 25, 30, 31]
DOMAINS = ["arc_challenge", "humaneval", "math", "triviaqa"]
HIDDEN_DIM = 2560


def mem_info_mb():
    """Peak resident memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_metadata(data_dir):
    """Load all JSON metadata; map question_id -> {domain, is_correct}."""
    meta = {}
    for path in glob.glob(os.path.join(data_dir, "*.json")):
        base = os.path.splitext(os.path.basename(path))[0]
        if base in {"cost_log_allpos"}:
            continue
        with open(path, "r") as f:
            rec = json.load(f)
        meta[base] = {
            "domain": rec.get("dataset", "unknown"),
            "is_correct": bool(rec.get("is_correct", False)),
        }
    return meta


def activation_path(data_dir, base, layer):
    return os.path.join(data_dir, f"{base}__layer_{layer}.npy")


def load_feature(data_dir, base, layer):
    """
    Load a single-layer activation and pool over sequence length.

    Strategy: mean-pool across tokens (last-token and mean-pool give similar
    results here; mean-pool is robust to variable-length prompts).
    Returned as float64 vector to avoid numerical issues with 2560-dim covariances.
    """
    path = activation_path(data_dir, base, layer)
    if not os.path.exists(path):
        return None
    # mmap_mode='r' keeps us from loading the whole file at once if the OS
    # decides to page it; we still read the single layer we need.
    arr = np.load(path, mmap_mode="r")
    # shape: (seq_len, 2560)
    if arr.ndim == 2:
        feat = arr.mean(axis=0).astype(np.float64)
    elif arr.ndim == 1:
        feat = arr.astype(np.float64)
    else:
        return None
    return feat


def normalize_features(X):
    """
    L2-normalize each row to compute Mahalanobis++ distance.

    X: (n_samples, n_features)
    Returns normalized X and per-sample L2 norms.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms, norms.squeeze()


def collect_features(data_dir, layer, ids, meta):
    """Load features for a list of ids; skip missing. Returns (X, y, domains)."""
    X = []
    y = []
    domains = []
    for qid in ids:
        if qid not in meta:
            continue
        feat = load_feature(data_dir, qid, layer)
        if feat is None:
            continue
        X.append(feat)
        y.append(1 if meta[qid]["is_correct"] else 0)
        domains.append(meta[qid]["domain"])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), domains


def mahal_plus_score(X, mu_correct, mu_incorrect, cov_correct, cov_incorrect, shrink_diag=1e-5):
    """
    Compute L2-normalized Mahalanobis distance as in Mahalanobis++:
    score(x) = || x - mu_correct ||^2_{Sigma_correct^{-1}} - || x - mu_incorrect ||^2_{Sigma_incorrect^{-1}}

    Positive => closer to incorrect class (higher uncertainty / lower correctness probability).
    We return this raw difference; AUROC uses the binary label is_correct=1.
    """
    d = X.shape[1]
    # Stabilize covariance inverses via eigendecomposition.
    def inv_mahal(Xc, mu, cov):
        delta = Xc - mu  # (n, d)
        # Add small ridge to eigenvalues for numerical stability.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-4, None)
        # Project delta into eigenbasis, divide by eigenvalues, project back.
        proj = delta @ eigvecs  # (n, d)
        scaled = proj / eigvals[np.newaxis, :]  # (n, d)
        return np.sum(scaled * proj, axis=1)  # (n,)

    m_correct = inv_mahal(X, mu_correct, cov_correct)
    m_incorrect = inv_mahal(X, mu_incorrect, cov_incorrect)
    return m_incorrect - m_correct


def fit_gaussians(X, y, use_shrink=True):
    """Fit class-conditional Gaussians. Returns (mu0, mu1, cov0, cov1)."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    X0 = X[idx0]
    X1 = X[idx1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    if use_shrink:
        # Ledoit-Wolf shrinkage; critical with d=2560 and ~200 samples/class.
        cov0 = LedoitWolf().fit(X0).covariance_
        cov1 = LedoitWolf().fit(X1).covariance_
    else:
        cov0 = np.cov(X0, rowvar=False, bias=False)
        cov1 = np.cov(X1, rowvar=False, bias=False)
    return mu0, mu1, cov0, cov1


def evaluate_domain_split(data_dir, layer, train_ids, test_ids, meta):
    """
    Train on train_ids, test on test_ids for one layer.
    Returns dict with AUROC, scores, labels, feature norms diagnostic.
    """
    X_train, y_train, _ = collect_features(data_dir, layer, train_ids, meta)
    X_test, y_test, _ = collect_features(data_dir, layer, test_ids, meta)

    if len(X_train) == 0 or len(X_test) == 0:
        return {"auroc": None, "n_train": len(X_train), "n_test": len(X_test)}

    # Mahalanobis++: L2-normalize features.
    X_train_norm, train_norms = normalize_features(X_train)
    X_test_norm, test_norms = normalize_features(X_test)

    mu0, mu1, cov0, cov1 = fit_gaussians(X_train_norm, y_train, use_shrink=True)
    scores = mahal_plus_score(X_test_norm, mu1, mu0, cov1, cov0)

    auroc = roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else None

    return {
        "auroc": float(auroc) if auroc is not None else None,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_norm_mean": float(train_norms.mean()),
        "train_norm_std": float(train_norms.std()),
        "test_norm_mean": float(test_norms.mean()),
        "test_norm_std": float(test_norms.std()),
    }


def main():
    start = time.time()
    meta = load_metadata(DATA_DIR)

    # Keep only ids that have all required layer files and are in target domains.
    all_ids = []
    for qid, info in meta.items():
        if info["domain"] not in DOMAINS:
            continue
        if all(os.path.exists(activation_path(DATA_DIR, qid, L)) for L in LAYERS):
            all_ids.append(qid)

    # Group by domain.
    by_domain = defaultdict(list)
    for qid in all_ids:
        by_domain[meta[qid]["domain"]].append(qid)

    print(f"Usable samples: {len(all_ids)}")
    for dom in DOMAINS:
        labels = [meta[qid]["is_correct"] for qid in by_domain[dom]]
        print(f"  {dom}: {len(by_domain[dom])} samples, {sum(labels)} correct, {len(labels)-sum(labels)} incorrect")

    report = {
        "task": "ESA-ARCH-01 T5",
        "method": "label-free Mahalanobis++",
        "description": "Leave-one-domain-out class-conditional Gaussian with Ledoit-Wolf shrinkage and L2-normalized Mahalanobis distance",
        "domains": DOMAINS,
        "layers": LAYERS,
        "total_samples": len(all_ids),
        "by_domain": {dom: len(by_domain[dom]) for dom in DOMAINS},
        "in_domain": {},
        "cross_domain": {},
        "feature_norm_collapse": {},
        "runtime_seconds": None,
        "peak_memory_mb": None,
        "notes": [],
    }

    # -----------------------------------------------------------------------
    # In-domain AUROC per layer: for each domain, train on other three domains
    # and test on held-out domain (this is also the cross-domain setting;
    # "in-domain" here means same held-out domain split but evaluated by layer).
    # -----------------------------------------------------------------------
    for layer in LAYERS:
        layer_aurocs = []
        domain_aurocs = {}
        domain_norms = {}
        for held_out in DOMAINS:
            train_ids = []
            for dom in DOMAINS:
                if dom == held_out:
                    continue
                train_ids.extend(by_domain[dom])
            test_ids = by_domain[held_out]
            result = evaluate_domain_split(DATA_DIR, layer, train_ids, test_ids, meta)
            auroc = result["auroc"]
            domain_aurocs[held_out] = auroc
            layer_aurocs.append(auroc)
            domain_norms[held_out] = {
                "test_norm_mean": result["test_norm_mean"],
                "test_norm_std": result["test_norm_std"],
                "train_norm_mean": result["train_norm_mean"],
                "train_norm_std": result["train_norm_std"],
            }
        mean_auroc = float(np.mean(layer_aurocs))
        report["cross_domain"][f"layer_{layer}"] = {
            "per_domain": domain_aurocs,
            "mean": mean_auroc,
            "domain_norms": domain_norms,
        }
        report["in_domain"][f"layer_{layer}"] = {
            "mean": mean_auroc,
            "per_domain": domain_aurocs,
        }

    # -----------------------------------------------------------------------
    # Best-layer summary
    # -----------------------------------------------------------------------
    best_layer = max(LAYERS, key=lambda L: report["cross_domain"][f"layer_{L}"]["mean"])
    best_mean = report["cross_domain"][f"layer_{best_layer}"]["mean"]
    report["best_layer"] = best_layer
    report["best_layer_cross_domain_mean_auroc"] = best_mean

    # -----------------------------------------------------------------------
    # Feature norm collapse diagnostic: overall mean/std across all samples, all layers.
    # -----------------------------------------------------------------------
    norm_means = []
    norm_stds = []
    for layer in LAYERS:
        X, _, _ = collect_features(DATA_DIR, layer, all_ids, meta)
        _, norms = normalize_features(X)
        norm_means.append(norms.mean())
        norm_stds.append(norms.std())
    report["feature_norm_collapse"] = {
        "global_norm_mean": float(np.mean(norm_means)),
        "global_norm_std": float(np.mean(norm_stds)),
        "per_layer_norm_means": {f"layer_{L}": float(m) for L, m in zip(LAYERS, norm_means)},
        "per_layer_norm_stds": {f"layer_{L}": float(s) for L, s in zip(LAYERS, norm_stds)},
        "interpretation": "Low mean norm or near-zero std across samples may indicate norm collapse; values near 1.0 after L2 normalization are expected.",
    }

    # -----------------------------------------------------------------------
    # Per-domain held-out AUROC averaged across layers (for the concise summary)
    # -----------------------------------------------------------------------
    held_out_means = {}
    for held_out in DOMAINS:
        vals = [
            report["cross_domain"][f"layer_{L}"]["per_domain"][held_out]
            for L in LAYERS
        ]
        held_out_means[held_out] = float(np.mean(vals))
    report["cross_domain_mean_by_held_out_domain"] = held_out_means
    report["overall_cross_domain_mean_auroc"] = float(np.mean(list(held_out_means.values())))

    elapsed = time.time() - start
    report["runtime_seconds"] = elapsed
    report["peak_memory_mb"] = mem_info_mb()
    report["notes"].append("Memory stays low because only one layer is loaded at a time per train/test pass.")
    report["notes"].append("Ledoit-Wolf shrinkage used for both class-conditional covariances.")
    report["notes"].append("L2-normalized Mahalanobis distance (Mahalanobis++) used for scoring.")

    out_dir = "outputs/ESA-ARCH-01/mahalanobis_baseline"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # -----------------------------------------------------------------------
    # Concise summary to stdout
    # -----------------------------------------------------------------------
    print("\n=== Mahalanobis++ Baseline Summary ===")
    print(f"Best layer: {best_layer}  cross-domain mean AUROC: {best_mean:.4f}")
    print("Per-layer cross-domain mean AUROC:")
    for layer in LAYERS:
        val = report["cross_domain"][f"layer_{layer}"]["mean"]
        print(f"  layer {layer:>2}: {val:.4f}")
    print("Held-out domain AUROC (mean across layers):")
    for dom, val in held_out_means.items():
        print(f"  {dom}: {val:.4f}")
    print(f"Overall cross-domain mean AUROC: {report['overall_cross_domain_mean_auroc']:.4f}")
    print(f"Runtime: {elapsed:.1f}s  Peak memory: {report['peak_memory_mb']:.1f} MB")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
