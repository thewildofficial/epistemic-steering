#!/usr/bin/env python3
"""Wave 3 analysis: T10, T11, T12, T14 on HumanEval data. CPU-only, scikit-learn + numpy."""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lvu_parser import LVUParser

BASE = os.path.join(os.path.dirname(__file__), "..")
META_PATH = os.path.join(BASE, "data", "humaneval_100", "multilayer_humaneval", "humaneval_metadata.jsonl")
SPLIT_PATH = os.path.join(BASE, "data", "humaneval_100", "multilayer_humaneval", "humaneval_split.json")
ACT_DIR = os.path.join(BASE, "data", "humaneval_100", "multilayer_humaneval", "activations")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

metadata = []
with open(META_PATH, "r") as f:
    for line in f:
        metadata.append(json.loads(line.strip()))

with open(SPLIT_PATH, "r") as f:
    split_map = json.load(f)

train_samples = [m for m in metadata if split_map.get(m["id"]) == "train"]
val_samples = [m for m in metadata if split_map.get(m["id"]) == "val"]

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")


def load_activation(sample_id, layer):
    path = os.path.join(ACT_DIR, f"{sample_id}__layer_{layer}.npy")
    return np.load(path).flatten()


def load_mean_pooled(sample_id):
    arrs = [load_activation(sample_id, l) for l in [15, 17, 19, 20]]
    return np.mean(arrs, axis=0)


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= low) & (y_prob <= high) if i == n_bins - 1 else (y_prob >= low) & (y_prob < high)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(y_prob[in_bin]) - np.mean(y_true[in_bin])) * prop_in_bin
    return ece


def compute_selective_accuracy(y_true, y_prob, fraction=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_select = max(1, int(len(y_prob) * fraction))
    idx = np.argsort(y_prob)[-n_select:]
    return accuracy_score(y_true[idx], (y_prob[idx] >= 0.5).astype(int))


print("\n=== T10: Baseline Probe A (Layer 25) ===")
X_train_A = np.stack([load_activation(m["id"], 25) for m in train_samples])
X_val_A = np.stack([load_activation(m["id"], 25) for m in val_samples])
y_train_A = np.array([m["correctness"] for m in train_samples]).astype(int)
y_val_A = np.array([m["correctness"] for m in val_samples]).astype(int)

clf_A = LogisticRegression(class_weight="balanced", max_iter=1000)
clf_A.fit(X_train_A, y_train_A)

proba_val_A = clf_A.predict_proba(X_val_A)[:, 1]
auroc_A = roc_auc_score(y_val_A, proba_val_A)
ece_A = compute_ece(y_val_A, proba_val_A)
sel_acc_A = compute_selective_accuracy(y_val_A, proba_val_A, fraction=0.5)

print(f"AUROC: {auroc_A:.4f}, ECE: {ece_A:.4f}, SelectiveAcc(50%): {sel_acc_A:.4f}")

result_T10 = {
    "auroc": round(auroc_A, 6),
    "ece": round(ece_A, 6),
    "selective_accuracy_50": round(sel_acc_A, 6),
    "n_train": len(train_samples),
    "n_val": len(val_samples),
    "layer": 25,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
with open(os.path.join(RESULTS_DIR, "task-10-baseline.json"), "w") as f:
    json.dump(result_T10, f, indent=2)

print("\n=== T11: Stacey Probe B (Middle layers mean-pooled) ===")
X_train_B = np.stack([load_mean_pooled(m["id"]) for m in train_samples])
X_val_B = np.stack([load_mean_pooled(m["id"]) for m in val_samples])
y_train_B = y_train_A.copy()
y_val_B = y_val_A.copy()

clf_B = LogisticRegression(class_weight="balanced", max_iter=1000)
clf_B.fit(X_train_B, y_train_B)

proba_val_B = clf_B.predict_proba(X_val_B)[:, 1]
auroc_B = roc_auc_score(y_val_B, proba_val_B)
ece_B = compute_ece(y_val_B, proba_val_B)
sel_acc_B = compute_selective_accuracy(y_val_B, proba_val_B, fraction=0.5)

print(f"AUROC: {auroc_B:.4f}, ECE: {ece_B:.4f}, SelectiveAcc(50%): {sel_acc_B:.4f}")

result_T11 = {
    "auroc": round(auroc_B, 6),
    "ece": round(ece_B, 6),
    "selective_accuracy_50": round(sel_acc_B, 6),
    "n_train": len(train_samples),
    "n_val": len(val_samples),
    "layers": [15, 17, 19, 20],
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "comparison_to_baseline_A": {
        "baseline_A_auroc": round(auroc_A, 6),
        "stacey_B_auroc": round(auroc_B, 6),
        "stacey_B_ge_baseline_A": bool(auroc_B >= auroc_A),
    }
}
with open(os.path.join(RESULTS_DIR, "task-11-stacey.json"), "w") as f:
    json.dump(result_T11, f, indent=2)

print("\n=== T12: Best single signal ===")
msp_all = np.array([m["msp"] for m in val_samples])
auroc_msp = roc_auc_score(y_val_A, msp_all)
print(f"MSP AUROC: {auroc_msp:.4f}")

entropy_all = np.array([m["entropy"] for m in val_samples])
auroc_entropy = roc_auc_score(y_val_A, -entropy_all)
print(f"Entropy (inverted) AUROC: {auroc_entropy:.4f}")

signals = {
    "Baseline_A": auroc_A,
    "Stacey_B": auroc_B,
    "MSP": auroc_msp,
    "Entropy": auroc_entropy,
}
best_signal = max(signals, key=signals.get)
print(f"Best single signal: {best_signal} (AUROC={signals[best_signal]:.4f})")

result_T12 = {
    "baseline_A_auroc": round(auroc_A, 6),
    "stacey_B_auroc": round(auroc_B, 6),
    "msp_auroc": round(auroc_msp, 6),
    "entropy_auroc": round(auroc_entropy, 6),
    "best_single_signal": best_signal,
    "best_single_signal_auroc": round(signals[best_signal], 6),
    "n_train": len(train_samples),
    "n_val": len(val_samples),
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
with open(os.path.join(RESULTS_DIR, "task-12-best-signal.json"), "w") as f:
    json.dump(result_T12, f, indent=2)

print("\n=== T14: LVU scoring ===")
parser = LVUParser()
lvu_scores = []
for m in metadata:
    lvu_scores.append(parser.score(m["generated_text"])["overall_score"])

lvu_scores_arr = np.array(lvu_scores)
lvu_variance = float(np.var(lvu_scores_arr))
print(f"LVU variance: {lvu_variance:.6f}")

result_T14 = {
    "lvu_variance": round(lvu_variance, 8),
    "n_samples": len(metadata),
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

if lvu_variance > 0:
    lvu_val = np.array([lvu_scores[i] for i, m in enumerate(metadata) if split_map.get(m["id"]) == "val"])
    auroc_lvu = roc_auc_score(y_val_A, -lvu_val)
    result_T14["lvu_auroc"] = round(auroc_lvu, 6)
    result_T14["h3_confirmed"] = False
    result_T14["h3_note"] = "Variance > 0, LVU has signal"
    print(f"LVU AUROC (inverted): {auroc_lvu:.4f}")
else:
    result_T14["lvu_auroc"] = 0.0
    result_T14["h3_confirmed"] = True
    result_T14["h3_note"] = "H3 CONFIRMED: variance ≈ 0 for code, LVU = 0 for all ensemble calculations"
    print("H3 CONFIRMED: variance ≈ 0")

with open(os.path.join(RESULTS_DIR, "task-14-lvu.json"), "w") as f:
    json.dump(result_T14, f, indent=2)

print("\n=== All tasks complete. Results saved to results/ ===")
