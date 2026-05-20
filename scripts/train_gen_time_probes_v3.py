"""Train per-position probes — simplified: load only all_results.pkl, handle ndarray."""
from __future__ import annotations
import json, pickle, os
from pathlib import Path
import modal
from modal import App, Image, Volume

app = App("probe-v3")
volume = Volume.from_name("epistemic-model-cache")
RESULTS_DIR = "/vol/results"
GEN_TIME_DIR = f"{RESULTS_DIR}/gen_time_gsm8k_layer25_qwen_prompt"
OUTPUT_DIR = f"{RESULTS_DIR}/gen_time_probe_results"

image = Image.debian_slim().pip_install("numpy", "scikit-learn", "scipy", "tqdm")


@app.function(image=image, volumes={"/vol": volume}, cpu=8.0, timeout=7200)
def train():
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from tqdm import tqdm

    def compute_ece(scores, labels, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            acc = labels[mask].mean()
            conf = scores[mask].mean()
            ece += (mask.sum() / len(scores)) * abs(acc - conf)
        return ece

    all_pkl_path = f"{GEN_TIME_DIR}/all_results.pkl"
    with open(all_pkl_path, 'rb') as f:
        all_results = pickle.load(f)
    if isinstance(all_results, list):
        all_results = {r.get('question_id', str(i)): r for i, r in enumerate(all_results)}
    print(f"Loaded {len(all_results)} entries")

    pos_data = {}
    n_with_hs = 0

    for qid, r in all_results.items():
        if not isinstance(r, dict):
            continue
        hs = r.get('hidden_states')
        if hs is None:
            continue

        correct = r.get('correct', False)

        if isinstance(hs, np.ndarray) and hs.ndim == 2:
            for pos in range(hs.shape[0]):
                pos_data.setdefault(pos, []).append((hs[pos, :].copy(), correct))
            n_with_hs += 1
        elif isinstance(hs, list) and len(hs) > 0:
            for pos, h in enumerate(hs):
                h_flat = np.asarray(h).squeeze()
                pos_data.setdefault(pos, []).append((h_flat, correct))
            n_with_hs += 1

    print(f"Questions with HS: {n_with_hs}")
    print(f"Token positions: {len(pos_data)}")

    position_results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for pos in tqdm(sorted(pos_data.keys()), desc="Training probes"):
        items = pos_data[pos]
        if len(items) < 20:
            continue
        X = np.array([item[0] for item in items])
        y = np.array([item[1] for item in items], dtype=bool)
        if len(np.unique(y)) < 2:
            continue

        test_preds = np.zeros(len(y))
        train_aurocs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegressionCV(Cs=10, cv=3, max_iter=1000, scoring="roc_auc", solver="lbfgs")
            clf.fit(X[train_idx], y[train_idx])
            test_preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            train_pred = clf.predict_proba(X[train_idx])[:, 1]
            try:
                train_aurocs.append(float(roc_auc_score(y[train_idx], train_pred)))
            except ValueError:
                train_aurocs.append(0.5)

        try:
            test_auroc = float(roc_auc_score(y, test_preds))
        except ValueError:
            continue
        if np.isnan(test_auroc):
            continue

        train_auroc_mean = float(np.nanmean(train_aurocs))
        gap = train_auroc_mean - test_auroc
        brier = float(brier_score_loss(y, test_preds))
        ece = float(compute_ece(test_preds, y))

        position_results.append({
            "token_index": int(pos),
            "test_auroc": round(test_auroc, 4),
            "train_auroc": round(train_auroc_mean, 4),
            "overfitting_gap": round(gap, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "n_samples": len(y),
            "n_correct": int(np.sum(y)),
        })

    aurocs = [p["test_auroc"] for p in position_results]
    indices = [p["token_index"] for p in position_results]

    if aurocs:
        best_pos = max(position_results, key=lambda x: x["test_auroc"])
        if max(aurocs) - min(aurocs) < 0.05:
            scenario = "C (flat)"
        else:
            first_q = np.mean([p["test_auroc"] for p in position_results if p["token_index"] <= max(indices) * 0.25])
            last_q = np.mean([p["test_auroc"] for p in position_results if p["token_index"] >= max(indices) * 0.75])
            if last_q > first_q + 0.05:
                scenario = "A (monotonic)"
            elif best_pos["token_index"] > len(indices) // 2:
                scenario = "B (late-jump)"
            else:
                scenario = "B (early-jump)"
    else:
        scenario = "C (flat)"
        best_pos = {}

    output = {
        "position_results": position_results,
        "optimal_position": best_pos,
        "scenario": scenario,
        "n_questions_with_hs": n_with_hs,
        "n_positions": len(pos_data),
        "avg_metrics": {
            "test_auroc_mean": float(np.mean(aurocs)) if aurocs else 0.0,
            "max_test_auroc": max(aurocs) if aurocs else 0.0,
            "overfitting_gap_mean": float(np.mean([p["overfitting_gap"] for p in position_results])) if position_results else 0.0,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/gen_time_sweep_full.json", "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nFULL RESULTS ({n_with_hs} questions):")
    print(f"  Optimal pos: {best_pos.get('token_index','N/A')}, AUROC={best_pos.get('test_auroc',0):.4f}")
    print(f"  Brier: {best_pos.get('brier_score',0):.4f}, ECE: {best_pos.get('ece',0):.4f}")
    print(f"  Scenario: {scenario}")
    if best_pos.get('ece', 1.0) < 0.05:
        print(f"  CALIBRATION: WELL-CALIBRATED (ECE < 0.05)")
    elif best_pos.get('ece', 1.0) < 0.10:
        print(f"  CALIBRATION: ACCEPTABLE (ECE < 0.10)")
    else:
        print(f"  CALIBRATION: NEEDS FIXING (ECE >= 0.10)")
    return output


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"Done. Qs={result['n_questions_with_hs']}, AUROC={result['optimal_position'].get('test_auroc','?')}")
