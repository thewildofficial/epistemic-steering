"""Train per-position LR probes on gen-time hidden states.
Runs on Modal CPU (no GPU needed). Saves only results (few MB) to volume.
"""
from __future__ import annotations
import json, pickle, os
from pathlib import Path
import modal
from modal import App, Image, Volume

app = App("gen-time-probe-training")
volume = Volume.from_name("epistemic-model-cache")
RESULTS_DIR = "/vol/results"

image = (
    Image.debian_slim()
    .pip_install("numpy", "scikit-learn", "scipy", "tqdm")
)

GEN_TIME_DIR = f"{RESULTS_DIR}/gen_time_gsm8k_layer25_qwen_prompt"
OUTPUT_DIR = f"{RESULTS_DIR}/gen_time_probe_results"


@app.function(image=image, volumes={"/vol": volume}, cpu=4.0, timeout=3600)
def train_probes():
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    all_results = {}
    all_pkl_path = f"{GEN_TIME_DIR}/all_results.pkl"

    if os.path.exists(all_pkl_path):
        print(f"Loading all_results.pkl...")
        with open(all_pkl_path, 'rb') as f:
            all_results = pickle.load(f)
        if isinstance(all_results, list):
            all_results = {r.get('question_id', str(i)): r for i, r in enumerate(all_results)}
        print(f"Loaded {len(all_results)} entries")

    pos_data: dict[int, list] = {}
    n_ok = 0
    n_no_hs = 0
    for qid, r in all_results.items():
        if not isinstance(r, dict):
            continue
        if 'hidden_states' not in r:
            n_no_hs += 1
            continue
        hs = r['hidden_states']
        if not isinstance(hs, list) or len(hs) == 0:
            n_no_hs += 1
            continue
        correct = r.get('correct', False)
        for pos, h in enumerate(hs):
            h_flat = h.squeeze()
            pos_data.setdefault(pos, []).append((h_flat, correct))
        n_ok += 1

    print(f"Questions with hidden states: {n_ok}")
    print(f"Questions without hidden states: {n_no_hs}")
    print(f"Total token positions: {len(pos_data)}")

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
            clf = LogisticRegressionCV(
                Cs=10, cv=3, max_iter=1000, scoring="roc_auc", solver="lbfgs"
            )
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

        position_results.append({
            "token_index": int(pos),
            "test_auroc": round(test_auroc, 4),
            "train_auroc": round(train_auroc_mean, 4),
            "overfitting_gap": round(gap, 4),
            "n_samples": len(y),
            "n_correct": int(np.sum(y)),
        })

    aurocs = [p["test_auroc"] for p in position_results]
    indices = [p["token_index"] for p in position_results]

    if aurocs:
        first_quarter_aurocs = [p["test_auroc"] for p in position_results if p["token_index"] <= max(indices) * 0.25]
        last_quarter_aurocs = [p["test_auroc"] for p in position_results if p["token_index"] >= max(indices) * 0.75]
        first_quarter = np.mean(first_quarter_aurocs) if first_quarter_aurocs else 0.0
        last_quarter = np.mean(last_quarter_aurocs) if last_quarter_aurocs else 0.0

        best_pos = max(position_results, key=lambda x: x["test_auroc"])
        max_idx = best_pos["token_index"]
        mid = len(indices) // 2

        if max(aurocs) - min(aurocs) < 0.05:
            scenario = "C (flat)"
        elif last_quarter > first_quarter + 0.05:
            scenario = "A (monotonic increase)"
        elif max_idx > mid:
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
        "n_questions": len(all_results),
        "n_positions": len(pos_data),
        "n_questions_with_hs": n_ok,
        "avg_metrics": {
            "test_auroc_mean": float(np.mean(aurocs)) if aurocs else 0.0,
            "train_auroc_mean": float(np.mean([p["train_auroc"] for p in position_results])) if position_results else 0.0,
            "overfitting_gap_mean": float(np.mean([p["overfitting_gap"] for p in position_results])) if position_results else 0.0,
            "max_test_auroc": max(aurocs) if aurocs else 0.0,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/gen_time_sweep.json", "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nOptimal position: {best_pos.get('token_index', 'N/A')} (AUROC={best_pos.get('test_auroc', 0):.4f})")
    print(f"Scenario: {scenario}")
    print(f"Avg test AUROC: {output['avg_metrics']['test_auroc_mean']:.4f}")
    print(f"Max test AUROC: {output['avg_metrics']['max_test_auroc']:.4f}")
    print(f"Avg overfitting gap: {output['avg_metrics']['overfitting_gap_mean']:.4f}")
    print(f"\nResults saved to {OUTPUT_DIR}/gen_time_sweep.json")
    return output


@app.local_entrypoint()
def main():
    result = train_probes.remote()
    print(f"\nDone. Scenario: {result.get('scenario')}")
    opt = result.get('optimal_position', {})
    print(f"Optimal position: {opt.get('token_index')}, AUROC={opt.get('test_auroc')}")
