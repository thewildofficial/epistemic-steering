import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

sys.path.insert(0, ".")
from src.probe import compute_auroc

DATA_DIR = Path("data")
GEN_TIME_DIR = DATA_DIR / "gen_time"
OUTPUT_PATH = DATA_DIR / "gen_time_probe_results.json"
VERIFICATION_PATH = DATA_DIR / "verification_results.json"
HIDDEN_DIM = 2560


def load_gen_time_data(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        return None

    pkl_files = list(data_dir.glob("*.pkl"))
    if not pkl_files:
        return None

    all_file = data_dir / "gen_time_all.pkl"
    if all_file.exists():
        files_to_load = [all_file]
    else:
        batch_files = [f for f in pkl_files if f.name.startswith("gen_time_batch_")]
        if batch_files:
            files_to_load = [sorted(batch_files)[-1]]
        else:
            files_to_load = pkl_files

    results = []
    for pkl_path in files_to_load:
        try:
            with open(pkl_path, "rb") as f:
                batch = pickle.load(f)
            if isinstance(batch, list):
                results.extend(batch)
            else:
                results.append(batch)
        except Exception as exc:
            print(f"  WARNING: could not load {pkl_path}: {exc}")

    if not results:
        return None

    hidden_states = {}
    labels = {}
    datasets = {}
    token_positions = {}

    for result in results:
        if not isinstance(result, dict) or "error" in result:
            continue
        qid = result["question_id"]
        hs_list = [np.array(hs).squeeze() for hs in result["hidden_states"]]
        hidden_states[qid] = hs_list
        labels[qid] = result["correct"]
        datasets[qid] = result["dataset"]
        token_positions[qid] = result["token_positions"]

    return {
        "hidden_states": hidden_states,
        "labels": labels,
        "datasets": datasets,
        "token_positions": token_positions,
    }


def generate_synthetic_data(n_questions=200, hidden_dim=HIDDEN_DIM, n_positions=20):
    rng = np.random.RandomState(42)
    direction = rng.randn(hidden_dim)
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    hidden_states = {}
    labels = {}
    datasets = {}
    token_positions = {}

    for qid in range(n_questions):
        dataset = "mmlu" if qid % 2 == 0 else "gsm8k"
        correct = rng.rand() > 0.5

        states = []
        positions = []
        for pos in range(0, n_positions * 5, 5):
            noise = rng.randn(hidden_dim) * 0.5
            signal = direction * (0.6 if correct else -0.6)
            drift = direction * (pos / 250.0)
            state = noise + signal + drift
            states.append(state)
            positions.append(pos)

        hidden_states[qid] = states
        labels[qid] = correct
        datasets[qid] = dataset
        token_positions[qid] = positions

    return {
        "hidden_states": hidden_states,
        "labels": labels,
        "datasets": datasets,
        "token_positions": token_positions,
    }


def get_prefill_auroc():
    if VERIFICATION_PATH.exists():
        try:
            with open(VERIFICATION_PATH, "r") as f:
                results = json.load(f)
            return {
                "mmlu": results.get("mmlu", {}).get("auroc"),
                "gsm8k": results.get("gsm8k", {}).get("auroc"),
            }
        except Exception:
            pass
    return {"mmlu": 0.827, "gsm8k": 0.654}


def train_probe_at_position(X, y):
    if len(np.unique(y)) < 2:
        return None

    n_min_class = min(np.sum(y), np.sum(~y))
    if n_min_class < 5:
        return None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(y))
    train_aurocs = []

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegressionCV(
            Cs=10,
            cv=3,
            max_iter=1000,
            scoring="roc_auc",
            solver="lbfgs",
        )
        clf.fit(X[train_idx], y[train_idx])
        test_preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        train_pred = clf.predict_proba(X[train_idx])[:, 1]
        train_aurocs.append(compute_auroc(train_pred, y[train_idx]))

    test_auroc = compute_auroc(test_preds, y)
    if np.isnan(test_auroc):
        return None

    train_auroc_mean = float(np.mean(train_aurocs))
    overfitting_gap = train_auroc_mean - test_auroc

    final_clf = LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=1000,
        scoring="roc_auc",
        solver="lbfgs",
    )
    final_clf.fit(X, y)

    return {
        "test_auroc": float(test_auroc),
        "train_auroc_mean": train_auroc_mean,
        "overfitting_gap": float(overfitting_gap),
        "coef": final_clf.coef_[0].tolist(),
        "intercept": float(final_clf.intercept_[0]),
    }


def main(synthetic=False):
    print("=" * 70)
    print("GENERATION-TIME PROBE TRAINING")
    print("=" * 70)

    if synthetic:
        print("\n--synthetic flag set: using synthetic data for pipeline validation\n")
        data = generate_synthetic_data()
    else:
        data = load_gen_time_data(GEN_TIME_DIR)
        if data is None:
            print(
                "\n" + "=" * 70
                + "\nGeneration-time data not found.\n"
                + f"Expected: {GEN_TIME_DIR}/\n\n"
                + "Run scripts/extract_gen_time_data.py on Modal first ($3-5 GPU cost).\n"
                + "Or run with --synthetic to validate the training pipeline.\n"
                + "=" * 70 + "\n"
            )
            return

    hidden_states = data["hidden_states"]
    labels = data["labels"]
    datasets = data["datasets"]
    token_positions = data["token_positions"]

    all_positions = set()
    for positions in token_positions.values():
        all_positions.update(positions)
    all_positions = sorted(all_positions)

    print(f"Questions loaded: {len(hidden_states)}")
    print(f"Token positions: {all_positions[:10]}... ({len(all_positions)} total)\n")

    prefill_auroc = get_prefill_auroc()

    position_results = []
    cv_results = []

    for position in all_positions:
        X_list = []
        y_list = []
        dataset_list = []

        for qid, states in hidden_states.items():
            if position not in token_positions[qid]:
                continue
            idx = token_positions[qid].index(position)
            X_list.append(states[idx])
            y_list.append(labels[qid])
            dataset_list.append(datasets[qid])

        if len(X_list) < 10:
            continue

        X = np.array(X_list)
        y = np.array(y_list, dtype=bool)
        datasets_arr = np.array(dataset_list)

        combined = train_probe_at_position(X, y)
        if combined is None:
            continue

        mmlu_mask = datasets_arr == "mmlu"
        gsm8k_mask = datasets_arr == "gsm8k"

        mmlu_res = (
            train_probe_at_position(X[mmlu_mask], y[mmlu_mask])
            if mmlu_mask.sum() >= 10
            else None
        )
        gsm8k_res = (
            train_probe_at_position(X[gsm8k_mask], y[gsm8k_mask])
            if gsm8k_mask.sum() >= 10
            else None
        )

        pos_result = {
            "token_index": int(position),
            "mmlu_auroc": float(mmlu_res["test_auroc"]) if mmlu_res else None,
            "gsm8k_auroc": float(gsm8k_res["test_auroc"]) if gsm8k_res else None,
            "combined_auroc": float(combined["test_auroc"]),
        }
        position_results.append(pos_result)
        cv_results.append({
            "token_index": int(position),
            "train_auroc_mean": combined["train_auroc_mean"],
            "test_auroc_mean": combined["test_auroc"],
            "overfitting_gap": combined["overfitting_gap"],
        })

        mmlu_str = f"{mmlu_res['test_auroc']:.3f}" if mmlu_res else "N/A"
        gsm8k_str = f"{gsm8k_res['test_auroc']:.3f}" if gsm8k_res else "N/A"
        print(
            f"  pos {position:4d}  n={len(y):4d}  "
            f"combined={combined['test_auroc']:.3f}  "
            f"mmlu={mmlu_str:>6}  "
            f"gsm8k={gsm8k_str:>6}  "
            f"gap={combined['overfitting_gap']:.3f}"
        )

    if not position_results:
        print("No valid positions found with enough data for training.")
        return

    optimal = max(position_results, key=lambda x: x["combined_auroc"])

    valid_mmlu = [p for p in position_results if p["mmlu_auroc"] is not None]
    valid_gsm8k = [p for p in position_results if p["gsm8k_auroc"] is not None]
    best_gen_mmlu = max(valid_mmlu, key=lambda x: x["mmlu_auroc"]) if valid_mmlu else None
    best_gen_gsm8k = max(valid_gsm8k, key=lambda x: x["gsm8k_auroc"]) if valid_gsm8k else None

    avg_cv = {
        "train_auroc_mean": float(np.mean([c["train_auroc_mean"] for c in cv_results])),
        "test_auroc_mean": float(np.mean([c["test_auroc_mean"] for c in cv_results])),
        "overfitting_gap": float(np.mean([c["overfitting_gap"] for c in cv_results])),
    }

    results = {
        "positions": position_results,
        "optimal_position": {
            "token_index": optimal["token_index"],
            "auroc": optimal["combined_auroc"],
        },
        "prefill_comparison": {
            "prefill_mmlu_auroc": prefill_auroc.get("mmlu"),
            "prefill_gsm8k_auroc": prefill_auroc.get("gsm8k"),
            "best_gen_time_mmlu_auroc": best_gen_mmlu["mmlu_auroc"] if best_gen_mmlu else None,
            "best_gen_time_gsm8k_auroc": best_gen_gsm8k["gsm8k_auroc"] if best_gen_gsm8k else None,
        },
        "cross_validation": avg_cv,
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Optimal position: {optimal['token_index']} (AUROC {optimal['combined_auroc']:.3f})")
    if prefill_auroc.get("mmlu") is not None:
        print(f"Prefill MMLU AUROC:   {prefill_auroc['mmlu']:.3f}")
    if best_gen_mmlu:
        print(f"Best gen-time MMLU:   {best_gen_mmlu['mmlu_auroc']:.3f}")
    if prefill_auroc.get("gsm8k") is not None:
        print(f"Prefill GSM8K AUROC:  {prefill_auroc['gsm8k']:.3f}")
    if best_gen_gsm8k:
        print(f"Best gen-time GSM8K:  {best_gen_gsm8k['gsm8k_auroc']:.3f}")
    print(f"\nAvg CV across positions:")
    print(f"  Train AUROC: {avg_cv['train_auroc_mean']:.3f}")
    print(f"  Test AUROC:  {avg_cv['test_auroc_mean']:.3f}")
    print(f"  Overfitting gap: {avg_cv['overfitting_gap']:.3f}")
    print("=" * 70)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    synthetic = "--synthetic" in sys.argv
    main(synthetic=synthetic)
