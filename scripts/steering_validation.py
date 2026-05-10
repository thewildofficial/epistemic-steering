import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LAYER = 25
HIGH_THRESHOLD = 0.7
LOW_THRESHOLD = 0.3
TOKENS_DIRECT = 8
TOKENS_COT = 120
HIDDEN_DIM = 2560


def load_heldout_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def apply_platt_scaling(scores: np.ndarray, labels: np.ndarray):
    scores_2d = scores.reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(scores_2d, labels)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return expit(a * scores + b), {"slope": a, "intercept": b}


def route_question(score: float, high: float, low: float) -> str:
    if score >= high:
        return "direct"
    elif score <= low:
        return "abstain"
    return "cot"


def compute_metrics_for_results(results, calibrated_scores, high, low):
    n = len(results)
    routes = [route_question(s, high, low) for s in calibrated_scores]
    labels = np.array([r["correct"] for r in results], dtype=bool)

    steering_correct = np.array([
        (routes[i] == "direct" and results[i]["correct"]) or
        (routes[i] == "cot" and results[i]["correct"]) or
        (routes[i] == "abstain" and not results[i]["correct"])
        for i in range(n)
    ])
    always_direct_correct = labels.copy()
    always_cot_correct = labels.copy()

    np.random.seed(42)
    random_routes = np.random.choice(["direct", "cot", "abstain"], size=n)
    random_correct = np.array([
        (random_routes[i] == "direct" and labels[i]) or
        (random_routes[i] == "cot" and labels[i]) or
        (random_routes[i] == "abstain" and not labels[i])
        for i in range(n)
    ])

    steering_tokens = np.array([r.get("tokens_used", 0) for r in results])
    always_direct_tokens = np.full(n, TOKENS_DIRECT)

    actual_cot_tokens = [r["tokens_used"] for r in results if r["route"] == "cot"]
    avg_cot_tokens = np.mean(actual_cot_tokens) if actual_cot_tokens else TOKENS_COT
    always_cot_tokens = np.full(n, avg_cot_tokens)

    random_tokens = np.array([
        TOKENS_DIRECT if random_routes[i] == "direct" else
        (avg_cot_tokens if random_routes[i] == "cot" else 0)
        for i in range(n)
    ])

    methods = {
        "Steering": {"correct": steering_correct, "tokens": steering_tokens},
        "Always-Direct": {"correct": always_direct_correct, "tokens": always_direct_tokens},
        "Always-CoT": {"correct": always_cot_correct, "tokens": always_cot_tokens},
        "Random": {"correct": random_correct, "tokens": random_tokens},
    }

    metrics = {}
    for name, data in methods.items():
        acc = float(np.mean(data["correct"]))
        avg_tokens = float(np.mean(data["tokens"]))
        always_cot_total = n * avg_cot_tokens
        savings = float((always_cot_total - np.sum(data["tokens"])) / always_cot_total) if always_cot_total > 0 else 0.0
        metrics[name] = {"accuracy": acc, "tokens_per_question": avg_tokens, "savings_vs_cot": savings}

    return metrics, avg_cot_tokens


def bootstrap_ci_metric(arr: np.ndarray, metric_fn, n_bootstrap=1000, ci=95):
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    rng = np.random.default_rng(42)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric_fn(arr[idx])

    alpha = (100 - ci) / 2
    lower = float(np.percentile(estimates, alpha))
    upper = float(np.percentile(estimates, 100 - alpha))
    mean = float(np.mean(estimates))

    return {"mean": mean, "lower": lower, "upper": upper}


def compute_all_bootstrap_cis(results, calibrated_scores, high, low, n_bootstrap=1000):
    n = len(results)
    routes = [route_question(s, high, low) for s in calibrated_scores]
    labels = np.array([r["correct"] for r in results], dtype=bool)

    steering_correct = np.array([
        (routes[i] == "direct" and results[i]["correct"]) or
        (routes[i] == "cot" and results[i]["correct"]) or
        (routes[i] == "abstain" and not results[i]["correct"])
        for i in range(n)
    ])
    always_direct_correct = labels.copy()
    always_cot_correct = labels.copy()

    np.random.seed(42)
    random_routes = np.random.choice(["direct", "cot", "abstain"], size=n)
    random_correct = np.array([
        (random_routes[i] == "direct" and labels[i]) or
        (random_routes[i] == "cot" and labels[i]) or
        (random_routes[i] == "abstain" and not labels[i])
        for i in range(n)
    ])

    steering_tokens = np.array([r.get("tokens_used", 0) for r in results])
    always_direct_tokens = np.full(n, TOKENS_DIRECT)

    actual_cot_tokens = [r["tokens_used"] for r in results if r["route"] == "cot"]
    avg_cot_tokens = np.mean(actual_cot_tokens) if actual_cot_tokens else TOKENS_COT
    always_cot_tokens = np.full(n, avg_cot_tokens)

    random_tokens = np.array([
        TOKENS_DIRECT if random_routes[i] == "direct" else
        (avg_cot_tokens if random_routes[i] == "cot" else 0)
        for i in range(n)
    ])

    method_data = {
        "Steering": {"correct": steering_correct, "tokens": steering_tokens},
        "Always-Direct": {"correct": always_direct_correct, "tokens": always_direct_tokens},
        "Always-CoT": {"correct": always_cot_correct, "tokens": always_cot_tokens},
        "Random": {"correct": random_correct, "tokens": random_tokens},
    }

    result = {}
    for name, data in method_data.items():
        acc_ci = bootstrap_ci_metric(data["correct"], np.mean, n_bootstrap)
        tok_ci = bootstrap_ci_metric(data["tokens"], np.mean, n_bootstrap)

        acc = float(np.mean(data["correct"]))
        avg_tokens = float(np.mean(data["tokens"]))
        always_cot_total = n * avg_cot_tokens
        savings = float((always_cot_total - np.sum(data["tokens"])) / always_cot_total) if always_cot_total > 0 else 0.0

        if name == "Steering":
            correct_idx = np.where(labels)[0]
            incorrect_idx = np.where(~labels)[0]
            caught_incorrect = len([i for i in incorrect_idx if routes[i] == "abstain"])
            prevention_rate = float(caught_incorrect / len(incorrect_idx)) if len(incorrect_idx) > 0 else 0.0
            blocked_correct = len([i for i in correct_idx if routes[i] == "abstain"])
            unnecessary_block_rate = float(blocked_correct / len(correct_idx)) if len(correct_idx) > 0 else 0.0
            try:
                auroc = float(roc_auc_score(labels, calibrated_scores))
            except ValueError:
                auroc = float("nan")

            direct_n = sum(1 for r in routes if r == "direct")
            cot_n = sum(1 for r in routes if r == "cot")
            abstain_n = sum(1 for r in routes if r == "abstain")

            prev_arr = np.array([1 if routes[i] == "abstain" and not labels[i] else 0 for i in range(n)])
            prev_ci = bootstrap_ci_metric(prev_arr, np.mean, n_bootstrap)
            block_arr = np.array([1 if routes[i] == "abstain" and labels[i] else 0 for i in range(n)])
            block_ci = bootstrap_ci_metric(block_arr, np.mean, n_bootstrap)

            result[name] = {
                "accuracy": round(acc, 4),
                "accuracy_ci": [round(acc_ci["lower"], 4), round(acc_ci["upper"], 4)],
                "tokens_per_question": round(avg_tokens, 1),
                "tokens_ci": [round(tok_ci["lower"], 1), round(tok_ci["upper"], 1)],
                "savings_vs_cot": round(savings, 4),
                "prevention_rate": round(prevention_rate, 4),
                "prevention_ci": [round(prev_ci["lower"], 4), round(prev_ci["upper"], 4)],
                "unnecessary_block_rate": round(unnecessary_block_rate, 4),
                "block_ci": [round(block_ci["lower"], 4), round(block_ci["upper"], 4)],
                "auroc": round(auroc, 4),
                "routing": {
                    "direct": direct_n, "cot": cot_n, "abstain": abstain_n,
                    "direct_pct": direct_n / n, "cot_pct": cot_n / n, "abstain_pct": abstain_n / n,
                },
            }
        else:
            result[name] = {
                "accuracy": round(acc, 4),
                "accuracy_ci": [round(acc_ci["lower"], 4), round(acc_ci["upper"], 4)],
                "tokens_per_question": round(avg_tokens, 1),
                "tokens_ci": [round(tok_ci["lower"], 1), round(tok_ci["upper"], 1)],
                "savings_vs_cot": round(savings, 4),
            }
    return result, avg_cot_tokens


def print_comparison_table(metrics_with_cis):
    print("\n" + "=" * 75)
    print("STEERING VALIDATION RESULTS (Held-out Data, Layer 25, Platt-scaled)")
    print("=" * 75)

    header = f"{'Method':<20} | {'Accuracy':^18} | {'Tokens/Q':^8} | {'Savings vs CoT':^12}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name in ["Steering", "Always-Direct", "Always-CoT", "Random"]:
        m = metrics_with_cis[name]
        acc = m["accuracy"]
        acc_lo, acc_hi = m["accuracy_ci"]
        acc_str = f"{acc:.4f} ± {(acc_hi - acc_lo) / 2:.4f}"
        tokens = m["tokens_per_question"]
        savings_str = f"{m['savings_vs_cot']:.1%}"

        print(f"{name:<20} | {acc_str:>18} | {tokens:>8.1f} | {savings_str:>12}")

    print(sep)

    steer = metrics_with_cis["Steering"]
    print(f"\nSteering-specific metrics:")
    print(f"  AUROC: {steer['auroc']:.4f}")
    print(f"  Prevention rate: {steer['prevention_rate']:.4f} [{steer['prevention_ci'][0]:.4f}, {steer['prevention_ci'][1]:.4f}]")
    print(f"  Unnecessary block rate: {steer['unnecessary_block_rate']:.4f} [{steer['block_ci'][0]:.4f}, {steer['block_ci'][1]:.4f}]")
    print(f"  Routing: direct={steer['routing']['direct']} ({steer['routing']['direct_pct']:.1%}), "
          f"CoT={steer['routing']['cot']} ({steer['routing']['cot_pct']:.1%}), "
          f"abstain={steer['routing']['abstain']} ({steer['routing']['abstain_pct']:.1%})")

    steering_acc = metrics_with_cis["Steering"]["accuracy"]
    direct_acc = metrics_with_cis["Always-Direct"]["accuracy"]
    if steering_acc >= direct_acc:
        print(f"\n Verification PASSED: Steering ({steering_acc:.4f}) >= Always-Direct ({direct_acc:.4f})")
    else:
        print(f"\n Verification FAILED: Steering ({steering_acc:.4f}) < Always-Direct ({direct_acc:.4f})")


def main():
    project_root = Path(__file__).resolve().parent.parent
    heldout_path = project_root / "data" / "heldout_eval" / "heldout_results.jsonl"
    output_path = project_root / "data" / "ablation_results" / "steering_validation.json"

    print("=" * 60)
    print("STEERING VALIDATION")
    print("=" * 60)

    print(f"\n Loading held-out data from {heldout_path}")
    results = load_heldout_results(str(heldout_path))
    n = len(results)
    print(f"  Loaded {n} held-out questions")

    raw_scores = np.array([r["probe_score"] for r in results])
    labels = np.array([r["correct"] for r in results])

    print("\n Applying Platt scaling to probe scores")
    calibrated_scores, cal_params = apply_platt_scaling(raw_scores, labels)
    print(f"  Platt params: slope={cal_params['slope']:.4f}, intercept={cal_params['intercept']:.4f}")

    try:
        raw_auroc = roc_auc_score(labels, raw_scores)
        cal_auroc = roc_auc_score(labels, calibrated_scores)
        print(f"  Raw AUROC: {raw_auroc:.4f}, Calibrated AUROC: {cal_auroc:.4f}")
    except ValueError:
        raw_auroc = cal_auroc = float("nan")

    print(f"\n Steering thresholds: HIGH={HIGH_THRESHOLD}, LOW={LOW_THRESHOLD}")

    print("\n Computing bootstrap CIs (n=1000)")
    metrics_with_cis, avg_cot_tokens = compute_all_bootstrap_cis(
        results, calibrated_scores, HIGH_THRESHOLD, LOW_THRESHOLD, n_bootstrap=1000
    )

    print_comparison_table(metrics_with_cis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "type": "steering_validation",
            "layer": LAYER,
            "probe": "mean-difference direction (training-free)",
            "calibration": "platt_scaling",
            "calibration_params": cal_params,
            "high_threshold": HIGH_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
            "n_heldout": n,
            "bootstrap_n": 1000,
            "auroc_raw": float(raw_auroc),
            "auroc_calibrated": float(cal_auroc),
        },
        "methods": metrics_with_cis,
        "raw_data_summary": {
            "n_total": n,
            "n_correct": int(labels.sum()),
            "n_incorrect": int((~labels).sum()),
            "accuracy_overall": float(labels.mean()),
            "avg_cot_tokens": float(avg_cot_tokens),
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=float)

    print(f"\n Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return output_data


if __name__ == "__main__":
    main()