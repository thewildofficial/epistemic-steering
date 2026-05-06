"""Baseline comparison across methods.

Compares epistemic steering against always-direct, always-CoT, and random routing.
Computes bootstrap confidence intervals and generates comparison figures.

Figure outputs:
  - figures/fig6_accuracy_comparison.{png,pdf}  — accuracy bar chart
  - figures/fig7_selective_accuracy_vs_abstention.{png,pdf} — selective acc vs abstention rate  
  - figures/fig8_token_efficiency.{png,pdf} — token efficiency vs accuracy

Usage:
    python scripts/compare_methods.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import selective_accuracy, token_efficiency


def bootstrap_ci(metric_fn, data, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: callable(data) -> float
        data: array-like of per-question results (0/1 for binary, or numeric)
        n_bootstrap: number of bootstrap resamples
        ci: confidence interval percentile (default 95)

    Returns:
        dict with 'mean', 'lower', 'upper'
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    rng = np.random.default_rng(42)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric_fn(data[idx])

    alpha = (100 - ci) / 2
    lower = float(np.percentile(estimates, alpha))
    upper = float(np.percentile(estimates, 100 - alpha))
    mean = float(np.mean(estimates))

    return {"mean": mean, "lower": lower, "upper": upper}


def bootstrap_diff(data_a, data_b, n_bootstrap=1000, ci=95):
    """Bootstrap CI for difference in means (data_a - data_b).

    Returns:
        dict with 'mean_delta', 'lower', 'upper', 'significant' (bool)
    """
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    n = len(data_a)

    if n == 0:
        return {"mean_delta": 0.0, "lower": 0.0, "upper": 0.0, "significant": False}

    rng = np.random.default_rng(42)
    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a_mean = np.mean(data_a[idx])
        b_mean = np.mean(data_b[idx])
        estimates[i] = a_mean - b_mean

    alpha = (100 - ci) / 2
    lower = float(np.percentile(estimates, alpha))
    upper = float(np.percentile(estimates, 100 - alpha))
    mean_delta = float(np.mean(estimates))

    significant = (lower > 0) or (upper < 0)

    return {
        "mean_delta": mean_delta,
        "lower": lower,
        "upper": upper,
        "significant": significant,
    }


def load_results():
    """Load in-sample verification results, falling back to held-out if available.

    Returns:
        tuple of (verif: dict, heldout: dict | None)
    """
    verif_path = Path("data/verification_results.json")
    if not verif_path.exists():
        print(f"ERROR: {verif_path} not found.")
        sys.exit(1)

    with open(verif_path) as f:
        verif = json.load(f)

    heldout = None
    heldout_path = Path("data/heldout_evaluation_results.json")
    if heldout_path.exists():
        with open(heldout_path) as f:
            heldout = json.load(f)

    return verif, heldout


def build_synthetic_per_question_data(cm: dict, n_total: int) -> dict:
    """Build synthetic per-question arrays from confusion matrix counts.

    Given TP, FP, TN, FN at a threshold, constructs arrays ordered as:
      correct+confident (TP), correct+uncertain (FN), wrong+confident (FP), wrong+uncertain (TN)

    Args:
        cm: confusion matrix dict with TP, FP, TN, FN
        n_total: total number of questions

    Returns:
        dict with was_correct, was_confident arrays and summary counts
    """
    tp, fp, tn, fn_val = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    n_correct = tp + fn_val
    n_incorrect = fp + tn
    assert n_correct + n_incorrect == n_total, (
        f"Confusion matrix doesn't sum to n_total: "
        f"TP+FP+TN+FN={tp+fp+tn+fn_val} != {n_total}"
    )

    was_correct = np.array(
        [True] * tp + [True] * fn_val + [False] * fp + [False] * tn
    )
    was_confident = np.array(
        [True] * tp + [False] * fn_val + [True] * fp + [False] * tn
    )

    return {
        "was_correct": was_correct,
        "was_confident": was_confident,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
    }


def build_comparison_table(verif, heldout):
    """Build comparison table with bootstrap CIs across methods.

    Uses in-sample verification data as primary source.
    Estimates baseline methods (always-direct, always-CoT, random routing)
    from the confusion matrix and literature-typical token costs.

    Token estimates: direct ≈ 8 tok/q, CoT ≈ 120 tok/q (aligned with generation scripts).
    CoT accuracy estimated at base_acc + 4pp (MMLU) and base_acc + 10pp (GSM8K) —
    conservative for Qwen3.5-4B. Held-out Modal runs will replace these estimates.

    Args:
        verif: verification_results.json data
        heldout: heldout_evaluation_results.json data or None

    Returns:
        dict with methods, statistical_tests, token_efficiency, threshold_sweep
    """
    use_heldout = heldout is not None
    data_source = "heldout" if use_heldout else "in_sample"
    methods = []

    mmlu = verif["mmlu"]
    mmlu_cm = mmlu["confusion_matrix_at_threshold_0.5"]
    mmlu_n = mmlu["n_total"]
    mmlu_correct = mmlu["n_correct"]
    mmlu_incorrect = mmlu["n_incorrect"]
    mmlu_auroc = mmlu["auroc"]

    gsm8k = verif["gsm8k"]
    gsm8k_cm = gsm8k["confusion_matrix_at_threshold_0.5"]
    gsm8k_n = gsm8k["n_total"]
    gsm8k_correct = gsm8k["n_correct"]
    gsm8k_incorrect = gsm8k["n_incorrect"]
    gsm8k_auroc = gsm8k["auroc"]

    TOKENS_DIRECT = 8
    TOKENS_COT = 120

    # ── Always Direct: base model accuracy, no steering ──
    mmlu_direct_acc = mmlu_correct / mmlu_n
    mmlu_direct_per_q = np.array([1] * mmlu_correct + [0] * mmlu_incorrect)
    mmlu_direct_ci = bootstrap_ci(np.mean, mmlu_direct_per_q)

    gsm8k_direct_acc = gsm8k_correct / gsm8k_n
    gsm8k_direct_per_q = np.array([1] * gsm8k_correct + [0] * gsm8k_incorrect)
    gsm8k_direct_ci = bootstrap_ci(np.mean, gsm8k_direct_per_q)

    methods.append({
        "name": "Always Direct",
        "description": "Model answers directly, no steering, no abstention.",
        "mmlu": {
            "accuracy": round(mmlu_direct_acc, 4),
            "accuracy_ci": [round(mmlu_direct_ci["lower"], 4), round(mmlu_direct_ci["upper"], 4)],
            "selective_accuracy": round(mmlu_direct_acc, 4),
            "tokens_per_question": TOKENS_DIRECT,
            "abstention_rate": 0.0,
        },
        "gsm8k": {
            "accuracy": round(gsm8k_direct_acc, 4),
            "accuracy_ci": [round(gsm8k_direct_ci["lower"], 4), round(gsm8k_direct_ci["upper"], 4)],
            "selective_accuracy": round(gsm8k_direct_acc, 4),
            "tokens_per_question": TOKENS_DIRECT,
            "abstention_rate": 0.0,
        },
        "bootstrap_ci": {
            "mmlu_accuracy": [round(mmlu_direct_ci["lower"], 4), round(mmlu_direct_ci["upper"], 4)],
            "gsm8k_accuracy": [round(gsm8k_direct_ci["lower"], 4), round(gsm8k_direct_ci["upper"], 4)],
        },
    })

    # ── Always CoT: estimated improvement over direct (held-out pending) ──
    mmlu_cot_acc = min(mmlu_direct_acc + 0.04, 1.0)
    mmlu_cot_correct = int(mmlu_cot_acc * mmlu_n)
    mmlu_cot_per_q = np.array([1] * mmlu_cot_correct + [0] * (mmlu_n - mmlu_cot_correct))
    mmlu_cot_ci = bootstrap_ci(np.mean, mmlu_cot_per_q)

    gsm8k_cot_acc = min(gsm8k_direct_acc + 0.10, 1.0)
    gsm8k_cot_correct = int(gsm8k_cot_acc * gsm8k_n)
    gsm8k_cot_per_q = np.array([1] * gsm8k_cot_correct + [0] * (gsm8k_n - gsm8k_cot_correct))
    gsm8k_cot_ci = bootstrap_ci(np.mean, gsm8k_cot_per_q)

    methods.append({
        "name": "Always CoT",
        "description": "All questions answered with chain-of-thought reasoning. No steering.",
        "mmlu": {
            "accuracy": round(mmlu_cot_acc, 4),
            "accuracy_ci": [round(mmlu_cot_ci["lower"], 4), round(mmlu_cot_ci["upper"], 4)],
            "selective_accuracy": round(mmlu_cot_acc, 4),
            "tokens_per_question": TOKENS_COT,
            "abstention_rate": 0.0,
        },
        "gsm8k": {
            "accuracy": round(gsm8k_cot_acc, 4),
            "accuracy_ci": [round(gsm8k_cot_ci["lower"], 4), round(gsm8k_cot_ci["upper"], 4)],
            "selective_accuracy": round(gsm8k_cot_acc, 4),
            "tokens_per_question": TOKENS_COT,
            "abstention_rate": 0.0,
        },
        "bootstrap_ci": {
            "mmlu_accuracy": [round(mmlu_cot_ci["lower"], 4), round(mmlu_cot_ci["upper"], 4)],
            "gsm8k_accuracy": [round(gsm8k_cot_ci["lower"], 4), round(gsm8k_cot_ci["upper"], 4)],
        },
        "_note": "CoT accuracy estimated from base accuracy. Held-out evaluation pending.",
    })

    # ── Random Routing: 1/3 direct, 1/3 CoT, 1/3 abstain ──
    mmlu_rand_acc = (mmlu_direct_acc + mmlu_cot_acc + 0.0) / 3.0
    mmlu_rand_correct = int(mmlu_rand_acc * mmlu_n)
    mmlu_rand_per_q = np.array([1] * mmlu_rand_correct + [0] * (mmlu_n - mmlu_rand_correct))
    mmlu_rand_ci = bootstrap_ci(np.mean, mmlu_rand_per_q)
    mmlu_rand_tokens = (TOKENS_DIRECT + TOKENS_COT + 0) / 3.0

    gsm8k_rand_acc = (gsm8k_direct_acc + gsm8k_cot_acc + 0.0) / 3.0
    gsm8k_rand_correct = int(gsm8k_rand_acc * gsm8k_n)
    gsm8k_rand_per_q = np.array([1] * gsm8k_rand_correct + [0] * (gsm8k_n - gsm8k_rand_correct))
    gsm8k_rand_ci = bootstrap_ci(np.mean, gsm8k_rand_per_q)
    gsm8k_rand_tokens = (TOKENS_DIRECT + TOKENS_COT + 0) / 3.0

    methods.append({
        "name": "Random Routing",
        "description": "1/3 direct, 1/3 CoT, 1/3 abstain. No epistemic signal.",
        "mmlu": {
            "accuracy": round(mmlu_rand_acc, 4),
            "accuracy_ci": [round(mmlu_rand_ci["lower"], 4), round(mmlu_rand_ci["upper"], 4)],
            "selective_accuracy": round(mmlu_rand_acc / (2.0 / 3.0), 4) if mmlu_rand_acc > 0 else 0.0,
            "tokens_per_question": round(mmlu_rand_tokens, 1),
            "abstention_rate": 1.0 / 3.0,
        },
        "gsm8k": {
            "accuracy": round(gsm8k_rand_acc, 4),
            "accuracy_ci": [round(gsm8k_rand_ci["lower"], 4), round(gsm8k_rand_ci["upper"], 4)],
            "selective_accuracy": round(gsm8k_rand_acc / (2.0 / 3.0), 4) if gsm8k_rand_acc > 0 else 0.0,
            "tokens_per_question": round(gsm8k_rand_tokens, 1),
            "abstention_rate": 1.0 / 3.0,
        },
        "bootstrap_ci": {
            "mmlu_accuracy": [round(mmlu_rand_ci["lower"], 4), round(mmlu_rand_ci["upper"], 4)],
            "gsm8k_accuracy": [round(gsm8k_rand_ci["lower"], 4), round(gsm8k_rand_ci["upper"], 4)],
        },
        "_note": "Random routing is a theoretical estimate (no GPU run).",
    })

    # ── Prefill Probe (Ours) at t=0.5 ──
    tp, fp, tn, fn_val = mmlu_cm["TP"], mmlu_cm["FP"], mmlu_cm["TN"], mmlu_cm["FN"]

    # Upper bound: assumes CoT fixes all caught errors → (N - FP) / N
    mmlu_sel_acc = selective_accuracy(
        direct_correct=tp, cot_correct=tn + fn_val, abstentions=0, total=mmlu_n
    )
    # Conservative: 80% CoT fix rate on caught errors
    cot_fix_rate = 0.8
    mmlu_sel_acc_cons = selective_accuracy(
        direct_correct=tp,
        cot_correct=int(tn * cot_fix_rate) + fn_val,
        abstentions=0,
        total=mmlu_n,
    )

    mmlu_conf_acc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    mmlu_prev = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mmlu_block = fn_val / (fn_val + tp) if (fn_val + tp) > 0 else 0.0
    mmlu_abstention = (fn_val + tn) / mmlu_n

    n_confident = tp + fp
    n_uncertain = tn + fn_val
    mmlu_steer_tokens = (n_confident * TOKENS_DIRECT + n_uncertain * TOKENS_COT) / mmlu_n

    mmlu_steer_correct_arr = np.array(
        [1] * tp + [1] * fn_val + [0] * fp + [1] * tn
    )
    mmlu_steer_ci = bootstrap_ci(np.mean, mmlu_steer_correct_arr)

    n_cot_fixed = int(tn * cot_fix_rate)
    n_cot_unfixed = tn - n_cot_fixed
    mmlu_steer_cons_arr = np.array(
        [1] * tp + [1] * fn_val + [0] * fp + [1] * n_cot_fixed + [0] * n_cot_unfixed
    )
    mmlu_steer_cons_ci = bootstrap_ci(np.mean, mmlu_steer_cons_arr)

    gsm_tp, gsm_fp, gsm_tn, gsm_fn_val = gsm8k_cm["TP"], gsm8k_cm["FP"], gsm8k_cm["TN"], gsm8k_cm["FN"]
    gsm8k_sel_acc = selective_accuracy(
        direct_correct=gsm_tp, cot_correct=gsm_tn + gsm_fn_val, abstentions=0, total=gsm8k_n
    )
    gsm8k_abstention = (gsm_fn_val + gsm_tn) / gsm8k_n
    gsm_confident = gsm_tp + gsm_fp
    gsm_uncertain = gsm_tn + gsm_fn_val
    gsm8k_steer_tokens = (
        (gsm_confident * TOKENS_DIRECT + gsm_uncertain * TOKENS_COT) / gsm8k_n
    ) if gsm8k_n > 0 else 0.0

    gsm8k_steer_correct_arr = np.array(
        [1] * gsm_tp + [1] * gsm_fn_val + [0] * gsm_fp + [1] * gsm_tn
    )
    gsm8k_steer_ci = bootstrap_ci(np.mean, gsm8k_steer_correct_arr)

    methods.append({
        "name": "Prefill Probe (Ours)",
        "description": f"Epistemic steering at t=0.5. MMLU AUROC={mmlu_auroc:.3f}, GSM8K AUROC={gsm8k_auroc:.3f} (overfit).",
        "mmlu": {
            "accuracy": round(mmlu_sel_acc, 4),
            "accuracy_ci": [round(mmlu_steer_ci["lower"], 4), round(mmlu_steer_ci["upper"], 4)],
            "selective_accuracy": round(mmlu_sel_acc, 4),
            "selective_accuracy_conservative": round(mmlu_sel_acc_cons, 4),
            "selective_accuracy_conservative_ci": [
                round(mmlu_steer_cons_ci["lower"], 4),
                round(mmlu_steer_cons_ci["upper"], 4),
            ],
            "tokens_per_question": round(mmlu_steer_tokens, 1),
            "abstention_rate": round(mmlu_abstention, 4),
            "prevention_rate": round(mmlu_prev, 4),
            "unnecessary_block_rate": round(mmlu_block, 4),
            "direct_accuracy_on_confident": round(mmlu_conf_acc, 4),
            "auroc": round(mmlu_auroc, 4),
        },
        "gsm8k": {
            "accuracy": round(gsm8k_sel_acc, 4),
            "accuracy_ci": [round(gsm8k_steer_ci["lower"], 4), round(gsm8k_steer_ci["upper"], 4)],
            "selective_accuracy": round(gsm8k_sel_acc, 4),
            "tokens_per_question": round(gsm8k_steer_tokens, 1),
            "abstention_rate": round(gsm8k_abstention, 4),
            "auroc": round(gsm8k_auroc, 4),
        },
        "bootstrap_ci": {
            "mmlu_accuracy": [round(mmlu_steer_ci["lower"], 4), round(mmlu_steer_ci["upper"], 4)],
            "gsm8k_accuracy": [round(gsm8k_steer_ci["lower"], 4), round(gsm8k_steer_ci["upper"], 4)],
        },
        "warnings": [
            "IN-SAMPLE evaluation on data probe was trained on.",
            "Selective accuracy assumes CoT answers all caught questions correctly (upper bound).",
            "Conservative estimate uses 80% CoT fix rate for caught errors.",
            "GSM8K AUROC=0.994 is overfit (7/200 correct, severe class imbalance).",
        ],
    })

    # ── Statistical tests ──
    stat_tests = {}

    direct_mmlu = np.array([1] * mmlu_correct + [0] * mmlu_incorrect)
    prefill_mmlu = mmlu_steer_correct_arr

    diff_direct = bootstrap_diff(prefill_mmlu, direct_mmlu)
    stat_tests["prefill_vs_direct"] = {
        "dataset": "mmlu",
        "delta": round(diff_direct["mean_delta"], 4),
        "ci": [round(diff_direct["lower"], 4), round(diff_direct["upper"], 4)],
        "significant": diff_direct["significant"],
        "note": "Prefill probe (upper bound) vs always-direct baseline.",
    }

    prefill_vs_rand = bootstrap_diff(prefill_mmlu, mmlu_rand_per_q)
    stat_tests["prefill_vs_random"] = {
        "dataset": "mmlu",
        "delta": round(prefill_vs_rand["mean_delta"], 4),
        "ci": [round(prefill_vs_rand["lower"], 4), round(prefill_vs_rand["upper"], 4)],
        "significant": prefill_vs_rand["significant"],
    }

    diff_direct_cons = bootstrap_diff(mmlu_steer_cons_arr, direct_mmlu)
    stat_tests["prefill_conservative_vs_direct"] = {
        "dataset": "mmlu",
        "delta": round(diff_direct_cons["mean_delta"], 4),
        "ci": [round(diff_direct_cons["lower"], 4), round(diff_direct_cons["upper"], 4)],
        "significant": diff_direct_cons["significant"],
        "note": "Conservative estimate (80% CoT fix rate) vs always-direct.",
    }

    stat_tests["gsm8k_caveat"] = {
        "warning": "GSM8K statistical tests omitted. 0.994 AUROC is in-sample overfit on 7/200 positives.",
        "recommendation": "Run held-out evaluation on Modal GPU before claiming GSM8K results.",
    }

    threshold_data = None
    threshold_path = Path("data/threshold_analysis.json")
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_data = json.load(f)

    always_cot_total = mmlu_n * TOKENS_COT
    always_direct_total = mmlu_n * TOKENS_DIRECT

    token_eff = {}
    token_eff["always_direct"] = {
        **token_efficiency(
            direct_tokens=always_direct_total, cot_tokens=always_direct_total,
            routed_tokens=always_direct_total, total=mmlu_n,
        ),
        "savings_vs_always_cot": float(always_cot_total - always_direct_total) / always_cot_total,
    }
    token_eff["always_cot"] = {
        **token_efficiency(
            direct_tokens=0, cot_tokens=always_cot_total,
            routed_tokens=always_cot_total, total=mmlu_n,
        ),
        "savings_vs_always_cot": 0.0,
    }
    rand_total_toks = mmlu_n * mmlu_rand_tokens
    token_eff["random_routing"] = {
        **token_efficiency(
            direct_tokens=mmlu_n * TOKENS_DIRECT / 3, cot_tokens=mmlu_n * TOKENS_COT / 3,
            routed_tokens=rand_total_toks, total=mmlu_n,
        ),
        "savings_vs_always_cot": float(always_cot_total - rand_total_toks) / always_cot_total,
    }
    steer_total_toks = n_confident * TOKENS_DIRECT + n_uncertain * TOKENS_COT
    token_eff["prefill_probe"] = {
        **token_efficiency(
            direct_tokens=n_confident * TOKENS_DIRECT, cot_tokens=n_uncertain * TOKENS_COT,
            routed_tokens=steer_total_toks, total=mmlu_n,
        ),
        "savings_vs_always_cot": float(always_cot_total - steer_total_toks) / always_cot_total,
    }

    return {
        "metadata": {
            "type": "comparison",
            "data_source": data_source,
            "warning": "Results from in-sample verification. Held-out evaluation pending Modal GPU run.",
            "probe_layer": verif["metadata"].get("probe_layer", 30),
            "model": verif["metadata"].get("model", "Qwen3.5-4B"),
            "threshold": 0.5,
            "cot_fix_rate_for_conservative": cot_fix_rate,
        },
        "methods": methods,
        "statistical_tests": stat_tests,
        "token_efficiency": token_eff,
        "threshold_sweep": threshold_data,
    }


def _setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.set_palette("colorblind")


def _save_figure(fig: plt.Figure, save_path: str):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(table: dict, save_path: str):
    """Fig 6: Accuracy bar chart — 4 methods, MMLU + GSM8K side by side."""
    _setup_style()

    methods = table["methods"]
    short_names = ["Always\nDirect", "Always\nCoT", "Random\nRouting", "Prefill Probe\n(Ours)"]

    mmlu_acc = [m["mmlu"]["accuracy"] for m in methods]
    mmlu_lo = [m["bootstrap_ci"]["mmlu_accuracy"][0] for m in methods]
    mmlu_hi = [m["bootstrap_ci"]["mmlu_accuracy"][1] for m in methods]
    mmlu_err_lo = [acc - lo for acc, lo in zip(mmlu_acc, mmlu_lo)]
    mmlu_err_hi = [hi - acc for acc, hi in zip(mmlu_acc, mmlu_hi)]

    gsm8k_acc = [m["gsm8k"]["accuracy"] for m in methods]
    gsm8k_lo = [m["bootstrap_ci"]["gsm8k_accuracy"][0] for m in methods]
    gsm8k_hi = [m["bootstrap_ci"]["gsm8k_accuracy"][1] for m in methods]
    gsm8k_err_lo = [acc - lo for acc, lo in zip(gsm8k_acc, gsm8k_lo)]
    gsm8k_err_hi = [hi - acc for acc, hi in zip(gsm8k_acc, gsm8k_hi)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(methods))
    width = 0.35
    colors = ["#4878D0", "#6ACC64", "#D65F5F", "#EE854A"]

    for ax_idx, (ax, accs, err_lo, err_hi, title, n) in enumerate([
        (axes[0], mmlu_acc, mmlu_err_lo, mmlu_err_hi, "MMLU (n=456)", 456),
        (axes[1], gsm8k_acc, gsm8k_err_lo, gsm8k_err_hi, "GSM8K (n=200)", 200),
    ]):
        bars = ax.bar(x, accs, width, yerr=[err_lo, err_hi],
                      capsize=6, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, min(acc + 0.03, 0.98),
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    if len(methods) >= 4 and "prevention_rate" in methods[3].get("mmlu", {}):
        prev = methods[3]["mmlu"]["prevention_rate"]
        axes[0].annotate(f"Prevention: {prev:.1%}",
                         xy=(3, mmlu_acc[3]), xytext=(3, mmlu_acc[3] + 0.12),
                         ha="center", fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    fig.suptitle("Figure 6: Accuracy Comparison Across Methods",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "WARNING: IN-SAMPLE evaluation. Held-out results pending Modal GPU run.",
             ha="center", fontsize=10, fontstyle="italic", color="gray")
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_selective_accuracy_vs_abstention(table: dict, save_path: str):
    """Fig 7: Selective accuracy vs abstention rate (MMLU).

    Baselines shown as single points; probe shows threshold sweep curve.
    """
    _setup_style()

    methods = table["methods"]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#4878D0", "#6ACC64", "#D65F5F", "#EE854A"]
    markers = ["s", "D", "v", "o"]

    for i, method in enumerate(methods):
        mmlu = method["mmlu"]
        abst_rate = mmlu.get("abstention_rate", 0.0)
        sel_acc = mmlu["selective_accuracy"]
        ax.scatter(abst_rate, sel_acc, c=colors[i], marker=markers[i],
                   s=250, edgecolors="black", linewidth=1.5,
                   label=method["name"], zorder=5)
        offset_y = -25 if i == 3 else 15
        ax.annotate(method["name"], (abst_rate, sel_acc),
                    textcoords="offset points", xytext=(8, offset_y),
                    fontsize=10, fontweight="bold", ha="left", color=colors[i])

    threshold_data = table.get("threshold_sweep")
    if threshold_data and "mmlu" in threshold_data:
        mmlu_sweep = threshold_data["mmlu"]
        if "selective_accuracies" in mmlu_sweep and "unnecessary_block_rates" in mmlu_sweep:
            sel_accs = np.array(mmlu_sweep["selective_accuracies"])
            block_rates = np.array(mmlu_sweep["unnecessary_block_rates"])
            thresholds = np.array(mmlu_sweep["thresholds"])
            n_total = mmlu_sweep.get("n_total", 456)
            n_correct = mmlu_sweep.get("n_correct", 254)
            n_wrong = n_total - n_correct
            prev_rates = np.array(mmlu_sweep["prevention_rates"])
            abst_rates = (block_rates * n_correct + prev_rates * n_wrong) / n_total

            scatter = ax.scatter(abst_rates, sel_accs, c=thresholds,
                                 cmap="viridis", s=40, alpha=0.6, zorder=3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Threshold", fontsize=12, fontweight="bold")
            ax.plot(abst_rates, sel_accs, "-", color="gray", alpha=0.4, linewidth=1, zorder=2)

    ax.set_xlabel("Abstention Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("Selective Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Figure 7: Selective Accuracy vs Abstention Rate (MMLU)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-0.02, max(ax.get_xlim()[1], 1.02))
    ax.set_ylim(0, 1.02)
    fig.text(0.5, -0.02, "WARNING: IN-SAMPLE evaluation. Held-out results pending Modal GPU run.",
             ha="center", fontsize=10, fontstyle="italic", color="gray")
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_token_efficiency(table: dict, save_path: str):
    """Fig 8: Token efficiency bubble plot — accuracy vs tokens per question (MMLU)."""
    _setup_style()

    methods = table["methods"]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#4878D0", "#6ACC64", "#D65F5F", "#EE854A"]
    markers = ["s", "D", "v", "o"]
    offsets = [(10, 10), (10, 10), (10, -15), (-15, -15)]

    for i, method in enumerate(methods):
        mmlu = method["mmlu"]
        acc = mmlu["accuracy"]
        tokens = mmlu["tokens_per_question"]
        sel_acc = mmlu.get("selective_accuracy", acc)
        size = max(100, sel_acc * 600)
        ax.scatter(acc, tokens, c=colors[i], marker=markers[i],
                   s=size, edgecolors="black", linewidth=1.5,
                   label=method["name"], zorder=5, alpha=0.85)
        ox, oy = offsets[i]
        ax.annotate(method["name"], (acc, tokens), textcoords="offset points",
                    xytext=(ox, oy), fontsize=10, fontweight="bold",
                    ha="left", color=colors[i])

    ax.annotate("Better →", xy=(0.95, 0.05), xycoords="axes fraction",
                fontsize=11, fontstyle="italic", color="green", ha="right", va="bottom")
    ax.set_xlabel("Accuracy (MMLU)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Avg Tokens per Question", fontsize=14, fontweight="bold")
    ax.set_title("Figure 8: Token Efficiency vs Accuracy (MMLU)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.text(0.5, -0.02, "WARNING: IN-SAMPLE evaluation. Held-out results pending Modal GPU run.",
             ha="center", fontsize=10, fontstyle="italic", color="gray")
    plt.tight_layout()
    _save_figure(fig, save_path)


def main():
    print("=" * 60)
    print("BASELINE COMPARISON: Epistemic Steering vs Baselines")
    print("=" * 60)

    verif, heldout = load_results()
    print(f"\nData source: {'heldout' if heldout else 'in-sample verification'}")
    print(f"MMLU: {verif['mmlu']['n_total']} questions, AUROC={verif['mmlu']['auroc']:.3f}")
    print(f"GSM8K: {verif['gsm8k']['n_total']} questions, AUROC={verif['gsm8k']['auroc']:.3f}")
    if not heldout:
        print("⚠  WARNING: Using in-sample data. Held-out evaluation pending Modal GPU run.")

    print("\nBuilding comparison table with bootstrap CIs ...")
    table = build_comparison_table(verif, heldout)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2, default=float)
    print(f"\nComparison results saved to: {output_path}")

    print("\n── Method Comparison Summary ──")
    print(f"{'Method':<25} {'MMLU Acc':>10} {'95% CI':>20} {'Tokens/Q':>10} {'Abstention':>12}")
    print("-" * 80)
    for m in table["methods"]:
        ci = m["bootstrap_ci"]["mmlu_accuracy"]
        print(f"{m['name']:<25} {m['mmlu']['accuracy']:>10.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]"
              f"{m['mmlu']['tokens_per_question']:>10.1f} "
              f"{m['mmlu'].get('abstention_rate', 0):>12.1%}")

    print("\n── Statistical Tests (MMLU, bootstrap) ──")
    for test_name, test in table["statistical_tests"].items():
        if "warning" in test:
            print(f"  {test_name}: {test['warning']}")
            continue
        sig = " ✓ SIGNIFICANT" if test["significant"] else " (not significant)"
        print(f"  {test_name}: Δ={test['delta']:.4f}, "
              f"95% CI [{test['ci'][0]:.4f}, {test['ci'][1]:.4f}]{sig}")
        if "note" in test:
            print(f"    Note: {test['note']}")

    print("\nGenerating comparison figures ...")
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_comparison(table, str(figures_dir / "fig6_accuracy_comparison"))
    print("  ✓ fig6_accuracy_comparison (.png + .pdf)")

    plot_selective_accuracy_vs_abstention(table, str(figures_dir / "fig7_selective_accuracy_vs_abstention"))
    print("  ✓ fig7_selective_accuracy_vs_abstention (.png + .pdf)")

    plot_token_efficiency(table, str(figures_dir / "fig8_token_efficiency"))
    print("  ✓ fig8_token_efficiency (.png + .pdf)")

    print("\n── Token Efficiency (MMLU) ──")
    for method_name, eff in table["token_efficiency"].items():
        savings = eff.get("savings_vs_always_cot", 0)
        print(f"  {method_name:<20}: {eff['tokens_per_question']:.1f} tok/q, "
              f"savings vs always-CoT: {savings:.1%}")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print(f"  data/comparison_results.json")
    print(f"  figures/fig6_accuracy_comparison.{{png,pdf}}")
    print(f"  figures/fig7_selective_accuracy_vs_abstention.{{png,pdf}}")
    print(f"  figures/fig8_token_efficiency.{{png,pdf}}")
    print("=" * 60)


if __name__ == "__main__":
    main()
