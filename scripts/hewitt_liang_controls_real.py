#!/usr/bin/env python3
"""
Out-of-sample Hewitt-Liang Controls on REAL 6-benchmark data.

Implements three held-out control tasks to validate that the probe's high AUROC
reflects genuine correctness signal, not spurious correlates or overfitting:

  1. Random-Labels Control — shuffle y_train only, train on (X_train, y_shuffled),
     evaluate AUROC on held-out (X_test, y_test). Expect ~0.50.
  2. Noise/Position Control — replace X with random noise N(0,1), same shape.
     Train on (X_noise_train, y_train). Evaluate on held-out (X_noise_test, y_test).
     Expect ~0.50.
  3. Permutation Control — shuffle rows of X (across samples). Train on
     (X_perm_train, y_train). Evaluate on held-out (X_perm_test, y_test).
     If AUROC persists, signal is marginal.

Usage:
    python scripts/hewitt_liang_controls_real.py
    python scripts/hewitt_liang_controls_real.py --benchmark mmlu
    python scripts/hewitt_liang_controls_real.py --output results/custom.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

BENCHMARKS = ["mmlu", "arc_challenge", "triviaqa", "gsm8k", "math", "humaneval"]

LAYER = 25
HIDDEN_DIM = 2560
SEED = 42

# Threshold for "spurious" conclusion
SPURIOUS_THRESHOLD = 0.60

# Real probe within-domain AUROCs (from user notes)
EXPECTED_REAL_AUROCS = {
    "mmlu": 0.847,
    "arc_challenge": 0.903,
    "triviaqa": 0.867,
    "gsm8k": 0.803,
    "math": 0.949,
    "humaneval": 0.976,
}

CONTROL_SEEDS = [42, 123, 456, 789, 2024]


def load_v2_benchmark_data(benchmark: str):
    """Load real activations and labels for a benchmark.

    Supports two directory layouts:
      - v2: data/benchmark_activations_v2/{benchmark}/ (*.npy + *.json side-by-side)
      - legacy: data/benchmark_activations_{benchmark}/{benchmark}/ (*.npy)
                + data/benchmark_activations_{benchmark}_results.jsonl

    Returns (X, y) where X is (n_samples, HIDDEN_DIM) and y is (n_samples,) bool.
    Returns (None, None) if data missing.
    """
    # ---- Try v2 layout first ----
    v2_dir = PROJECT_ROOT / "data" / "benchmark_activations_v2" / benchmark
    if v2_dir.is_dir():
        npy_files = sorted(v2_dir.glob("*.npy"))
        X_list, y_list = [], []
        for npy_path in npy_files:
            json_path = npy_path.with_suffix(".json")
            if not json_path.exists():
                continue
            try:
                activation = np.load(npy_path).astype(np.float64)
                with open(json_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            # If 2D (seq_len, hidden), take last token
            if activation.ndim == 2:
                activation = activation[-1]
            if activation.shape[0] != HIDDEN_DIM:
                continue
            # Handle 0-dim numpy arrays for correct field
            correct_val = meta.get("correct", False)
            if isinstance(correct_val, np.ndarray):
                if correct_val.ndim == 0:
                    correct_val = correct_val.item()
                else:
                    correct_val = bool(correct_val)
            else:
                correct_val = bool(correct_val)
            y_list.append(correct_val)
            X_list.append(activation)
        if X_list:
            return np.stack(X_list, axis=0), np.array(y_list, dtype=bool)

    alt_dir = PROJECT_ROOT / "data" / "benchmark_activations" / benchmark
    alt_results = PROJECT_ROOT / "data" / "benchmark_activations" / f"{benchmark}_results.jsonl"
    X, y = _try_load_npy_jsonl(alt_dir, alt_results)
    if X is not None:
        return X, y

    legacy_dir = PROJECT_ROOT / "data" / f"benchmark_activations_{benchmark}"
    legacy_results = PROJECT_ROOT / "data" / f"benchmark_activations_{benchmark}_results.jsonl"
    sub_dir = legacy_dir / benchmark
    act_dir = sub_dir if sub_dir.is_dir() else legacy_dir
    X, y = _try_load_npy_jsonl(act_dir, legacy_results)
    if X is not None:
        return X, y

    return None, None


def _try_load_npy_jsonl(act_dir: Path, results_file: Path):
    """Try to load (X, y) from a directory of .npy files and a results JSONL.

    Returns (None, None) if either missing or no matching samples found.
    """
    if not act_dir.is_dir() or not results_file.is_file():
        return None, None

    label_map = {}
    with open(results_file) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                correct_val = rec.get("correct", False)
                if isinstance(correct_val, np.ndarray):
                    correct_val = correct_val.item() if correct_val.ndim == 0 else bool(correct_val)
                label_map[rec["question_id"]] = bool(correct_val)

    npy_files = sorted(act_dir.glob("*.npy"))
    X_list, y_list = [], []
    for npy_path in npy_files:
        stem = npy_path.stem
        qid = stem.rsplit("__layer_", 1)[0] if "__layer_" in stem else stem
        if qid not in label_map:
            continue
        try:
            activation = np.load(npy_path).astype(np.float64)
        except Exception:
            continue
        if activation.ndim == 2:
            activation = activation[-1]
        if activation.shape[0] != HIDDEN_DIM:
            continue
        X_list.append(activation)
        y_list.append(label_map[qid])

    if X_list:
        return np.stack(X_list, axis=0), np.array(y_list, dtype=bool)
    return None, None


def _train_and_evaluate_auroc(X_train, y_train, X_test, y_test):
    """Train LogisticRegressionCV on training data and return AUROC on test data.

    Uses SAME hyperparameters as real probes: Cs=10, cv=StratifiedKFold(3),
    scoring='roc_auc', max_iter=1000, random_state=SEED.
    """
    if len(np.unique(y_train)) < 2:
        return float("nan")

    # Check if test set has at least 2 classes
    if len(np.unique(y_test)) < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    model = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        scoring="roc_auc",
        max_iter=1000,
        random_state=SEED,
    )
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"    [WARN] LogisticRegressionCV failed: {e}")
        return float("nan")

    y_prob = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def conclusion(auroc: float) -> str:
    """Return a conclusion string based on AUROC."""
    if np.isnan(auroc):
        return "SKIP — invalid AUROC (single class or training failure)"
    if auroc > SPURIOUS_THRESHOLD:
        return "FAILED — AUROC > 0.60, probe may read spurious correlates"
    elif auroc > 0.55:
        return "WEAK — slight above-chance signal, possible partial confound"
    else:
        return "PASS — near-chance AUROC, control passed"


def run_controls_for_benchmark(benchmark: str):
    """Run all three controls + real probe baseline for a single benchmark.

    Returns a dict with results.
    """
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {benchmark}")
    print(f"{'=' * 70}")

    X, y = load_v2_benchmark_data(benchmark)
    if X is None:
        print(f"  [ERROR] No data found for benchmark '{benchmark}'")
        return {
            "benchmark": benchmark,
            "status": "failed",
            "error": "no_data",
        }

    n_samples, n_features = X.shape
    n_correct = int(y.sum())
    n_incorrect = int((~y).sum())
    print(f"  Samples: {n_samples}, Features: {n_features}")
    print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")

    # Can't stratify single-class data
    if len(np.unique(y)) < 2:
        print(f"  [SKIP] Only one class present (all {y[0]})")
        return {
            "benchmark": benchmark,
            "status": "failed",
            "error": "single_class",
            "n_samples": n_samples,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
        }

    if n_samples < 20:
        print(f"  [ERROR] Too few samples ({n_samples}) for 80/20 stratified split + 3-fold CV")
        return {
            "benchmark": benchmark,
            "status": "failed",
            "error": "too_few_samples",
        }

    # --- 80/20 stratified train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Test correct: {y_test.sum()}, Test incorrect: {(~y_test).sum()}")

    results = {
        "benchmark": benchmark,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "expected_real_auroc": EXPECTED_REAL_AUROCS.get(benchmark, None),
    }

    # ---- REAL PROBE BASELINE (single run) ----
    print(f"\n  --- Real Probe Baseline ---")
    real_auroc = _train_and_evaluate_auroc(X_train, y_train, X_test, y_test)
    real_result = {
        "auroc": real_auroc,
        "conclusion": "baseline",
    }
    results["real_probe"] = real_result
    print(f"    AUROC: {real_auroc:.4f}")

        # ---- CONTROL 1: Random-Labels (multi-seed) ----
    print(f"\n  --- Control 1: Random-Labels (n_seeds={len(CONTROL_SEEDS)}) ---")
    random_aurocs = []
    n_flipped_total = 0
    for ctrl_seed in CONTROL_SEEDS:
        shuffle_rng = np.random.RandomState(seed=ctrl_seed)
        y_train_shuffled = y_train.copy()
        shuffle_rng.shuffle(y_train_shuffled)
        n_flipped = int((y_train != y_train_shuffled).sum())
        n_flipped_total += n_flipped
        auroc = _train_and_evaluate_auroc(X_train, y_train_shuffled, X_test, y_test)
        random_aurocs.append({"seed": ctrl_seed, "auroc": auroc})
        print(f"    seed={ctrl_seed} flipped={n_flipped} auroc={auroc:.4f}")

    random_mean = float(np.nanmean([r["auroc"] for r in random_aurocs]))
    random_std = float(np.nanstd([r["auroc"] for r in random_aurocs]))
    random_result = {
        "per_seed": random_aurocs,
        "auroc_mean": random_mean,
        "auroc_std": random_std,
        "mean_n_flipped": n_flipped_total / len(CONTROL_SEEDS),
        "conclusion": conclusion(random_mean),
    }
    results["random_labels"] = random_result
    print(f"    Mean AUROC: {random_mean:.4f} ± {random_std:.4f} | {random_result['conclusion']}")

    # ---- CONTROL 2: Noise/Position (multi-seed) ----
    print(f"\n  --- Control 2: Noise/Position (n_seeds={len(CONTROL_SEEDS)}) ---")
    noise_aurocs = []
    for ctrl_seed in CONTROL_SEEDS:
        noise_rng = np.random.RandomState(seed=ctrl_seed)
        X_noise = noise_rng.randn(*X.shape).astype(np.float64)
        X_noise_train, X_noise_test, _, _ = train_test_split(
            X_noise, y, test_size=0.2, stratify=y, random_state=SEED
        )
        auroc = _train_and_evaluate_auroc(X_noise_train, y_train, X_noise_test, y_test)
        noise_aurocs.append({"seed": ctrl_seed, "auroc": auroc})
        print(f"    seed={ctrl_seed} auroc={auroc:.4f}")

    noise_mean = float(np.nanmean([r["auroc"] for r in noise_aurocs]))
    noise_std = float(np.nanstd([r["auroc"] for r in noise_aurocs]))
    noise_result = {
        "per_seed": noise_aurocs,
        "auroc_mean": noise_mean,
        "auroc_std": noise_std,
        "conclusion": conclusion(noise_mean),
    }
    results["noise_position"] = noise_result
    print(f"    Mean AUROC: {noise_mean:.4f} ± {noise_std:.4f} | {noise_result['conclusion']}")

    # ---- CONTROL 3: Permutation (multi-seed) ----
    print(f"\n  --- Control 3: Permutation (n_seeds={len(CONTROL_SEEDS)}) ---")
    perm_aurocs = []
    for ctrl_seed in CONTROL_SEEDS:
        perm_rng = np.random.RandomState(seed=ctrl_seed)
        perm_indices = perm_rng.permutation(len(X))
        X_perm = X[perm_indices]
        X_perm_train, X_perm_test, _, _ = train_test_split(
            X_perm, y, test_size=0.2, stratify=y, random_state=SEED
        )
        auroc = _train_and_evaluate_auroc(X_perm_train, y_train, X_perm_test, y_test)
        perm_aurocs.append({"seed": ctrl_seed, "auroc": auroc})
        print(f"    seed={ctrl_seed} auroc={auroc:.4f}")

    perm_mean = float(np.nanmean([r["auroc"] for r in perm_aurocs]))
    perm_std = float(np.nanstd([r["auroc"] for r in perm_aurocs]))
    perm_result = {
        "per_seed": perm_aurocs,
        "auroc_mean": perm_mean,
        "auroc_std": perm_std,
        "conclusion": conclusion(perm_mean),
    }
    results["permutation"] = perm_result
    print(f"    Mean AUROC: {perm_mean:.4f} ± {perm_std:.4f} | {perm_result['conclusion']}")

    # ---- Overall assessment for this benchmark ----
    controls = [random_result, noise_result, perm_result]
    control_conclusions = [c["conclusion"] for c in controls]
    any_failed = any(c.startswith("FAILED") for c in control_conclusions)
    any_weak = any(c.startswith("WEAK") for c in control_conclusions)
    if any_failed:
        overall = f"FAILED — at least one control AUROC ≥ {SPURIOUS_THRESHOLD}"
    elif any_weak:
        overall = f"WEAK — at least one control AUROC > 0.55"
    else:
        overall = "PASS — all controls near chance"
    results["overall_assessment"] = overall
    print(f"\n  OVERALL: {overall}")

    return results


def generate_markdown_report(all_results: list[dict]) -> str:
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# Hewitt-Liang Controls (Real Data, Held-Out Evaluation)")
    lines.append("")
    lines.append("**Methodology:** 80/20 stratified train/test split. LogisticRegressionCV(Cs=10, cv=StratifiedKFold(3), scoring='roc_auc', max_iter=1000). All control AUROCs evaluated on the **held-out test set** -- never in-sample. Each control is averaged over 5 randomization seeds.")
    lines.append("")
    lines.append("| Benchmark | N | Real AUROC | Random-Labels | Noise | Permutation | Overall |")
    lines.append("|-----------|---|------------|---------------|-------|-------------|---------|")

    for res in all_results:
        bench = res.get("benchmark", "?")
        n = res.get("n_samples", "?")
        real = res.get("real_probe", {}).get("auroc", float("nan"))

        def ctrl_auroc(key):
            if key in res and isinstance(res[key], dict):
                return res[key].get("auroc_mean", float("nan"))
            return float("nan")

        rand = ctrl_auroc("random_labels")
        noise = ctrl_auroc("noise_position")
        perm = ctrl_auroc("permutation")
        overall = res.get("overall_assessment", "?")

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}" if not np.isnan(v) else "N/A"
            return str(v)

        lines.append(
            f"| {bench} | {n} | {fmt(real)} | {fmt(rand)} | {fmt(noise)} | {fmt(perm)} | {overall} |"
        )

    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for res in all_results:
        bench = res.get("benchmark", "?")
        lines.append(f"### {bench}")
        lines.append(f"- **Samples:** {res.get('n_samples', '?')} (train={res.get('n_train', '?')}, test={res.get('n_test', '?')})")
        if "real_probe" in res:
            lines.append(f"- **Real probe AUROC:** {fmt(res['real_probe'].get('auroc', float('nan')))} (expected: {res.get('expected_real_auroc', 'N/A')})")

        for ctrl_name, ctrl_key in [
            ("Random-Labels", "random_labels"),
            ("Noise/Position", "noise_position"),
            ("Permutation", "permutation"),
        ]:
            if ctrl_key in res:
                ctrl = res[ctrl_key]
                mean = ctrl.get("auroc_mean", float("nan"))
                std = ctrl.get("auroc_std", float("nan"))
                conc = ctrl.get("conclusion", "")
                lines.append(f"- **{ctrl_name} AUROC:** {fmt(mean)} ± {fmt(std)} — {conc}")

        lines.append(f"- **Overall:** {res.get('overall_assessment', '')}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by scripts/hewitt_liang_controls_real.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Out-of-sample Hewitt-Liang Controls on real benchmark data"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help="Benchmark to run (default: all)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(RESULTS_DIR / "hewitt_liang_controls_real.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(RESULTS_DIR / "hewitt_liang_controls_real.md"),
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks_to_run = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    print("=" * 70)
    print("HEWITT-LIANG CONTROLS — REAL DATA, HELD-OUT EVALUATION")
    print("=" * 70)
    print(f"  Benchmarks: {benchmarks_to_run}")
    print(f"  Layer: {LAYER}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Split: 80/20 stratified, random_state={SEED}")
    print(f"  Probe: LogisticRegressionCV(Cs=10, cv=StratifiedKFold(3), scoring='roc_auc', max_iter=1000)")
    print(f"  Control seeds: {CONTROL_SEEDS}")
    print(f"  Spurious threshold: AUROC > {SPURIOUS_THRESHOLD}")
    print("")

    all_results = []
    for benchmark in benchmarks_to_run:
        result = run_controls_for_benchmark(benchmark)
        all_results.append(result)

    # Global assessment
    evaluations = [
        r.get("overall_assessment", "")
        for r in all_results
        if "overall_assessment" in r
    ]
    if not evaluations:
        global_assessment = "NO VALID RESULTS"
    elif all(e.startswith("PASS") for e in evaluations):
        global_assessment = "ALL BENCHMARKS PASSED — probe signal is genuine"
    else:
        global_assessment = "SOME BENCHMARKS FAILED — probe may read spurious correlates"

    output_data = {
        "metadata": {
            "task": "hewitt_liang_controls_real",
            "layer": LAYER,
            "hidden_dim": HIDDEN_DIM,
            "split": "80/20 stratified",
            "random_state": SEED,
            "probe_type": "LogisticRegressionCV",
            "probe_params": {
                "Cs": 10,
                "cv": "StratifiedKFold(3)",
                "scoring": "roc_auc",
                "max_iter": 1000,
                "random_state": SEED,
            },
            "control_seeds": CONTROL_SEEDS,
            "spurious_threshold": SPURIOUS_THRESHOLD,
        },
        "results": all_results,
        "global_assessment": global_assessment,
    }

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2, default=float)
    print(f"\n{'=' * 70}")
    print(f"JSON saved to {args.output_json}")

    # Save Markdown
    md_report = generate_markdown_report(all_results)
    with open(args.output_md, "w") as f:
        f.write(md_report)
    print(f"Markdown saved to {args.output_md}")

    print(f"Global assessment: {global_assessment}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
