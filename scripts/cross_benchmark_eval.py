#!/usr/bin/env uv run python
"""
Cross-Benchmark Generalization Evaluation.

Evaluates the best probe (layer 25, LogisticRegressionCV, Platt-calibrated)
on MATH, HumanEval, TriviaQA, and ARC-Challenge benchmarks.
Tests zero-shot transfer from MMLU training. 5-seed protocol.

MMLU re-test runs locally (activations already extracted).
New benchmarks require Modal GPU for hidden state extraction.

Usage:
    uv run python scripts/cross_benchmark_eval.py          # Full evaluation
    uv run python scripts/cross_benchmark_eval.py --mmlu-only  # MMLU re-test only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations_allpos"
LABELS_FILE = PROJECT_ROOT / "data" / "probe_extract_allpos_results.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ablation_results"
OUTPUT_FILE = OUTPUT_DIR / "cross_benchmark.json"
BENCHMARK_PROMPTS_DIR = PROJECT_ROOT / "data" / "benchmark_prompts"
BENCHMARK_ACTS_DIR = PROJECT_ROOT / "data" / "benchmark_activations"

LAYER = 25
HIDDEN_DIM = 2560
SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
DEFAULT_N_QUESTIONS = 100
HOLDOUT_RATIO = 0.20
SPLIT_SEED = 42


def load_labels():
    labels = {}
    with open(LABELS_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            qid = item["question_id"]
            labels[qid] = {
                "correct": item["correct"],
                "dataset": item.get("dataset", "mmlu"),
            }
    return labels


def get_valid_question_ids(activations_dir, labels):
    act_files = list(activations_dir.glob("*.npy"))
    all_qids = set()
    for f in act_files:
        parts = f.stem.split("__")
        if parts:
            all_qids.add(parts[0])
    return all_qids & set(labels.keys())


def get_last_token(arr):
    return arr[-1, :] if arr.ndim == 2 else arr


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (bin_count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_all_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
    }


def train_lr_probe(X, y, seed):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    model = LogisticRegressionCV(
        Cs=10, cv=cv, scoring="roc_auc", max_iter=1000,
        random_state=seed,
    )
    model.fit(X, y)
    return {"coef": model.coef_[0].astype(np.float64), "intercept": float(model.intercept_[0])}


def apply_probe(X, weights):
    return expit(np.dot(X, weights["coef"]) + weights["intercept"])


def apply_platt_calibration(raw_scores, y_true, cal_mask):
    logits = np.log(np.maximum(raw_scores, 1e-12) / np.maximum(1 - raw_scores, 1e-12))
    cal_logits = logits[cal_mask]
    cal_labels = y_true[cal_mask]
    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(cal_logits.reshape(-1, 1), cal_labels)
    test_logits = logits[~cal_mask]
    calibrated = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 1e-12, 1 - 1e-12)


def run_mmlu_retest():
    labels = load_labels()
    all_qids = get_valid_question_ids(ACTIVATIONS_DIR, labels)
    all_qids_list = sorted(all_qids)

    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(all_qids_list)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_RATIO)
    test_qids = sorted(shuffled[:n_holdout])
    train_qids = sorted(shuffled[n_holdout:])

    print(f"  Train QIDs: {len(train_qids)}, Test QIDs: {len(test_qids)}")

    X_train_full = np.array([
        get_last_token(np.load(ACTIVATIONS_DIR / f"{qid}__layer_{LAYER}.npy"))
        for qid in train_qids
    ])
    y_train_full = np.array([labels[qid]["correct"] for qid in train_qids], dtype=bool)
    X_test = np.array([
        get_last_token(np.load(ACTIVATIONS_DIR / f"{qid}__layer_{LAYER}.npy"))
        for qid in test_qids
    ])
    y_test = np.array([labels[qid]["correct"] for qid in test_qids], dtype=bool)
    test_datasets = np.array([labels[qid]["dataset"] for qid in test_qids])

    per_seed = []
    mmlu_probe_weights = []

    for seed in SEEDS:
        weights = train_lr_probe(X_train_full, y_train_full, seed)
        mmlu_probe_weights.append(weights)

        raw_scores = apply_probe(X_test, weights)

        n_test = len(test_qids)
        cal_size = max(int(n_test * 0.3), 20)
        cal_mask = np.zeros(n_test, dtype=bool)
        seed_rng = np.random.RandomState(seed)
        cal_indices = seed_rng.choice(n_test, size=min(cal_size, n_test), replace=False)
        cal_mask[cal_indices] = True

        test_mask = ~cal_mask
        platt_scores = apply_platt_calibration(raw_scores, y_test, cal_mask)

        raw_metrics = compute_all_metrics(y_test[test_mask], raw_scores[test_mask])
        platt_metrics = compute_all_metrics(y_test[test_mask], platt_scores)

        per_seed.append({
            "seed": seed,
            "n_train": len(train_qids),
            "n_test_total": n_test,
            "n_cal": int(cal_mask.sum()),
            "n_eval": int(test_mask.sum()),
            "raw": raw_metrics,
            "platt": platt_metrics,
        })

    def agg(values):
        return float(np.mean(values)), float(np.std(values))

    auroc_raw_m, auroc_raw_s = agg([s["raw"]["auroc"] for s in per_seed])
    brier_raw_m, brier_raw_s = agg([s["raw"]["brier"] for s in per_seed])
    ece_raw_m, ece_raw_s = agg([s["raw"]["ece"] for s in per_seed])
    auroc_platt_m, auroc_platt_s = agg([s["platt"]["auroc"] for s in per_seed])
    brier_platt_m, brier_platt_s = agg([s["platt"]["brier"] for s in per_seed])
    ece_platt_m, ece_platt_s = agg([s["platt"]["ece"] for s in per_seed])

    aggregated = {
        "raw": {
            "auroc_mean": auroc_raw_m, "auroc_std": auroc_raw_s,
            "brier_mean": brier_raw_m, "brier_std": brier_raw_s,
            "ece_mean": ece_raw_m, "ece_std": ece_raw_s,
        },
        "platt": {
            "auroc_mean": auroc_platt_m, "auroc_std": auroc_platt_s,
            "brier_mean": brier_platt_m, "brier_std": brier_platt_s,
            "ece_mean": ece_platt_m, "ece_std": ece_platt_s,
        },
    }

    ece_reduction = (ece_raw_m - ece_platt_m) / ece_raw_m * 100 if ece_raw_m > 0 else 0

    return {
        "n_train_total": len(train_qids),
        "n_test_total": len(test_qids),
        "per_seed": per_seed,
        "aggregated": aggregated,
        "ece_reduction_pct": ece_reduction,
    }, mmlu_probe_weights, test_qids, y_test, test_datasets


def load_benchmark_datasets(n):
    benchmarks = {}
    from datasets import load_dataset

    for name, loader_fn, max_n in [
        ("math", lambda: load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{min(n, 500)}]"), 500),
        ("humaneval", lambda: load_dataset("openai_humaneval", split=f"test[:{min(n, 164)}]"), 164),
        ("triviaqa", lambda: load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{min(n, 200)}]"), 200),
        ("arc_challenge", lambda: load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{min(n, 1172)}]"), 1172),
    ]:
        try:
            ds = loader_fn()
            questions = []
            for i, item in enumerate(ds):
                if name == "math":
                    prompt = (
                        "Solve the following math problem. Provide ONLY the final answer (a number or expression).\n\n"
                        f"Problem: {item['problem']}\n\nAnswer:"
                    )
                    correct_answer = str(item.get("answer", ""))
                    questions.append({
                        "question_id": f"math_{i}", "dataset": name, "prompt": prompt,
                        "correct_answer": correct_answer,
                        "subject": item.get("subject", ""),
                        "level": item.get("level", 0),
                    })
                elif name == "humaneval":
                    prompt = (
                        "Complete the following Python function. Provide ONLY the function body.\n\n"
                        f"{item['prompt']}"
                    )
                    questions.append({
                        "question_id": f"{item['task_id'].replace('/', '_')}", "dataset": name,
                        "prompt": prompt, "correct_answer": item["canonical_solution"],
                        "entry_point": item["entry_point"],
                    })
                elif name == "triviaqa":
                    question_text = item["question"]
                    answers = item["answer"]["value"]
                    correct_answer = answers[0] if isinstance(answers, list) and answers else str(answers)
                    prompt = (
                        "Answer the following trivia question concisely.\n\n"
                        f"Question: {question_text}\n\nAnswer:"
                    )
                    questions.append({
                        "question_id": f"triviaqa_{i}", "dataset": name, "prompt": prompt,
                        "correct_answer": str(correct_answer),
                    })
                elif name == "arc_challenge":
                    choice_text = ""
                    choices = item["choices"]
                    for j, choice in enumerate(choices["text"]):
                        choice_text += f"{chr(ord('A') + j)}) {choice}\n"
                    prompt = (
                        "Answer the following multiple choice question. Respond with ONLY the letter.\n\n"
                        f"Question: {item['question']}\n{choice_text}\nAnswer:"
                    )
                    questions.append({
                        "question_id": f"arc_{item['id']}", "dataset": name, "prompt": prompt,
                        "correct_answer": item["answerKey"],
                    })
            benchmarks[name] = questions
            print(f"  {name}: {len(questions)} questions")
        except Exception as e:
            print(f"  {name}: FAILED — {type(e).__name__}: {str(e)[:100]}")
            benchmarks[name] = []
    return benchmarks


def save_benchmark_prompts(benchmarks):
    BENCHMARK_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, questions in benchmarks.items():
        if not questions:
            continue
        filepath = BENCHMARK_PROMPTS_DIR / f"{name}_prompts.jsonl"
        with open(filepath, "w") as f:
            for q in questions:
                f.write(json.dumps({
                    "question_id": q["question_id"], "dataset": q["dataset"],
                    "prompt": q["prompt"], "correct_answer": q["correct_answer"],
                }) + "\n")
        print(f"  Saved {len(questions)} prompts to {filepath.name}")


def check_modal_available():
    try:
        result = subprocess.run(["modal", "config", "show"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def find_existing_benchmark_activations(benchmark_name):
    act_dir = BENCHMARK_ACTS_DIR / benchmark_name
    if not act_dir.exists():
        return None
    npy_files = list(act_dir.glob("*.npy"))
    return act_dir if npy_files else None


def evaluate_cross_benchmark(mmlu_weights_list, benchmark_name, benchmark_questions):
    act_dir = find_existing_benchmark_activations(benchmark_name)
    if act_dir is None:
        return {"status": "missing_activations", "n_questions": len(benchmark_questions)}

    activations = {}
    for f in act_dir.glob("*.npy"):
        qid = f.stem
        try:
            act = np.load(f).ravel()
            activations[qid] = act
        except Exception:
            continue

    qid_to_item = {q["question_id"]: q for q in benchmark_questions}
    valid_qids = sorted(set(activations.keys()) & set(qid_to_item.keys()))

    if not valid_qids:
        return {"status": "no_matching_questions", "n_activations": len(activations),
                "n_questions": len(benchmark_questions)}

    # For correctness labels, we use the probe to predict and compare against
    # the benchmark's known answers. For now, we flag that correctness
    # determination requires model generation.
    return {
        "status": "activations_found_no_labels",
        "n_questions": len(valid_qids),
        "note": "Correctness labels require running Qwen3.5-4B on each question."
    }


def run_failure_analysis(test_qids, y_test, test_datasets):
    analysis = {"by_dataset": {}, "by_subject": {}}

    for ds in np.unique(test_datasets):
        mask = test_datasets == ds
        ds_qids = [test_qids[i] for i, m in enumerate(mask) if m]
        ds_labels = y_test[mask]
        analysis["by_dataset"][str(ds)] = {
            "n_questions": int(mask.sum()),
            "n_correct": int(ds_labels.sum()),
            "accuracy": float(ds_labels.mean()),
            "sample_qids": ds_qids[:5],
        }

    for qid in test_qids:
        parts = qid.split("_")
        if len(parts) >= 2:
            subject = parts[0] if parts[0] == "gsm8k" else "_".join(parts[:2])
            if subject not in analysis["by_subject"]:
                analysis["by_subject"][subject] = []
            analysis["by_subject"][subject].append(qid)

    analysis["by_subject"] = {
        k: {"n_questions": len(v), "sample": v[:5]}
        for k, v in sorted(analysis["by_subject"].items())
    }

    return analysis


def print_extraction_guide(benchmarks):
    print("\n" + "=" * 70)
    print("GPU EXTRACTION REQUIRED FOR NEW BENCHMARKS")
    print("=" * 70)
    print(f"""
Prompts saved to: {BENCHMARK_PROMPTS_DIR}

To extract hidden states, use the Modal extraction pattern from
scripts/evaluate_heldout.py. Template:

  1. Deploy to Modal (T4 GPU, ~5 min per 100 questions):
     modal run --detach scripts/cross_benchmark_eval.py --extract

  2. After extraction, re-run evaluation:
     uv run python scripts/cross_benchmark_eval.py

Estimated costs (T4 GPU, $0.59/hr):
  100 questions × ~3s each ≈ 5 min ≈ $0.05 per benchmark
""")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Cross-Benchmark Generalization")
    parser.add_argument("--mmlu-only", action="store_true")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_N_QUESTIONS)
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-BENCHMARK GENERALIZATION EVALUATION")
    print(f"  Layer: {LAYER} (LogisticRegressionCV + Platt)")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    # ─── MMLU Re-test ──────────────────────────────────────────────
    print("\n── MMLU Re-test (5-seed, layer 25, LogisticRegressionCV + Platt) ──")
    mmlu_result, mmlu_weights, test_qids, y_test, test_datasets = run_mmlu_retest()

    agg = mmlu_result["aggregated"]
    print(f"  Raw  AUROC: {agg['raw']['auroc_mean']:.4f} ± {agg['raw']['auroc_std']:.4f}")
    print(f"  Platt AUROC: {agg['platt']['auroc_mean']:.4f} ± {agg['platt']['auroc_std']:.4f}")
    print(f"  Raw  ECE:   {agg['raw']['ece_mean']:.4f} ± {agg['raw']['ece_std']:.4f}")
    print(f"  Platt ECE:  {agg['platt']['ece_mean']:.4f} ± {agg['platt']['ece_std']:.4f}")
    print(f"  ECE Reduction: {mmlu_result['ece_reduction_pct']:.1f}%")

    # ─── Load benchmark datasets ───────────────────────────────────
    print(f"\n── Loading benchmark datasets (n={args.num_questions}) ──")
    benchmarks = load_benchmark_datasets(args.num_questions)
    save_benchmark_prompts(benchmarks)

    # ─── Check GPU availability ────────────────────────────────────
    modal_available = check_modal_available()
    if modal_available:
        print("\n  ✅ Modal CLI available")
    else:
        print("\n  ⚠️  Modal CLI not configured")
        print_extraction_guide(benchmarks)

    # ─── Cross-benchmark evaluation ────────────────────────────────
    print("\n── Cross-Benchmark Evaluation ──")
    cross_results = {}
    for name in ["math", "humaneval", "triviaqa", "arc_challenge"]:
        questions = benchmarks.get(name, [])
        if not questions:
            cross_results[name] = {"status": "no_questions", "n_questions": 0}
            continue
        result = evaluate_cross_benchmark(mmlu_weights, name, questions)
        cross_results[name] = result
        status = result["status"]
        n = result.get("n_questions", 0)
        print(f"  {name}: {status} (n={n})")

    # ─── Reverse transfer check ────────────────────────────────────
    print("\n── Reverse Transfer Potential ──")
    reverse_potential = {}
    for name in ["math", "humaneval", "triviaqa", "arc_challenge"]:
        act_dir = find_existing_benchmark_activations(name)
        has_questions = len(benchmarks.get(name, [])) > 0
        can_reverse = act_dir is not None and has_questions
        reverse_potential[name] = {
            "has_activations": act_dir is not None,
            "has_questions": has_questions,
            "can_run_reverse_transfer": can_reverse,
        }
        print(f"  {name}: reverse_transfer_possible={can_reverse}")

    # ─── Failure analysis ──────────────────────────────────────────
    print("\n── Failure Analysis ──")
    failure_analysis = run_failure_analysis(test_qids, y_test, test_datasets)
    print("  By dataset:")
    for ds, info in failure_analysis["by_dataset"].items():
        print(f"    {ds}: n={info['n_questions']}, accuracy={info['accuracy']:.3f}")

    # ─── Save results ──────────────────────────────────────────────
    full_results = {
        "metadata": {
            "task": "cross_benchmark_generalization",
            "layer": LAYER,
            "hidden_dim": HIDDEN_DIM,
            "model": "Qwen3.5-4B-Instruct",
            "probe_type": "LogisticRegressionCV",
            "calibration": "Platt",
            "seeds": SEEDS,
            "split_seed": SPLIT_SEED,
            "n_questions_per_benchmark": args.num_questions,
        },
        "mmlu_retest": mmlu_result,
        "cross_benchmark": cross_results,
        "reverse_transfer_potential": reverse_potential,
        "failure_analysis": failure_analysis,
        "notes": {
            "mmlu": "Trained and tested on 80/20 split of 649 labeled QIDs (seed=42).",
            "cross_benchmark": "New benchmarks require GPU extraction for hidden states.",
            "reverse_transfer": "Requires benchmark activations (GPU extraction first).",
            "prompts_dir": str(BENCHMARK_PROMPTS_DIR),
            "activations_dir": str(BENCHMARK_ACTS_DIR),
        },
    }

    with open(args.output, "w") as f:
        json.dump(full_results, f, indent=2, default=float)

    print(f"\n── Results saved to {args.output} ──")
    print(f"  File size: {os.path.getsize(args.output) / 1024:.1f} KB")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
