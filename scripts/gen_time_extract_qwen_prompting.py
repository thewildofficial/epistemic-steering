"""Generation-time probing with official Qwen3.5 prompting.

Fixes underperformance by using correct chat template, thinking mode,
4-shot CoT, and proper sampling. Extracts layer-25 hidden states at every
token position, trains per-position LR probes, and reports AUROC trajectory.

Usage:  uv run modal run --detach scripts/gen_time_extract_qwen_prompting.py
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

import modal
from modal import App, Image, Volume

app = App("gen-time-qwen-prompting")

volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/models/Qwen_Qwen3.5-4B"
RESULTS_DIR = "/vol/results"

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        "accelerate",
        "scikit-learn",
        "scipy",
    )
)

LAYER = 25
HIDDEN_DIM = 2560
MAX_NEW_TOKENS = 4096


def load_fewshot_prompt(path: str, n_shots: int = 4) -> str:
    with open(path, "r") as f:
        text = f.read().strip()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return "\n\n".join(blocks[:n_shots])


def parse_gsm8k_question(raw_prompt: str) -> str:
    match = re.search(r"Question:\s*(.*?)\n\nAnswer:", raw_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_prompt.strip()


def format_gsm8k_chat_prompt(question: str, fewshot_text: str, tokenizer) -> str:
    user_content = (
        f"{fewshot_text}\n\n"
        f"Question: {question}\n"
        "Let's think step by step"
    )
    messages = [{"role": "user", "content": user_content}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return chat_prompt + "<think>\n"


def extract_answer_gsm8k(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def check_correctness(predicted: str | None, correct: str) -> bool:
    if predicted is None:
        return False
    try:
        return float(predicted) == float(correct)
    except (ValueError, TypeError):
        return predicted.strip() == correct.strip()


def compute_auroc_safe(scores, labels):
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def train_per_position_probes(results: list[dict]) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold

    valid = [r for r in results if "error" not in r and "hidden_states" in r]
    if not valid:
        return {"error": "No valid results for probe training"}

    pos_data: dict[int, list[tuple]] = {}
    for r in valid:
        for pos, hs in enumerate(r["hidden_states"]):
            pos_data.setdefault(pos, []).append((hs.squeeze(), r["correct"]))

    position_results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for pos in sorted(pos_data.keys()):
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
                Cs=10,
                cv=3,
                max_iter=1000,
                scoring="roc_auc",
                solver="lbfgs",
            )
            clf.fit(X[train_idx], y[train_idx])
            test_preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            train_pred = clf.predict_proba(X[train_idx])[:, 1]
            train_aurocs.append(compute_auroc_safe(train_pred, y[train_idx]))

        test_auroc = compute_auroc_safe(test_preds, y)
        if test_auroc != test_auroc:
            continue

        train_auroc_mean = float(np.nanmean(train_aurocs))
        gap = train_auroc_mean - test_auroc

        position_results.append(
            {
                "token_index": int(pos),
                "test_auroc": float(test_auroc),
                "train_auroc": train_auroc_mean,
                "overfitting_gap": float(gap),
                "n_samples": len(y),
                "n_correct": int(np.sum(y)),
            }
        )

        print(
            f"  pos {pos:4d}  n={len(y):4d}  "
            f"test={test_auroc:.3f}  train={train_auroc_mean:.3f}  "
            f"gap={gap:.3f}"
        )

    if not position_results:
        return {"error": "No positions with enough data"}

    optimal = max(position_results, key=lambda x: x["test_auroc"])
    avg_test = float(np.mean([p["test_auroc"] for p in position_results]))
    avg_train = float(np.mean([p["train_auroc"] for p in position_results]))
    avg_gap = float(np.mean([p["overfitting_gap"] for p in position_results]))

    return {
        "position_results": position_results,
        "optimal_position": optimal,
        "avg_metrics": {
            "test_auroc_mean": avg_test,
            "train_auroc_mean": avg_train,
            "overfitting_gap_mean": avg_gap,
        },
    }


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=10800,
)
def extract_and_train(questions: list[dict], fewshot_text: str) -> dict:
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("GEN-TIME EXTRACTION: Official Qwen3.5 Prompting")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"  Model loaded on {next(model.parameters()).device}")

    out_dir = Path(RESULTS_DIR) / "gen_time_qwen_prompt_activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    extraction_results: list[dict] = []
    start_time = time.time()

    for i, q in enumerate(tqdm(questions, desc="Generating")):
        chat_prompt = format_gsm8k_chat_prompt(
            q["question_text"], fewshot_text, tokenizer
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        gen_hidden_states: list[np.ndarray] = []
        step_counter = [0]

        def hook_fn(module, input, output):
            step_counter[0] += 1
            if step_counter[0] == 1:
                return
            hidden_states = output[0] if isinstance(output, tuple) else output
            hs = hidden_states[:, -1, :].detach().cpu().numpy()
            gen_hidden_states.append(hs)

        layer = model.model.layers[LAYER]
        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            think_end_pos = None
            if "</think>" in generated_text:
                think_part = (
                    generated_text.split("</think>")[0] + "</think>"
                )
                think_tokens = tokenizer(
                    think_part, add_special_tokens=False
                )["input_ids"]
                think_end_pos = len(think_tokens)

            token_labels = []
            for pos in range(len(gen_hidden_states)):
                if think_end_pos is not None and pos < think_end_pos:
                    token_labels.append("think")
                elif think_end_pos is not None:
                    token_labels.append("answer")
                else:
                    token_labels.append("think")

            model_answer = extract_answer_gsm8k(generated_text)
            correct = check_correctness(
                model_answer, q.get("correct_answer", "")
            )

            extraction_results.append(
                {
                    "question_id": q["question_id"],
                    "dataset": "gsm8k",
                    "hidden_states": gen_hidden_states,
                    "token_positions": list(range(len(gen_hidden_states))),
                    "token_labels": token_labels,
                    "generated_text": generated_text,
                    "model_answer": model_answer,
                    "correct": correct,
                    "correct_answer": q.get("correct_answer", ""),
                    "n_generated_tokens": len(gen_hidden_states),
                    "think_end_pos": think_end_pos,
                }
            )

        except Exception as exc:
            print(f"\nERROR on {q['question_id']}: {exc}")
            extraction_results.append(
                {
                    "question_id": q["question_id"],
                    "dataset": "gsm8k",
                    "error": str(exc),
                    "correct": False,
                }
            )
        finally:
            handle.remove()

        if (i + 1) % 50 == 0:
            batch_path = out_dir / f"batch_{i + 1:04d}.pkl"
            with open(batch_path, "wb") as f:
                pickle.dump(extraction_results, f)
            elapsed = time.time() - start_time
            n_ok = sum(1 for r in extraction_results if "error" not in r)
            print(
                f"\nCheckpoint {i + 1}/{len(questions)} — "
                f"elapsed {elapsed / 60:.1f} min  ok={n_ok}"
            )

    final_pkl = out_dir / "all.pkl"
    with open(final_pkl, "wb") as f:
        pickle.dump(extraction_results, f)

    print("\n" + "=" * 60)
    print("TRAINING PER-POSITION LR PROBES")
    print("=" * 60)
    probe_results = train_per_position_probes(extraction_results)

    probe_path = out_dir / "probe_results.json"
    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2, default=float)

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164
    n_ok = sum(1 for r in extraction_results if "error" not in r)
    n_correct = sum(
        1 for r in extraction_results if r.get("correct", False)
    )

    print("\n" + "=" * 60)
    print("EXTRACTION + PROBE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Questions:      {len(extraction_results)}")
    print(f"Successful:     {n_ok}")
    print(f"Correct:        {n_correct} / {n_ok}")
    if n_ok:
        print(f"Accuracy:       {n_correct / n_ok:.1%}")
    if probe_results.get("optimal_position"):
        opt = probe_results["optimal_position"]
        print(
            f"Optimal pos:    {opt['token_index']} "
            f"(AUROC {opt['test_auroc']:.3f})"
        )
    print(f"GPU time:       {elapsed / 60:.1f} min")
    print(f"Est. cost:      ${cost_usd:.2f}")
    print("=" * 60)

    return {
        "n_questions": len(extraction_results),
        "n_ok": n_ok,
        "n_correct": n_correct,
        "accuracy": n_correct / n_ok if n_ok else 0.0,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": cost_usd,
        "probe_results": probe_results,
    }


@app.local_entrypoint()
def main():
    import subprocess

    PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")
    DATA_DIR = PROJECT_ROOT / "data"

    fewshot_path = DATA_DIR / "gsm8k_fewshot_prompt.txt"
    fewshot_text = load_fewshot_prompt(str(fewshot_path), n_shots=4)
    print(f"Loaded {fewshot_text.count('Question:')} few-shot examples")

    questions = []
    probe_path = DATA_DIR / "probe_extract_results.jsonl"
    with open(probe_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("dataset") == "gsm8k":
                questions.append(
                    {
                        "question_id": q["question_id"],
                        "question_text": parse_gsm8k_question(q["prompt"]),
                        "correct_answer": q.get("correct_answer", ""),
                        "correct": q.get("correct", False),
                    }
                )

    questions = questions[:200]
    print(f"Selected {len(questions)} GSM8K questions for extraction")

    result = extract_and_train.remote(questions, fewshot_text)
    print(f"\nRemote result: {result}")

    local_act_dir = DATA_DIR / "gen_time_qwen_prompt_activations"
    local_act_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading activations to {local_act_dir} ...")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "epistemic-model-cache",
            "results/gen_time_qwen_prompt_activations/",
            str(local_act_dir),
            "--recursive",
        ],
        check=False,
    )

    ablation_dir = DATA_DIR / "ablation_results"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = ablation_dir / "gen_time_sweep.json"

    sweep = {
        "metadata": {
            "script": "gen_time_extract_qwen_prompting.py",
            "model": "Qwen/Qwen3.5-4B-Instruct",
            "layer": LAYER,
            "n_questions": len(questions),
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            },
            "prompting": {
                "chat_template": True,
                "add_generation_prompt": True,
                "thinking_mode": True,
                "few_shot": 4,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "extraction_summary": {
            "n_questions": result.get("n_questions", 0),
            "n_ok": result.get("n_ok", 0),
            "n_correct": result.get("n_correct", 0),
            "accuracy": result.get("accuracy", 0.0),
            "elapsed_seconds": result.get("elapsed_seconds", 0.0),
            "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
        },
        "probe_results": result.get("probe_results", {}),
    }

    with open(sweep_path, "w") as f:
        json.dump(sweep, f, indent=2, default=float)
    print(f"Saved sweep results to {sweep_path}")

    probe = result.get("probe_results", {})
    opt = probe.get("optimal_position", {})
    avg = probe.get("avg_metrics", {})

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Extraction accuracy: {result.get('accuracy', 0):.1%}")
    if opt:
        print(
            f"Optimal gen-time pos: {opt.get('token_index')}  "
            f"AUROC={opt.get('test_auroc', 0):.3f}"
        )
    if avg:
        print(f"Avg test AUROC:  {avg.get('test_auroc_mean', 0):.3f}")
        print(f"Avg train AUROC: {avg.get('train_auroc_mean', 0):.3f}")
        print(f"Avg overfitting: {avg.get('overfitting_gap_mean', 0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    print(
        "This script runs on Modal GPU.\n"
        "To execute (detached):\n"
        "  uv run modal run --detach scripts/gen_time_extract_qwen_prompting.py\n"
    )
