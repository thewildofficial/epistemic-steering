"""Extract benchmark activations (layer 25) on Modal GPU.

Generates answers, checks correctness, and saves hidden states
for MATH, HumanEval, TriviaQA, and ARC-Challenge benchmarks.

Usage:
    modal run --detach scripts/extract_benchmark_activations.py

Results saved to Modal volume: /vol/results/benchmark_activations/
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path

import modal
from modal import App, Image, Volume

app = App("benchmark-extraction")

volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/models/Qwen_Qwen3.5-4B"
RESULTS_DIR = "/vol/results"
ACTIVATIONS_DIR = f"{RESULTS_DIR}/benchmark_activations"

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        "accelerate",
        "datasets",
    )
)

LAYER = 25
HIDDEN_DIM = 2560
BATCH_SIZE = 4
MAX_NEW_TOKENS = 256
N_QUESTIONS = 50


def normalize_math_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("$", "").replace(" ", "")
    text = text.replace("^", "").replace("_", "")
    return text


def extract_answer_math(generated: str) -> str:
    if not generated:
        return ""
    generated = generated.strip()
    lines = [l.strip() for l in generated.split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return generated


def check_correctness_math(generated: str, correct: str) -> bool:
    gen = extract_answer_math(generated)
    gen_norm = normalize_math_answer(gen)
    correct_norm = normalize_math_answer(correct)
    if not gen_norm or not correct_norm:
        return False
    if gen_norm == correct_norm:
        return True
    if gen_norm in correct_norm or correct_norm in gen_norm:
        return True
    try:
        if float(gen_norm) == float(correct_norm):
            return True
    except ValueError:
        pass
    return False


def extract_answer_arc(generated: str) -> str | None:
    if not generated:
        return None
    text = generated.strip()
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches:
        return matches[0]
    return None


def check_correctness_arc(generated: str, correct: str) -> bool:
    ans = extract_answer_arc(generated)
    if ans is None:
        return False
    return ans.upper() == correct.upper()


def check_correctness_triviaqa(generated: str, correct: str) -> bool:
    if not generated or not correct:
        return False
    gen = generated.strip().lower()
    corr = correct.strip().lower()
    if gen == corr:
        return True
    if corr in gen or gen in corr:
        return True
    gen_clean = re.sub(r"[^\w\s]", "", gen)
    corr_clean = re.sub(r"[^\w\s]", "", corr)
    if gen_clean == corr_clean:
        return True
    if corr_clean in gen_clean or gen_clean in corr_clean:
        return True
    return False


def check_correctness_humaneval(prompt: str, generated: str, correct: str) -> bool:
    body = generated.strip()
    if not body:
        return False

    body = re.sub(r"^```python\n?", "", body)
    body = re.sub(r"^```\n?", "", body)
    body = re.sub(r"```$", "", body)
    body = body.strip()

    sig_match = re.search(
        r"^(def\s+\w+\s*\([^)]*\)\s*(->\s*\w+\s*)?:\s*)",
        prompt,
        re.MULTILINE,
    )
    if not sig_match:
        return body.strip() == correct.strip()

    signature = sig_match.group(1).strip()
    docstring_match = re.search(r'(""".*?""")', prompt, re.DOTALL)
    docstring = docstring_match.group(1) if docstring_match else '""""""'

    full_code = (
        f"{signature}\n    {docstring}\n{textwrap.indent(body, '    ')}\n"
    )

    try:
        compile(full_code, "<string>", "exec")
    except SyntaxError:
        return False

    namespace = {}
    try:
        exec(full_code, namespace)
    except Exception:
        return False

    examples = re.findall(
        r">>>\s*(.+?)\n\s*(.+?)(?=\n\s*>>>|\n\s*\"\"\"|$)",
        docstring,
        re.DOTALL,
    )
    if not examples:
        return body.strip() == correct.strip()

    passed = 0
    for call_str, expected_str in examples:
        call_str = call_str.strip()
        expected_str = expected_str.strip()
        try:
            result = eval(call_str, namespace)
            if repr(result) == expected_str:
                passed += 1
        except Exception:
            pass

    return passed >= max(1, len(examples) // 2)


def load_benchmark_questions(benchmark_name: str, n: int):
    from datasets import load_dataset

    if benchmark_name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{min(n, 500)}]")
        questions = []
        for i, item in enumerate(ds):
            prompt = (
                "Solve the following math problem. Provide ONLY the final answer (a number or expression).\n\n"
                f"Problem: {item['problem']}\n\nAnswer:"
            )
            questions.append({
                "question_id": f"math_{i}",
                "dataset": "math",
                "prompt": prompt,
                "correct_answer": str(item.get("answer", "")),
            })
        return questions

    elif benchmark_name == "humaneval":
        ds = load_dataset("openai_humaneval", split=f"test[:{min(n, 164)}]")
        questions = []
        for i, item in enumerate(ds):
            prompt = (
                "Complete the following Python function. Provide ONLY the function body.\n\n"
                f"{item['prompt']}"
            )
            questions.append({
                "question_id": f"{item['task_id'].replace('/', '_')}",
                "dataset": "humaneval",
                "prompt": prompt,
                "correct_answer": item["canonical_solution"],
            })
        return questions

    elif benchmark_name == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{min(n, 200)}]")
        questions = []
        for i, item in enumerate(ds):
            answers = item["answer"]["value"]
            correct_answer = answers[0] if isinstance(answers, list) and answers else str(answers)
            prompt = (
                "Answer the following trivia question concisely.\n\n"
                f"Question: {item['question']}\n\nAnswer:"
            )
            questions.append({
                "question_id": f"triviaqa_{i}",
                "dataset": "triviaqa",
                "prompt": prompt,
                "correct_answer": str(correct_answer),
            })
        return questions

    elif benchmark_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=f"test[:{min(n, 1172)}]")
        questions = []
        for i, item in enumerate(ds):
            choice_text = ""
            choices = item["choices"]
            for j, choice in enumerate(choices["text"]):
                choice_text += f"{chr(ord('A') + j)}) {choice}\n"
            prompt = (
                "Answer the following multiple choice question. Respond with ONLY the letter.\n\n"
                f"Question: {item['question']}\n{choice_text}\nAnswer:"
            )
            questions.append({
                "question_id": f"arc_{item['id']}",
                "dataset": "arc_challenge",
                "prompt": prompt,
                "correct_answer": item["answerKey"],
            })
        return questions

    return []


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=7200,
)
def extract_all_benchmarks(benchmark_names: list[str], n_questions: int) -> list[dict]:
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("BENCHMARK ACTIVATION EXTRACTION")
    print("=" * 60)

    print("\n── Loading model ──")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"  Model loaded on {next(model.parameters()).device}")

    all_results = []

    for benchmark_name in benchmark_names:
        questions = load_benchmark_questions(benchmark_name, n_questions)
        if not questions:
            print(f"  Skipping {benchmark_name}: no questions loaded")
            continue

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"EXTRACTING: {benchmark_name} ({len(questions)} questions)")
        print(f"{'='*60}")

        act_dir = Path(ACTIVATIONS_DIR) / benchmark_name
        act_dir.mkdir(parents=True, exist_ok=True)
        results_path = Path(ACTIVATIONS_DIR) / f"{benchmark_name}_results.jsonl"
        with open(results_path, "w") as _:
            pass

        results = []
        num_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_iter = tqdm(
            range(0, len(questions), BATCH_SIZE),
            total=num_batches,
            desc=f"{benchmark_name} batches",
        )

        for batch_start in batch_iter:
            batch = questions[batch_start:batch_start + BATCH_SIZE]
            prompts = [q["prompt"] for q in batch]

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[LAYER][:, -1, :].cpu().numpy()

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs.input_ids.shape[1]

            for idx, q in enumerate(batch):
                qid = q["question_id"]
                hs = hidden_states[idx]
                np.save(act_dir / f"{qid}__layer_{LAYER}.npy", hs)

                generated_ids = gen_ids[idx, input_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                correct_answer = q.get("correct_answer", "")
                if benchmark_name == "math":
                    correct = check_correctness_math(generated_text, correct_answer)
                elif benchmark_name == "arc_challenge":
                    correct = check_correctness_arc(generated_text, correct_answer)
                elif benchmark_name == "triviaqa":
                    correct = check_correctness_triviaqa(generated_text, correct_answer)
                elif benchmark_name == "humaneval":
                    correct = check_correctness_humaneval(
                        q["prompt"], generated_text, correct_answer
                    )
                else:
                    correct = False

                result = {
                    "question_id": qid,
                    "dataset": benchmark_name,
                    "generated_text": generated_text,
                    "correct_answer": correct_answer,
                    "correct": correct,
                }
                results.append(result)
                with open(results_path, "a") as f:
                    f.write(json.dumps(result, default=float) + "\n")

        elapsed = time.time() - start_time
        n_correct = sum(1 for r in results if r["correct"])
        accuracy = n_correct / len(results) if results else 0.0

        print(f"\n── {benchmark_name} complete ──")
        print(f"  Questions: {len(results)}")
        print(f"  Correct: {n_correct} ({accuracy:.1%})")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Activations: {act_dir}")
        print(f"  Results: {results_path}")

        all_results.append({
            "benchmark": benchmark_name,
            "n_questions": len(results),
            "n_correct": n_correct,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
        })

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 60)

    return all_results


@app.local_entrypoint()
def main():
    extract_all_benchmarks.remote(
        ["math", "humaneval", "triviaqa", "arc_challenge"], N_QUESTIONS
    )


if __name__ == "__main__":
    print(
        "This script is designed to run on Modal GPU.\n"
        "To execute (detached):\n"
        "  modal run --detach scripts/extract_benchmark_activations.py\n"
        "Or attached:\n"
        "  modal run scripts/extract_benchmark_activations.py\n"
    )
