"""Combined Modal extraction for T6 (cross-benchmark) and T8 (generation-time probing).

Extracts layer 25 hidden states from Qwen3.5-4B for ALL benchmarks + GSM8K CoT generation
with corrected token limits.

Usage:
    uv run modal run --detach scripts/combined_modal_extraction.py

Results saved to Modal volume and downloaded to local data/.
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path

import modal
from modal import App, Image, Volume

app = App("combined-extraction-t6-t8")

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
    )
)

LAYER = 25
HIDDEN_DIM = 2560
BATCH_SIZE = 4

MAX_TOKENS = {
    "math": 1024,
    "humaneval": 256,
    "triviaqa": 128,
    "arc_challenge": 256,
    "gsm8k": 512,
}


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


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=10800,
)
def extract_all(
    benchmark_questions: dict[str, list[dict]],
    gsm8k_questions: list[dict],
) -> dict:
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("COMBINED EXTRACTION: T6 (benchmarks) + T8 (gen-time)")
    print("=" * 60)

    print("\nLoading model...")
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

    summary = {"benchmarks": {}, "gsm8k": {}}

    print("\n" + "=" * 60)
    print("PART 1: BENCHMARK EXTRACTION (T6)")
    print("=" * 60)

    benchmark_act_dir = Path(RESULTS_DIR) / "benchmark_activations"
    benchmark_act_dir.mkdir(parents=True, exist_ok=True)

    for benchmark_name, questions in benchmark_questions.items():
        if not questions:
            continue

        max_new = MAX_TOKENS.get(benchmark_name, 256)
        start_time = time.time()
        print(f"\n{benchmark_name}: {len(questions)} questions, max_new_tokens={max_new}")

        act_dir = benchmark_act_dir / benchmark_name
        act_dir.mkdir(parents=True, exist_ok=True)
        results_path = benchmark_act_dir / f"{benchmark_name}_results.jsonl"
        with open(results_path, "w") as _:
            pass

        results = []
        num_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_iter = tqdm(
            range(0, len(questions), BATCH_SIZE),
            total=num_batches,
            desc=f"{benchmark_name}",
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
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
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

        print(f"  Done: {len(results)} questions, {n_correct} correct ({accuracy:.1%}), {elapsed:.1f}s")
        summary["benchmarks"][benchmark_name] = {
            "n_questions": len(results),
            "n_correct": n_correct,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
        }

    print("\n" + "=" * 60)
    print("PART 2: GSM8K GENERATION-TIME EXTRACTION (T8)")
    print("=" * 60)

    gen_time_dir = Path(RESULTS_DIR) / "gen_time_activations"
    gen_time_dir.mkdir(parents=True, exist_ok=True)
    gen_time_meta_path = gen_time_dir / "gen_time_gsm8k_metadata.jsonl"
    with open(gen_time_meta_path, "w") as _:
        pass

    gsm8k_results = []
    gsm8k_start = time.time()

    for i, q in enumerate(tqdm(gsm8k_questions, desc="GSM8K gen-time")):
        prompt = q["prompt"]
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
                    max_new_tokens=MAX_TOKENS["gsm8k"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            qid = q["question_id"]
            for pos_idx, hs in enumerate(gen_hidden_states):
                np.save(gen_time_dir / f"{qid}_pos_{pos_idx}__layer_{LAYER}.npy", hs.squeeze())

            think_end_pos = None
            answer_start_pos = None
            if "</think>" in generated_text:
                think_part = generated_text.split("</think>")[0] + "</think>"
                think_tokens = tokenizer(think_part, add_special_tokens=False)["input_ids"]
                think_end_pos = len(think_tokens)
                answer_start_pos = think_end_pos

            result = {
                "question_id": qid,
                "dataset": "gsm8k",
                "subject": q.get("subject"),
                "generated_text": generated_text,
                "correct": q["correct"],
                "correct_answer": q.get("correct_answer"),
                "n_generated_tokens": len(gen_hidden_states),
                "think_end_pos": think_end_pos,
                "answer_start_pos": answer_start_pos,
            }
            gsm8k_results.append(result)

            with open(gen_time_meta_path, "a") as f:
                f.write(json.dumps(result, default=float) + "\n")

        except Exception as exc:
            print(f"\nERROR on {q['question_id']}: {exc}")
            gsm8k_results.append({
                "question_id": q["question_id"],
                "dataset": "gsm8k",
                "error": str(exc),
                "correct": q["correct"],
            })
        finally:
            handle.remove()

    gsm8k_elapsed = time.time() - gsm8k_start
    summary["gsm8k"] = {
        "n_questions": len(gsm8k_results),
        "elapsed_seconds": gsm8k_elapsed,
    }

    print(f"\n  GSM8K done: {len(gsm8k_results)} questions, {gsm8k_elapsed:.1f}s")

    total_elapsed = sum(b["elapsed_seconds"] for b in summary["benchmarks"].values()) + gsm8k_elapsed
    cost_usd = total_elapsed * 0.000164

    print("\n" + "=" * 60)
    print("ALL EXTRACTION COMPLETE")
    print("=" * 60)
    for name, info in summary["benchmarks"].items():
        print(f"  {name}: {info['n_questions']} questions, {info['n_correct']} correct ({info['accuracy']:.1%})")
    print(f"  gsm8k: {summary['gsm8k']['n_questions']} questions")
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    print(f"  Est. cost: ${cost_usd:.2f}")
    print("=" * 60)

    return summary


@app.local_entrypoint()
def main():
    import subprocess

    PROJECT_ROOT = Path("/Users/aban/drive/Projects/epistemic-steering")

    print("Loading benchmark prompts locally...")
    benchmark_questions = {}
    for name in ["math", "humaneval", "triviaqa", "arc_challenge"]:
        path = PROJECT_ROOT / "data" / "benchmark_prompts" / f"{name}_prompts.jsonl"
        questions = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        benchmark_questions[name] = questions
        print(f"  {name}: {len(questions)} questions")

    print("Loading GSM8K questions locally...")
    gsm8k_questions = []
    probe_path = PROJECT_ROOT / "data" / "probe_extract_results.jsonl"
    with open(probe_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("dataset") == "gsm8k":
                gsm8k_questions.append(q)
    print(f"  gsm8k: {len(gsm8k_questions)} questions")

    print("\nLaunching Modal GPU extraction...")
    result = extract_all.remote(benchmark_questions, gsm8k_questions)
    print(f"Remote result: {result}")

    print("\nDownloading benchmark activations...")
    local_benchmark_dir = PROJECT_ROOT / "data" / "benchmark_activations"
    local_benchmark_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "modal", "volume", "get", "epistemic-model-cache",
        "results/benchmark_activations/", str(local_benchmark_dir),
        "--recursive",
    ], check=False)

    print("\nDownloading generation-time activations...")
    local_gen_time_dir = PROJECT_ROOT / "data" / "gen_time_activations"
    local_gen_time_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "modal", "volume", "get", "epistemic-model-cache",
        "results/gen_time_activations/", str(local_gen_time_dir),
        "--recursive",
    ], check=False)

    print("\nDownload complete.")
    print(f"  Benchmarks: {local_benchmark_dir}")
    print(f"  Gen-time:   {local_gen_time_dir}")


if __name__ == "__main__":
    print(
        "This script is designed to run on Modal GPU.\n"
        "To execute (detached):\n"
        "  uv run modal run --detach scripts/combined_modal_extraction.py\n"
    )
