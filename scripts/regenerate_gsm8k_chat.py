"""Regenerate GSM8K answers with proper chat template prompting on Modal.

Uses Qwen3.5-4B-Instruct with tokenizer.apply_chat_template() to fix the
3.5% accuracy issue caused by raw text prompting.

Also extracts prefill hidden states at every layer for probe training.

Cost estimate:
    ~200 GSM8K questions × ~10s each ≈ 33 minutes on T4
    T4 cost: $0.000164/sec ≈ $0.59/hr
    Estimated total: $0.32 (plus overhead)
    Budget ceiling: $5

Output:
    Saved to Modal volume ``epistemic-model-cache`` under ``results/gsm8k_chat/``:
    - gsm8k_chat_results.jsonl  : generation results with accuracy
    - activations/              : per-question, per-layer prefill hidden states
"""

import modal
from modal import App, Volume, Image

app = App("gsm8k-regenerate")

volume = Volume.from_name("epistemic-model-cache")
MODEL_PATH = "/vol/models/Qwen_Qwen3.5-4B"

image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "numpy", "tqdm", "accelerate")
)


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=7_200,
)
def regenerate_gsm8k():
    import time
    import json
    import re
    import numpy as np
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    start_time = time.time()

    print("Loading model from volume ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {next(model.parameters()).device}")

    print("Loading questions from volume ...")
    with open("/vol/results/probe_extract_results.jsonl", "r") as f:
        all_qs = [json.loads(line) for line in f if line.strip()]

    gsm8k = [q for q in all_qs if q["dataset"] == "gsm8k"]
    print(f"Found {len(gsm8k)} GSM8K questions (of {len(all_qs)} total)")

    out_dir = Path("/vol/results/gsm8k_chat")
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_layers = model.config.num_hidden_layers

    for i, q in enumerate(tqdm(gsm8k, desc="Regenerating GSM8K")):
        messages = [{"role": "user", "content": q["prompt"]}]
        chat_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_input, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
            )
            prefill_states = {}
            for layer_idx in range(n_layers):
                hs = outputs.hidden_states[layer_idx + 1][:, -1, :].cpu().numpy()
                prefill_states[str(layer_idx)] = hs

        act_path = act_dir / f"{q['question_id']}_prefill.npz"
        np.savez(act_path, **prefill_states)

        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        generated = full_text[len(input_text):].strip()

        # Extract final answer: look for "answer" pattern or take last standalone number
        import re
        lower = generated.lower()
        # Try patterns like "final answer: X", "answer: X", "= X"
        answer_patterns = [
            r'final\s+answer\s*(?:is\s*)?:?\s*(\d+(?:\.\d+)?)',
            r'(?:the\s+)?answer\s*(?:is\s*)?:?\s*(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)\s*$',
            r'\*\*(\d+(?:\.\d+)?)\*\*',
        ]
        extracted = None
        for pat in answer_patterns:
            m = re.search(pat, lower)
            if m:
                extracted = m.group(1)
                break
        if extracted is None:
            # Fallback: last number on its own line
            lines = generated.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                nums = re.findall(r'\b\d+(?:\.\d+)?\b', line)
                if nums and len(line) < 50:  # short line with a number
                    extracted = nums[-1]
                    break
        if extracted is None:
            # Ultimate fallback
            nums = re.findall(r'\b\d+(?:\.\d+)?\b', generated)
            extracted = nums[-1] if nums else "?"
        extracted_answer = extracted

        correct = str(extracted_answer).strip() == str(q["correct_answer"]).strip()

        result = {
            "question_id": q["question_id"],
            "dataset": "gsm8k",
            "prompt": q["prompt"],
            "chat_formatted": chat_input[:200] + "..." if len(chat_input) > 200 else chat_input,
            "generated_text": generated,
            "correct_answer": q["correct_answer"],
            "model_answer": extracted_answer,
            "correct": correct,
        }
        results.append(result)

        if i < 3:
            print(f"\n[DEBUG Q{i}] Generated: {generated[:200]}...")
            print(f"[DEBUG Q{i}] Extracted: {extracted_answer}, Correct: {q['correct_answer']}, Match: {correct}")

        if (i + 1) % 20 == 0:
            acc = sum(1 for r in results if r["correct"]) / len(results)
            print(
                f"\nProgress: {i + 1}/{len(gsm8k)}, "
                f"accuracy: {acc:.1%} ({sum(1 for r in results if r['correct'])}/{len(results)})"
            )
            checkpoint_path = out_dir / f"gsm8k_chat_checkpoint_{i + 1:04d}.jsonl"
            with open(checkpoint_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = 100 * correct_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"Final accuracy: {correct_count}/{total} = {accuracy:.1f}%")
    print("=" * 60)

    results_path = out_dir / "gsm8k_chat_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164

    print(f"Saved results to {results_path}")
    print(f"Saved activations to {act_dir}")
    print(f"Total GPU time: {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} hr)")
    print(f"Estimated Modal cost: ${cost_usd:.2f}")

    return {
        "n_questions": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": cost_usd,
    }


@app.local_entrypoint()
def main():
    result = regenerate_gsm8k.remote()
    print(f"\nRemote result: {result}")

    import subprocess

    local_dir = Path("data/gsm8k_chat")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading results to {local_dir.resolve()} ...")
    cmd = [
        "modal", "volume", "get", "epistemic-model-cache",
        "results/gsm8k_chat/", str(local_dir), "--recursive",
    ]
    subprocess.run(cmd, check=False)
    print("Download complete.")
