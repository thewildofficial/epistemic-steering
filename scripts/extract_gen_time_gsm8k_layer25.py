import modal
from modal import App, Volume, Image

app = App("gen-time-gsm8k-layer25")

volume = Volume.from_name("epistemic-model-cache")
MODEL_PATH = "/vol/models/Qwen_Qwen3.5-4B"

image = Image.debian_slim().pip_install("torch", "transformers", "numpy", "tqdm", "accelerate")


@app.function(gpu="T4", volumes={"/vol": volume}, image=image, timeout=10_800)
def extract_gsm8k_layer25():
    import time
    import json
    import pickle
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Model loaded on {next(model.parameters()).device}")

    print("Loading questions from volume ...")
    questions_path = Path("/vol/results/probe_extract_results.jsonl")
    with open(questions_path, "r") as f:
        all_questions = [json.loads(line) for line in f if line.strip()]

    questions = [q for q in all_questions if q.get("dataset") == "gsm8k"]
    print(f"Loaded {len(questions)} GSM8K questions (of {len(all_questions)} total)")

    out_dir = Path("/vol/results/gen_time_gsm8k_layer25")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    target_layer = 25

    for i, q in enumerate(tqdm(questions, desc="Generating")):
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

        layer = model.model.layers[target_layer]
        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            sampled_hs = gen_hidden_states
            token_positions = list(range(len(gen_hidden_states)))

            result = {
                "question_id": q["question_id"],
                "dataset": q["dataset"],
                "subject": q.get("subject"),
                "hidden_states": sampled_hs,
                "token_positions": token_positions,
                "generated_text": generated_text,
                "correct": q["correct"],
                "correct_answer": q.get("correct_answer"),
                "n_generated_tokens": len(gen_hidden_states),
            }
            results.append(result)

        except Exception as exc:
            print(f"\nERROR on {q['question_id']}: {exc}")
            results.append(
                {
                    "question_id": q["question_id"],
                    "dataset": q["dataset"],
                    "error": str(exc),
                    "correct": q["correct"],
                }
            )
        finally:
            handle.remove()

        if (i + 1) % 50 == 0:
            batch_path = out_dir / f"gen_time_gsm8k_layer25_batch_{i + 1:04d}.pkl"
            with open(batch_path, "wb") as f:
                pickle.dump(results, f)
            elapsed = time.time() - start_time
            print(f"\nCheckpoint {i + 1}/{len(questions)} — elapsed {elapsed / 60:.1f} min")

    final_path = out_dir / "gen_time_gsm8k_layer25_all.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164

    print("\n" + "=" * 60)
    print(f"Done. {len(results)}/{len(questions)} questions processed.")
    print(f"Total GPU time: {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} hr)")
    print(f"Estimated Modal cost: ${cost_usd:.2f}")
    print("=" * 60)

    return {
        "n_questions": len(results),
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": cost_usd,
    }


@app.local_entrypoint()
def main():
    result = extract_gsm8k_layer25.remote()
    print(f"Remote result: {result}")

    import subprocess
    from pathlib import Path

    local_dir = Path("data/gen_time_gsm8k_layer25")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading results to {local_dir.resolve()} ...")
    cmd = [
        "modal", "volume", "get", "epistemic-model-cache",
        "results/gen_time_gsm8k_layer25/", str(local_dir), "--recursive",
    ]
    subprocess.run(cmd, check=False)
    print("Download complete.")
