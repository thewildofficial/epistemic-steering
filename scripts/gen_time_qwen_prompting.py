from __future__ import annotations
import json, os, re, pickle
from pathlib import Path
import modal
from modal import App, Image, Volume

app = App("gen-time-qwen-prompting")
volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/models/Qwen_Qwen3.5-4B"
RESULTS_DIR = "/vol/results"
GEN_TIME_DIR = f"{RESULTS_DIR}/gen_time_gsm8k_layer25_qwen_prompt"
FEWSHOT_PATH = "/vol/results/gsm8k_fewshot_prompt.txt"

LAYER = 25
HIDDEN_DIM = 2560
MAX_NEW_TOKENS = 4096
GPU_CONFIG = ["RTX-PRO-6000", "H100", "A100-80GB", "L40S"]
QUESTION_RESULTS_DIR = f"{GEN_TIME_DIR}/questions"
COMMIT_EVERY = 5

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.11.0",
        "transformers>=5.8.0",
        "triton>=3.6.0",
        "numpy",
        "tqdm",
        "scikit-learn",
        "scipy",
        "pandas",
        "accelerate",
        "ninja",
        "packaging",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)


def _safe_qid(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(qid))


def _atomic_pickle(obj, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _normalize_result(result: dict) -> dict:
    result.setdefault("dataset", "gsm8k")
    n_tokens = int(result.get("n_tokens", len(result.get("hidden_states", []))))
    result.setdefault("n_tokens", n_tokens)
    result.setdefault("token_positions", list(range(n_tokens)))
    return result


def _load_completed_results() -> dict:
    all_results = {}
    all_pkl_path = f"{GEN_TIME_DIR}/all_results.pkl"
    if os.path.exists(all_pkl_path):
        with open(all_pkl_path, "rb") as f:
            loaded = pickle.load(f)
        all_results.update(
            {qid: _normalize_result(result) for qid, result in loaded.items()}
        )

    question_dir = Path(QUESTION_RESULTS_DIR)
    if question_dir.exists():
        for path in question_dir.glob("*.pkl"):
            with open(path, "rb") as f:
                result = _normalize_result(pickle.load(f))
            all_results[result["question_id"]] = result
    return all_results


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU_CONFIG, timeout=36000)
def extract_gen_time_qwen_prompting(shard_index: int = 0, num_shards: int = 1):
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    for package_name in ("fla", "causal_conv1d"):
        try:
            __import__(package_name)
            print(f"{package_name}: available")
        except Exception as exc:
            print(f"{package_name}: unavailable ({exc})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    eos_token_ids = {tokenizer.eos_token_id}
    unk_token_id = tokenizer.unk_token_id
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(stop_token)
        if token_id is not None and token_id != unk_token_id:
            eos_token_ids.add(token_id)
    eos_token_ids = sorted(eos_token_ids)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    with open(FEWSHOT_PATH) as f:
        fewshot_prompt = f.read()

    train_path = f"{RESULTS_DIR}/probe_extract_results.jsonl"
    gsm8k_questions = [
        json.loads(line) for line in open(train_path)
        if json.loads(line).get("dataset") == "gsm8k"
    ][:200]

    os.makedirs(GEN_TIME_DIR, exist_ok=True)
    os.makedirs(QUESTION_RESULTS_DIR, exist_ok=True)

    # Resume from last checkpoint if exists
    all_results = _load_completed_results()
    completed_ids = set(all_results.keys())
    shard_questions = [
        q for idx, q in enumerate(gsm8k_questions)
        if idx % num_shards == shard_index
    ]
    remaining = [q for q in shard_questions if q["question_id"] not in completed_ids]
    print(
        f"Shard {shard_index + 1}/{num_shards}: "
        f"{len(completed_ids)} completed globally, "
        f"{len(remaining)}/{len(shard_questions)} remaining in shard"
    )

    processed = 0
    for i, q in enumerate(tqdm(remaining, desc="GSM8K gen-time")):
        qid = q["question_id"]
        messages = [{"role": "user", "content": f"{fewshot_prompt}\nQuestion: {q['prompt']}\nLet's think step by step"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + "<think>\n"

        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        prefill_len = input_ids.shape[1]

        hidden_state_tensors = []
        step_counter = [0]

        def hook_fn(module, input, output):
            step_counter[0] += 1
            if step_counter[0] == 1:  # Skip prefill forward pass
                return
            h = output[0] if isinstance(output, tuple) else output
            hidden_state_tensors.append(
                h[:, -1, :].detach().squeeze(0).to(torch.float16).clone()
            )

        layer = model.model.layers[LAYER]
        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=eos_token_ids,
                )
        finally:
            handle.remove()

        if hidden_state_tensors:
            hidden_states = torch.stack(hidden_state_tensors).float().cpu().numpy()
        else:
            hidden_states = np.empty((0, HIDDEN_DIM), dtype=np.float32)
        n_tokens = int(hidden_states.shape[0])

        generated_ids = output_ids[0, prefill_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        def parse_num(s):
            if isinstance(s, (int, float)):
                return float(s)
            return float(str(s).replace(",", "").strip())

        answer_part = generated_text.split("The answer is")[-1] if "The answer is" in generated_text else generated_text
        answer_match = re.findall(r'(-?\d+(?:,\d{3})*\.?\d*)', answer_part)
        predicted = parse_num(answer_match[-1]) if answer_match else None
        correct = bool(predicted is not None and abs(predicted - parse_num(q["correct_answer"])) < 0.001)

        all_results[qid] = {
            "question_id": qid,
            "dataset": "gsm8k",
            "correct": correct,
            "correct_answer": q["correct_answer"],
            "model_answer": predicted,
            "generated_text": generated_text,
            "hidden_states": hidden_states,
            "n_tokens": n_tokens,
            "token_positions": list(range(n_tokens)),
        }

        pred_str = str(predicted) if predicted is not None else "?"
        gold_str = q["correct_answer"]
        status = "✓" if correct else "✗"
        completed_now = len(all_results)
        running_acc = sum(bool(r["correct"]) for r in all_results.values()) / completed_now
        print(f"  [{completed_now}/200] {status} pred={pred_str} gold={gold_str} | acc={running_acc:.1%} | tokens={n_tokens}")

        result_path = f"{QUESTION_RESULTS_DIR}/{_safe_qid(qid)}.pkl"
        _atomic_pickle(all_results[qid], result_path)

        processed += 1
        if processed % COMMIT_EVERY == 0:
            volume.commit()

    volume.commit()
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "processed": processed,
        "known_completed": len(all_results),
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=1800)
def merge_gen_time_qwen_prompting():
    import numpy as np

    all_results = _load_completed_results()
    _atomic_pickle(all_results, f"{GEN_TIME_DIR}/all_results.pkl")
    n = len(all_results)
    correct = sum(bool(r["correct"]) for r in all_results.values())
    summary = {"total": n, "correct": correct, "accuracy": correct/n if n else 0,
               "avg_tokens": float(np.mean([r["n_tokens"] for r in all_results.values()]))}
    with open(f"{GEN_TIME_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    print(f"Merged {n} results. Accuracy={summary['accuracy']:.1%}, avg_tokens={summary['avg_tokens']:.1f}")
    return summary


@app.local_entrypoint()
def main(shards: int = 1):
    if shards < 1:
        raise ValueError("shards must be >= 1")

    if shards == 1:
        print(extract_gen_time_qwen_prompting.remote(0, 1))
    else:
        shard_indices = list(range(shards))
        print(f"Launching {shards} GPU shards...")
        for result in extract_gen_time_qwen_prompting.map(
            shard_indices,
            [shards] * shards,
            order_outputs=False,
        ):
            print(result)
    print(merge_gen_time_qwen_prompting.remote())
